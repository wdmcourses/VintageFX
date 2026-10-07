
from __future__ import annotations

import contextvars

import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter1d, uniform_filter1d

EPS = 1e-12

# Global trim for every characteristic noise layer presets add
# (two trims of 25 %: 0.75 * 0.75 = 0.5625 of the original level).
NOISE_TRIM = 0.5625


NOISE_ENABLED: contextvars.ContextVar = contextvars.ContextVar("VFX_NOISE", default=True)
HUM_ENABLED: contextvars.ContextVar = contextvars.ContextVar("VFX_HUM", default=True)


def noise_on() -> bool:
    return bool(NOISE_ENABLED.get())


def lerp(a: float, b: float, s: float) -> float:
    return a + (b - a) * s


def db(x_db: float) -> float:
    return float(10.0 ** (x_db / 20.0))


def wide_band(sr: int) -> tuple[float, float]:
    return 18.0, min(sr * 0.475, 20000.0)


def as_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[np.newaxis, :]
    return x


def to_mono(x: np.ndarray) -> np.ndarray:
    if x.shape[0] > 1:
        return x.mean(axis=0, keepdims=True)
    return x


def collapse_mono(x: np.ndarray) -> np.ndarray:
    if x.shape[0] == 1:
        return x
    m = x.mean(axis=0, keepdims=True)
    return np.repeat(m, x.shape[0], axis=0)


def normalize(x: np.ndarray, peak: float = 0.95) -> np.ndarray:
    m = float(np.max(np.abs(x))) if x.size else 0.0
    if m < EPS:
        return x
    return x * (peak / m)


def _guard_length(x: np.ndarray, kind: str, **kw) -> bool:
    return x.shape[-1] > 512


def _sos_apply(sos, x: np.ndarray, zero_phase: bool = True) -> np.ndarray:
    n = x.shape[-1]
    padlen = 3 * (2 * len(sos) + 1)
    if zero_phase and n > padlen:
        return signal.sosfiltfilt(sos, x, axis=-1)
    return signal.sosfilt(sos, x, axis=-1)


def bandpass(x: np.ndarray, sr: int, lo: float, hi: float, order: int = 2) -> np.ndarray:
    nyq = sr / 2.0
    lo = max(float(lo), 3.0)
    hi = min(float(hi), nyq * 0.98)
    if hi <= lo * 1.05:
        return x
    if not _guard_length(x, "bp"):
        return x
    sos = signal.butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")
    return _sos_apply(sos, x)


def lowpass(x: np.ndarray, sr: int, hi: float, order: int = 2) -> np.ndarray:
    nyq = sr / 2.0
    hi = min(float(hi), nyq * 0.98)
    if hi >= nyq * 0.97 or not _guard_length(x, "lp"):
        return x
    sos = signal.butter(order, hi / nyq, btype="low", output="sos")
    return _sos_apply(sos, x)


def highpass(x: np.ndarray, sr: int, lo: float, order: int = 2,
             zero_phase: bool = True) -> np.ndarray:
    nyq = sr / 2.0
    lo = max(float(lo), 2.0)
    if lo <= 3.0 or not _guard_length(x, "hp"):
        return x
    sos = signal.butter(order, lo / nyq, btype="high", output="sos")
    return _sos_apply(sos, x, zero_phase)


def _biquad_rbj(x: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    b = b / a[0]
    a = a / a[0]
    return signal.lfilter(b, a, x, axis=-1)


def peak_eq(x: np.ndarray, sr: int, f0: float, gain_db: float, Q: float = 1.0) -> np.ndarray:
    if abs(gain_db) < 0.05:
        return x
    f0 = min(max(f0, 5.0), sr * 0.48)
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / sr
    alpha = np.sin(w0) / (2.0 * Q)
    cw = np.cos(w0)
    b = np.array([1 + alpha * A, -2 * cw, 1 - alpha * A])
    a = np.array([1 + alpha / A, -2 * cw, 1 - alpha / A])
    return _biquad_rbj(x, b, a)


def lowshelf(x: np.ndarray, sr: int, f0: float, gain_db: float, Q: float = 0.707) -> np.ndarray:
    if abs(gain_db) < 0.05:
        return x
    f0 = min(max(f0, 5.0), sr * 0.48)
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / sr
    alpha = np.sin(w0) / (2.0 * Q)
    cw = np.cos(w0)
    sa = 2.0 * np.sqrt(A) * alpha
    b = A * np.array([(A + 1) - (A - 1) * cw + sa,
                      2 * ((A - 1) - (A + 1) * cw),
                      (A + 1) - (A - 1) * cw - sa])
    a = np.array([(A + 1) + (A - 1) * cw + sa,
                  -2 * ((A - 1) + (A + 1) * cw),
                  (A + 1) + (A - 1) * cw - sa])
    return _biquad_rbj(x, b, a)


def highshelf(x: np.ndarray, sr: int, f0: float, gain_db: float, Q: float = 0.707) -> np.ndarray:
    if abs(gain_db) < 0.05:
        return x
    f0 = min(max(f0, 5.0), sr * 0.48)
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / sr
    alpha = np.sin(w0) / (2.0 * Q)
    cw = np.cos(w0)
    sa = 2.0 * np.sqrt(A) * alpha
    b = A * np.array([(A + 1) + (A - 1) * cw + sa,
                      -2 * ((A - 1) + (A + 1) * cw),
                      (A + 1) + (A - 1) * cw - sa])
    a = np.array([(A + 1) - (A - 1) * cw + sa,
                  2 * ((A - 1) - (A + 1) * cw),
                  (A + 1) - (A - 1) * cw - sa])
    return _biquad_rbj(x, b, a)


def saturate(x: np.ndarray, drive: float = 2.0, bias: float = 0.0, mix: float = 1.0) -> np.ndarray:
    if drive <= 1.001 and abs(bias) < 1e-6:
        return x
    peak = float(np.max(np.abs(x))) + EPS
    xn = x / peak
    y = np.tanh((xn + bias) * drive) - np.tanh(np.array(bias) * drive)
    y = y / (np.tanh(drive * (1.0 + abs(bias))) + EPS)
    y = y * peak
    if mix >= 0.999:
        return y
    return (1.0 - mix) * x + mix * y


def compress(x: np.ndarray, sr: int, thresh_db: float = -18.0,
             ratio: float = 3.0, amount: float = 1.0, win: float = 0.012) -> np.ndarray:
    if amount <= 0.01:
        return x
    w = max(1, int(sr * win))
    ref = float(np.max(np.abs(x))) + EPS
    if x.ndim == 2:
        env = np.max(np.abs(x) / ref, axis=0)
    else:
        env = np.abs(x) / ref
    env = uniform_filter1d(env, size=w, mode="nearest")
    env_db = 20.0 * np.log10(env + EPS)
    over = np.maximum(env_db - thresh_db, 0.0)
    gr_db = -over * (1.0 - 1.0 / ratio)
    gr_db = uniform_filter1d(gr_db, size=w, mode="nearest")
    gain = db(0.0) * np.power(10.0, (gr_db * amount) / 20.0)
    if x.ndim == 2:
        return x * gain[np.newaxis, :]
    return x * gain


def pink_noise(n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    if n < 8:
        return np.zeros(n)
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    f = np.fft.rfftfreq(n, 1.0 / sr)
    scale = np.ones_like(f)
    scale[1:] = 1.0 / np.sqrt(f[1:])
    spec *= scale
    spec[0] = 0.0
    out = np.fft.irfft(spec, n)
    m = float(np.max(np.abs(out))) + EPS
    return out / m


def shaped_noise(n: int, sr: int, rng: np.random.Generator, kind: str = "white") -> np.ndarray:
    x = rng.standard_normal(n)
    if kind == "pink":
        return pink_noise(n, sr, rng)
    return x / (float(np.max(np.abs(x))) + EPS)


def hiss_noise(n: int, sr: int, rng: np.random.Generator, lo: float,
               hi: float, amp: float) -> np.ndarray:
    if amp <= 0 or not noise_on():
        return np.zeros(n)
    x = shaped_noise(n, sr, rng, "white")
    x = bandpass(x, sr, lo, hi, order=2)
    m = float(np.max(np.abs(x))) + EPS
    return x / m * amp


def hum_noise(n: int, sr: int, rng: np.random.Generator, amp: float,
              freq: float = 50.0, harmonics: int = 4) -> np.ndarray:
    if amp <= 0 or not noise_on():
        return np.zeros(n)
    t = np.arange(n, dtype=np.float64) / sr
    out = np.zeros(n)
    for h in range(1, harmonics + 1):
        f = freq * h
        if f >= sr * 0.48:
            break
        out += np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi)) / h
    m = float(np.max(np.abs(out))) + EPS
    return out / m * amp


def fading_envelope(n: int, sr: int, rng: np.random.Generator, rate_hz: float,
                    depth_db: float) -> np.ndarray:
    if rate_hz <= 0 or depth_db <= 0:
        return np.ones(n)
    w = rng.standard_normal(n)
    cut = min(rate_hz, sr * 0.4)
    sos = signal.butter(2, cut / (sr / 2.0), btype="low", output="sos")
    w = _sos_apply(sos, w)
    m = float(np.max(np.abs(w))) + EPS
    w = w / m
    return np.power(10.0, (depth_db * w) / 20.0)


def whistle_tone(n: int, sr: int, rng: np.random.Generator, f0: float = 1200.0,
                 sweep: float = 500.0, rate: float = 0.07) -> np.ndarray:
    if not noise_on():
        return np.zeros(n)
    t = np.arange(n, dtype=np.float64) / sr
    f = f0 + sweep * np.sin(2 * np.pi * rate * t + rng.uniform(0, 2 * np.pi))
    f = np.clip(f, 60.0, sr * 0.45)
    phase = 2.0 * np.pi * np.cumsum(f) / sr
    return np.sin(phase)


def dropout_env(n: int, sr: int, rng: np.random.Generator, density: float,
                depth_db: float, seed: int = 3) -> np.ndarray:
    if density <= 0:
        return np.ones(n)
    dur = n / sr
    count = int(rng.poisson(density * dur))
    if count == 0:
        return np.ones(n)
    imp = np.zeros(n)
    for _ in range(count):
        pos = int(rng.integers(0, n - 1))
        length = int(sr * rng.uniform(0.002, 0.03))
        length = max(2, min(length, n - pos))
        imp[pos:pos + length] += 1.0
    k = int(sr * 0.004)
    kernel = np.exp(-np.arange(k) / (k / 3.0))
    dip = np.convolve(imp, kernel, mode="same")
    dip = np.clip(dip, 0.0, 1.0)
    floor = db(-abs(depth_db))
    return floor + (1.0 - floor) * (1.0 - dip)


def _add(x: np.ndarray, noise_fn, rng: np.random.Generator) -> np.ndarray:
    n = x.shape[-1]
    for c in range(x.shape[0]):
        x[c] += noise_fn(n) * NOISE_TRIM
    return x


def _asym_smooth(g: np.ndarray, sr: int, attack_ms: float = 5.0,
                 release_ms: float = 150.0) -> np.ndarray:
    g = np.asarray(g, dtype=np.float64)
    if g.size == 0:
        return g
    a_att = float(np.exp(-1.0 / max(sr * attack_ms / 1000.0, 1.0)))
    a_rel = float(np.exp(-1.0 / max(sr * release_ms / 1000.0, 1.0)))
    b_f, a_f = np.array([1.0 - a_att]), np.array([1.0, -a_att])
    b_s, a_s = np.array([1.0 - a_rel]), np.array([1.0, -a_rel])
    g0 = float(g[0])
    gf = signal.lfilter(b_f, a_f, g, zi=signal.lfilter_zi(b_f, a_f) * g0)[0]
    gs = signal.lfilter(b_s, a_s, g, zi=signal.lfilter_zi(b_s, a_s) * g0)[0]
    return np.minimum(gf, gs)


def compressor(x: np.ndarray, sr: int, thresh_db: float = -18.0, ratio: float = 3.0,
               attack_ms: float = 10.0, release_ms: float = 150.0,
               makeup_db: float = 0.0, knee_db: float = 6.0,
               amount: float = 1.0, win_ms: float = 8.0) -> np.ndarray:
    if amount <= 0.01 or ratio <= 1.001:
        return x
    if x.ndim == 2:
        env_src = np.max(np.abs(x), axis=0)
    else:
        env_src = np.abs(x)
    w = max(3, int(sr * win_ms / 1000.0))
    if w % 2 == 0:
        w += 1
    env = uniform_filter1d(env_src, size=w, mode="nearest")
    ref = float(np.max(env)) + EPS
    lvl = 20.0 * np.log10(env / ref + EPS)

    slope = 1.0 - 1.0 / ratio
    over = lvl - thresh_db
    gr = np.zeros_like(over)
    half = knee_db * 0.5
    big = over >= half
    mid = (over > -half) & ~big
    gr[big] = slope * over[big]
    if knee_db > 0.0:
        gr[mid] = slope * (over[mid] + half) ** 2 / (2.0 * knee_db)
    else:
        gr[mid] = slope * over[mid]

    gain = np.power(10.0, (-gr * amount) / 20.0)
    gain = _asym_smooth(gain, sr, attack_ms, release_ms)
    out = x * gain[np.newaxis, :] if x.ndim == 2 else x * gain
    if makeup_db:
        out = out * db(makeup_db)
    return out


def limiter(x: np.ndarray, sr: int, ceiling: float = 0.97,
            lookahead_ms: float = 3.0, release_ms: float = 80.0) -> np.ndarray:
    peak = np.max(np.abs(x), axis=0) if x.ndim == 2 else np.abs(x)
    la = max(1, int(sr * lookahead_ms / 1000.0))
    peak_la = maximum_filter1d(peak, size=2 * la + 1, mode="nearest")
    gain = np.minimum(1.0, ceiling / np.maximum(peak_la, EPS))
    gain = _asym_smooth(gain, sr, attack_ms=0.4, release_ms=release_ms)
    return x * gain[np.newaxis, :] if x.ndim == 2 else x * gain


def amp_sag(x: np.ndarray, sr: int, amount: float = 1.0, attack_ms: float = 25.0,
            release_ms: float = 220.0) -> np.ndarray:
    if amount <= 0.01:
        return x
    env = np.max(np.abs(x), axis=0) if x.ndim == 2 else np.abs(x)
    w = max(3, int(sr * 0.006))
    env = uniform_filter1d(env, size=w, mode="nearest")
    ref = float(np.percentile(env, 88.0)) + EPS
    lvl = env / ref
    gain = 1.0 / (1.0 + amount * np.maximum(lvl - 0.35, 0.0))
    gain = _asym_smooth(gain, sr, attack_ms, release_ms)
    return x * gain[np.newaxis, :] if x.ndim == 2 else x * gain


def agc(x: np.ndarray, sr: int, amount: float = 1.0, window_ms: float = 250.0,
        max_gain_db: float = 10.0) -> np.ndarray:
    if amount <= 0.01:
        return x
    env = np.max(np.abs(x), axis=0) if x.ndim == 2 else np.abs(x)
    w = max(3, int(sr * window_ms / 1000.0))
    env = uniform_filter1d(env, size=w, mode="nearest")
    ref = float(np.percentile(env, 75.0)) + EPS
    gain = ref / (env + EPS)
    gain = np.clip(gain, db(-max_gain_db), db(max_gain_db))
    gain = uniform_filter1d(gain, size=w, mode="nearest")
    gain = 1.0 + amount * (gain - 1.0)
    return x * gain[np.newaxis, :] if x.ndim == 2 else x * gain


def hard_clip(x: np.ndarray, drive: float = 2.0, mix: float = 1.0) -> np.ndarray:
    if drive <= 1.001 or mix <= 0.001:
        return x
    peak = float(np.max(np.abs(x))) + EPS
    xn = x / peak
    y = np.clip(xn * drive, -1.0, 1.0) * (peak / drive)
    if mix >= 0.999:
        return y
    return (1.0 - mix) * x + mix * y


def crossover(x: np.ndarray, threshold: float = 0.05, mix: float = 1.0) -> np.ndarray:
    if threshold <= 0.0 or mix <= 0.001:
        return x
    peak = float(np.max(np.abs(x))) + EPS
    t = threshold * peak
    y = np.sign(x) * np.maximum(np.abs(x) - t, 0.0) * (peak / (peak - t + EPS))
    if mix >= 0.999:
        return y
    return (1.0 - mix) * x + mix * y


def tape_soft(x: np.ndarray, drive: float = 1.6, bias: float = 0.04,
              mix: float = 1.0) -> np.ndarray:
    return saturate(x, drive=drive, bias=bias, mix=mix)


def reverb(x: np.ndarray, sr: int, decay: float = 1.0, brightness: float = 5000.0,
           mix: float = 0.25, predelay_ms: float = 15.0, seed: int = 5) -> np.ndarray:
    if mix <= 0.005:
        return x
    n_ir = int(sr * max(0.1, decay))
    if n_ir < 64:
        return x
    rng = np.random.default_rng(seed)
    ir = rng.standard_normal(n_ir)
    t = np.arange(n_ir, dtype=np.float64) / sr
    ir *= np.exp(-t * (6.908 / decay))
    ir = lowpass(ir, sr, brightness, order=2)
    pre = int(sr * predelay_ms / 1000.0)
    if pre > 0:
        ir = np.concatenate([np.zeros(pre), ir])
    ir /= float(np.sqrt(np.sum(ir ** 2))) + EPS
    out = np.empty_like(x)
    for c in range(x.shape[0]):
        wet = signal.fftconvolve(x[c], ir, mode="full")[: x.shape[-1]]
        out[c] = (1.0 - mix) * x[c] + mix * wet
    return out


def echo(x: np.ndarray, sr: int, delays_ms=(150.0, 310.0, 470.0),
         gains=(0.45, 0.28, 0.18), mix: float = 0.35,
         feedback: float = 0.0) -> np.ndarray:
    if mix <= 0.005:
        return x
    n = x.shape[-1]
    wet = np.zeros_like(x)
    for ms, g in zip(delays_ms, gains):
        d = int(sr * ms / 1000.0)
        if d < 1 or d >= n:
            continue
        wet[:, d:] += x[:, :-d] * g
        if feedback > 0.0:
            k = g * feedback
            cur = d
            while abs(k) > 0.02 and cur + d < n:
                wet[:, cur + d:] += x[:, : n - (cur + d)] * k
                k *= feedback
                cur += d
    return x + mix * wet


def stereo_width(x: np.ndarray, w: float = 1.0) -> np.ndarray:
    if x.shape[0] != 2 or abs(w - 1.0) < 0.02:
        return x
    L, R = x[0], x[1]
    mid = (L + R) * 0.5
    side = (L - R) * 0.5 * w
    return np.stack([mid + side, mid - side])


def bitcrush(x: np.ndarray, bits: float = 8.0, dsample: int = 1,
             mix: float = 1.0) -> np.ndarray:
    if mix <= 0.001:
        return x
    y = x
    if bits and bits < 16.0:
        q = 2.0 ** (bits - 1.0)
        y = np.round(y * q) / q
    if dsample and dsample > 1:
        ds = int(dsample)
        if y.ndim == 2:
            y = np.repeat(y[:, ::ds], ds, axis=-1)[:, : x.shape[-1]]
        else:
            y = np.repeat(y[::ds], ds)[: x.shape[-1]]
    if mix >= 0.999:
        return y
    return (1.0 - mix) * x + mix * y


def proximity(x: np.ndarray, sr: int, amount: float = 1.0) -> np.ndarray:
    if abs(amount) < 0.01:
        return x
    x = lowshelf(x, sr, 130.0, 6.0 * amount, Q=0.7)
    return highshelf(x, sr, 7500.0, -2.5 * amount, Q=0.7)


def distant(x: np.ndarray, sr: int, amount: float = 1.0, seed: int = 9) -> np.ndarray:
    if amount <= 0.01:
        return x
    x = highshelf(x, sr, 200.0, -5.0 * amount, Q=0.7)
    x = peak_eq(x, sr, 3200.0, 1.8 * amount, Q=1.0)
    return reverb(x, sr, decay=0.55, brightness=5200.0,
                  mix=0.16 * amount, predelay_ms=9.0, seed=seed)


def box_color(x: np.ndarray, sr: int, amount: float = 1.0) -> np.ndarray:
    if amount <= 0.01:
        return x
    x = peak_eq(x, sr, 240.0, 5.0 * amount, Q=0.9)
    x = peak_eq(x, sr, 760.0, 4.0 * amount, Q=1.3)
    return highshelf(x, sr, 4000.0, -4.0 * amount, Q=0.7)


def inner_groove(x: np.ndarray, sr: int, amount: float = 1.0, start: float = 0.6,
                 lo_hz: float = 4500.0) -> np.ndarray:
    if amount <= 0.01:
        return x
    lp = lowpass(x, sr, lo_hz, order=2)
    n = x.shape[-1]
    w = np.linspace(0.0, 1.0, n, dtype=np.float64)
    w = amount * np.clip((w - start) / max(1e-6, 1.0 - start), 0.0, 1.0)
    w = w[np.newaxis, :]
    return x * (1.0 - w) + lp * w


def azimuth_loss(x: np.ndarray, sr: int, amount: float = 1.0) -> np.ndarray:
    if amount <= 0.01:
        return x
    return lowpass(x, sr, lerp(20000.0, 5500.0, amount), order=2)


def sanitize(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    m = float(np.max(np.abs(x))) if x.size else 0.0
    if m > 1e3:
        x = x * (1e3 / m)
    return x


def load_audio(path: str) -> tuple[np.ndarray, int]:
    import soundfile as sf
    try:
        data, sr = sf.read(path, dtype="float64", always_2d=True)
        return data.T, int(sr)
    except Exception:
        import shutil
        import subprocess
        import tempfile
        import os
        if shutil.which("ffmpeg") is None:
            raise
        with tempfile.TemporaryDirectory() as td:
            wav = os.path.join(td, "decoded.wav")
            subprocess.run(
                ["ffmpeg", "-y", "-i", path, "-vn", "-acodec", "pcm_f32le", wav],
                check=True, capture_output=True,
            )
            data, sr = sf.read(wav, dtype="float64", always_2d=True)
            return data.T, int(sr)


def decode_bytes(raw: bytes) -> tuple[np.ndarray, int]:
    import io
    import shutil
    import subprocess
    import soundfile as sf
    try:
        data, sr = sf.read(io.BytesIO(raw), dtype="float64", always_2d=True)
        return data.T, int(sr)
    except Exception:
        if shutil.which("ffmpeg") is None:
            raise
        p = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", "pipe:0", "-f", "wav", "pipe:1"],
            input=raw, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if p.returncode != 0 or not p.stdout:
            raise ValueError("Could not decode this audio format")
        data, sr = sf.read(io.BytesIO(p.stdout), dtype="float64", always_2d=True)
        return data.T, int(sr)


def save_audio(path: str, x: np.ndarray, sr: int) -> str:
    import soundfile as sf
    x = np.clip(x, -1.0, 1.0)
    data = x.T
    if path.lower().endswith(".mp3"):
        try:
            subtypes = sf.available_subtypes("MP3")
            if "MPEG_LAYER_III" in subtypes:
                sf.write(path, data, sr, format="MP3", subtype="MPEG_LAYER_III")
                return "mp3"
        except Exception:
            pass
        path = path[:-4] + ".wav"
    sf.write(path, data, sr, format="WAV", subtype="PCM_16")
    return "wav"


def encode_audio(x: np.ndarray, sr: int) -> tuple[bytes, str]:
    import io
    import soundfile as sf
    x = np.clip(x, -1.0, 1.0)
    data = x.T
    buf = io.BytesIO()
    try:
        subtypes = sf.available_subtypes("MP3")
        if "MPEG_LAYER_III" in subtypes:
            sf.write(buf, data, sr, format="MP3", subtype="MPEG_LAYER_III")
            return buf.getvalue(), "mp3"
    except Exception:
        pass
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue(), "wav"
