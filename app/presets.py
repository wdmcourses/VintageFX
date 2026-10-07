
from __future__ import annotations

import contextvars
import os

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import lfilter

from engine import (
    NOISE_ENABLED, HUM_ENABLED, EPS,
    bandpass, lowpass, highpass, peak_eq, lowshelf, highshelf,
    saturate, compress, hiss_noise, hum_noise,
    fading_envelope, whistle_tone, dropout_env, pink_noise,
    lerp, db, wide_band, collapse_mono, normalize, _add,
    compressor, amp_sag, agc, hard_clip, crossover, reverb, echo,
    stereo_width, proximity, distant, box_color, inner_groove,
    azimuth_loss, sanitize,
)

# No preset may finish louder than the source by more than ~1 dB;
# apply_chain trims any chain that would outrun the original level.
LOUD_TOL = 1.12


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _ref(x: np.ndarray) -> float:
    return float(np.max(np.abs(x))) + EPS


def _band(x, sr, s, lo, hi, order=2):
    w_lo, w_hi = wide_band(sr)
    return bandpass(x, sr, lerp(w_lo, lo, s), lerp(w_hi, hi, s), order=order)


def _hp(x, sr, s, f, order=2):
    return highpass(x, sr, lerp(wide_band(sr)[0], f, s), order=order)


def _lp(x, sr, s, f, order=2):
    return lowpass(x, sr, lerp(wide_band(sr)[1], f, s), order=order)


def _noise_on() -> bool:
    return bool(NOISE_ENABLED.get())


# Per-preset extra trim applied to every characteristic noise layer a preset
# adds. apply_chain sets it around each era/device call so a single knob can
# quieten hiss, rumble, hum, whistle and static for one preset at a time.
NOISE_FACTOR: contextvars.ContextVar = contextvars.ContextVar(
    "VFX_NOISE_FACTOR", default=1.0)


def _nf() -> float:
    return float(NOISE_FACTOR.get())


def _his(x, sr, rng, ref, s, lo, ldb, hi=9000.0):
    if not _noise_on() or s <= 0.001:
        return x
    return _add(x, lambda n: hiss_noise(n, sr, rng, lo, hi,
                                        ref * db(ldb) * s * _nf()), rng)


def _rumble(x, sr, rng, ref, s, ldb, hi=130.0):
    if not _noise_on() or s <= 0.001:
        return x
    return _add(x, lambda n: lowpass(pink_noise(n, sr, rng), sr, hi, order=2)
                * ref * db(ldb) * s * _nf(), rng)


def _hum(x, sr, rng, ref, s, ldb, freq=50.0, harm=4):
    if not _noise_on() or not bool(HUM_ENABLED.get()) or s <= 0.001:
        return x
    return _add(x, lambda n: hum_noise(n, sr, rng, ref * db(ldb) * s * _nf(),
                                       freq=freq, harmonics=harm), rng)


def _whistle(x, sr, rng, ref, s, ldb, f0=1350.0, sweep=450.0, rate=0.06):
    if not _noise_on() or s <= 0.001:
        return x
    return _add(x, lambda n: whistle_tone(n, sr, rng, f0=f0, sweep=sweep,
                                          rate=rate) * ref * db(ldb) * s
                * _nf(), rng)


def _static(x, sr, rng, ref, s, lo, hi, ldb, fade_hz=0.6, fade_db=7.0):
    if not _noise_on() or s <= 0.001:
        return x

    def fn(n):
        st = hiss_noise(n, sr, rng, lo, hi, ref * db(ldb) * s * _nf())
        env = fading_envelope(n, sr, rng, rate_hz=fade_hz,
                              depth_db=fade_db * max(s, 0.15))
        return st * env

    return _add(x, fn, rng)


def _breath(x, sr, rng, ref, s, ldb, lo=1500.0, hi=9000.0):
    if not _noise_on() or s <= 0.001:
        return x
    src = x[0] if x.ndim == 2 else x
    w = max(3, int(sr * 0.06))
    env = uniform_filter1d(np.abs(src), size=w, mode="nearest")
    ref_e = float(np.percentile(env, 85.0)) + EPS
    ref_e = min(ref_e, float(np.max(env)) + EPS)
    mod = np.clip(env / ref_e, 0.15, 2.5)

    def fn(n):
        nz = hiss_noise(n, sr, rng, lo, hi, ref * db(ldb) * s * _nf())
        return nz * mod[:n]

    return _add(x, fn, rng)


def _drift(x, sr, rng, rate, depth_db):
    n = x.shape[-1]
    return x * fading_envelope(n, sr, rng, rate_hz=rate, depth_db=depth_db)


def cylinder(x, sr, s):
    rng = _rng(1201)
    ref = _ref(x)
    x = _band(x, sr, s, 260, 2500, order=3)
    x = peak_eq(x, sr, 700, 5.0 * s, Q=1.2)
    x = highshelf(x, sr, 1800, -10.0 * s)
    x = lowshelf(x, sr, 200, -8.0 * s)
    x = saturate(x, drive=lerp(1.0, 4.4, s), bias=0.18 * s, mix=min(1.0, s * 1.2))
    x = _his(x, sr, rng, ref, s, 300, -23.5, hi=4500)
    x = _rumble(x, sr, rng, ref, s, -30.5, hi=90)
    return collapse_mono(x)


def acoustic1910(x, sr, s):
    rng = _rng(1101)
    ref = _ref(x)
    x = _band(x, sr, s, 300, 3200, order=2)
    x = peak_eq(x, sr, 850, 5.5 * s, Q=1.05)
    x = peak_eq(x, sr, 2100, 3.0 * s, Q=1.6)
    x = highshelf(x, sr, 2500, -8.0 * s)
    x = lowshelf(x, sr, 280, -6.0 * s)
    x = saturate(x, drive=lerp(1.0, 3.4, s), bias=0.14 * s, mix=min(1.0, s * 1.1))
    x = _his(x, sr, rng, ref, s, 400, -26.5, hi=5000)
    x = _rumble(x, sr, rng, ref, s, -36.5, hi=95)
    return collapse_mono(x)


def det_radio(x, sr, s):
    rng = _rng(1303)
    ref = _ref(x)
    x = _band(x, sr, s, 220, 3200, order=3)
    x = peak_eq(x, sr, 1050, 4.0 * s, Q=1.0)
    x = hard_clip(x, drive=lerp(1.0, 3.2, s), mix=0.85 * s)
    x = agc(x, sr, amount=0.55 * s, window_ms=320)
    x = _his(x, sr, rng, ref, s, 220, -27.5, hi=3200)
    x = _rumble(x, sr, rng, ref, s, -36.5, hi=150)
    x = collapse_mono(x)
    return x


def electrecord(x, sr, s):
    rng = _rng(1404)
    ref = _ref(x)
    x = _band(x, sr, s, 90, 5200, order=2)
    x = peak_eq(x, sr, 3000, 4.0 * s, Q=0.9)
    x = peak_eq(x, sr, 1400, 2.0 * s, Q=1.2)
    x = saturate(x, drive=lerp(1.0, 2.4, s), bias=0.10 * s, mix=0.6 * s)
    x = _his(x, sr, rng, ref, s, 350, -28, hi=6000)
    return collapse_mono(x)


def shellac78(x, sr, s):
    rng = _rng(2202)
    ref = _ref(x)
    x = _band(x, sr, s, 120, 5600, order=2)
    x = peak_eq(x, sr, 1400, 3.5 * s, Q=0.9)
    x = saturate(x, drive=lerp(1.0, 2.4, s), bias=0.06 * s, mix=0.6 * s)
    x = _his(x, sr, rng, ref, s, 300, -27, hi=6500)
    return collapse_mono(x)


def lamp_radio(x, sr, s):
    rng = _rng(5505)
    ref = _ref(x)
    x = _band(x, sr, s, 300, 4500, order=3)
    x = highshelf(x, sr, 2800, -7.0 * s)
    x = saturate(x, drive=lerp(1.0, 3.6, s), bias=0.10 * s, mix=0.75 * s)
    x = agc(x, sr, amount=0.45 * s, window_ms=280)
    x = _static(x, sr, rng, ref, s, 300, 4500, -25, fade_hz=0.6, fade_db=7.0)
    x = _whistle(x, sr, rng, ref, s, -46, f0=1350, sweep=450, rate=0.06)
    x = _hum(x, sr, rng, ref, s, -40, freq=50.0, harm=4)
    return collapse_mono(x)


def radio_am(x, sr, s):
    rng = _rng(5505)
    ref = _ref(x)

    x = _band(x, sr, s, 300, 4500, order=3)
    x = highshelf(x, sr, 2800, -7.0 * s, Q=0.7)
    x = saturate(x, drive=lerp(1.0, 3.6, s), bias=0.10 * s, mix=0.75 * s)

    def _static_v1(n):
        st = hiss_noise(n, sr, rng, 300, 4500, ref * db(-25) * s * _nf())
        env = fading_envelope(n, sr, rng, rate_hz=0.6, depth_db=7.0 * s)
        return st * env

    x = _add(x, _static_v1, rng)
    x = _add(x, lambda n: whistle_tone(n, sr, rng, f0=1350, sweep=450,
                                       rate=0.06) * ref * db(-42) * s
             * _nf(), rng)
    x = _hum(x, sr, rng, ref, s, -40, freq=50.0, harm=4)
    return collapse_mono(x)


def wire_rec(x, sr, s):
    rng = _rng(1707)
    ref = _ref(x)
    x = _band(x, sr, s, 220, 4200, order=2)
    x = peak_eq(x, sr, 1600, 3.5 * s, Q=1.0)
    x = saturate(x, drive=lerp(1.0, 3.0, s), bias=0.12 * s, mix=0.7 * s)
    x = compress(x, sr, thresh_db=-22, ratio=3.0, amount=s)
    x = _his(x, sr, rng, ref, s, 300, -25, hi=5500)
    return collapse_mono(x)


def sw_radio(x, sr, s):
    rng = _rng(1606)
    ref = _ref(x)
    x = _band(x, sr, s, 350, 2900, order=3)
    x = peak_eq(x, sr, 1800, 3.0 * s, Q=1.1)
    x = saturate(x, drive=lerp(1.0, 3.8, s), bias=0.08 * s, mix=0.7 * s)
    x = agc(x, sr, amount=0.8 * s, window_ms=220, max_gain_db=12.0)
    x = _static(x, sr, rng, ref, s, 350, 2900, -24, fade_hz=0.9, fade_db=9.0)
    x = _whistle(x, sr, rng, ref, s, -42, f0=1150, sweep=700, rate=0.05)
    x = _hum(x, sr, rng, ref, s, -42, freq=50.0, harm=3)
    if s > 0.01:
        x = echo(x, sr, delays_ms=(43.0, 87.0), gains=(0.32, 0.18),
                 mix=0.30 * s)
    return collapse_mono(x)


def optical35(x, sr, s):
    rng = _rng(1160)
    ref = _ref(x)
    x = _band(x, sr, s, 70, 8000, order=2)
    x = peak_eq(x, sr, 2500, 3.0 * s, Q=1.0)
    x = highshelf(x, sr, 6000, -3.0 * s)
    x = saturate(x, drive=lerp(1.0, 2.6, s), bias=0.09 * s, mix=0.6 * s)
    x = _drift(x, sr, rng, rate=0.7, depth_db=4.5 * s)
    x = _his(x, sr, rng, ref, s, 500, -32.5, hi=8000)
    return collapse_mono(x)


def optical8(x, sr, s):
    rng = _rng(8808)
    ref = _ref(x)
    x = _band(x, sr, s, 250, 7200, order=2)
    x = peak_eq(x, sr, 1800, 3.0 * s, Q=1.2)
    x = saturate(x, drive=lerp(1.0, 2.8, s), bias=0.09 * s, mix=0.65 * s)
    x = _his(x, sr, rng, ref, s, 800, -31, hi=7000)
    return collapse_mono(x)


def lp_vinyl(x, sr, s):
    rng = _rng(3303)
    ref = _ref(x)
    x = _hp(x, sr, s, 28)
    x = _lp(x, sr, s, 13500)
    x = lowshelf(x, sr, 110, 2.5 * s)
    x = highshelf(x, sr, 4500, -3.0 * s)
    x = saturate(x, drive=lerp(1.0, 1.9, s), bias=0.05 * s, mix=0.45 * s)
    x = _his(x, sr, rng, ref, s, 200, -36, hi=7000)
    x = _rumble(x, sr, rng, ref, s, -49, hi=120)
    x = inner_groove(x, sr, amount=0.7 * s, start=0.6, lo_hz=4500)
    return x


def reel_tape(x, sr, s):
    rng = _rng(1120)
    ref = _ref(x)
    x = _hp(x, sr, s, 30)
    x = _lp(x, sr, s, 15000)
    x = peak_eq(x, sr, 60, 4.0 * s, Q=1.0)
    x = peak_eq(x, sr, 2500, 1.5 * s, Q=1.0)
    x = saturate(x, drive=lerp(1.0, 2.2, s), bias=0.05 * s, mix=0.6 * s)
    x = azimuth_loss(x, sr, 0.35 * s)
    x = _his(x, sr, rng, ref, s, 2500, -40, hi=14000)
    return x


def tv_mono(x, sr, s):
    rng = _rng(10101)
    ref = _ref(x)
    x = _band(x, sr, s, 180, 7500, order=2)
    x = peak_eq(x, sr, 700, 4.0 * s, Q=1.0)
    x = highshelf(x, sr, 5500, -5.0 * s)
    x = saturate(x, drive=lerp(1.0, 2.5, s), bias=0.10 * s, mix=0.7 * s)
    x = _hum(x, sr, rng, ref, s, -36, freq=50.0, harm=5)
    x = _his(x, sr, rng, ref, s, 1000, -42, hi=7500)
    return collapse_mono(x)


def stereo_hifi(x, sr, s):
    rng = _rng(1110)
    ref = _ref(x)
    x = _hp(x, sr, s, 32)
    x = _lp(x, sr, s, 16000)
    x = lowshelf(x, sr, 120, 2.5 * s)
    x = highshelf(x, sr, 9000, 2.0 * s)
    x = saturate(x, drive=lerp(1.0, 1.8, s), bias=0.06 * s, mix=0.4 * s)
    x = _his(x, sr, rng, ref, s, 4000, -40, hi=16000)
    x = stereo_width(x, 1.0 + 0.15 * s)
    return x


def cassette(x, sr, s):
    rng = _rng(6606)
    ref = _ref(x)
    x = _hp(x, sr, s, 45)
    x = _lp(x, sr, s, 11800)
    x = peak_eq(x, sr, 90, 2.0 * s, Q=1.0)
    x = saturate(x, drive=lerp(1.0, 2.3, s), bias=0.05 * s, mix=0.55 * s)
    x = compress(x, sr, thresh_db=-20, ratio=2.2, amount=0.8 * s, win=0.02)
    x = _his(x, sr, rng, ref, s, 3000, -40, hi=11000)
    if _noise_on() and s > 0.001:
        x = x * dropout_env(x.shape[-1], sr, rng, density=0.05 * s,
                            depth_db=-9 * s, seed=44)
    return x


def dolby_cass(x, sr, s):
    rng = _rng(1140)
    ref = _ref(x)
    x = _hp(x, sr, s, 45)
    x = _lp(x, sr, s, 13500)
    x = peak_eq(x, sr, 90, 2.0 * s, Q=1.0)
    x = saturate(x, drive=lerp(1.0, 2.0, s), bias=0.04 * s, mix=0.5 * s)
    x = compress(x, sr, thresh_db=-22, ratio=2.5, amount=0.8 * s, win=0.018)
    x = _his(x, sr, rng, ref, s, 4000, -44, hi=13000)
    x = _breath(x, sr, rng, ref, s, -47)
    return x


def track8(x, sr, s):
    rng = _rng(8080)
    ref = _ref(x)
    x = _band(x, sr, s, 80, 7800, order=3)
    x = peak_eq(x, sr, 1500, 3.0 * s, Q=0.9)
    x = highshelf(x, sr, 6000, -5.0 * s)
    x = saturate(x, drive=lerp(1.0, 3.0, s), bias=0.07 * s, mix=0.75 * s)
    x = compress(x, sr, thresh_db=-17, ratio=2.8, amount=0.7 * s, win=0.012)
    x = _his(x, sr, rng, ref, s, 2000, -33, hi=8000)
    x = _rumble(x, sr, rng, ref, s, -40, hi=120)
    if _noise_on() and s > 0.001:
        tt = np.arange(x.shape[-1]) / sr
        dd = np.mod(tt, 45.0)
        th = np.sin(2 * np.pi * 55.0 * dd) * np.exp(-dd / 0.10) * (dd < 0.8)
        x = x + th * ref * db(-26) * s
    return collapse_mono(x)


def transistor_radio(x, sr, s):
    rng = _rng(1150)
    ref = _ref(x)
    x = _band(x, sr, s, 260, 4000, order=3)
    x = peak_eq(x, sr, 900, 4.0 * s, Q=1.1)
    x = peak_eq(x, sr, 3000, 3.0 * s, Q=1.4)
    x = crossover(x, threshold=0.10 * s, mix=0.7 * s)
    x = hard_clip(x, drive=lerp(1.0, 2.4, s), mix=0.5 * s)
    x = agc(x, sr, amount=0.5 * s, window_ms=180)
    x = _his(x, sr, rng, ref, s, 400, -33, hi=4200)
    return collapse_mono(x)


def mic_carbon(x, sr, s):
    rng = _rng(2101)
    ref = _ref(x)
    x = _band(x, sr, s, 250, 3000, order=3)
    x = peak_eq(x, sr, 1600, 5.0 * s, Q=1.1)
    x = highshelf(x, sr, 2400, -9.0 * s)
    x = lowshelf(x, sr, 300, -4.0 * s)
    x = saturate(x, drive=lerp(1.0, 4.5, s), bias=0.0, mix=min(1.0, s * 1.2))
    x = compress(x, sr, thresh_db=-20, ratio=4.0, amount=s, win=0.006)
    x = _his(x, sr, rng, ref, s, 500, -30, hi=3000)
    x = _hum(x, sr, rng, ref, s, -36, freq=50.0, harm=3)
    return collapse_mono(x)


def mic_crystal(x, sr, s):
    rng = _rng(2102)
    ref = _ref(x)
    x = _band(x, sr, s, 120, 4200, order=2)
    x = peak_eq(x, sr, 3200, 7.0 * s, Q=3.0)
    x = peak_eq(x, sr, 900, 4.0 * s, Q=1.5)
    x = highshelf(x, sr, 5000, -8.0 * s)
    x = hard_clip(x, drive=lerp(1.0, 2.2, s), mix=0.45 * s)
    x = _his(x, sr, rng, ref, s, 400, -36, hi=4500)
    x = _hum(x, sr, rng, ref, s, -38, freq=50.0, harm=2)
    return collapse_mono(x)


def mic_dynamic55(x, sr, s):
    rng = _rng(2106)
    ref = _ref(x)
    x = _hp(x, sr, s, 55)
    x = _lp(x, sr, s, 13000)
    x = peak_eq(x, sr, 2600, 5.0 * s, Q=1.3)
    x = peak_eq(x, sr, 90, 3.0 * s, Q=1.0)
    x = saturate(x, drive=lerp(1.0, 2.3, s), bias=0.08 * s, mix=0.55 * s)
    x = compress(x, sr, thresh_db=-18, ratio=3.5, amount=0.8 * s, win=0.012)
    x = _his(x, sr, rng, ref, s, 3000, -46, hi=13000)
    return collapse_mono(x)


def mic_ribbon44(x, sr, s):
    rng = _rng(2103)
    ref = _ref(x)
    x = _hp(x, sr, s, 40)
    x = _lp(x, sr, s, 12000)
    x = lowshelf(x, sr, 150, 5.0 * s)
    x = highshelf(x, sr, 6000, -5.0 * s)
    x = peak_eq(x, sr, 3000, -2.0 * s, Q=1.0)
    x = saturate(x, drive=lerp(1.0, 2.0, s), bias=0.16 * s, mix=0.5 * s)
    x = compress(x, sr, thresh_db=-22, ratio=2.5, amount=0.7 * s, win=0.02)
    x = _his(x, sr, rng, ref, s, 3000, -48, hi=12000)
    x = _hum(x, sr, rng, ref, s, -46, freq=50.0, harm=3)
    return x


def mic_ribbon4038(x, sr, s):
    rng = _rng(2104)
    ref = _ref(x)
    x = _hp(x, sr, s, 35)
    x = _lp(x, sr, s, 15000)
    x = lowshelf(x, sr, 120, 3.5 * s)
    x = highshelf(x, sr, 7000, -6.0 * s)
    x = peak_eq(x, sr, 180, 3.0 * s, Q=0.9)
    x = saturate(x, drive=lerp(1.0, 1.9, s), bias=0.10 * s, mix=0.45 * s)
    x = compress(x, sr, thresh_db=-24, ratio=2.0, amount=0.5 * s, win=0.025)
    x = _his(x, sr, rng, ref, s, 5000, -50, hi=15000)
    return x


def mic_cond_u47(x, sr, s):
    rng = _rng(2105)
    ref = _ref(x)
    x = _hp(x, sr, s, 28)
    x = _lp(x, sr, s, 15500)
    x = peak_eq(x, sr, 5000, 2.5 * s, Q=0.9)
    x = lowshelf(x, sr, 140, 2.5 * s)
    x = saturate(x, drive=lerp(1.0, 2.0, s), bias=0.20 * s, mix=0.5 * s)
    x = amp_sag(x, sr, amount=0.6 * s)
    x = compress(x, sr, thresh_db=-24, ratio=2.2, amount=0.6 * s, win=0.03)
    x = _his(x, sr, rng, ref, s, 4000, -52, hi=15500)
    x = _hum(x, sr, rng, ref, s, -50, freq=50.0, harm=2)
    return x


def mic_sm58(x, sr, s):
    rng = _rng(2107)
    ref = _ref(x)
    x = _hp(x, sr, s, 60)
    x = _lp(x, sr, s, 14000)
    x = peak_eq(x, sr, 4000, 4.5 * s, Q=1.2)
    x = peak_eq(x, sr, 200, -3.0 * s, Q=1.0)
    x = lowshelf(x, sr, 120, 3.0 * s)
    x = saturate(x, drive=lerp(1.0, 2.5, s), bias=0.06 * s, mix=0.6 * s)
    x = compress(x, sr, thresh_db=-16, ratio=4.0, amount=0.8 * s, win=0.01)
    x = _his(x, sr, rng, ref, s, 5000, -52, hi=14000)
    return x


def mic_electret(x, sr, s):
    rng = _rng(2108)
    ref = _ref(x)
    x = _band(x, sr, s, 150, 7500, order=2)
    x = peak_eq(x, sr, 3000, 4.0 * s, Q=1.4)
    x = peak_eq(x, sr, 9000, 5.0 * s, Q=2.5)
    x = hard_clip(x, drive=lerp(1.0, 2.2, s), mix=0.4 * s)
    x = _his(x, sr, rng, ref, s, 3500, -40, hi=8000)
    x = _hum(x, sr, rng, ref, s, -44, freq=50.0, harm=2)
    return collapse_mono(x)


def mic_c12(x, sr, s):
    rng = _rng(2109)
    ref = _ref(x)
    x = _hp(x, sr, s, 30)
    x = _lp(x, sr, s, 16000)
    x = peak_eq(x, sr, 4000, 3.0 * s, Q=1.1)
    x = highshelf(x, sr, 9000, 3.5 * s)
    x = lowshelf(x, sr, 150, 2.0 * s)
    x = saturate(x, drive=lerp(1.0, 1.9, s), bias=0.18 * s, mix=0.5 * s)
    x = amp_sag(x, sr, amount=0.4 * s)
    x = compress(x, sr, thresh_db=-24, ratio=2.0, amount=0.5 * s, win=0.03)
    x = _his(x, sr, rng, ref, s, 5000, -52, hi=16000)
    x = _hum(x, sr, rng, ref, s, -52, freq=50.0, harm=2)
    return x


def mic_u67(x, sr, s):
    rng = _rng(2110)
    ref = _ref(x)
    x = _hp(x, sr, s, 40)
    x = _lp(x, sr, s, 15000)
    x = peak_eq(x, sr, 3000, 2.0 * s, Q=1.0)
    x = peak_eq(x, sr, 250, 2.0 * s, Q=0.9)
    x = highshelf(x, sr, 8000, -2.0 * s)
    x = saturate(x, drive=lerp(1.0, 2.0, s), bias=0.22 * s, mix=0.55 * s)
    x = amp_sag(x, sr, amount=0.5 * s)
    x = compress(x, sr, thresh_db=-22, ratio=2.2, amount=0.6 * s, win=0.028)
    x = _his(x, sr, rng, ref, s, 4000, -50, hi=15000)
    x = _hum(x, sr, rng, ref, s, -50, freq=50.0, harm=2)
    return x


def mic_u87(x, sr, s):
    rng = _rng(2111)
    ref = _ref(x)
    x = _hp(x, sr, s, 50)
    x = _lp(x, sr, s, 16500)
    x = peak_eq(x, sr, 3500, 3.5 * s, Q=1.2)
    x = peak_eq(x, sr, 200, 3.0 * s, Q=1.0)
    x = highshelf(x, sr, 10000, 2.0 * s)
    x = saturate(x, drive=lerp(1.0, 2.0, s), bias=0.10 * s, mix=0.5 * s)
    x = compress(x, sr, thresh_db=-20, ratio=2.5, amount=0.6 * s, win=0.02)
    x = _his(x, sr, rng, ref, s, 4500, -52, hi=16500)
    return x


def mic_md421(x, sr, s):
    rng = _rng(2112)
    ref = _ref(x)
    x = _hp(x, sr, s, 50)
    x = _lp(x, sr, s, 14500)
    x = peak_eq(x, sr, 5000, 3.5 * s, Q=1.3)
    x = peak_eq(x, sr, 150, 2.0 * s, Q=1.0)
    x = saturate(x, drive=lerp(1.0, 2.1, s), bias=0.08 * s, mix=0.55 * s)
    x = compress(x, sr, thresh_db=-18, ratio=3.0, amount=0.7 * s, win=0.014)
    x = _his(x, sr, rng, ref, s, 4000, -50, hi=14500)
    return collapse_mono(x)


def mic_sm7(x, sr, s):
    rng = _rng(2113)
    ref = _ref(x)
    x = _hp(x, sr, s, 50)
    x = _lp(x, sr, s, 11500)
    x = peak_eq(x, sr, 2500, 1.5 * s, Q=1.1)
    x = peak_eq(x, sr, 400, 2.0 * s, Q=0.9)
    x = lowshelf(x, sr, 150, 2.0 * s)
    x = highshelf(x, sr, 5500, -3.5 * s)
    x = saturate(x, drive=lerp(1.0, 1.9, s), bias=0.12 * s, mix=0.5 * s)
    x = compress(x, sr, thresh_db=-20, ratio=2.5, amount=0.6 * s, win=0.02)
    x = _his(x, sr, rng, ref, s, 6000, -56, hi=11500)
    return x


def mic_re20(x, sr, s):
    rng = _rng(2114)
    ref = _ref(x)
    x = _hp(x, sr, s, 45)
    x = _lp(x, sr, s, 14000)
    x = peak_eq(x, sr, 2000, 1.5 * s, Q=0.9)
    x = peak_eq(x, sr, 70, -3.0 * s, Q=0.8)
    x = lowshelf(x, sr, 130, -2.0 * s)
    x = saturate(x, drive=lerp(1.0, 1.9, s), bias=0.06 * s, mix=0.5 * s)
    x = compress(x, sr, thresh_db=-20, ratio=2.5, amount=0.5 * s, win=0.022)
    x = _his(x, sr, rng, ref, s, 5000, -54, hi=14000)
    return x


def amp_lamp(x, sr, s):
    rng = _rng(2201)
    ref = _ref(x)
    x = saturate(x, drive=lerp(1.0, 2.0, s), bias=0.20 * s, mix=0.7 * s)
    x = amp_sag(x, sr, amount=0.6 * s, attack_ms=30.0, release_ms=260.0)
    x = compress(x, sr, thresh_db=-16, ratio=2.0, amount=0.5 * s, win=0.02)
    x = highshelf(x, sr, 6500, -4.0 * s)
    x = lowshelf(x, sr, 110, 2.0 * s)
    x = peak_eq(x, sr, 1200, 1.5 * s, Q=0.8)
    x = _his(x, sr, rng, ref, s, 2500, -48, hi=14000)
    x = _hum(x, sr, rng, ref, s, -44, freq=50.0, harm=4)
    return x


def amp_transistor(x, sr, s):
    rng = _rng(2202)
    ref = _ref(x)
    x = crossover(x, threshold=0.05 * s, mix=0.25 * s)
    x = hard_clip(x, drive=lerp(1.0, 1.5, s), mix=0.25 * s)
    x = compress(x, sr, thresh_db=-14, ratio=2.0, amount=0.35 * s, win=0.012)
    x = peak_eq(x, sr, 3500, 2.5 * s, Q=1.0)
    x = highshelf(x, sr, 9000, 2.0 * s)
    x = lowshelf(x, sr, 90, -2.0 * s)
    x = _his(x, sr, rng, ref, s, 5000, -54, hi=15000)
    return x


def amp_jp(x, sr, s):
    rng = _rng(2203)
    ref = _ref(x)
    x = _hp(x, sr, s, 25)
    x = peak_eq(x, sr, 32, 3.0 * s, Q=0.9)
    x = highshelf(x, sr, 10000, 2.5 * s)
    x = compress(x, sr, thresh_db=-14, ratio=2.0, amount=0.5 * s, win=0.008)
    x = _his(x, sr, rng, ref, s, 6000, -55, hi=17000)
    return x


def horn_gg(x, sr, s):
    rng = _rng(2301)
    ref = _ref(x)
    x = _band(x, sr, s, 280, 3600, order=2)
    x = peak_eq(x, sr, 900, 6.0 * s, Q=1.0)
    x = peak_eq(x, sr, 2200, 3.0 * s, Q=1.5)
    x = lowshelf(x, sr, 350, -8.0 * s)
    x = saturate(x, drive=lerp(1.0, 3.6, s), bias=0.14 * s, mix=min(1.0, s * 1.1))
    x = _his(x, sr, rng, ref, s, 500, -40, hi=4000)
    return collapse_mono(x)


def cone_speaker(x, sr, s):
    rng = _rng(7707)
    ref = _ref(x)
    x = _band(x, sr, s, 165, 6800, order=2)
    x = peak_eq(x, sr, 450, 5.5 * s, Q=1.0)
    x = peak_eq(x, sr, 2800, 3.5 * s, Q=1.4)
    x = highshelf(x, sr, 5000, -6.0 * s)
    x = saturate(x, drive=lerp(1.0, 3.0, s), bias=0.08 * s, mix=0.8 * s)
    x = compress(x, sr, thresh_db=-16, ratio=3.2, amount=s, win=0.01)
    x = _his(x, sr, rng, ref, s, 1200, -40, hi=7000)
    return collapse_mono(x)


def open_baffle(x, sr, s):
    rng = _rng(2304)
    ref = _ref(x)
    x = _hp(x, sr, s, 110)
    x = _lp(x, sr, s, 9000)
    x = peak_eq(x, sr, 500, 4.0 * s, Q=1.0)
    x = peak_eq(x, sr, 1600, -3.0 * s, Q=1.2)
    x = saturate(x, drive=lerp(1.0, 2.4, s), bias=0.10 * s, mix=0.55 * s)
    x = reverb(x, sr, decay=0.5, brightness=4500.0, mix=0.14 * s, seed=61)
    x = _his(x, sr, rng, ref, s, 1500, -44, hi=9000)
    return x


def pa_speaker(x, sr, s):
    rng = _rng(2305)
    ref = _ref(x)
    x = _band(x, sr, s, 70, 9000, order=2)
    x = peak_eq(x, sr, 110, 5.0 * s, Q=1.0)
    x = peak_eq(x, sr, 1400, -5.0 * s, Q=1.1)
    x = peak_eq(x, sr, 4500, 5.0 * s, Q=1.6)
    x = hard_clip(x, drive=lerp(1.0, 3.6, s), mix=0.75 * s)
    x = reverb(x, sr, decay=0.8, brightness=4200.0, mix=0.12 * s, seed=63)
    x = _his(x, sr, rng, ref, s, 2000, -44, hi=9000)
    return x


def telephone(x, sr, s):
    rng = _rng(9909)
    ref = _ref(x)
    x = _band(x, sr, s, 330, 3400, order=3)
    x = hard_clip(x, drive=lerp(1.0, 5.0, s), mix=min(1.0, s * 1.1))
    x = compress(x, sr, thresh_db=-20, ratio=4.5, amount=0.8 * s, win=0.006)
    x = _his(x, sr, rng, ref, s, 330, -50, hi=3400)
    x = _hum(x, sr, rng, ref, s, -56, freq=50.0, harm=2)
    return collapse_mono(x)


def cb_radio(x, sr, s):
    rng = _rng(2407)
    ref = _ref(x)
    x = _band(x, sr, s, 320, 2800, order=3)
    x = saturate(x, drive=lerp(1.0, 4.0, s), bias=0.0, mix=0.8 * s)
    x = compress(x, sr, thresh_db=-16, ratio=3.0, amount=0.6 * s, win=0.012)
    x = agc(x, sr, amount=0.3 * s, window_ms=160)
    x = _static(x, sr, rng, ref, s, 320, 2800, -38, fade_hz=1.2, fade_db=5.0)
    x = _whistle(x, sr, rng, ref, s, -44, f0=1900, sweep=300, rate=0.15)
    x = _hum(x, sr, rng, ref, s, -38, freq=50.0, harm=3)
    return collapse_mono(x)


def dictaphone(x, sr, s):
    rng = _rng(2408)
    ref = _ref(x)
    x = _band(x, sr, s, 250, 4200, order=2)
    x = peak_eq(x, sr, 1200, 4.0 * s, Q=1.2)
    x = hard_clip(x, drive=lerp(1.0, 1.6, s), mix=0.25 * s)
    x = compress(x, sr, thresh_db=-20, ratio=2.5, amount=0.6 * s, win=0.02)
    x = _his(x, sr, rng, ref, s, 700, -46, hi=4500)
    x = _rumble(x, sr, rng, ref, s, -44, hi=120)
    x = _hum(x, sr, rng, ref, s, -44, freq=50.0, harm=4)
    return collapse_mono(x)


def jukebox(x, sr, s):
    rng = _rng(2410)
    ref = _ref(x)
    x = _band(x, sr, s, 130, 7000, order=2)
    x = peak_eq(x, sr, 800, 4.0 * s, Q=1.1)
    x = lowshelf(x, sr, 200, 3.0 * s)
    x = saturate(x, drive=lerp(1.0, 3.0, s), bias=0.16 * s, mix=0.7 * s)
    x = reverb(x, sr, decay=0.7, brightness=5000.0, mix=0.18 * s, seed=77)
    x = _his(x, sr, rng, ref, s, 400, -34, hi=7000)
    return collapse_mono(x)


def boombox(x, sr, s):
    rng = _rng(2409)
    ref = _ref(x)
    x = _hp(x, sr, s, 75)
    x = _lp(x, sr, s, 14000)
    x = peak_eq(x, sr, 100, 6.0 * s, Q=1.0)
    x = peak_eq(x, sr, 3000, 4.0 * s, Q=1.4)
    x = saturate(x, drive=lerp(1.0, 2.5, s), bias=0.06 * s, mix=0.5 * s)
    x = compress(x, sr, thresh_db=-13, ratio=3.0, amount=0.42 * s, win=0.01)
    x = _his(x, sr, rng, ref, s, 4000, -42, hi=14000)
    x = stereo_width(x, 1.0 + 0.2 * s)
    return x


def megafoon(x, sr, s):
    rng = _rng(2411)
    ref = _ref(x)
    x = _band(x, sr, s, 500, 3000, order=3)
    x = peak_eq(x, sr, 1500, 6.0 * s, Q=1.2)
    x = hard_clip(x, drive=lerp(1.0, 4.5, s), mix=min(1.0, s * 1.2))
    x = agc(x, sr, amount=0.9 * s, window_ms=90, max_gain_db=8.0)
    x = _his(x, sr, rng, ref, s, 600, -30, hi=3200)
    x = _hum(x, sr, rng, ref, s, -36, freq=50.0, harm=2)
    return collapse_mono(x)


def _deemph(x, sr, s, tau_us=75.0):
    """First-order FM de-emphasis (75 us): the receiver's fixed high roll-off,
    blended in with the effect strength."""
    f0 = 1.0e6 / (2.0 * np.pi * tau_us)
    a = float(np.exp(-2.0 * np.pi * f0 / sr))
    lp = lfilter([1.0 - a], [1.0, -a], x, axis=-1)
    return x + (lp - x) * (0.8 * s)


def fm_receiver(x, sr, s):
    rng = _rng(2601)
    ref = _ref(x)
    x = _band(x, sr, s, 45, 15000, order=4)
    x = _deemph(x, sr, s, tau_us=75.0)
    x = compress(x, sr, thresh_db=-22, ratio=4.0, amount=0.85 * s, win=0.015)
    x = saturate(x, drive=lerp(1.0, 2.6, s), bias=0.04 * s, mix=0.55 * s)
    x = highshelf(x, sr, 9000, -2.0 * s)
    x = _his(x, sr, rng, ref, s, 700, -40, hi=11500)
    if _noise_on() and s > 0.001:
        # A trace of the 19 kHz stereo pilot leaking into the audio output.
        x = _add(x, lambda n: np.sin(2 * np.pi * 19000.0 * np.arange(n) / sr)
                 * ref * db(-46) * s * _nf(), rng)
    return x


def space_hall(x, sr, s):
    x = reverb(x, sr, decay=2.2, brightness=4300.0, mix=0.22 * s,
               predelay_ms=34.0, seed=81)
    x = echo(x, sr, delays_ms=(36.0, 76.0), gains=(0.17, 0.08),
             mix=0.11 * s)
    x = stereo_width(x, 1.0 + 0.12 * s)
    x = lowshelf(x, sr, 160, -1.0 * s)
    return x


def space_stadium(x, sr, s):
    rng = _rng(3102)
    ref = _ref(x)
    x = reverb(x, sr, decay=2.4, brightness=3500.0, mix=0.18 * s,
               predelay_ms=45.0, seed=82)
    x = echo(x, sr, delays_ms=(115.0, 245.0), gains=(0.4, 0.2),
             mix=0.26 * s)
    x = _his(x, sr, rng, ref, s, 2000, -44, hi=9000)
    x = lowshelf(x, sr, 120, -3.0 * s)
    return x


def space_cathedral(x, sr, s):
    x = reverb(x, sr, decay=3.2, brightness=3600.0, mix=0.27 * s,
               predelay_ms=40.0, seed=83)
    x = highshelf(x, sr, 4000, -1.5 * s)
    x = lowshelf(x, sr, 200, 1.0 * s)
    return x


def space_club(x, sr, s):
    x = reverb(x, sr, decay=0.5, brightness=4500.0, mix=0.22 * s,
               predelay_ms=8.0, seed=84)
    x = peak_eq(x, sr, 80, 4.0 * s, Q=0.9)
    x = peak_eq(x, sr, 400, -2.0 * s, Q=1.1)
    return x


def space_bathroom(x, sr, s):
    x = reverb(x, sr, decay=0.7, brightness=8000.0, mix=0.38 * s,
               predelay_ms=6.0, seed=85)
    x = highshelf(x, sr, 3000, 3.0 * s)
    x = peak_eq(x, sr, 900, 2.0 * s, Q=1.2)
    return x


def space_warehouse(x, sr, s):
    x = reverb(x, sr, decay=1.9, brightness=3800.0, mix=0.26 * s,
               predelay_ms=15.0, seed=86)
    x = lowshelf(x, sr, 150, -3.0 * s)
    x = peak_eq(x, sr, 2500, -2.0 * s, Q=1.0)
    return x


def space_garage(x, sr, s):
    x = reverb(x, sr, decay=1.1, brightness=2600.0, mix=0.26 * s,
               predelay_ms=9.0, seed=87)
    x = peak_eq(x, sr, 320, 4.0 * s, Q=1.3)
    x = highshelf(x, sr, 5000, -3.0 * s)
    return x


def space_cave(x, sr, s):
    x = reverb(x, sr, decay=2.2, brightness=2400.0, mix=0.26 * s,
               predelay_ms=26.0, seed=88)
    x = echo(x, sr, delays_ms=(230.0, 470.0), gains=(0.3, 0.15),
             mix=0.22 * s)
    x = highshelf(x, sr, 4000, -3.0 * s)
    x = lowshelf(x, sr, 160, 1.5 * s)
    return x


def space_slapback(x, sr, s):
    x = echo(x, sr, delays_ms=(72.0, 148.0), gains=(0.45, 0.20),
             mix=0.45 * s)
    x = peak_eq(x, sr, 1800, -1.5 * s, Q=1.0)
    return x


def space_plate(x, sr, s):
    x = reverb(x, sr, decay=1.4, brightness=9000.0, mix=0.34 * s,
               predelay_ms=3.0, seed=89)
    x = highshelf(x, sr, 6000, 2.0 * s)
    return x


def space_spring(x, sr, s):
    x = reverb(x, sr, decay=0.9, brightness=5500.0, mix=0.30 * s,
               predelay_ms=2.0, seed=90)
    x = peak_eq(x, sr, 2500, 3.0 * s, Q=1.4)
    x = lowshelf(x, sr, 140, -2.0 * s)
    return x


def space_amphitheater(x, sr, s):
    x = echo(x, sr, delays_ms=(128.0, 266.0), gains=(0.34, 0.16),
             mix=0.34 * s)
    x = reverb(x, sr, decay=1.6, brightness=4000.0, mix=0.20 * s,
               predelay_ms=26.0, seed=93)
    x = highshelf(x, sr, 5000, -2.0 * s)
    return x


def space_tape_echo(x, sr, s):
    x = lowpass(x, sr, lerp(wide_band(sr)[1], 7000.0, s), order=2)
    x = echo(x, sr, delays_ms=(96.0, 210.0), gains=(0.5, 0.25),
             mix=0.5 * s, feedback=0.3)
    x = highpass(x, sr, 120.0, order=1)
    return x


def _m(name, emoji, desc, tags, fn, special: bool = False):
    return {"name": name, "emoji": emoji, "desc": desc, "tags": tags,
            "fn": fn, "special": special}


EPOCHS: dict[str, dict] = {
    "cylinder": _m("Wax cylinder", "🕯️",
                   "1900-1920. Edison: hand-cranked drive, 2.5 kHz ceiling, coarse surface noise.",
                   ["era", "1900s"], cylinder),
    "acoustic1910": _m("Acoustic gramophone", "📯",
                       "1900-1925. Non-electric horn: 250-3200 Hz, throat resonance, spring-motor rumble.",
                       ["era", "1920s"], acoustic1910),
    "det_radio": _m("Crystal radio", "📻",
                    "1920s. Crystal detector: headphone only, heavy nonlinearity, noise, total dependence on the air.",
                    ["era", "radio"], det_radio),
    "electrecord": _m("Electrical recording", "⚡",
                      "1925-1945. Piezo pickup and microphone: wider band, clear mids, gentle surface hiss.",
                      ["era", "1930s"], electrecord),
    "shellac78": _m("Shellac 78 rpm", "💿",
                "1925-1950. Shellac compound: fine surface hiss, brittle bright top end.",
                ["era", "record"], shellac78),
    "lamp_radio": _m("Tube AM radio", "📻",
                     "1930-1955. 300-4500 Hz, fading static, heterodyne whistle, mains hum.",
                     ["era", "radio"], lamp_radio),
    "optical35": _m("35 mm film optical", "🎬",
                    "1930-1970. Optical soundtrack: 24 fps, variable density, surface hiss, gate flicker.",
                    ["era", "film"], optical35),
    "wire_rec": _m("Wire recording", "🧵",
                   "1940s. Steel wire: narrow band, hard midrange, a thin wiry tone.",
                   ["era", "1940s"], wire_rec),
    "sw_radio": _m("Shortwave radio", "📡",
                   "1940-1975. Narrow band, ionospheric fading, beat tones, line echo.",
                   ["era", "radio"], sw_radio),
    "lp_vinyl": _m("LP vinyl", "🎛️",
                   "1948-1980. RIAA warmth, quiet rumble, top end fades toward the end of the side.",
                   ["era", "record"], lp_vinyl),
    "reel_tape": _m("Reel tape", "🎚️",
                    "1950-1975. Wide band, 60 Hz head bump, soft saturation, tape hiss.",
                    ["era", "tape"], reel_tape),
    "optical8": _m("8 mm film optical", "🎞️",
                   "1950-1980. Home movies: 18 fps, faint surface hiss, narrow band.",
                   ["era", "film"], optical8),
    "tv_mono": _m("Vintage TV", "📺",
                  "1955-1975. Small speaker, line-frequency buzz and 50 Hz hum, soft top end.",
                  ["era", "1960s"], tv_mono),
    "stereo_hifi": _m("Early stereo hi-fi", "🎶",
                      "1958-1975. Wide band, warm response, wide stage, light tape noise.",
                      ["era", "stereo"], stereo_hifi),
    "amp_jp": _m("1970s Japanese hi-fi", "🇯🇵",
                 "1970-1980. Japanese hi-fi separates: flat response, high "
                 "damping factor, tight and clean clinical bass.",
                 ["era", "1970s"], amp_jp),
    "transistor_radio": _m("Transistor radio", "🔲",
                           "1960-1975. Small speaker, cold tone, cross-modulation, dying battery.",
                           ["era", "radio"], transistor_radio),
    "radio_am": _m("AM radio 1960s", "📻",
                   "300-4500 Hz, faint fading static, heterodyne whistle, tube receiver.",
                   ["era", "radio"], radio_am),
    "cassette": _m("Cassette", "📼",
                   "1963-1990. Oxide: top-end hiss, soft saturation, dropouts.",
                   ["era", "tape"], cassette),
    "dolby_cass": _m("Dolby B cassette", "📼",
                     "1975-1995. Noise reduction quiets the floor but adds breathing.",
                     ["era", "tape"], dolby_cass),
    "track8": _m("8-track cartridge", "🚗",
                 "1965-1981. Endless loop cart: warm tape saturation, muffled top, program-change thump.",
                 ["era", "1970s"], track8),
}

DEVICES: dict[str, dict] = {
    "mic_carbon": _m("Carbon microphone", "📞",
                     "1900-1940. Carbon granules: 250-3000 Hz, self-noise, built-in compression, telephonic voice.",
                     ["microphone", "1920s"], mic_carbon),
    "mic_crystal": _m("Crystal (piezo)", "💎",
                      "1930-1960. Quartz / Rochelle salt element: sharp resonance, high output, little bass.",
                      ["microphone", "1930s"], mic_crystal),
    "mic_dynamic55": _m("Unidyne dynamic", "🎤",
                        "1937-1960. Shure 545: 50-15000 Hz, horn element, pronounced presence peak.",
                        ["microphone", "1950s"], mic_dynamic55),
    "mic_ribbon44": _m("RCA 44 ribbon", "🎙️",
                       "1932-1960. Figure-8: warm close bass, soft top end, tube path.",
                       ["microphone", "1940s"], mic_ribbon44),
    "mic_ribbon4038": _m("BBC 4038 ribbon", "🎙️",
                         "1950s. Flat 30-15000 Hz, 6 dB/oct roll-off, natural neutral tone.",
                         ["microphone", "1950s"], mic_ribbon4038),
    "mic_cond_u47": _m("Tube condenser U47", "🎚️",
                       "1949-1965. Neumann U47: air and detail, tube even harmonics, amp sag.",
                       ["microphone", "1950s"], mic_cond_u47),
    "mic_sm58": _m("SM58 dynamic", "🎤",
                   "1966+. Shure SM58: 4 kHz vocal peak, dense close bass, indestructible capsule.",
                   ["microphone", "1960s"], mic_sm58),
    "mic_electret": _m("Electret", "📎",
                       "1968-1990. Tiny dictaphone / headset capsule: narrow range, hiss.",
                       ["microphone", "1970s"], mic_electret),
    "mic_c12": _m("AKG C12 tube condenser", "🎚️",
                  "1953. Airy top end, silk presence, tube warmth and a hint of sag.",
                  ["microphone", "1950s"], mic_c12),
    "mic_u67": _m("Neumann U67 tube", "🎚️",
                  "1960. Warm midrange, soft top, even tube harmonics, close and intimate.",
                  ["microphone", "1960s"], mic_u67),
    "mic_u87": _m("Neumann U87 FET", "🎚️",
                  "1967. Tight lows, 3.5 kHz presence peak, detailed and slightly forward.",
                  ["microphone", "1970s"], mic_u87),
    "mic_md421": _m("Sennheiser MD 421", "🎤",
                    "1954. Wide dynamic range, five-position presence switch, punchy mids.",
                    ["microphone", "1960s"], mic_md421),
    "mic_sm7": _m("Shure SM7", "🎤",
                  "1973. Dark, warm and smooth, soft top end, gentle 2.5 kHz lift, great isolation.",
                  ["microphone", "1970s"], mic_sm7),
    "mic_re20": _m("Electro-Voice RE20", "🎤",
                   "1968. Variable-D, flat broadcast response, no proximity boom.",
                   ["microphone", "1970s"], mic_re20),

    "amp_lamp": _m("Tube amplifier", "🔥",
                   "Even harmonics, soft saturation, sag on loud phrases, warm top end.",
                   ["amplifier", "tube"], amp_lamp),
    "amp_transistor": _m("Transistor amplifier", "❄️",
                         "1960-1975. Class B: zero-crossing distortion, cold mids, hard ceiling.",
                         ["amplifier", "1960s"], amp_transistor),

    "horn_gg": _m("Horn speaker", "📯",
                  "1920-1930. 280-3600 Hz, 900 Hz throat, no bass at all, very efficient.",
                  ["speaker", "1920s"], horn_gg),
    "cone_speaker": _m("Paper-cone speaker", "🔊",
                       "1950s. Paper cone in a cabinet: 450 Hz boom, boxy resonance, soft top end.",
                       ["speaker", "1950s"], cone_speaker),

    "telephone": _m("Telephone line", "☎️",
                    "330-3400 Hz, compandor, hard clipping, mains hum.",
                    ["line"], telephone),
    "cb_radio": _m("CB radio", "📻",
                   "1970s. Compandor and squelch: sound pops up, then drowns in static.",
                   ["line", "1970s"], cb_radio),
    "dictaphone": _m("1960s dictaphone", "🎙️",
                     "Narrow band, motor hum, heavy AGC, tape hiss.",
                     ["device", "1960s"], dictaphone),
    "boombox": _m("Boombox", "📻",
                  "1980s. 105 Hz bass reflex, small drivers, clipping at full level.",
                  ["device", "1980s"], boombox),
    "megafoon": _m("Megaphone", "📢",
                   "500-3000 Hz, horn throat, hard ceiling, level AGC.",
                   ["device"], megafoon),
    "fm_receiver": _m("FM stereo receiver", "📻",
                      "1961-1990. Multiplex stereo: 30 Hz-15 kHz, 75 us "
                      "de-emphasis, glue compression, a trace of pilot tone.",
                      ["device", "1970s"], fm_receiver),
}


SPACES: dict[str, dict] = {
    "pa_speaker": _m("PA system", "🎚️",
                     "1970s club stack: woofer box, mid dip, horn tweeter, hard limit.",
                     ["space", "1970s"], pa_speaker),
    "open_baffle": _m("Open baffle", "🔘",
                      "Dipole in a room: bass radiates backward, everything under 110 Hz is lost.",
                      ["space", "1940s"], open_baffle),
    "jukebox": _m("Jukebox room", "🕹️",
                  "1950s arcade: warmth, echo, lively room tone.",
                  ["space", "1950s"], jukebox),
    "hall": _m("Concert hall", "🏛️",
               "Large measured hall: 2.2 s tail, early reflections, a wide and breathing stage.",
               ["space"], space_hall),
    "stadium": _m("Stadium", "🏟️",
                  "Open air: distant slap echoes, long wash, crowd murmur.",
                  ["space"], space_stadium),
    "cathedral": _m("Cathedral", "⛪",
                    "3.2 s of soft stone reflections, airy and endless.",
                    ["space"], space_cathedral),
    "club": _m("Club", "🪩",
               "Small and loud: short bright room, punchy 80 Hz floor.",
               ["space"], space_club),
    "bathroom": _m("Bathroom", "🛁",
                   "Tiles: short bright reflections with a ringing midrange.",
                   ["space"], space_bathroom),
    "warehouse": _m("Warehouse", "🏭",
                    "Big empty concrete: long wash, hollow mids.",
                    ["space"], space_warehouse),
    "garage": _m("Garage", "🚗",
                 "Tight boxy room, resonant 320 Hz, dull top end.",
                 ["space"], space_garage),
    "cave": _m("Cave", "🕳️",
               "Natural cavern: dark water echoes, distant long reflections.",
               ["space"], space_cave),
    "slapback": _m("Street slapback", "🏙️",
                   "Hard wall at 70-150 ms: the classic 1950s double.",
                   ["space"], space_slapback),
    "amphitheater": _m("Amphitheater", "🏺",
                       "Greek open-air theatre: marble slap echoes, warm distant wash.",
                       ["space"], space_amphitheater),
    "tape_echo": _m("Tape echo", "🎛️",
                    "1950s Echoplex: analog tape delay, repeats bloom and dull.",
                    ["space"], space_tape_echo),
    "plate": _m("Plate reverb", "🎛️",
                "Dense bright metallic plate with a smooth 1.4 s tail.",
                ["space"], space_plate),
    "spring": _m("Spring reverb", "🪝",
                 "Tank springs: splashy bright boing, guitar-amp classic.",
                 ["space"], space_spring),
}


_VINYL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vinyl.mp3")
_VINYL_BEDS: dict[int, np.ndarray | None] = {}


def _vinyl_bed(sr: int):
    """Mono loop of the user's vinyl surface (app/vinyl.mp3), resampled to the
    working rate. The recording is played back untouched -- no level evening,
    no clipping -- and tiled end-to-start, so the whole file is heard before
    it repeats. Cached per sample rate; None when the file is missing so the
    Vinyl button simply stays silent."""
    if sr in _VINYL_BEDS:
        return _VINYL_BEDS[sr]
    bed = None
    try:
        import soundfile as sf
        data, fsr = sf.read(_VINYL_PATH, dtype="float64", always_2d=True)
        mono = data.mean(axis=1)
        if fsr != sr and mono.size > 1:
            n_new = max(1, int(round(mono.size * sr / float(fsr))))
            mono = np.interp(np.arange(n_new) / float(sr),
                             np.arange(mono.size) / float(fsr), mono)
        bed = mono - float(np.mean(mono))
    except Exception:
        bed = None
    _VINYL_BEDS[sr] = bed
    return bed


# Fixed make-up applied to the Vinyl layer so the user's file sits at a usable
# level in the mix. The file itself still defines its character and dynamics.
VINYL_GAIN = 1.5


def _vinyl_layer(n: int, sr: int, s: float):
    """The surface layer for ``n`` samples: the file looped and lifted by the
    fixed make-up gain, or None when there is nothing to add."""
    if s <= 0.001:
        return None
    bed = _vinyl_bed(sr)
    if bed is None or bed.size < 32:
        return None
    return bed[np.arange(n) % bed.size] * (VINYL_GAIN * s)


def v_vinyl(x, sr, s):
    """Marker for the Vinyl option. The actual surface layer is mixed in after
    the main chain has been levelled (see apply_chain), so it can never pull
    the programme's level down."""
    return x


def v_mains_hum(x, sr, s):
    return x


def v_boxy(x, sr, s):
    return box_color(x, sr, s)


def v_bright(x, sr, s):
    return highshelf(x, sr, 6000, 5.0 * s, Q=0.7)


def v_dark(x, sr, s):
    return highshelf(x, sr, 5000, -8.0 * s, Q=0.7)


def _brick_wall(x, sr, hz=90.0):
    """FabFilter-style 96 dB/oct low cut at ``hz`` (default 90 Hz), plus a
    gentle low shelf that takes the boom out of the lows.  Minimum-phase, so
    the attack stays crisp, and applied last in the chain (see apply_chain) so
    nothing can put lows back under the corner.  Deliberately independent of
    the effect strength: the wall is always at full depth."""
    fc = float(np.clip(hz, 20.0, sr * 0.45))
    x = highpass(x, sr, fc, order=16, zero_phase=False)
    return lowshelf(x, sr, 280.0, -2.0, Q=0.7)


def v_brick(x, sr, s):
    """Marker for the Brick option; the wall itself is applied at the very end
    of apply_chain, whatever else is in the chain."""
    return x


def _opt(oid, name, emoji, fn, desc=""):
    return {"id": oid, "name": name, "emoji": emoji, "fn": fn, "desc": desc}


VARIANT_GROUPS: list[dict] = [
    {
        "id": "signal",
        "name": "Signal and geometry",
        "stage": 2,
        "hint": "Final tonal balance of the chain.",
        "pairs": [
            {"id": "brick", "name": "Brick", "pos": 1,
             "a": _opt("brick", "Brick", "\U0001f9f1", v_brick,
                       "Brickwall: 96 dB/oct low cut at {hz} Hz. Click to type "
                       "your own corner frequency.")},
        ],
    },
    {
        "id": "character",
        "name": "Character",
        "stage": 1,
        "hint": "Mains hum, record surface and cabinet colouring.",
        "pairs": [
            {"id": "supply", "name": "Power", "pos": 2,
             "a": _opt("hum", "50 Hz", "\u26a1", v_mains_hum,
                       "Mains hum: 50 Hz and its harmonics.")},
            {"id": "surface", "name": "Vinyl", "pos": 0,
             "a": _opt("vinyl", "Vinyl", "\U0001f4bf", v_vinyl,
                       "Vinyl surface looped from app/vinyl.mp3.")},
            {"id": "body", "name": "Cabinet", "pos": 3,
             "a": _opt("boxy", "Cabinet", "\U0001f4e6", v_boxy,
                       "Cabinet / box colouring.")},
        ],
    },
    {
        "id": "brightness",
        "name": "Brightness",
        "stage": 2,
        "hint": "Dark and bright as the last word of the chain.",
        "pairs": [
            {"id": "tone", "name": "Brightness", "pos": 4,
             "a": _opt("dark", "Dark", "\U0001f311", v_dark,
                       "Softer, darker top end."),
             "b": _opt("bright", "Bright", "\u2728", v_bright,
                       "Brighter, airier top end.")},
        ],
    },
]

# Render order is independent of the processing stages above: the flat row of
# character buttons follows ``pos`` (Vinyl first, then Brick, 50 Hz, Cabinet,
# Dark/Bright), while the DSP still runs each group in its own stage.
_PAIRS_BY_POS = sorted(
    (p for g in VARIANT_GROUPS for p in g["pairs"]), key=lambda p: p["pos"])
VARIANT_ORDER: list[str] = [p["id"] for p in _PAIRS_BY_POS]
assert len(set(VARIANT_ORDER)) == len(VARIANT_ORDER), "variant pair ids unique"
assert [p["pos"] for p in _PAIRS_BY_POS] == list(range(len(_PAIRS_BY_POS))), \
    "variant pair pos values must be unique and contiguous"

VARIANT_INDEX: dict[str, dict] = {}
for _g in VARIANT_GROUPS:
    for _p in _g["pairs"]:
        for _side in ("a", "b"):
            if _side not in _p:
                continue
            _o = _p[_side]
            if _o["id"] in VARIANT_INDEX:
                raise RuntimeError(
                    f"Duplicate variant id: {_o['id']!r} "
                    f"(pairs {VARIANT_INDEX[_o['id']]['pair']!r} and {_p['id']!r})")
            VARIANT_INDEX[_o["id"]] = {"group": _g["id"], "stage": _g["stage"],
                                       "pair": _p["id"], "opt": _o}

_N_OPTS = sum(1 for _g in VARIANT_GROUPS
              for _p in _g["pairs"]
              for _s in ("a", "b") if _s in _p)
assert len(VARIANT_INDEX) == _N_OPTS, "variant ids must be unique"


COMPRESSIONS: list[dict] = [
    {"id": "off", "name": "None", "emoji": "◻️",
     "desc": "No compression - dynamics stay exactly as in the original.",
     "params": None},
    {"id": "light", "name": "Light", "emoji": "🌤️",
     "desc": "1.6:1 - gently lifts the quiet parts.",
     "params": {"thresh_db": -17.0, "ratio": 1.6, "attack_ms": 30.0,
                "release_ms": 280.0, "makeup_db": 2.0, "knee_db": 8.0}},
    {"id": "medium", "name": "Medium", "emoji": "⛅",
     "desc": "3:1 - studio density, even delivery.",
     "params": {"thresh_db": -20.0, "ratio": 3.0, "attack_ms": 12.0,
                "release_ms": 160.0, "makeup_db": 4.0, "knee_db": 6.0}},
]

COMPRESSION_BY_ID = {c["id"]: c for c in COMPRESSIONS}

PRESETS: dict[str, dict] = {**EPOCHS, **DEVICES, **SPACES}


def apply_compression(x: np.ndarray, sr: int, compression_id: str) -> np.ndarray:
    meta = COMPRESSION_BY_ID.get(compression_id or "off")
    if not meta or not meta.get("params"):
        return x
    return compressor(x, sr, **dict(meta["params"]))


def _apply_variants(x, sr, stage, chosen, strength):
    wanted = set(chosen)
    applied = []
    for grp in VARIANT_GROUPS:
        if grp["stage"] != stage:
            continue
        for pair in grp["pairs"]:
            sel = None
            if pair["a"]["id"] in wanted:
                sel = pair["a"]
            elif pair.get("b") and pair["b"]["id"] in wanted:
                sel = pair["b"]
            if sel is None:
                continue
            x = sanitize(sel["fn"](x, sr, strength))
            applied.append(f"{pair['name']}: {sel['name']}")
    return x, applied


# Extra per-preset noise trims layered on top of the global engine NOISE_TRIM.
# Each value is the fraction of the previous noise level, from the listening
# pass that asked for quieter surfaces and devices.
ERA_NOISE_TRIM: dict[str, float] = {
    "cylinder": 0.50, "acoustic1910": 0.50, "det_radio": 0.50,     # -50 %
    "lamp_radio": 0.75, "sw_radio": 0.75, "cassette": 0.75,        # -25 %
    "electrecord": 0.50,                                           # -50 %
    "shellac78": 0.25, "optical8": 0.25, "transistor_radio": 0.25,
    "radio_am": 0.25, "track8": 0.25,                              # -75 %
    "optical35": 0.85, "reel_tape": 0.85,
    "tv_mono": 0.85, "stereo_hifi": 0.85, "amp_jp": 0.85,
    "dolby_cass": 0.85,                                            # -15 %
    "lp_vinyl": 0.31875,                                           # -15/-25/-50 %
    "wire_rec": 0.20,                                              # -80 %
}
DEVICE_NOISE_TRIM = 0.50   # every Device-section effect: -50 %

assert set(ERA_NOISE_TRIM) == set(EPOCHS), "era noise trims must cover every era"


def _run_preset(x, sr, s, meta, trim):
    if s <= 0.001:
        return x
    tok = NOISE_FACTOR.set(trim)
    try:
        return sanitize(meta["fn"](x, sr, s))
    finally:
        NOISE_FACTOR.reset(tok)


def apply_chain(x: np.ndarray, sr: int, epochs=(), devices=(), spaces=(),
                variants=(), strength: float = 0.7, compression: str = "off",
                no_noise: bool = False, volume: float = 1.0,
                spaces_strength: float = 1.0,
                brick_hz: float = 90.0) -> tuple[np.ndarray, list[str]]:
    s = float(np.clip(strength, 0.0, 1.0))
    ss = float(np.clip(spaces_strength, 0.0, 1.0))
    vs = 0.55 + 0.45 * s
    vol = float(np.clip(volume, 0.0, 2.0))
    steps: list[str] = []

    _src = x.mean(axis=0) if x.ndim == 2 else x
    src_pk = float(np.max(np.abs(_src))) + EPS
    src_rms = float(np.sqrt(np.mean(_src * _src)))

    token = NOISE_ENABLED.set(not bool(no_noise))
    h_token = HUM_ENABLED.set("hum" in variants)
    try:
        x, done = _apply_variants(x, sr, 0, variants, vs)
        steps += done

        for pid in epochs:
            meta = EPOCHS.get(pid)
            if meta is None:
                raise KeyError(f"Unknown era: {pid}")
            x = _run_preset(x, sr, s, meta, ERA_NOISE_TRIM.get(pid, 1.0))
            steps.append(meta["name"])

        for pid in devices:
            meta = DEVICES.get(pid)
            if meta is None:
                raise KeyError(f"Unknown device: {pid}")
            x = _run_preset(x, sr, s, meta, DEVICE_NOISE_TRIM)
            steps.append(meta["name"])

        for pid in spaces:
            meta = SPACES.get(pid)
            if meta is None:
                raise KeyError(f"Unknown space: {pid}")
            x = _run_preset(x, sr, s * ss, meta, 1.0)
            steps.append(meta["name"])

        for stage in (1, 2):
            x, done = _apply_variants(x, sr, stage, variants, vs)
            steps += done

        if "hum" in variants:
            x = sanitize(_hum(x, sr, _rng(9011), _ref(x), s, -24,
                              freq=50.0, harm=5))

        comp = COMPRESSION_BY_ID.get(compression or "off")
        if comp and comp["id"] != "off":
            x = sanitize(apply_compression(x, sr, comp["id"]))
            steps.append(f"Compression: {comp['name']}")

        x = normalize(sanitize(x), peak=0.97)
        if src_rms > EPS and src_pk > EPS:
            cap = 0.97 * (src_rms / src_pk) * LOUD_TOL
            y = x.mean(axis=0) if x.ndim == 2 else x
            r = float(np.sqrt(np.mean(y * y))) + EPS
            if r > cap:
                x = x * (cap / r)

        # The Vinyl surface is mixed in only now, after the programme has been
        # levelled, so switching it on can never make the programme quieter.
        # It rides on top and is soft-limited to the headroom above the
        # programme, so it stays as loud as fits without clipping.
        if "vinyl" in variants and not bool(no_noise):
            layer = _vinyl_layer(x.shape[-1], sr, vs)
            if layer is not None:
                main = x.mean(axis=0) if x.ndim == 2 else x
                room = np.maximum(0.985 - np.abs(main), 1e-4)
                x = x + np.tanh(layer / room) * room

        # The Brick wall is always applied last, whatever else is selected, so
        # nothing downstream can put lows back under the corner.  It ignores
        # the effect strength: the wall is always at full depth.
        if "brick" in variants:
            x = sanitize(_brick_wall(x, sr, brick_hz))

        x = x * vol
    finally:
        NOISE_ENABLED.reset(token)
        HUM_ENABLED.reset(h_token)

    if not steps and not bool(no_noise):
        raise ValueError("Select an era, device, space, variant, "
                         "a compression level or enable noise removal")
    return x, steps


def apply_preset(x: np.ndarray, sr: int, preset_id: str, strength: float) -> np.ndarray:
    if preset_id in EPOCHS:
        out, _ = apply_chain(x, sr, epochs=[preset_id], strength=strength,
                             compression="off")
    else:
        out, _ = apply_chain(x, sr, devices=[preset_id], strength=strength,
                             compression="off")
    return out
