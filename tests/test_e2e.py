from __future__ import annotations

import ast
import json
import os
import re
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "app"))
sys.dont_write_bytecode = True

import numpy as np
import soundfile as sf
import presets
import server

PORT = 8791
BASE = "http://127.0.0.1:%d" % PORT
TMP = os.path.join(tempfile.gettempdir(), "vfx_e2e")
os.makedirs(TMP, exist_ok=True)

FAILS = []


def check(cond, msg):
    print("[%s] %s" % ("OK   " if cond else "FAIL ", msg))
    if not cond:
        FAILS.append(msg)


def req(path, data=None, headers=None, method=None):
    h = dict(headers or {})
    body = None
    if data is not None and not isinstance(data, (bytes, bytearray)):
        body = json.dumps(data).encode("utf-8")
        h.setdefault("Content-Type", "application/json")
    elif isinstance(data, (bytes, bytearray)):
        body = data
    r = urllib.request.Request(BASE + path, data=body, headers=h, method=method)
    try:
        with urllib.request.urlopen(r, timeout=300) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def jreq(path, data=None, headers=None, method=None):
    st, hd, body = req(path, data, headers, method=method)
    try:
        return st, hd, json.loads(body.decode("utf-8"))
    except Exception:
        return st, hd, {"_raw": body[:300]}


sr = 44100
dur = 4.0
t = np.arange(int(sr * dur)) / sr
mono = 0.5 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 930 * t)
mono += 0.03 * np.random.default_rng(1).standard_normal(mono.size)
src = os.path.join(TMP, "e2e_src.wav")
sf.write(src, mono.astype(np.float32), sr, subtype="PCM_16")
WAV = open(src, "rb").read()
print("test file: %.1f s, mono, %d Hz, %d bytes" % (dur, sr, len(WAV)))


def main():
    httpd = server.Server(("127.0.0.1", PORT), server.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    time.sleep(0.4)

    try:
        run()
    finally:
        httpd.shutdown()

    print()
    if FAILS:
        print("FAILED: %d" % len(FAILS))
        for f in FAILS:
            print("  -", f)
        sys.exit(1)
    print("ALL CHECKS PASSED")


def run():
    st, hd, P = jreq("/api/presets")
    check(st == 200, "/api/presets -> 200")
    check("epochs" in P and "devices" in P, "epochs and devices present")
    check("spaces" in P, "spaces present")
    check("variants" in P and "compressions" in P, "variants and compressions present")
    print("     eras=%d devices=%d spaces=%d variant groups=%d compressions=%d"
          % (len(P["epochs"]), len(P["devices"]), len(P["spaces"]),
             len(P["variants"]), len(P["compressions"])))
    check(len(P["epochs"]) == 20, "eras == 20")
    check(len(P["devices"]) == 24, "devices == 24")
    check(len(P["spaces"]) == 16, "spaces == 16")
    check(len(P["variants"]) == 3, "variant groups == 3")
    check(len(P["compressions"]) == 3, "compressions == 3")

    spec = [g["id"] for g in P["variants"]]
    check(len(spec) == len(set(spec)), "variant group ids unique")
    npairs = sum(len(g["pairs"]) for g in P["variants"])
    check(npairs == 5,
          "exactly 5 character pairs (Brick/Power/Vinyl/Cabinet/Brightness)")
    check(all("a" in p for g in P["variants"] for p in g["pairs"]),
          "every pair has side a")
    check(any(p["b"] is None for g in P["variants"] for p in g["pairs"]),
          "single-sided pairs are serialized with b as null")

    opts = []
    pairs = [p for g in P["variants"] for p in g["pairs"]]
    check(all("pos" in p for p in pairs),
          "every character pair carries a display position")
    check(sorted(p["pos"] for p in pairs) == list(range(len(pairs))),
          "character display positions are unique and contiguous")
    pairs.sort(key=lambda p: p["pos"])
    for p in pairs:
        opts.append(p["a"])
        if p.get("b"):
            opts.append(p["b"])
    check(len(opts) == 6,
          "6 character options (Vinyl/Brick/50 Hz/Cabinet/Dark/Bright)")
    check(not any(o["id"] == "neutral" for o in opts), "Neutral option removed")
    check(not any(o["id"] == "quiet" for o in opts), "Clean option removed")
    check(any(o["id"] == "vinyl" for o in opts), "Vinyl surface option present")
    check(any(o["id"] == "boxy" for o in opts), "Cabinet (boxy) option kept")
    check(any(o["id"] == "hum" for o in opts), "'50 Hz' option kept")
    check(any(o["id"] == "brick" for o in opts), "Brick option present")
    oid_list = [o["id"] for o in opts]
    check(oid_list[0] == "vinyl" and oid_list[1] == "brick",
          "Vinyl leads the row, right before Brick: %s" % oid_list)
    check(oid_list.index("dark") < oid_list.index("bright"),
          "Dark sits before Bright")
    check(oid_list.index("brick") < oid_list.index("dark"),
          "Brick sits before Dark/Bright")
    check(all(len(o["name"]) <= 10 for o in opts),
          "character option names are short (single row fit)")
    check(all(o.get("desc") for o in opts),
          "every character option carries a tooltip description")

    special = [e for e in P["epochs"] if e.get("special")]
    check(len(special) == 0, "no special/golden era card")
    eorder = [e["id"] for e in P["epochs"]]
    check(eorder.index("radio_am") == eorder.index("transistor_radio") + 1,
          "radio_am sits right after transistor_radio")
    check(eorder.index("amp_jp") == eorder.index("stereo_hifi") + 1,
          "1970s Japanese hi-fi sits right after Early stereo hi-fi")
    check("remaster" not in eorder and "vhs" not in eorder,
          "no remaster / VHS card in the era section")
    check(any(s["id"] == "amphitheater" for s in P["spaces"]),
          "new space: amphitheater")
    check(any(s["id"] == "tape_echo" for s in P["spaces"]),
          "new space: tape echo")

    eids = eorder
    dids = [d["id"] for d in P["devices"]]
    sids = [s["id"] for s in P["spaces"]]
    vids = [o["id"] for o in opts]
    cids = [c["id"] for c in P["compressions"]]
    check(len(set(eids + dids + sids)) == len(eids) + len(dids) + len(sids),
          "era/device/space ids are globally unique")
    check(len(set(vids)) == len(vids), "variant option ids unique")
    check(cids == ["off", "light", "medium"], "compressions are off/light/medium")
    check(any(m["id"] == "mic_sm7" for m in P["devices"]), "Shure SM7 present")
    check(any(m["id"] == "mic_re20" for m in P["devices"]), "EV RE20 present")
    check(any(m["id"] == "hall" for m in P["spaces"]), "concert hall present")
    check(any(m["id"] == "slapback" for m in P["spaces"]), "street slapback present")

    blob = json.dumps(P, ensure_ascii=False)
    check(not re.search(r"[\u0400-\u04FF]", blob), "no Cyrillic in the API payload")

    with open(os.path.join(ROOT, "app", "presets.py"), "r",
              encoding="utf-8") as fh:
        src_text = fh.read()
    check("_crk(" not in src_text, "hand-rolled crackle generator removed")
    check("vinyl.mp3" in src_text, "Vinyl mixes the bundled vinyl.mp3")
    with open(os.path.join(ROOT, "app", "engine.py"), "r",
              encoding="utf-8") as fh:
        engine_text = fh.read()
    check("def crackle(" not in engine_text,
          "dead crackle generator removed from the engine too")
    check("def cone_speaker(" in src_text and "isodynamic" not in src_text,
          "the paper-cone speaker preset is named consistently")
    tree = ast.parse(src_text)
    vinyl_fns = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name != "_vinyl_bed":
            seg = ast.get_source_segment(src_text, node) or ""
            if "_vinyl_bed(" in seg:
                vinyl_fns.add(node.name)
    check(vinyl_fns == {"_vinyl_layer"},
          "the vinyl bed is read only by the Vinyl layer: %s"
          % sorted(vinyl_fns))

    whistles = re.findall(r"_whistle\([^)]*?,\s*(-[\d.]+),", src_text)
    check(len(whistles) == 3, "heterodyne whistle in 3 radios: %s" % whistles)
    check(all(float(w) <= -42.0 for w in whistles),
          "heterodyne whistle is quieter everywhere (%s)" % whistles)

    st, hd, j = jreq("/api/upload", data=WAV,
                     headers={"X-Filename": "e2e%20test.wav"})
    check(st == 200 and "id" in j, "upload")
    uid = j["id"]
    check(re.fullmatch(r"[0-9a-f]{12}", uid or ""), "uid format %s" % uid)
    check(j.get("channels") == 1 and abs(j.get("sr") - sr) < 2, "file metadata")
    orig_url = j.get("url") or ""
    check(bool(orig_url), "upload returns the original url")

    st, hd, body = req(orig_url)
    check(st == 200 and body == WAV,
          "original bytes served in memory (%d bytes)" % len(body))
    r = urllib.request.Request(BASE + orig_url, headers={"Range": "bytes=0-99"})
    try:
        with urllib.request.urlopen(r, timeout=60) as resp:
            part = resp.read()
            check(resp.status == 206 and len(part) == 100
                  and part == WAV[:100],
                  "in-memory original Range -> 206, %d bytes" % len(part))
    except urllib.error.HTTPError as e:
        check(False, "original Range -> %d" % e.code)

    st, _, j = jreq("/api/process", data={"id": "zz", "epochs": ["radio_am"]})
    check(st == 400, "broken id -> 400")

    st, _, j = jreq("/api/process", data={"id": uid, "epochs": ["nope"]})
    check(st == 400 and "nope" in j.get("error", ""), "unknown era -> 400")

    st, _, j = jreq("/api/process", data={"id": uid, "spaces": ["nope"]})
    check(st == 400 and "nope" in j.get("error", ""), "unknown space -> 400")

    st, _, j = jreq("/api/process", data={"id": uid})
    check(st == 400, "empty selection -> 400")

    st, _, jc1 = jreq("/api/process", data={"id": uid, "compression": "light"})
    check(st == 200 and len(jc1["result"]["steps"]) == 1
          and "Compression" in jc1["result"]["steps"][0],
          "compression-only (no era/device) -> 200")
    st, _, jn1 = jreq("/api/process", data={"id": uid, "no_noise": True})
    check(st == 200 and jn1["result"]["no_noise"] is True,
          "noise-removal-only -> 200")
    st, _, jbn = jreq("/api/process",
                      data={"id": uid, "compression": "light",
                            "no_noise": True})
    check(st == 200, "compression + noise-removal only -> 200")

    st, _, j = jreq("/api/process",
                    data={"id": uid, "epochs": ["radio_am"],
                          "compression": "mega"})
    check(st == 400 and "mega" in j.get("error", ""), "bad compression -> 400")

    st, _, j = jreq("/api/process",
                    data={"id": uid, "variants": ["not_a_variant"]})
    check(st == 400, "unknown variant -> 400")

    bystage = {}
    for g in P["variants"]:
        bystage.setdefault(g["stage"], g["pairs"][0]["a"]["id"])
    picked = [bystage[k] for k in sorted(bystage)]
    check(len(picked) >= 1, "one variant per stage: %s" % picked)

    payload = {
        "id": uid,
        "epochs": ["cylinder", "shellac78", "lp_vinyl", "radio_am"],
        "devices": ["mic_ribbon44", "mic_sm7", "amp_lamp", "cone_speaker"],
        "spaces": ["hall", "slapback", "amphitheater"],
        "variants": picked,
        "strength": 1.0,
        "compression": "medium",
        "volume": 1.0,
        "no_noise": False,
    }
    t0 = time.time()
    st, hd, j = jreq("/api/process", data=payload)
    dt = time.time() - t0
    check(st == 200 and "result" in j, "full chain -> 200 ( %.1f s)" % dt)
    res = j.get("result", {})
    print("     steps:", " -> ".join(res.get("steps", [])) or "-")
    check(len(res.get("steps", [])) >= 1, "steps list present")
    check(len(res.get("url", "")) > 0 and res.get("download"), "url and download")
    check(res.get("format") in ("mp3", "wav"), "format mp3/wav")
    check(res.get("compression") == "medium", "compression echoed")
    check(abs(res.get("volume", 1.0) - 1.0) < 1e-6, "volume echoed")
    check(not re.search(r"[\u0400-\u04FF]", res.get("label", "")),
          "step labels are English")
    check("results" not in j, "no legacy results key (one file)")

    st, hd, body = req(res["download"])
    check(st == 200 and len(body) > 10000, "download: %d bytes" % len(body))
    check(body[:3] == b"ID3" or body[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"),
          "mp3 header: %r" % body[:3])

    st, hd, body = req(res["url"])
    check(st == 200 and len(body) > 10000, "playback by url: %d bytes" % len(body))

    r = urllib.request.Request(BASE + res["download"],
                               headers={"Range": "bytes=0-99"})
    try:
        with urllib.request.urlopen(r, timeout=60) as resp:
            part = resp.read()
            check(resp.status == 206 and len(part) == 100,
                  "Range -> 206, %d bytes" % len(part))
    except urllib.error.HTTPError as e:
        check(False, "Range -> %d" % e.code)

    st, _, j2 = jreq("/api/process", data=payload)
    check(st == 200 and j2["result"]["file"] == res["file"],
          "same request -> same file (in-memory, no copies)")

    quiet = dict(payload, no_noise=True)
    st, _, jq = jreq("/api/process", data=quiet)
    check(st == 200 and jq["result"]["no_noise"] is True, "no_noise=True accepted")
    check(jq["result"]["file"] != res["file"], "noise-free render is another file")

    half = dict(payload, volume=0.5)
    st, _, jh = jreq("/api/process", data=half)
    check(st == 200 and abs(jh["result"]["volume"] - 0.5) < 1e-6,
          "volume=0.5 accepted and echoed")

    full = dict(payload, volume=2.0)
    st, _, jf = jreq("/api/process", data=full)
    check(st == 200 and abs(jf["result"]["volume"] - 2.0) < 1e-6,
          "volume=2.0 accepted and echoed (max)")
    hot = dict(payload, volume=5.0)
    st, _, jh2 = jreq("/api/process", data=hot)
    check(st == 200 and abs(jh2["result"]["volume"] - 2.0) < 1e-6,
          "volume above 2.0 is clamped to 2.0")

    st, _, jss = jreq("/api/process",
                      data=dict(payload, spaces=["hall"], spaces_strength=0.4))
    check(st == 200
          and abs(jss["result"].get("spaces_strength", 1.0) - 0.4) < 1e-6,
          "spaces_strength accepted and echoed")
    st, _, jss2 = jreq("/api/process", data=dict(payload, spaces_strength=9.0))
    check(st == 200
          and abs(jss2["result"].get("spaces_strength", 1.0) - 1.0) < 1e-6,
          "spaces_strength above 1.0 is clamped")

    st, _, jbz = jreq("/api/process",
                      data=dict(payload, variants=["brick"], brick_hz=250))
    check(st == 200 and abs(jbz["result"].get("brick_hz", 90.0) - 250.0) < 1e-6,
          "brick_hz accepted and echoed")
    st, _, jbz2 = jreq("/api/process",
                       data=dict(payload, variants=["brick"], brick_hz=99999))
    check(st == 200 and abs(jbz2["result"].get("brick_hz", 90.0) - 2000.0) < 1e-6,
          "brick_hz is clamped to 2000")

    for c in cids:
        st, _, jc = jreq("/api/process", data=dict(payload, compression=c))
        check(st == 200, "compression %-6s -> 200 (%s)"
              % (c, jc.get("result", {}).get("file", "?")))

    for s in (0.0, 0.05, 0.5, 1.0):
        st, _, js = jreq("/api/process", data=dict(payload, strength=s))
        check(st == 200 and abs(js["result"]["strength"] - s) < 1e-6,
              "strength=%.2f -> 200" % s)

    for space in sids:
        st, _, jsp = jreq("/api/process",
                          data={"id": uid, "spaces": [space], "strength": 1.0})
        check(st == 200, "space only (%s) -> 200" % space)

    st, _, jv = jreq("/api/process",
                     data={"id": uid, "variants": [vids[0]],
                           "compression": "off", "strength": 1.0})
    check(st == 200 and len(jv["result"]["steps"]) >= 1,
          "variant only (no era/device) -> 200")

    st, _, jl = jreq("/api/process",
                     data={"id": uid, "presets": ["radio_am"], "strength": 0.8})
    check(st == 200, "legacy presets field -> 200")
    st, _, jl = jreq("/api/process", data={"id": uid, "presets": ["hall"]})
    check(st == 200, "legacy presets with a space -> 200")

    hx = (0.3 * np.sin(2 * np.pi * 440 * t))[None, :].astype(np.float32)

    def hum_line(y):
        y2 = y.mean(axis=0) if y.ndim == 2 else y
        Y = np.abs(np.fft.rfft(y2))
        f = np.fft.rfftfreq(y2.size, 1.0 / sr)
        i = int(round(50.0 * (y2.size / sr)))
        m = (f >= 34) & (f <= 66) & (np.abs(f - 50.0) > 0.5)
        med = float(np.median(Y[m]) + 1e-12)
        return float(Y[i] / med)

    hy_off, _ = presets.apply_chain(hx, sr, epochs=["lamp_radio"],
                                    compression="off")
    hy_on, _ = presets.apply_chain(hx, sr, epochs=["lamp_radio"],
                                   variants=["hum"], compression="off")
    hy_base, _ = presets.apply_chain(hx.copy(), sr, variants=["hum"],
                                     compression="off")
    check(hum_line(hy_on) > 5 * hum_line(hy_off) and hum_line(hy_on) > 5,
          "50 Hz line appears only when the '50 Hz' character is on")
    check(hum_line(hy_off) < 5,
          "era hum stays muted without the '50 Hz' character")
    check(hum_line(hy_base) > 5,
          "'50 Hz' injects hum even when no era supplies it")

    vy_off, _ = presets.apply_chain(hx, sr, epochs=["lp_vinyl"],
                                    compression="off")
    vy_on, _ = presets.apply_chain(hx, sr, epochs=["lp_vinyl"],
                                   variants=["vinyl"], compression="off")

    vdiff = float(np.max(np.abs(vy_on - vy_off)))
    check(vdiff > 0.01,
          "'Vinyl' option adds surface crackle transients (%.4f)" % vdiff)

    vbed = presets._vinyl_bed(sr)
    vtile = vbed[np.arange(hx.shape[-1]) % vbed.size]
    vlayer = presets._vinyl_layer(hx.shape[-1], sr, 1.0)
    check(float(np.max(np.abs(vlayer - presets.VINYL_GAIN * vtile))) < 1e-6,
          "Vinyl plays the file as authored, lifted %.2fx (%.1f dB)"
          % (presets.VINYL_GAIN, 20.0 * np.log10(presets.VINYL_GAIN)))

    # Switching Vinyl on must not change the programme's level: it is mixed in
    # after the programme is levelled, riding on the headroom above it.
    base, _ = presets.apply_chain(hx.copy(), sr, variants=["dark"],
                                  compression="off", strength=1.0)
    withv, _ = presets.apply_chain(hx.copy(), sr, variants=["dark", "vinyl"],
                                   compression="off", strength=1.0)
    room = np.maximum(0.985 - np.abs(base.mean(axis=0)), 1e-4)
    expected = base + np.tanh(vlayer / room) * room
    check(float(np.max(np.abs(withv - expected))) < 1e-6,
          "Vinyl rides on top without ducking the programme")

    room_full, _ = presets.apply_chain(hx.copy(), sr, spaces=["hall"],
                                       compression="off", strength=1.0)
    room_low, _ = presets.apply_chain(hx.copy(), sr, spaces=["hall"],
                                      compression="off", strength=1.0,
                                      spaces_strength=0.3)
    check(float(np.max(np.abs(room_full - room_low))) > 1e-4,
          "spaces strength changes how much the room bleeds into the mix")

    rng_r = np.random.default_rng(9)
    gate = 0.32 * np.sin(2 * np.pi * 500 * t) * (np.mod(t, 2.0) < 1.0)
    noisysrc = (gate + 0.012 * rng_r.standard_normal(t.size)
                )[None, :].astype(np.float32)
    plain, _ = presets.apply_chain(noisysrc.copy(), sr, no_noise=True,
                                   compression="off")
    fsrc = (gate + 0.0005 * np.random.default_rng(11)
            .standard_normal(t.size))[None, :].astype(np.float32)
    lp_on, _ = presets.apply_chain(fsrc.copy(), sr, epochs=["lp_vinyl"],
                                   strength=1.0, compression="off")
    lp_off, _ = presets.apply_chain(fsrc.copy(), sr, epochs=["lp_vinyl"],
                                    strength=1.0, compression="off",
                                    no_noise=True)
    sh_on, _ = presets.apply_chain(fsrc.copy(), sr, epochs=["shellac78"],
                                   strength=1.0, compression="off")
    sh_off, _ = presets.apply_chain(fsrc.copy(), sr, epochs=["shellac78"],
                                    strength=1.0, compression="off",
                                    no_noise=True)

    def seg_rms(y, a, b):
        y2 = y.mean(axis=0) if y.ndim == 2 else y
        return float(np.sqrt(np.mean(y2[int(a * sr):int(b * sr)] ** 2)))

    def noise_floor(on, off):
        a, b = seg_rms(on, 1.05, 1.95), seg_rms(off, 1.05, 1.95)
        return float(np.sqrt(max(a * a - b * b, 0.0)))

    lvl_in = seg_rms(plain, 0.05, 0.95)
    lvl_out = seg_rms(lp_on, 0.05, 0.95)
    check(lvl_out > 0.6 * lvl_in,
          "LP vinyl keeps the programme intact (%.2f -> %.2f)"
          % (lvl_in, lvl_out))
    fl_lp = noise_floor(lp_on, lp_off)
    fl_sh = noise_floor(sh_on, sh_off)
    check(fl_lp > 1e-6,
          "LP vinyl still has a surface noise floor (%.2e)" % fl_lp)

    # The requested per-era cuts are wired in and actually take effect.
    check(presets.ERA_NOISE_TRIM.get("shellac78") == 0.25
          and presets.ERA_NOISE_TRIM.get("wire_rec") == 0.2
          and presets.ERA_NOISE_TRIM.get("electrecord") == 0.5
          and presets.ERA_NOISE_TRIM.get("lp_vinyl") == 0.31875
          and presets.ERA_NOISE_TRIM.get("cylinder") == 0.5
          and presets.ERA_NOISE_TRIM.get("acoustic1910") == 0.5
          and presets.ERA_NOISE_TRIM.get("det_radio") == 0.5,
          "era noise trims match the requested cuts")
    check(presets.DEVICE_NOISE_TRIM == 0.5,
          "every Device-section effect is trimmed -50%")
    saved_trim = presets.ERA_NOISE_TRIM["shellac78"]
    presets.ERA_NOISE_TRIM["shellac78"] = 1.0
    sh_full, _ = presets.apply_chain(fsrc.copy(), sr, epochs=["shellac78"],
                                     strength=1.0, compression="off")
    presets.ERA_NOISE_TRIM["shellac78"] = saved_trim
    fl_sh_full = noise_floor(sh_full, sh_off)
    check(fl_sh < 0.45 * fl_sh_full,
          "shellac floor really dropped after the -75%% cut (%.2e vs %.2e)"
          % (fl_sh, fl_sh_full))

    # Brick: a 96 dB/oct low cut at 95 Hz plus a low shelf that tames the
    # boom in the lows.  Probe with on-bin sines so the FFT has no leakage.
    probe = (0.45 * np.sin(2 * np.pi * 30.0 * t)
             + 0.45 * np.sin(2 * np.pi * 60.0 * t)
             + 0.45 * np.sin(2 * np.pi * 190.0 * t)
             + 0.45 * np.sin(2 * np.pi * 1000.0 * t))[None, :].astype(np.float32)
    pb, _ = presets.apply_chain(probe.copy(), sr, variants=["brick"],
                                strength=1.0, compression="off", no_noise=True)
    steady = pb.mean(axis=0)[int(0.5 * sr):int(3.5 * sr)]
    PB = np.abs(np.fft.rfft(steady))
    pf = np.fft.rfftfreq(steady.size, 1.0 / sr)

    def pamp(fr):
        return float(PB[np.argmin(np.abs(pf - fr))]) + 1e-12

    d1k = pamp(1000.0)
    d190 = 20.0 * np.log10(pamp(190.0) / d1k)
    d60 = 20.0 * np.log10(pamp(60.0) / d1k)
    check(-3.5 <= d190 <= -0.5,
          "Brick low shelf tames the boom (%.1f dB at 190 Hz)" % d190)
    check(d60 <= -50.0,
          "Brick is a very steep wall (%.0f dB at 60 Hz)" % d60)

    # The Brick wall is exempt from the effect strength: full depth always.
    b0, _ = presets.apply_chain(probe.copy(), sr, variants=["brick"],
                                strength=0.0, compression="off", no_noise=True)
    b1, _ = presets.apply_chain(probe.copy(), sr, variants=["brick"],
                                strength=1.0, compression="off", no_noise=True)
    check(float(np.max(np.abs(b0 - b1))) < 1e-6,
          "Brick ignores the effect strength")

    # Loudness balance: no era/device/space may finish louder than the
    # source once both are normalised to the same peak.
    lrng = np.random.default_rng(5)
    lsrc = (0.22 * np.sin(2 * np.pi * 110 * t)
            + 0.10 * np.sin(2 * np.pi * 440 * t)
            + 0.008 * lrng.standard_normal(t.size))[None, :].astype(np.float32)
    src_r = float(np.sqrt(np.mean(
        (lsrc.mean(axis=0)) ** 2)))
    ref_l = src_r * (0.97 / (float(np.max(np.abs(lsrc))) + 1e-12))
    worst = ("none", -99.0)
    for key in ("epochs", "devices", "spaces"):
        for pid_ in [p["id"] for p in P[key]]:
            y, _ = presets.apply_chain(lsrc.copy(), sr, strength=1.0,
                                       compression="off", **{key: [pid_]})
            r = float(np.sqrt(np.mean(
                (y.mean(axis=0) if y.ndim == 2 else y) ** 2)))
            d = 20.0 * np.log10(r / ref_l + 1e-12)
            if d > worst[1]:
                worst = (pid_, d)
    check(worst[1] <= 1.5,
          "no preset outruns the source loudness (worst %s %+.1f dB)"
          % worst)

    churn_uids = []
    for i in range(7):
        st, _, ju = jreq("/api/upload", data=WAV,
                         headers={"X-Filename": "churn%d.wav" % i})
        churn_uids.append(ju.get("id"))
        check(st == 200 and churn_uids[-1], "upload churn %d -> 200" % i)
    check(len(server._uploads) <= server.MAX_UPLOADS,
          "uploads are LRU-capped at %d (now %d)"
          % (server.MAX_UPLOADS, len(server._uploads)))
    evicted_url = orig_url
    st, hd, body = req(evicted_url)
    check(st != 200, "oldest upload is evicted from memory (no disk residue)")

    st, _, jres = jreq("/api/results")
    check(st == 404, "/api/results is gone (no history) -> 404")

    st, hd, body = req("/")
    check(st == 200 and b"VintageFX" in body, "index.html served from root")
    html = body.decode("utf-8")
    check(not re.search(r"[\u0400-\u04FF]", html), "index.html is fully English")
    check("nothing selected" not in html,
          "'nothing selected' placeholders removed")
    check("localStorage" not in html,
          "no settings persistence (nothing saved across reloads)")
    check(".drop" in html and "cursor:pointer" in html,
          "drop zone has a pointer cursor across the whole area")
    check("class=\"top\"" not in html and "<header" not in html,
          "top header removed")
    check("All recordings" not in html and "api/results" not in html,
          "history section removed")
    check('id="srcPlayer"' in html, "standard player for the original exists")
    check('id="spacesStrength"' in html and 'id="spacesQuick"' in html,
          "separate Rooms & spaces strength control is present")
    check('class="brand"' in html,
          "title block sits where the drop zone used to be")
    check("Era effects for your sound" not in html and "◉" not in html,
          "brand is just the plain VintageFX text")
    mbox = re.search(r'id="srcBox"(.{0,500}?)id="drop"', html, re.S)
    check(bool(mbox) and "</div>" not in mbox.group(1),
          "file drop zone lives inside the srcbox")
    check("Mutes hiss, clicks, hum and vinyl crackle." in html
          and "The era character stays" not in html,
          "noise caption trimmed to the one useful sentence")
    check('id="volume" min="0" max="200"' in html,
          "volume slider spans 0-200 with 100 in the middle")
    check('(v / 2) + "%"' in html, "volume midpoint mapping (100% -> center)")
    check("chip-x" in html and "data-set=" in html,
          "chain chips have a remove button")
    check("optrow" in html, "character options live in a single row")
    check("File accepted" not in html, "old 'File accepted' text removed")
    check("{v: 0, l: \"0%\"}" in html and "{v: 100, l: \"100%\"}" in html,
          "strength steps 0/25/50/75/100")
    check("<footer" not in html, "footer block removed")
    check("<!--" not in html, "no comments in index.html")

    for path in ("/static/nothing.js", "/api/none"):
        st, _, _ = req(path)
        check(st == 404, "%s -> 404" % path)

    st, hd, body = req("/in/missing.mp3")
    check(st == 404, "missing original -> 404")

    st, hd, body = req(res["url"] + ".restart-check")
    check(st == 404, "unknown result key -> 404")

    check(not os.path.isdir(os.path.join(ROOT, "data")),
          "no data folder was ever created on disk")
    check(not os.path.isdir(os.path.join(ROOT, "static")),
          "static folder removed")
    check(os.path.isfile(os.path.join(ROOT, "app", "index.html")),
          "index.html lives inside app/")


if __name__ == "__main__":
    main()