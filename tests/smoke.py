# -*- coding: utf-8 -*-
"""Import smoke: era count/order, device swap, option order, constants."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(os.path.dirname(HERE), "app")
os.chdir(APP)
sys.path.insert(0, APP)
sys.dont_write_bytecode = True
import presets as P  # noqa: E402

eras = list(P.EPOCHS.keys())
assert len(eras) == 20, len(eras)
assert "remaster" not in eras and "vhs" not in eras, eras
assert "amp_jp" in eras
assert eras[eras.index("stereo_hifi") + 1] == "amp_jp", eras
print("eras:", len(eras))
print("after stereo_hifi:", eras[eras.index("stereo_hifi") + 1])

assert "amp_jp" not in P.DEVICES, "amp_jp must leave the Device section"
assert "fm_receiver" in P.DEVICES
assert len(P.DEVICES) == 24, len(P.DEVICES)
assert len(P.SPACES) == 16

pairs = sorted((pr for g in P.VARIANT_GROUPS for pr in g["pairs"]),
               key=lambda p: p["pos"])
oids = []
for pr in pairs:
    if "a" in pr:
        oids.append(pr["a"]["id"])
    if "b" in pr:
        oids.append(pr["b"]["id"])
print("opts:", oids)
assert oids == ["vinyl", "brick", "hum", "boxy", "dark", "bright"], oids
assert P.VARIANT_ORDER == ["surface", "brick", "supply", "body", "tone"], P.VARIANT_ORDER

assert set(P.ERA_NOISE_TRIM) == set(P.EPOCHS), "era noise trims cover every era"
assert P.DEVICE_NOISE_TRIM == 0.5, P.DEVICE_NOISE_TRIM
print("noise trims:", len(P.ERA_NOISE_TRIM), "eras,", P.DEVICE_NOISE_TRIM, "devices")

assert "cone_speaker" in P.DEVICES and "isodynamic" not in P.DEVICES
for pid, meta in {**P.EPOCHS, **P.DEVICES, **P.SPACES}.items():
    assert meta["name"], pid
    assert meta["desc"] and len(meta["desc"]) > 12, pid
    assert callable(meta["fn"]), pid
    assert isinstance(meta.get("tags"), list), pid
print("presets consistent:", len(P.EPOCHS) + len(P.DEVICES) + len(P.SPACES))

sig = [g for g in P.VARIANT_GROUPS if g["id"] == "signal"][0]
assert [pr["id"] for pr in sig["pairs"]] == ["brick"], sig["pairs"]
chr_ = [g for g in P.VARIANT_GROUPS if g["id"] == "character"][0]
assert [pr["id"] for pr in chr_["pairs"]] == ["supply", "surface", "body"]
bri = [g for g in P.VARIANT_GROUPS if g["id"] == "brightness"][0]
assert [pr["id"] for pr in bri["pairs"]] == ["tone"]

assert abs(P.LOUD_TOL - 1.12) < 1e-9
print("LOUD_TOL:", P.LOUD_TOL)
print("OK")
