from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
import sys
import threading
import time
import traceback
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)
sys.dont_write_bytecode = True   # keep the app folder free of __pycache__

import engine
import presets

MAX_UPLOAD = 300 * 1024 * 1024
MAX_UPLOADS = 6
MAX_RESULTS = 8
ALLOWED_EXT = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".opus",
               ".aiff", ".wma"}

_CTYPES = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".opus": "audio/ogg",
    ".aiff": "audio/aiff",
    ".wma": "audio/x-ms-wma",
}

_lock = threading.Lock()
_uploads: dict[str, dict] = {}
_upload_files: dict[str, str] = {}
_results: list[str] = []
_results_map: dict[str, dict] = {}

BROWSER_OPENED = {"done": False}


def _uid() -> str:
    return uuid.uuid4().hex[:12]


def _ctype(name: str) -> str:
    ext = os.path.splitext(name)[1].lower()
    return (_CTYPES.get(ext)
            or mimetypes.guess_type(name)[0]
            or "application/octet-stream")


def _safe_name(name: str) -> str:
    name = os.path.basename(name)
    name = re.sub(r"[^\w\-.() \[\]]+", "_", name, flags=re.UNICODE)
    return name[:120] or "audio"


def upload_add(rec: dict) -> dict:
    with _lock:
        while len(_uploads) >= MAX_UPLOADS:
            old = next(iter(_uploads))
            victim = _uploads.pop(old, None)
            if victim:
                _upload_files.pop(victim["file"], None)
        uid = rec["id"]
        _uploads[uid] = rec
        _upload_files[rec["file"]] = uid
        rec["ts"] = int(time.time())
    return rec


def upload_get(uid: str):
    with _lock:
        return _uploads.get(uid)


def result_add(key: str, data: bytes, meta: dict) -> dict:
    with _lock:
        _results_map[key] = {"data": data, "meta": meta}
        if key in _results:
            _results.remove(key)
        _results.append(key)
        while len(_results) > MAX_RESULTS:
            k = _results.pop(0)
            _results_map.pop(k, None)
    return meta


def result_get(key: str):
    with _lock:
        r = _results_map.get(key)
        if not r:
            return None
        return r["data"], r["meta"]


def process_track(uid: str, spec: dict) -> dict:
    rec = upload_get(uid)
    if rec is None:
        raise LookupError("File not found. Upload the track again.")
    x, sr = rec["audio"], rec["sr"]
    out, steps = presets.apply_chain(
        x, sr,
        epochs=spec.get("epochs") or [],
        devices=spec.get("devices") or [],
        spaces=spec.get("spaces") or [],
        variants=spec.get("variants") or [],
        strength=float(spec.get("strength", 0.7)),
        compression=str(spec.get("compression") or "off"),
        no_noise=bool(spec.get("no_noise", False)),
        volume=float(spec.get("volume", 1.0)),
        spaces_strength=float(spec.get("spaces_strength", 1.0)),
        brick_hz=float(spec.get("brick_hz", 90.0)),
    )
    dur = out.shape[-1] / float(sr)

    token = hashlib.md5(
        json.dumps(spec, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:8]
    key = f"{uid}__vfx__{token}.mp3"
    data, fmt = engine.encode_audio(out, sr)
    if fmt == "wav":
        key = key[:-4] + ".wav"

    meta = {
        "file": key,
        "url": "/out/" + key,
        "download": "/download/" + key,
        "format": fmt,
        "duration": round(dur, 2),
        "steps": list(steps),
        "label": " → ".join(steps),
        "strength": round(float(spec.get("strength", 0.7)), 3),
        "compression": str(spec.get("compression") or "off"),
        "volume": round(float(spec.get("volume", 1.0)), 3),
        "spaces_strength": round(float(spec.get("spaces_strength", 1.0)), 3),
        "brick_hz": round(float(spec.get("brick_hz", 90.0)), 2),
        "no_noise": bool(spec.get("no_noise", False)),
        "uid": uid,
        "src": {"name": rec.get("name") or uid, "url": rec.get("url") or ""},
        "ts": int(time.time()),
    }
    result_add(key, data, meta)
    return meta


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "VintageFX/1.0"

    def _json(self, obj, code: int = 200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_bytes(self, data: bytes, ctype: str, name: str = None,
                    download: bool = False):
        size = len(data)
        start, end = 0, max(0, size - 1)
        rng = self.headers.get("Range")
        code = 200
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng.strip())
            if m:
                if m.group(1):
                    start = int(m.group(1))
                    if m.group(2):
                        end = min(int(m.group(2)), size - 1)
                elif m.group(2):
                    start = max(0, size - int(m.group(2)))
                if size == 0 or start > end or start >= size:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                code = 206
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        if code == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if download and name:
            fn = urllib.parse.quote(name)
            self.send_header("Content-Disposition",
                             f"attachment; filename*=UTF-8''{fn}")
        self.end_headers()
        self.wfile.write(data[start:end + 1])

    def _send_page(self):
        page = os.path.join(APP_DIR, "index.html")
        try:
            with open(page, "rb") as fh:
                html = fh.read()
        except OSError:
            self._json({"error": "index.html is missing"}, 500)
            return
        self._send_bytes(html, "text/html; charset=utf-8")

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return b""
        if length > MAX_UPLOAD:
            raise ValueError("File is too large (300 MB limit)")
        data = b""
        while len(data) < length:
            chunk = self.rfile.read(min(1 << 20, length - len(data)))
            if not chunk:
                break
            data += chunk
        return data

    def do_GET(self):
        try:
            parsed = urllib.parse.urlparse(self.path)
            path = urllib.parse.unquote(parsed.path)

            if path in ("/", "/index.html"):
                self._send_page()
                return

            if path == "/api/presets":
                def _cards(reg):
                    return [{
                        "id": pid,
                        "name": m["name"],
                        "emoji": m["emoji"],
                        "desc": m["desc"],
                        "tags": m.get("tags", []),
                        "special": bool(m.get("special", False)),
                    } for pid, m in reg.items()]

                groups = [{
                    "id": g["id"],
                    "name": g["name"],
                    "hint": g["hint"],
                    "stage": g["stage"],
                    "pairs": [{
                        "id": p["id"],
                        "name": p["name"],
                        "pos": p["pos"],
                        "a": {k: p["a"][k]
                              for k in ("id", "name", "emoji", "desc")},
                        "b": ({k: p["b"][k]
                               for k in ("id", "name", "emoji", "desc")}
                              if "b" in p else None),
                    } for p in g["pairs"]],
                } for g in presets.VARIANT_GROUPS]

                self._json({
                    "presets": _cards(presets.PRESETS),
                    "epochs": _cards(presets.EPOCHS),
                    "devices": _cards(presets.DEVICES),
                    "spaces": _cards(presets.SPACES),
                    "variants": groups,
                    "compressions": [{
                        "id": c["id"],
                        "name": c["name"],
                        "emoji": c["emoji"],
                        "desc": c["desc"],
                    } for c in presets.COMPRESSIONS],
                })
                return

            if path.startswith("/out/"):
                got = result_get(os.path.basename(path))
                if got is None:
                    self._json({"error": "Result not found"}, 404)
                    return
                data, meta = got
                self._send_bytes(data, _ctype(meta["file"]))
                return

            if path.startswith("/download/"):
                got = result_get(os.path.basename(path[len("/download/"):]))
                if got is None:
                    self._json({"error": "Result not found"}, 404)
                    return
                data, meta = got
                self._send_bytes(data, _ctype(meta["file"]),
                                 name=meta["file"], download=True)
                return

            if path.startswith("/in/"):
                fname = os.path.basename(path)
                with _lock:
                    uid = _upload_files.get(fname)
                    rec = _uploads.get(uid) if uid else None
                if rec is None:
                    self._json({"error": "File not found"}, 404)
                    return
                self._send_bytes(rec["raw"], _ctype(fname), name=fname)
                return

            self._json({"error": "Not found"}, 404)
        except BrokenPipeError:
            pass
        except Exception:
            traceback.print_exc()
            try:
                self._json({"error": "Internal server error"}, 500)
            except Exception:
                pass

    def do_POST(self):
        try:
            path = urllib.parse.urlparse(self.path).path

            if path == "/api/upload":
                raw = self._read_body()
                name = urllib.parse.unquote(
                    self.headers.get("X-Filename")
                    or self.headers.get("X-File-Name") or ""
                )
                name = _safe_name(name or "audio.mp3")
                ext = os.path.splitext(name)[1].lower()
                if ext not in ALLOWED_EXT:
                    self._json({"error": f"Format {ext or '?'} is not supported. "
                                         f"Use mp3, wav, flac, ogg, m4a..."}, 400)
                    return
                if not raw:
                    self._json({"error": "Empty file"}, 400)
                    return
                try:
                    x, sr = engine.decode_bytes(raw)
                except Exception as e:
                    self._json({"error": f"Could not read audio: {e}"}, 400)
                    return
                uid = _uid()
                rec = {
                    "id": uid,
                    "uid": uid,
                    "name": name,
                    "file": uid + ext,
                    "duration": round(x.shape[-1] / float(sr), 2),
                    "sr": sr,
                    "channels": int(x.shape[0]),
                    "size": len(raw),
                    "url": "/in/" + uid + ext,
                    "download": "/in/" + uid + ext,
                    "raw": raw,
                    "audio": x,
                }
                rec = upload_add(rec)
                self._json({k: rec[k] for k in (
                    "id", "uid", "name", "duration", "sr", "channels",
                    "size", "url", "download", "ts")})
                return

            if path == "/api/process":
                body = self._read_body()
                req = json.loads(body.decode("utf-8"))
                uid = str(req.get("id") or "")
                if not re.fullmatch(r"[0-9a-f]{12}", uid):
                    self._json({"error": "Invalid file id"}, 400)
                    return

                def _list(key):
                    v = req.get(key)
                    if v is None:
                        return []
                    if isinstance(v, str):
                        v = [v]
                    if not isinstance(v, list):
                        raise ValueError(f"Field {key} must be a list")
                    return [str(i) for i in v]

                epochs = _list("epochs")
                devices = _list("devices")
                spaces = _list("spaces")
                variants = _list("variants")

                if not (epochs or devices or spaces):
                    for pid in _list("presets"):
                        if pid in presets.EPOCHS:
                            epochs.append(pid)
                        elif pid in presets.DEVICES:
                            devices.append(pid)
                        elif pid in presets.SPACES:
                            spaces.append(pid)

                bad = ([p for p in epochs if p not in presets.EPOCHS]
                       + [p for p in devices if p not in presets.DEVICES]
                       + [p for p in spaces if p not in presets.SPACES]
                       + [v for v in variants if v not in presets.VARIANT_INDEX])
                if bad:
                    self._json({"error": "Unknown options: "
                                         + ", ".join(sorted(set(bad)))}, 400)
                    return

                compression = str(req.get("compression") or "off")
                if compression not in presets.COMPRESSION_BY_ID:
                    self._json({"error": "Unknown compression level: "
                                         + compression}, 400)
                    return

                no_noise = bool(req.get("no_noise", False))

                if not (epochs or devices or spaces or variants
                        or compression != "off" or no_noise):
                    self._json({"error": "Select an era, device, space, variant, "
                                         "a compression level or enable noise removal"},
                               400)
                    return

                strength = float(req.get("strength", 0.7))
                strength = max(0.0, min(1.0, strength))
                volume = float(req.get("volume", 1.0))
                volume = max(0.0, min(2.0, volume))
                spaces_strength = float(req.get("spaces_strength", 1.0))
                spaces_strength = max(0.0, min(1.0, spaces_strength))
                brick_hz = float(req.get("brick_hz", 90.0))
                brick_hz = max(20.0, min(2000.0, brick_hz))

                spec = {
                    "epochs": epochs,
                    "devices": devices,
                    "spaces": spaces,
                    "variants": variants,
                    "strength": round(strength, 3),
                    "compression": compression,
                    "volume": round(volume, 3),
                    "spaces_strength": round(spaces_strength, 3),
                    "brick_hz": round(brick_hz, 2),
                    "no_noise": no_noise,
                }
                meta = process_track(uid, spec)
                self._json({"result": meta})
                return

            self._json({"error": "Not found"}, 404)
        except BrokenPipeError:
            pass
        except ValueError as e:
            self._json({"error": str(e)}, 400)
        except KeyError as e:
            self._json({"error": str(e)}, 400)
        except Exception:
            traceback.print_exc()
            try:
                self._json({"error": "Could not process the track"}, 500)
            except Exception:
                pass

    def log_message(self, fmt, *args):
        try:
            sys.stdout.write("%s - %s\n" % (self.log_date_time_string(), fmt % args))
        except Exception:
            pass


class Server(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1],
                      (ConnectionResetError, BrokenPipeError, TimeoutError)):
            return
        super().handle_error(request, client_address)


def open_browser(port: int):
    import webbrowser
    time.sleep(0.8)
    if os.environ.get("VFX_NO_BROWSER"):
        return
    if not BROWSER_OPENED["done"]:
        BROWSER_OPENED["done"] = True
        try:
            webbrowser.open(f"http://127.0.0.1:{port}/")
        except Exception:
            pass


def find_port(start: int = 8765, attempts: int = 30) -> int:
    import socket
    for p in range(start, start + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise RuntimeError("No free port")


def main():
    port = find_port()
    httpd = Server(("127.0.0.1", port), Handler)
    print("=" * 60)
    print("  VintageFX - era effects for your sound")
    print("  Everything runs in memory. Nothing is saved on disk")
    print(f"  Open:    http://127.0.0.1:{port}/")
    print("  Stop:    Ctrl+C or close this window")
    print("=" * 60)
    threading.Thread(target=open_browser, args=(port,), daemon=True).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nBye!")
        httpd.server_close()


if __name__ == "__main__":
    main()