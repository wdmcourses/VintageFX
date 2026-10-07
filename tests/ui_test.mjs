import { spawn } from "node:child_process";
import net from "node:net";
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const CHROME = "C:/Program Files/Google/Chrome/Application/chrome.exe";
const BASE = process.argv[2] || "http://127.0.0.1:8765/";
const WS_PORT = 9333;
const PROFILE = "C:/Users/alex/AppData/Local/Temp/opencode/cdpprofile";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let failures = 0;

function check(cond, msg) {
  console.log(`[${cond ? "OK  " : "FAIL"}] ${msg}`);
  if (!cond) failures++;
}

class CDP {
  constructor(url) {
    this.url = url;
    this.id = 0;
    this.pending = new Map();
    this.buf = Buffer.alloc(0);
    this.upgraded = false;
  }
  connect() {
    const u = new URL(this.url);
    return new Promise((res, rej) => {
      this.sock = net.connect(Number(u.port), u.hostname);
      this.sock.once("error", rej);
      this.sock.once("connect", () => {
        const key = crypto.randomBytes(16).toString("base64");
        this.sock.write(
          `GET ${u.pathname} HTTP/1.1\r\nHost: ${u.host}\r\n` +
          "Upgrade: websocket\r\nConnection: Upgrade\r\n" +
          `Sec-WebSocket-Key: ${key}\r\nSec-WebSocket-Version: 13\r\n\r\n`);
      });
      this.sock.on("data", (d) => {
        if (!this.upgraded) {
          this.buf = Buffer.concat([this.buf, d]);
          const i = this.buf.indexOf("\r\n\r\n");
          if (i < 0) return;
          const head = this.buf.slice(0, i).toString();
          if (!/101/.test(head)) { rej(new Error("handshake: " + head)); return; }
          this.upgraded = true;
          this.buf = this.buf.slice(i + 4);
          res();
          this.parse();
        } else {
          this.buf = Buffer.concat([this.buf, d]);
          this.parse();
        }
      });
    });
  }
  parse() {
    for (;;) {
      if (this.buf.length < 2) return;
      const b0 = this.buf[0], b1 = this.buf[1];
      const opcode = b0 & 0x0f;
      const masked = (b1 & 0x80) !== 0;
      let len = b1 & 0x7f, off = 2;
      if (len === 126) {
        if (this.buf.length < 4) return;
        len = this.buf.readUInt16BE(2); off = 4;
      } else if (len === 127) {
        if (this.buf.length < 10) return;
        len = Number(this.buf.readBigUInt64BE(2)); off = 10;
      }
      const maskLen = masked ? 4 : 0;
      if (this.buf.length < off + maskLen + len) return;
      let payload = this.buf.slice(off + maskLen, off + maskLen + len);
      if (masked) {
        const mask = this.buf.slice(off, off + 4);
        payload = Buffer.from(payload.map((b, i) => b ^ mask[i % 4]));
      }
      this.buf = this.buf.slice(off + maskLen + len);
      if (opcode === 1) this.onMessage(payload.toString());
      else if (opcode === 8) return;
    }
  }
  onMessage(text) {
    let m;
    try { m = JSON.parse(text); } catch { return; }
    if (m.id !== undefined && this.pending.has(m.id)) {
      const { res, rej } = this.pending.get(m.id);
      this.pending.delete(m.id);
      if (m.error) rej(new Error(JSON.stringify(m.error)));
      else res(m.result);
    }
  }
  send(method, params = {}) {
    const id = ++this.id;
    const payload = Buffer.from(JSON.stringify({ id, method, params }));
    const mask = crypto.randomBytes(4);
    const n = payload.length;
    let header;
    if (n < 126) header = Buffer.from([0x81, 0x80 | n]);
    else if (n < 65536) {
      header = Buffer.alloc(4);
      header[0] = 0x81; header[1] = 0x80 | 126; header.writeUInt16BE(n, 2);
    } else {
      header = Buffer.alloc(10);
      header[0] = 0x81; header[1] = 0x80 | 127; header.writeBigUInt64BE(BigInt(n), 2);
    }
    const masked = Buffer.from(payload.map((b, i) => b ^ mask[i % 4]));
    this.sock.write(Buffer.concat([header, mask, masked]));
    return new Promise((res, rej) => this.pending.set(id, { res, rej }));
  }
  close() { try { this.sock.destroy(); } catch {} }
}

async function evaluate(cdp, expression) {
  const r = await cdp.send("Runtime.evaluate", {
    expression, returnByValue: true, awaitPromise: true
  });
  if (r.exceptionDetails) throw new Error(JSON.stringify(r.exceptionDetails));
  return r.result.value;
}

async function wait(cdp, expression, tries = 30) {
  for (let i = 0; i < tries; i++) {
    try { if (await evaluate(cdp, expression)) return true; } catch {}
    await sleep(300);
  }
  return false;
}

async function main() {
  fs.rmSync(PROFILE, { recursive: true, force: true });
  const chrome = spawn(CHROME, [
    "--headless=new", "--disable-gpu", "--no-sandbox",
    `--remote-debugging-port=${WS_PORT}`,
    `--user-data-dir=${PROFILE}`,
    "--autoplay-policy=no-user-gesture-required",
    "--no-first-run", "--no-default-browser-check",
    "about:blank"
  ], { stdio: "ignore" });

  let list;
  for (let i = 0; i < 60; i++) {
    try {
      const r = await fetch(`http://127.0.0.1:${WS_PORT}/json/list`);
      list = await r.json();
      if (list.length) break;
    } catch {}
    await sleep(250);
  }
  const page = (list || []).find((t) => t.type === "page");
  if (!page) { console.log("FAIL no devtools page"); process.exit(1); }

  const cdp = new CDP(page.webSocketDebuggerUrl);
  await cdp.connect();
  await cdp.send("Page.enable");
  await cdp.send("Runtime.enable");
  await cdp.send("Emulation.setDeviceMetricsOverride",
    { width: 1440, height: 900, deviceScaleFactor: 1, mobile: false });

  try {
    await cdp.send("Page.navigate", { url: BASE });
    await wait(cdp, "document.querySelectorAll('#epochCards .card').length === 20");

    const counts = JSON.parse(await evaluate(cdp, `JSON.stringify({
      eras: document.querySelectorAll('#epochCards .card').length,
      devices: document.querySelectorAll('#deviceCards .card').length,
      spaces: document.querySelectorAll('#spaceCards .card').length,
      opts: document.querySelectorAll('#toggles .opt').length,
      pairRows: document.querySelectorAll('.pair').length,
      toneboxes: document.querySelectorAll('#toggles .tonebox').length,
      toneOpts: document.querySelectorAll('#toggles .tonebox .opt').length,
      rowOrder: [...document.querySelectorAll('#toggles .opt')].map(b => b.dataset.v),
      quick: document.querySelectorAll('#strengthQuick .qbtn').length,
      spacesQuick: document.querySelectorAll('#spacesQuick .qbtn').length,
      comp: document.querySelectorAll('#compBtns .qbtn').length,
      quickOn: [...document.querySelectorAll('#strengthQuick .qbtn.on')].map(b => b.textContent),
      spacesQuickOn: [...document.querySelectorAll('#spacesQuick .qbtn.on')].map(b => b.textContent),
      compOn: document.querySelectorAll('#compBtns .qbtn.on').length,
      strength: document.getElementById('strengthVal').textContent,
      spacesVal: document.getElementById('spacesVal').textContent,
      spacesValue: document.getElementById('spacesStrength').value,
      spacesCtlHidden: document.getElementById('spacesCtl').hidden,
      volume: document.getElementById('volumeVal').textContent,
      volMin: document.getElementById('volume').min,
      volMax: document.getElementById('volume').max,
      volValue: document.getElementById('volume').value,
      volP: document.getElementById('volume').style.getPropertyValue('--p'),
      hasHeader: !!document.querySelector('header'),
      hasTop: !!document.querySelector('.top'),
      hasHist: !!document.getElementById('hist'),
      hasSrc: !!document.getElementById('srcPlayer'),
      dropInCol: !!document.querySelector('main .col .drop'),
      dropInBox: !!document.querySelector('#srcBox #drop'),
      brand: !!document.querySelector('.brand'),
      qiInline: getComputedStyle(document.querySelector('#compBtns .qi')).display,
      footer: !!document.querySelector('footer'),
      apply: !!document.getElementById('go')
    })`));
    check(counts.eras === 20, `20 era cards (${counts.eras})`);
    check(counts.devices === 24, `24 device cards (${counts.devices})`);
    check(counts.spaces === 16, `16 space cards (${counts.spaces})`);
    check(counts.opts === 6 && counts.pairRows === 0,
          `character options: one row of 6, no pair separation (${counts.opts})`);
    check(counts.toneboxes === 1 && counts.toneOpts === 2,
          `Dark+Bright grouped in one outlined box (${counts.toneboxes}/${counts.toneOpts})`);
    check(counts.rowOrder.join(",") === "vinyl,brick,hum,boxy,dark,bright",
          `option row order: ${counts.rowOrder.join(",")}`);
    check(counts.quick === 5 && counts.quickOn.join(" ") === "100%",
          `5 strength steps, 100% on (${counts.quickOn.join(" ")})`);
    check(counts.spacesQuick === 5 && counts.spacesVal === "100%"
          && counts.spacesValue === "100",
          `5 rooms & spaces steps, 100% default (${counts.spacesQuick}/${counts.spacesVal})`);
    check(counts.spacesCtlHidden === true,
          "rooms & spaces strength is hidden until a room is picked");
    check(counts.comp === 3 && counts.compOn === 1,
          `3 compression buttons, one selected (${counts.comp})`);
    check(counts.qiInline !== "block",
          `compression icon sits to the left of the label (${counts.qiInline})`);
    check(counts.strength === "100%" && counts.volume === "100%",
          `defaults strength ${counts.strength} / volume ${counts.volume}`);
    check(counts.volMin === "0" && counts.volMax === "200" && counts.volValue === "100",
          `volume slider 0-200, 100 default (${counts.volMin}/${counts.volValue}/${counts.volMax})`);
    check(counts.volP.replace("%", "").trim() === "50",
          `volume fill at 100% sits at the middle mark (--p ${counts.volP})`);
    check(counts.hasHeader === false && counts.hasTop === false,
          "no top header bar");
    check(counts.hasHist === false, "no 'All recordings' history section");
    check(counts.dropInCol && counts.dropInBox && counts.hasSrc,
          "drop zone lives inside the srcbox in the left column");
    check(counts.brand, "title block sits above the srcbox");
    const cols = await evaluate(cdp,
      "getComputedStyle(document.querySelector('main.grid')).gridTemplateColumns");
    check(/\s378px$/.test(cols), `sidebar column is 378px (${cols})`);

    const brandStatus = JSON.parse(await evaluate(cdp, `(() => {
      const b = document.querySelector('.brand');
      const st = document.getElementById('status');
      const h1 = b && b.querySelector('h1');
      if (!b || !st || !h1) return JSON.stringify({ok:false});
      const rb = b.getBoundingClientRect();
      const rs = st.getBoundingClientRect();
      const rh = h1.getBoundingClientRect();
      return JSON.stringify({ok:true, inBrand: b.contains(st),
        sameRow: Math.abs(rs.top - rh.top) < 30,
        rightOfLogo: rs.left >= rh.right - 1,
        rightEdge: Math.abs(rs.right - rb.right) < 3});
    })()`));
    check(brandStatus.ok && brandStatus.inBrand && brandStatus.sameRow
          && brandStatus.rightOfLogo && brandStatus.rightEdge,
          `status sits on the logo line, opposite side (${JSON.stringify(brandStatus)})`);
    check(counts.footer === false && counts.apply, "no footer; apply button present");

    const compact = JSON.parse(await evaluate(cdp, `JSON.stringify({
      labels: [...document.querySelectorAll('#toggles .opt')].map(b => b.textContent.trim())
    })`));
    check(compact.labels.every(l => l.length <= 8),
          `character button labels are short (${compact.labels.join(" / ")})`);
    const sizing = JSON.parse(await evaluate(cdp, `JSON.stringify((() => {
      const o = document.querySelector('#toggles .opt');
      const q = document.querySelector('#strengthQuick .qbtn');
      const a = o.getBoundingClientRect(), b = q.getBoundingClientRect();
      const so = getComputedStyle(o), sq = getComputedStyle(q);
      return {oh:a.height, qh:b.height, of:so.fontSize, qf:sq.fontSize,
              op:so.paddingTop, qp:sq.paddingTop, or:so.borderRadius, qr:sq.borderRadius};
    })())`));
    check(Math.abs(sizing.oh - sizing.qh) <= 1 && sizing.of === sizing.qf
          && sizing.op === sizing.qp && sizing.or === sizing.qr,
          `Character buttons match the percent buttons (h ${sizing.oh}/${sizing.qh}, font ${sizing.of}/${sizing.qf})`);
    check(compact.labels[0] === "Vinyl" && compact.labels[1] === "Brick",
          `Vinyl leads the row, Brick second (${compact.labels.join(" / ")})`);
    check(compact.labels.join(" ").includes("50 Hz")
          && compact.labels.includes("Vinyl") && compact.labels.includes("Brick")
          && !compact.labels.includes("Neutral") && !compact.labels.includes("Clean"),
          `Vinyl/Brick in, Neutral/Clean out, '50 Hz' kept (${compact.labels.join(" / ")})`);

    const noEmpty = await evaluate(cdp,
      "!document.body.innerText.includes('nothing selected')");
    check(noEmpty, "'nothing selected' placeholders are gone");

    const noStore = await evaluate(cdp,
      "![...document.querySelectorAll('script')].map(s => s.textContent).join('').includes('localStorage')");
    check(noStore, "the app never touches localStorage (no settings persistence)");

    const curs = await evaluate(cdp,
      "getComputedStyle(document.getElementById('drop')).cursor");
    check(curs === "pointer", `drop zone cursor is pointer (${curs})`);

    await evaluate(cdp, `(() => {
      document.querySelectorAll('#epochCards .card')[1].click();
      document.querySelectorAll('#epochCards .card')[16].click();
      document.querySelectorAll('#spaceCards .card')[3].click();
      document.querySelectorAll('#spaceCards .card')[9].click();
      const s = document.getElementById('strength');
      s.value = 75; s.dispatchEvent(new Event('input', {bubbles:true}));
      const v = document.getElementById('volume');
      v.value = 160; v.dispatchEvent(new Event('input', {bubbles:true}));
      const sp = document.getElementById('spacesStrength');
      sp.value = 40; sp.dispatchEvent(new Event('input', {bubbles:true}));
      document.querySelectorAll('#compBtns .qbtn')[1].click();
      document.querySelector('#toggles .opt[data-v="hum"]').click();
      document.getElementById('noNoise').checked = true;
      document.getElementById('noNoise').dispatchEvent(new Event('change', {bubbles:true}));
      return 1;
    })()`);

    const stored = await evaluate(cdp, "localStorage.getItem('vintagefx.settings.v1')");
    check(stored === null, "nothing is written to localStorage while using the page");

    const spacesShown = await evaluate(cdp,
      "document.getElementById('spacesCtl').hidden === false");
    check(spacesShown, "rooms & spaces strength appears once a room is picked");

    const chain1 = await evaluate(cdp,
      "document.querySelectorAll('#chain .chip:not(.sep)').length");
    check(chain1 >= 6, `chain shows every step as a removable chip (${chain1})`);
    const chain1Text = await evaluate(cdp, "document.getElementById('chain').textContent");
    check(chain1Text.includes("50 Hz"), `'50 Hz' chip in the chain (${chain1Text.slice(0, 70)})`);

    const rem = JSON.parse(await evaluate(cdp, `(() => {
      const x = document.querySelector('#chain .chip .chip-x');
      const set = x.closest('.chip').dataset.set;
      const id = x.closest('.chip').dataset.id;
      x.click();
      return JSON.stringify({
        set, id,
        chips: document.querySelectorAll('#chain .chip:not(.sep)').length,
        eras: document.querySelectorAll('#epochCards .card.on').length
      });
    })()`));
    check(rem.set === "epochs" && rem.chips === chain1 - 1 && rem.eras === 1,
          `x on a chain chip removes that step (${rem.set}, chips ${rem.chips}, eras ${rem.eras})`);

    await evaluate(cdp, `(() => {
      const btn = [...document.querySelectorAll('#chain .chip')].find(c =>
        c.dataset.set === "compression");
      if (btn) btn.querySelector('.chip-x').click();
      return 1;
    })()`);
    const compAfter = await evaluate(cdp,
      "document.querySelector('#compBtns .qbtn.on').dataset.id");
    check(compAfter === "off", "removing the compression chip switches compression to off");

    const titles = JSON.parse(await evaluate(cdp, `JSON.stringify({
      quick: [...document.querySelectorAll('#strengthQuick .qbtn')].every(b => b.title),
      spaces: [...document.querySelectorAll('#spacesQuick .qbtn')].every(b => b.title),
      comp: [...document.querySelectorAll('#compBtns .qbtn')].every(b => b.title),
      opts: [...document.querySelectorAll('#toggles .opt')].every(b => b.title),
      pick: !!document.getElementById('pick').title,
      go: !!document.getElementById('go').title,
      dl: !!document.getElementById('dl').title,
      presets: !!document.getElementById('presetOpen').title
        && !!document.getElementById('presetSave').title,
      sliders: ['strength', 'spacesStrength', 'volume'].every(id => document.getElementById(id).title)
    })`));
    check(titles.quick && titles.spaces && titles.comp && titles.opts && titles.pick
          && titles.go && titles.dl && titles.presets && titles.sliders,
          `control buttons carry titles (${JSON.stringify(titles)})`);

    const brickEdit = JSON.parse(await evaluate(cdp, `(() => {
      const sib = document.querySelector('#toggles .opt[data-v="boxy"]').getBoundingClientRect();
      document.querySelector('#toggles .opt[data-v="brick"]').click();
      const inp = document.querySelector('#toggles input[data-v="brick"]');
      if (!inp) return JSON.stringify({ok:false});
      const r = inp.getBoundingClientRect();
      const val = inp.value, title = inp.title;
      inp.value = '300'; inp.dispatchEvent(new Event('input', {bubbles:true}));
      const titleAfter = inp.title;
      return JSON.stringify({ok:true, val, title, titleAfter, hz: S.brickHz,
        dw: Math.abs(r.width - sib.width), dh: Math.abs(r.height - sib.height)});
    })()`));
    check(brickEdit.ok && brickEdit.val === "90" && brickEdit.hz === 300
          && /90 Hz/.test(brickEdit.title) && /300 Hz/.test(brickEdit.titleAfter),
          `clicking Brick turns it into a titled frequency field (${JSON.stringify(brickEdit)})`);
    check(brickEdit.ok && brickEdit.dw <= 1 && brickEdit.dh <= 1,
          `the field keeps the button geometry (dw=${brickEdit.dw}, dh=${brickEdit.dh})`);

    const brickToggle = JSON.parse(await evaluate(cdp, `(() => {
      const inp = document.querySelector('#toggles input[data-v="brick"]');
      inp.focus(); inp.click();
      const btn = document.querySelector('#toggles .opt[data-v="brick"]');
      const off = !!btn && btn.tagName === "BUTTON"
        && !document.querySelector('#toggles input[data-v="brick"]');
      const kept = S.brickHz, offTitle = btn && btn.title;
      btn.click();
      const inp2 = document.querySelector('#toggles input[data-v="brick"]');
      return JSON.stringify({off, kept, offTitle, back: !!inp2,
        val2: inp2 && inp2.value, onTitle: inp2 && inp2.title});
    })()`));
    check(brickToggle.off && brickToggle.kept === 300 && brickToggle.back
          && brickToggle.val2 === "300"
          && /300 Hz/.test(brickToggle.offTitle || "")
          && /300 Hz/.test(brickToggle.onTitle || ""),
          `a second click turns Brick off but keeps the frequency (${JSON.stringify(brickToggle)})`);

    const lastChip = await evaluate(cdp,
      "[...document.querySelectorAll('#chain .chip:not(.sep)')].pop().dataset.id");
    check(lastChip === "brick", `Brick sits last in the chain (last=${lastChip})`);

    const roundTrip = JSON.parse(await evaluate(cdp, `(() => {
      const snap = () => JSON.stringify({e:[...S.epochs], d:[...S.devices], sp:[...S.spaces],
        v:[...S.variants], st:S.strength, ss:S.spacesStrength, bz:S.brickHz,
        c:S.compression, nn:S.noNoise, vol:S.volume});
      const before = snap();
      const xml = presetXml();
      S.epochs = new Set(); S.devices = new Set(); S.spaces = new Set(); S.variants = new Set();
      S.strength = 0; S.spacesStrength = 0; S.brickHz = 20;
      S.compression = 'off'; S.noNoise = false; S.volume = 0;
      syncUIFromState();
      const wiped = [...S.epochs].length === 0 && S.strength === 0 && S.volume === 0;
      applyPresetXml(xml);
      return JSON.stringify({xmlHead: xml.slice(0, 40), wiped, ok: snap() === before});
    })()`));
    check(roundTrip.xmlHead.startsWith("<?xml") && roundTrip.wiped && roundTrip.ok,
          `Save/Open round-trips every setting through XML (${JSON.stringify(roundTrip)})`);

    await cdp.send("Page.reload");
    await wait(cdp, "document.querySelectorAll('#epochCards .card').length === 20");
    await sleep(400);

    const after = JSON.parse(await evaluate(cdp, `JSON.stringify({
      eras: document.querySelectorAll('#epochCards .card.on').length,
      spaces: document.querySelectorAll('#spaceCards .card.on').length,
      strength: document.getElementById('strengthVal').textContent,
      spacesVal: document.getElementById('spacesVal').textContent,
      volume: document.getElementById('volumeVal').textContent,
      volP: document.getElementById('volume').style.getPropertyValue('--p'),
      quickOn: [...document.querySelectorAll('#strengthQuick .qbtn.on')].map(b => b.textContent),
      spacesQuickOn: [...document.querySelectorAll('#spacesQuick .qbtn.on')].map(b => b.textContent),
      compOn: [...document.querySelectorAll('#compBtns .qbtn.on')].map(b => b.dataset.id),
      optsOn: [...document.querySelectorAll('#toggles .opt.on')].map(b => b.dataset.v),
      noNoise: document.getElementById('noNoise').checked,
      chainText: document.getElementById('chain').textContent,
      trackVisible: document.getElementById('srcTrack').hidden === false
    })`));
    check(after.eras === 0 && after.spaces === 0,
          `after reload nothing is restored (eras=${after.eras}, spaces=${after.spaces})`);
    check(after.strength === "100%" && after.volume === "100%"
          && after.spacesVal === "100%",
          `after reload strength ${after.strength} / spaces ${after.spacesVal} / volume ${after.volume} (defaults)`);
    check(after.quickOn.join(" ") === "100%" && after.spacesQuickOn.join(" ") === "100%",
          `quick steps back to 100% (${after.quickOn} / ${after.spacesQuickOn})`);
    check(after.compOn.length === 1 && after.compOn[0] === "off",
          `compression back to off (${after.compOn.join(",")})`);
    check(after.optsOn.length === 0, "no character option restored");
    check(after.noNoise === false, "no-noise checkbox not restored");
    check(after.chainText.trim() === "", "chain is empty after reload");
    check(after.trackVisible === false, "no track is remembered after reload");

    await evaluate(cdp, "document.querySelectorAll('#epochCards .card')[1].click();");

    const wav = fs.readFileSync(path.join(HERE, "_ui_test.wav"));
    const b64 = wav.toString("base64");
    await evaluate(cdp, `(() => {
      const bin = atob("${b64}");
      const arr = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
      upload(new File([arr], "ui test.wav", {type: "audio/wav"}));
      return 1;
    })()`);
    const hintBefore = await evaluate(cdp,
      "document.getElementById('dropTxt').textContent");
    check(/Drop an audio/.test(hintBefore),
          `drop hint shown before upload (${hintBefore})`);
    const uploaded = await wait(cdp,
      "document.getElementById('srcTrack').hidden === false", 40);
    check(uploaded, "upload reveals the standard player under the drop zone");
    const srcName = await evaluate(cdp, "document.getElementById('dropTxt').textContent");
    check(/ui test\.wav/.test(srcName), `file name shown in the drop zone (${srcName})`);
    const srcHasSrc = await evaluate(cdp,
      "!!document.getElementById('srcPlayer').getAttribute('src')");
    check(srcHasSrc, "standard player has the original audio src");
    const srcMeta = await evaluate(cdp, "document.getElementById('srcMeta').textContent");
    check(/kHz/.test(srcMeta), `track meta next to the name (${srcMeta})`);

    await evaluate(cdp, "document.getElementById('srcPlayer').play().catch(()=>{})");
    await sleep(900);
    const origPlaying = await evaluate(cdp,
      "!document.getElementById('srcPlayer').paused");
    check(origPlaying, "original plays in the standard player");

    await evaluate(cdp, "document.getElementById('go').click()");
    const done = await wait(cdp,
      "document.getElementById('status').textContent.indexOf('Done:') === 0", 80);
    const statusText = await evaluate(cdp, "document.getElementById('status').textContent");
    check(done, `apply finishes (${statusText})`);
    const meta = await evaluate(cdp,
      "document.getElementById('resMeta').textContent");
    check(/STEPS/i.test(meta), `result meta (${meta})`);
    const resSrc = await evaluate(cdp,
      "!!document.getElementById('player').getAttribute('src')");
    check(resSrc, "result player has audio src");

    const solo1 = JSON.parse(await evaluate(cdp, `JSON.stringify({
      srcPaused: document.getElementById('srcPlayer').paused,
      resPlaying: !document.getElementById('player').paused
    })`));
    check(solo1.srcPaused === true && solo1.resPlaying === true,
          "result playback stops the source player (one at a time)");

    await evaluate(cdp, "document.getElementById('srcPlayer').play().catch(()=>{})");
    await sleep(800);
    const solo2 = JSON.parse(await evaluate(cdp, `JSON.stringify({
      srcPlaying: !document.getElementById('srcPlayer').paused,
      resPaused: document.getElementById('player').paused
    })`));
    check(solo2.srcPlaying === true && solo2.resPaused === true,
          "starting the source stops the result player (one at a time)");
    const audioCount = await evaluate(cdp,
      "document.querySelectorAll('audio').length");
    check(audioCount === 2, `exactly two players on the page (${audioCount})`);

    await cdp.send("Page.reload");
    await wait(cdp, "document.querySelectorAll('#epochCards .card').length === 20");
    await sleep(900);
    const final = JSON.parse(await evaluate(cdp, `JSON.stringify({
      trackVisible: document.getElementById('srcTrack').hidden === false,
      name: document.getElementById('dropTxt').textContent,
      eras: document.querySelectorAll('#epochCards .card.on').length,
      chainText: document.getElementById('chain').textContent,
      histRows: document.querySelectorAll('.hrow').length
    })`));
    check(final.trackVisible === false && !/ui test\.wav/.test(final.name),
          `track is not remembered after reload (name='${final.name}')`);
    check(final.eras === 0, `selections reset after reload (${final.eras} eras)`);
    check(final.chainText.trim() === "", "chain empty after final reload");
    check(final.histRows === 0, "no history rows rendered anywhere");

    // sticky sidebar: with a viewport shorter than the panel it scrolls with
    // the page, then pins so its bottom rests on the viewport bottom.
    await cdp.send("Emulation.setDeviceMetricsOverride",
      { width: 1440, height: 500, deviceScaleFactor: 1, mobile: false });
    await sleep(400);
    await evaluate(cdp, "syncPin && syncPin()");
    const st = JSON.parse(await evaluate(cdp, `JSON.stringify((() => {
      const el = document.querySelector('.sec.sticky');
      if (!el) return {found:false};
      const vh = window.innerHeight;
      const h = el.offsetHeight;
      const sh = document.scrollingElement.scrollHeight;
      const styleTop = el.style.top;
      window.scrollTo(0, 0);
      const top0 = el.getBoundingClientRect().top;
      window.scrollTo(0, 160);
      const r1 = el.getBoundingClientRect();
      const pos = getComputedStyle(el).position;
      window.scrollTo(0, 0);
      return {found:true, pos, vh, h, sh, styleTop,
              top0, top1:r1.top, bottom1:r1.bottom};
    })())`));
    check(st.found && st.pos === "sticky",
          `settings panel uses position: sticky (${st.pos})`);
    check(st.found && st.sh > st.vh + 80,
          `page scrolls past the panel (scrollH=${st.sh}, vh=${st.vh})`);
    check(st.found && st.h > st.vh - 28,
          `panel is taller than the viewport (h=${st.h}, vh=${st.vh})`);
    check(st.found && st.top0 >= 12 && st.top0 <= 16,
          `panel starts at the top of the page (top0=${st.top0.toFixed(1)})`);
    const expectTop = st.vh - st.h - 14;
    check(st.found && Math.abs(st.top1 - expectTop) <= 2,
          `panel pins where its bottom meets the viewport bottom (top=${st.top1.toFixed(1)}, expected ${expectTop.toFixed(1)})`);
    check(st.found && Math.abs(st.bottom1 - (st.vh - 14)) <= 2,
          `pinned panel bottom rests on the viewport bottom (bottom=${st.bottom1.toFixed(1)}, vh-14=${st.vh - 14})`);
    check(st.found && st.styleTop === expectTop + "px",
          `pin offset is computed for the bottom rule (${st.styleTop})`);
  } catch (e) {
    check(false, "exception: " + e.message);
  } finally {
    cdp.close();
    chrome.kill();
  }

  console.log(failures ? `FAILED: ${failures}` : "UI CHECKS PASSED");
  process.exit(failures ? 1 : 0);
}

main();