// Copyright 2026 Michael Zhang
// SPDX-License-Identifier: Apache-2.0
// Live J-space visualizer frontend. No build step, no dependencies.

const $ = (id) => document.getElementById(id);
const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

let data = null;          // last /api/read response
let mode = "jlens";
let pinned = [];          // [{id, str}]
let rankViewId = null;    // token id whose ranks the grid shows, or null
let STATIC = false;       // no backend: serve precomputed example grids
let INFO = null;          // /api/info (or data/index.json) payload
let selectedSlug = null;  // active example in static mode
let colLimit = null;      // static demo: cap visible grid columns during reveal
let halu = new Map();     // generated position -> {score, flag} at the current threshold

// Render a token for display: make whitespace visible, trim length.
function tokStr(s) {
  if (s === undefined) return "?";
  let out = s.replace(/\n/g, "⏎").replace(/\t/g, "⇥");
  if (out.startsWith(" ")) out = "·" + out.slice(1);
  if (out.trim() === "") out = out.replace(/ /g, "·") || "∅";
  return out.length > 12 ? out.slice(0, 11) + "…" : out;
}

function setStatus(msg, err = false, html = false) {
  const el = $("status");
  if (html) el.innerHTML = msg;
  else el.textContent = msg;
  el.className = err ? "err" : "";
}

// The model's actual next words (output row, last position) — shown in the
// status line so "what does it answer" never has to be hunted for.
function modelSays() {
  if (data.continuation) {
    const text = data.continuation.replace(/\s+/g, " ").trim();
    return `model continues: “…${text}”`;
  }
  const out = data.grid[data.grid.length - 1];
  const t = data.seq_len - 1;
  const words = out.top_ids[t].slice(0, 3).map((id) => tokStr(data.vocab[id]));
  return `model says: ${words.map((w) => `“${w}”`).join(" ")}`;
}

async function init() {
  try {
    const res = await fetch("api/info");
    if (!res.ok) throw new Error();
    INFO = await res.json();
  } catch {
    // No backend — static demo mode: precomputed example grids only.
    STATIC = true;
    INFO = await (await fetch("data/index.json")).json();
    INFO.device = "precomputed";
    // These need a live model run; the demo is precomputed. Keep them visible
    // but locked, and explain why (hover + click) instead of doing nothing.
    $("live-label").hidden = false;
    for (const id of ["topk", "chat", "live"]) {
      const label = $(id).closest("label");
      $(id).readOnly = true; // blocks typing in the number box (no-op on checkboxes)
      label.classList.add("demo-locked");
      label.title = "Precomputed demo — clone and run it locally to use this";
      label.addEventListener("click", (e) => {
        e.preventDefault(); // don't toggle/spin — explain instead
        setStatus("top-k · chat · live only work when you run it locally — the demo is precomputed");
      });
    }
  }
  for (const link of INFO.links || []) {
    const a = document.createElement("a");
    a.href = link.url;
    a.className = "chip";
    a.textContent = link.name;
    document.querySelector("header").appendChild(a);
  }
  $("model-chip").textContent = `${INFO.model_id} · ${INFO.n_layers}L · d=${INFO.d_model} · ${INFO.device}`;
  $("lens-chip").textContent = `lens: ${INFO.fitted_layers.length} layers, ${INFO.lens_n_prompts} prompts`;
  $("english").checked = mode !== "jlens";
  if (!INFO.fitted_layers.length) {
    // No fitted lens yet: logit lens is the only readout available.
    mode = "logit";
    const jl = document.querySelector('#mode-toggle button[data-mode="jlens"]');
    jl.disabled = true;
    jl.title = "no fitted lens yet — run scripts/fit_lens.py --preset qwen-1.8b-chat";
    for (const b of $("mode-toggle").children) b.classList.toggle("active", b.dataset.mode === mode);
    $("lens-chip").textContent = "lens: none (logit-lens only)";
  }
  INFO.examples.forEach((ex, i) => {
    const opt = document.createElement("option");
    opt.value = String(i);
    opt.textContent = ex.name;
    $("examples").appendChild(opt);
  });
  // Shareable links: /?prompt=... pre-fills and reads immediately.
  const shared = new URLSearchParams(location.search).get("prompt");
  if (shared && !STATIC) {
    $("prompt").value = shared;
    read();
  } else {
    renderWelcome();
  }
}

// Blank-slate hint shown before the first read: what this is, plus the
// examples as one-click chips.
function renderWelcome() {
  const chips = INFO.examples
    .map((ex, i) => `<button class="ex-chip" data-i="${i}">${esc(ex.name)}</button>`)
    .join(" ");
  const typeHint = STATIC
    ? 'Pick an example below. To type <em>your own</em> prompts, clone the repo — <a href="https://github.com/Festyve/jspace-viz" target="_blank" rel="noopener">github.com/Festyve/jspace-viz</a> — and run it locally (two commands).'
    : "Ask it something above and press <b>Read</b> (⌘↵), or start from an example. Easy facts read green; obscure ones usually don't.";
  $("grid").innerHTML =
    `<div class="welcome"><h2>Ask a short factual question</h2><p>Each cell shows what one layer of <b>${esc(INFO.model_id.split("/").pop())}</b> ` +
    `is <i>about to say</i> at each word. Generated words are scored for hallucination from the same grid.</p>` +
    `<p>${typeHint}</p><p>${chips}</p></div>`;
}
$("grid").addEventListener("click", (e) => {
  const chip = e.target.closest(".ex-chip");
  if (!chip) return;
  loadExample(+chip.dataset.i);
});

// Selecting an example types it out while the model reads along, so you watch
// the workspace assemble as context accumulates. The static demo has no
// backend, but the model is causal — each precomputed column equals a fresh
// read of that prefix — so we reveal columns as the words are typed, same
// effect without a server.
let playing = false;
let playToken = 0;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function loadExample(i) {
  const ex = INFO.examples[i];
  $("examples").value = String(i);
  const token = ++playToken;
  playing = true;
  const ta = $("prompt");
  ta.value = "";

  // Static demo: fetch the full precomputed grid up front, then reveal it
  // column by column as the prompt types out (see note above). tokEnd[t] is the
  // typed-character count at which column t's token is fully on screen.
  let tokEnd = null;
  if (STATIC) {
    selectedSlug = ex.slug;
    data = await (await fetch(`data/${selectedSlug}_${mode}.json`)).json();
    if (token !== playToken) return;
    applyStaticPins();
    tokEnd = [];
    let acc = 0;
    for (let t = 0; t < data.seq_len; t++) {
      if (t > 0) acc += (data.vocab[data.context_ids[t]] || "").length;
      tokEnd[t] = acc; // the begin-of-sequence token adds no typed characters
    }
    colLimit = 1; // show that first column immediately, then grow with typing
    renderGrid();
    renderWorkspace();
  }

  // Clock-based so playback stays smooth at any frame rate and can't stall
  // when the browser throttles timers in unfocused tabs.
  const CHARS_PER_SEC = 70;
  const start = performance.now();
  let wordsSinceRead = 0;
  while (true) {
    if (token !== playToken) { colLimit = null; return; } // superseded
    const target = Math.min(
      ex.prompt.length,
      Math.floor(((performance.now() - start) / 1000) * CHARS_PER_SEC),
    );
    if (target > ta.value.length) {
      const added = ex.prompt.slice(ta.value.length, target);
      ta.value = ex.prompt.slice(0, target);
      ta.scrollTop = ta.scrollHeight;
      if (STATIC) {
        // grow the grid to every column whose token has now been typed
        let n = colLimit;
        while (n < data.seq_len && tokEnd[n] <= target) n++;
        if (n !== colLimit) {
          colLimit = Math.max(1, n);
          renderGrid();
          renderWorkspace();
        }
      } else {
        wordsSinceRead += (added.match(/\s+/g) || []).length;
        const nWords = ta.value.trim().split(/\s+/).length;
        if ($("live").checked && !reading && wordsSinceRead >= 2 && nWords >= 4) {
          wordsSinceRead = 0;
          read({ live: true });
        }
      }
    }
    if (target >= ex.prompt.length) break;
    await sleep(16);
  }
  playing = false;
  if (token !== playToken) { colLimit = null; return; }
  if (STATIC) {
    colLimit = null; // reveal the full grid now that the prompt is complete
    renderGrid();
    renderWorkspace();
    renderMetrics();
    setStatus(
      `${data.continuation ? modelSays() + " · " : ""}` +
      `${data.seq_len} tokens × ${data.layers.length} layers · precomputed demo`,
    );
  } else {
    while (reading) await sleep(100); // let the last live read settle
    read();
  }
}

// Static mode: ranks were precomputed for every token in any top-k cell;
// materialize pinned_ranks rows in the shape the renderer expects.
function applyStaticPins() {
  if (!pinned.length) return;
  data.grid.forEach((row, li) => {
    row.pinned_ranks = [];
    for (let t = 0; t < data.seq_len; t++) {
      row.pinned_ranks.push(pinned.map((p) => data.ranks[p.id]?.[li]?.[t] ?? null));
    }
  });
}

let reading = false;   // one read in flight at a time; latest text wins after
let rerunLive = false;

async function read(opts = {}) {
  if (!$("prompt").value.trim()) {
    renderWelcome();
    setStatus("type a prompt first, or pick an example");
    return;
  }
  if (reading) { rerunLive = true; return; }
  reading = true;
  if (!opts.live) setStatus("reading…");
  $("read").disabled = true;
  const t0 = performance.now();
  try {
    if (STATIC) {
      // Only precomputed prompts exist here; anything else → honest pointer.
      const match = INFO.examples.find((ex) => ex.prompt === $("prompt").value);
      if (!match) {
        setStatus(
          'custom prompts need a local clone — <a href="https://github.com/Festyve/jspace-viz" target="_blank" rel="noopener">github.com/Festyve/jspace-viz</a>',
          true, true,
        );
        $("read").disabled = false;
        return;
      }
      selectedSlug = match.slug;
      data = await (await fetch(`data/${selectedSlug}_${mode}.json`)).json();
      applyStaticPins();
      renderGrid();
      renderWorkspace();
      renderMetrics();
      setStatus(`${modelSays()} · ${data.seq_len} tokens × ${data.layers.length} layers · precomputed demo`);
      return;
    }
    const res = await fetch("api/read", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: $("prompt").value,
        mode,
        top_k: +$("topk").value,
        chat: $("chat").checked,
        pinned_ids: pinned.map((p) => p.id),
        continuation: !opts.live,
        max_new_tokens: +$("newtok").value,
        threshold: +$("halu-thresh").value,
      }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    data = await res.json();
    {
      const pl = data.prompt_len ?? data.seq_len, words = new Set();
      for (const row of data.grid) for (let t = Math.max(0, pl - 2); t < data.seq_len; t++) for (const id of row.top_ids[t].slice(0, 2)) { const w = (data.vocab[id] || "").trim(); if (CJK.test(w) && !GLOSS.has(w)) words.add(w); }
      glossTokens([...words]);
    }
    renderHalu();
    renderGrid();
    renderWorkspace();
    renderMetrics();
    if (opts.live && !playing) {
      for (const id of ["workspace-panel", "grid-wrap"]) {
        const el = $(id);
        el.classList.remove("flash");
        void el.offsetWidth; // restart the animation
        el.classList.add("flash");
      }
    }
    if (data.continuation) scrollToGeneration(); // the answer is what you came for
    const says = data.continuation ? `${modelSays()} · ` : "";
    setStatus(`${says}${data.seq_len} tokens × ${data.layers.length} layers · ${((performance.now() - t0) / 1000).toFixed(1)}s${opts.live ? " · live" : ""}`);
  } catch (e) {
    setStatus(e.message, true);
  } finally {
    $("read").disabled = false;
    reading = false;
    if (rerunLive) {
      rerunLive = false;
      read({ live: true });
    }
  }
}

// "English" = printable ASCII with at least one letter/digit (includes the
// chat-template tokens, which are informative). CJK/Hangul/etc. are skipped.
const ENGLISH = /^[\x20-\x7e\n\t]*[A-Za-z0-9][\x20-\x7e\n\t]*$/;
$("english").addEventListener("change", () => { if (data) renderGrid(); });

// Cell shading returns [background, darkCell]; dark cells get white text.
// Top-1 probability (0..1) — blue ramp.
function probColor(p) {
  const a = Math.min(0.92, Math.pow(p, 0.6));
  return [`rgba(31, 79, 128, ${a.toFixed(3)})`, a > 0.5];
}
// Rank (0 = top) — warm grey ramp, fades out past ~1000.
function rankColor(r) {
  const a = Math.max(0, 0.92 - 0.92 * (Math.log10(r + 1) / 3));
  return [`rgba(47, 43, 40, ${a.toFixed(3)})`, a > 0.5];
}

function renderGrid() {
  const grid = $("grid");
  const { layers, grid: rows, context_ids, vocab, seq_len } = data;
  const cols = colLimit == null ? seq_len : Math.min(seq_len, colLimit);
  grid.style.gridTemplateColumns = `64px repeat(${cols}, 80px)`;
  // Columns past prompt_len are the model's own generated words — tinted, so
  // you can see what it was thinking *while it spoke*.
  const promptLen = data.prompt_len ?? seq_len;
  const genCls = (t) => (t === promptLen ? " gen genb" : t > promptLen ? " gen" : "");
  const parts = ['<div class="hcell corner">layer ╲ pos</div>'];
  for (let t = 0; t < cols; t++) {
    const label = t === promptLen ? '<span class="idx">↳ model says</span>' : `<span class="idx">${t}</span>`;
    const h = halu.get(t);
    const hcls = h && h.flag ? " halu" : "";
    const hstyle = h && h.flag ? ` style="background:${haluColor(h.score)}" title="hallucination score ${h.score}"` : "";
    parts.push(`<div class="hcell${genCls(t)}${hcls}"${hstyle}>${label}${esc(tokStr(vocab[context_ids[t]]))}</div>`);
  }
  for (let li = 0; li < layers.length; li++) {
    const row = rows[li];
    const cls = row.is_output ? " outrow" : "";
    parts.push(`<div class="lcell${cls}">${row.is_output ? "output" : "L" + row.layer}</div>`);
    for (let t = 0; t < cols; t++) {
      let text, bg, dk = false;
      const pi = rankViewId === null ? -1 : pinned.findIndex((p) => p.id === rankViewId);
      if (pi >= 0 && row.pinned_ranks && row.pinned_ranks[t][pi] != null) {
        const r = row.pinned_ranks[t][pi];
        text = r === 0 ? "★0" : String(r);
        [bg, dk] = rankColor(r);
      } else if (pi >= 0 && row.pinned_ranks) {
        text = "–";
        bg = "transparent";
      } else {
        let i = 0;
        if ($("english").checked) {
          i = row.top_ids[t].findIndex((id) => ENGLISH.test(vocab[id] || ""));
        }
        text = i < 0 ? "·" : tokStr(vocab[row.top_ids[t][i]]);
        if (i >= 0) [bg, dk] = probColor(row.top_probs[t][i]); else bg = "transparent";
      }
      const flip = t >= promptLen - 1 && li > 0 && pi < 0 && rows[li - 1].top_ids[t][0] !== row.top_ids[t][0];
      parts.push(`<div class="cell${cls}${genCls(t)}${flip ? " flip" : ""}${dk ? " dk" : ""}" data-l="${li}" data-t="${t}" style="background:${bg}">${esc(text)}</div>`);
    }
  }
  const wrap = $("grid-wrap");
  const { scrollLeft, scrollTop } = wrap;
  grid.innerHTML = parts.join("");
  if (playing) {
    // follow the newest column while an example types itself out
    wrap.scrollLeft = wrap.scrollWidth;
  } else {
    wrap.scrollLeft = scrollLeft;
    wrap.scrollTop = scrollTop;
  }
  // The generated columns are the highlight but sit far to the right — offer a
  // jump when they exist and aren't already the whole grid.
  $("jump-gen").hidden = !(promptLen < seq_len);
}

// Scroll the grid so the "↳ model says" boundary sits near the left edge,
// keeping a couple of prompt columns visible for context. Uses live rects so
// it's immune to the sticky-header / CSS-grid offsetParent quirks.
function scrollToGeneration() {
  const boundary = document.querySelector(".hcell.genb");
  if (!boundary) return;
  const wrap = $("grid-wrap");
  const delta = boundary.getBoundingClientRect().left - wrap.getBoundingClientRect().left - 160;
  wrap.scrollLeft = Math.max(0, wrap.scrollLeft + delta);
}
$("jump-gen").addEventListener("click", scrollToGeneration);

function renderPinned() {
  $("pinned-row").hidden = pinned.length === 0;
  $("view-top1").hidden = rankViewId === null;
  $("pinned-chips").innerHTML = pinned
    .map(
      (p) =>
        `<span class="pin-chip${p.id === rankViewId ? " sel" : ""}" data-id="${p.id}">` +
        `${esc(tokStr(p.str))}<span class="x" data-x="${p.id}">✕</span></span>`
    )
    .join("");
}

// "What is it thinking about": aggregate mid-band lens tokens that are real
// words, not words from the prompt — i.e. the J-space content. Click to trace.
const STOP = new Set("the a an is are was were be been of to in on at for and or but not it its this that with as by from he she they we you i his her their".split(" "));
function renderWorkspace() {
  const { grid, vocab, context_ids, seq_len } = data;
  const cols = colLimit == null ? seq_len : Math.min(seq_len, colLimit);
  const fitted = grid.filter((r) => !r.is_output);
  const band = fitted.slice(Math.floor(fitted.length * 0.25), Math.ceil(fitted.length * 0.85));
  const promptWords = new Set(context_ids.slice(0, cols).map((id) => (vocab[id] || "").trim().toLowerCase()));
  const scores = new Map(); // word -> {score, id, str}
  for (const row of band) {
    for (let t = 4; t < cols; t++) {
      row.top_ids[t].forEach((id, i) => {
        const str = vocab[id] || "";
        const w = str.trim().toLowerCase();
        // word-start tokens (leading space in BPE) or CJK concept tokens (the J-space of a bilingual model)
        const cjk = CJK.test(str.trim());
        if (!cjk && !str.startsWith(" ") && !str.startsWith("▁")) return;
        if (!cjk && !/^[a-z][a-z'’-]{2,}$/i.test(str.trim())) return;
        if (promptWords.has(w) || STOP.has(w)) return;
        const p = row.top_probs[t][i];
        if (p < 0.03) return; // diffuse noise doesn't accumulate
        const e = scores.get(w) || { score: 0, id, str, peakP: 0, peakLayer: row.layer };
        e.score += p;
        if (p > e.peakP) { e.peakP = p; e.peakLayer = row.layer; } // where it lives
        if (!scores.has(w)) scores.set(w, e);
      });
    }
  }
  // Subtract each word's baseline hum on neutral text: glitch tokens appear
  // on every prompt and cancel out; prompt-specific content survives.
  const BASE = INFO.chip_baseline || {};
  const top = [...scores.entries()]
    .map(([w, e]) => ({ ...e, adj: e.score - (BASE[w] || 0) }))
    .filter((e) => e.adj > 0.15)
    .sort((a, b) => b.adj - a.adj)
    .slice(0, 10);
  const panel = $("workspace-panel");
  panel.hidden = top.length === 0;
  const previous = renderWorkspace._last || new Set();
  renderWorkspace._last = new Set(top.map((e) => e.str.trim().toLowerCase()));
  $("workspace-chips").innerHTML = top
    .map((e) => {
      const word = e.str.trim();
      const pop = previous.has(word.toLowerCase()) ? "" : " pop";
      const g = GLOSS.get(word);
      return `<span class="ws-chip${pop}" data-id="${e.id}" title="strength ${e.adj.toFixed(1)}, peaks at layer ${e.peakLayer}">${esc(word)}${g ? `<span class="gloss">≈ ${esc(g)}</span>` : ""}<span class="score">${e.adj.toFixed(1)}</span><span class="peak">L${e.peakLayer}</span></span>`;
    })
    .join("");
  glossTokens(top.map((e) => e.str.trim()).filter((w) => CJK.test(w) && !GLOSS.has(w)));
}
// ---- glosses for non-English workspace tokens (Gemini, cached server-side) ----
const CJK = /[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]/;
const GLOSS = new Map();
let glossing = false;
async function glossTokens(words) {
  if (!words.length || glossing || !(INFO && INFO.verifier)) return;
  glossing = true;
  try {
    const res = await fetch("api/gloss", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ tokens: words.slice(0, 80) }) });
    const g = await res.json();
    for (const [k, v] of Object.entries(g)) GLOSS.set(k, v);
    if (Object.keys(g).length) renderWorkspace();
  } catch {} finally { glossing = false; }
}
$("workspace-chips").addEventListener("click", (e) => {
  const chip = e.target.closest(".ws-chip");
  if (!chip || !data) return;
  const id = +chip.dataset.id;
  if (!pinned.some((p) => p.id === id)) {
    if (pinned.length >= 16) pinned.shift();
    pinned.push({ id, str: data.vocab[id] ?? "?" });
  }
  rankViewId = id;
  renderPinned();
  read();
});

// ---- hallucination panel ---------------------------------------------------
// Scores come from the server (jspace_viz/hallucination.py); flags are
// re-derived here so the threshold slider works without another read.
const WORDLIKE = /[A-Za-z0-9]/;
function haluColor(score) {
  return `rgba(185, 28, 28, ${(0.08 + 0.3 * score).toFixed(3)})`;
}
function renderHalu() {
  halu = new Map();
  const panel = $("halu-panel");
  const H = data.hallucination;
  if (!H) { panel.hidden = true; return; }
  const thr = +$("halu-thresh").value;
  $("halu-thresh-val").textContent = thr.toFixed(2);
  const toks = H.tokens.map((t) => ({ ...t, flag: t.score >= thr && WORDLIKE.test(t.token) && !t.in_prompt }));
  // whole-word flags, mirroring hallucination.py
  let ws = 0;
  toks.forEach((t, i) => { if (i > 0 && (t.token.startsWith(" ") || !WORDLIKE.test(t.token))) ws = i; t.word = ws; });
  const flaggedWords = new Set(toks.filter((t) => t.flag).map((t) => t.word));
  for (const t of toks) t.flag = flaggedWords.has(t.word) && WORDLIKE.test(t.token);
  for (const t of toks) halu.set(t.pos, { score: t.score, flag: t.flag });
  // inline answer with flagged tokens marked
  $("halu-answer").innerHTML = toks
    .map((t) => {
      const s = esc(t.token);
      if (!t.flag) return s;
      return `<mark title="score ${t.score} · p=${(t.p_final * 100).toFixed(0)}% · committed at ${(t.commit_depth * 100).toFixed(0)}% depth · workspace support ${(t.support * 100).toFixed(0)}% · instability ${(100 * (t.instability ?? 0)).toFixed(0)}% (${t.flips ?? 0} flips in the last third)">${s}</mark>`;
    })
    .join("");
  // contiguous flagged tokens -> spans
  const spans = [];
  let cur = null;
  for (const t of toks) {
    if (t.flag && cur && cur.end === t.pos - 1) { cur.end = t.pos; cur.text += t.token; cur.score = Math.max(cur.score, t.score); cur.n++; }
    else if (t.flag) { cur = { start: t.pos, end: t.pos, text: t.token, score: t.score, n: 1, p: t.p_final, d: t.commit_depth, s: t.support, i: t.instability ?? 0, f: t.flips ?? 0 }; spans.push(cur); }
    else cur = null;
  }
  // Verdict: calibrated P(wrong) from PopQA bins when available, else the raw score.
  const { cal, metric, pWrong, band } = verdictFor(H, thr);
  panel.className = "card " + band;
  const label = band === "low" ? "likely reliable" : band === "mid" ? "uncertain" : "likely hallucinated";
  const needle = pWrong != null ? pWrong : Math.min(1, metric);
  drawGauge($("halu-gauge"), needle, band);
  $("halu-verdict").innerHTML = pWrong != null
    ? `<span class="pct">${Math.round(pWrong * 100)}%</span>${esc(label)}<span class="note">estimated chance this answer is wrong · calibrated on ${cal.n} ${esc(cal.source)} questions</span>`
    : `<span class="pct">${metric.toFixed(2)}</span>${esc(label)}<span class="note">raw answer score, uncalibrated — run scripts/eval_popqa.py</span>`;
  const nWord = toks.filter((t) => WORDLIKE.test(t.token) && !t.in_prompt).length;
  const nFlag = toks.filter((t) => t.flag).length;
  $("halu-summary").textContent = `${nFlag}/${nWord} tokens flagged · max score ${H.max_score.toFixed(2)}`;
  const bar = (name, v, good, text) => `<div class="bar${good ? " good" : ""}">${name}<i style="--w:${Math.round(v * 100)}%"></i><b>${text}</b></div>`;
  $("halu-chips").innerHTML = spans.length
    ? spans.map((sp) => `<div class="sig-row"><div><div class="sig-word">${esc(sp.text.trim())}</div><div class="sig-score">score ${sp.score.toFixed(2)}</div></div><div class="bars">` +
        bar("confidence", sp.p, sp.p >= 0.5, `${(sp.p * 100).toFixed(0)}%`) +
        bar("decided at", sp.d, sp.d < 0.7, `${(sp.d * 100).toFixed(0)}% depth`) +
        bar("in workspace", sp.s, sp.s >= 0.3, `${(sp.s * 100).toFixed(0)}%`) +
        bar("instability", sp.i, sp.i < 0.3, `${(sp.i * 100).toFixed(0)}% · ${sp.f} flips`) +
      `</div></div>`).join("")
    : `<div class="sig-empty">no generated word is over the threshold</div>`;
  $("signals-card").hidden = false;
  $("fix-card").hidden = false;
  $("right-empty").hidden = true;
  panel.hidden = false;
  $("verify-out").textContent = "";
  $("verify-out").className = "";
  $("fix-out").hidden = true;
  renderHalu._pWrong = pWrong;
  $("verify").disabled = !(INFO && INFO.verifier);
  $("verify").title = INFO && INFO.verifier ? "ask a stronger model (Gemini) whether this answer is factually correct" : "put GEMINI_API_KEY in .env to enable";
}
$("fix").addEventListener("click", async () => {
  if (!data || !data.continuation) return;
  const out = $("fix-out");
  out.hidden = false; out.innerHTML = '<span class="k">working…</span><span>re-decoding with layer contrast (~30 s)</span>';
  $("fix").disabled = true;
  try {
    const H = data.hallucination;
    const res = await fetch("api/fix", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: $("prompt").value, chat: $("chat").checked, max_new_tokens: +$("newtok").value,
        p_wrong: renderHalu._pWrong, score: H.answer_score ?? H.max_score }) });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const f = await res.json();
    const orig = data.continuation.trim();
    // Order = what actually works on this model: verified correction, then abstention; the
    // layer-contrast re-decode is kept as a measured negative result (see docs §9b).
    const v = $("verify-out").className;
    const rows = [["original", esc(orig)]];
    rows.push(["<span class=\"ok\">✔ verified fix</span>", v === "wrong" ? esc($("verify-out").textContent) : v === "correct" ? '<span class="same">judge says the original is correct</span>' : '<span class="same">press Verify answer first — the judge supplies the correction</span>']);
    rows.push(["<span class=\"ok\">✔ safer answer</span>", f.abstain ? esc(f.abstain) : '<span class="same">risk is low — no need to abstain</span>']);
    const fs = INFO && INFO.fix_summary;
    const note = fs ? ` — measured on ${fs.n} questions: fixed ${fs.fixed}, broke ${fs.broken}, changed ${fs.changed}` : "";
    rows.push(["✘ DoLa re-decode<br><span class=\"k\">(experimental)</span>", (f.dola === orig ? '<span class="same">same answer</span>' : esc(f.dola)) + `<br><span class="k">re-ranks words by how much they emerged through the layers; can only pick among answers the model already had${esc(note)}</span>`]);
    out.innerHTML = rows.map(([k, v]) => `<span class="k">${k}</span><span>${v}</span>`).join("");
  } catch (e) { out.innerHTML = `<span class="k">error</span><span>${esc(e.message)}</span>`; }
  finally { $("fix").disabled = false; }
});
$("verify").addEventListener("click", async () => {
  if (!data || !data.continuation) return;
  const out = $("verify-out");
  out.className = ""; out.textContent = "asking the judge…";
  $("verify").disabled = true;
  try {
    const res = await fetch("api/verify", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: $("prompt").value, answer: data.continuation }) });
    const v = await res.json();
    out.className = v.verdict;
    const label = { correct: "✔ correct", wrong: "✘ wrong", unsure: "? unsure" }[v.verdict] || v.verdict;
    out.textContent = `${label}` + (v.correct_answer && v.verdict !== "correct" ? ` · correct answer: ${v.correct_answer}` : "") + (v.explanation ? ` — ${v.explanation}` : "");
  } catch (e) { out.className = "unsure"; out.textContent = e.message; }
  finally { $("verify").disabled = false; }
});
$("halu-thresh").addEventListener("input", () => {
  if (!data) { $("halu-thresh-val").textContent = (+$("halu-thresh").value).toFixed(2); return; }
  renderHalu();
  renderGrid();
});

// Calibrated P(wrong) from the PopQA bins when available, else the raw score.
function verdictFor(H, thr) {
  const cal = INFO && ((mode === "jlens" && INFO.calibration_jlens) || INFO.calibration);
  const metric = cal ? (H[cal.metric] ?? H.max_score) : H.max_score;
  const bin = cal && cal.bins.find((b) => metric >= b.lo && metric < b.hi);
  const pWrong = bin ? bin.p_wrong : null;
  const band = pWrong != null ? (pWrong < 0.25 ? "low" : pWrong < 0.6 ? "mid" : "high") : (H.max_score < thr ? "low" : "high");
  return { cal, metric, pWrong, band };
}

// Half-arc gauge: needle at P(wrong), arc coloured by band.
function drawGauge(svg, v, band) {
  const cx = 120, cy = 122, R = 88, w = 14;
  const col = band === "low" ? "#15803d" : band === "mid" ? "#c2703d" : "#b91c1c";
  const pt = (f, r) => { const a = Math.PI * (1 - f); return [cx + r * Math.cos(a), cy - r * Math.sin(a)]; };
  const arc = (f0, f1, r) => { const [x0, y0] = pt(f0, r), [x1, y1] = pt(f1, r); return `M${x0.toFixed(1)},${y0.toFixed(1)} A${r},${r} 0 0 1 ${x1.toFixed(1)},${y1.toFixed(1)}`; };
  const v1 = Math.max(0.005, Math.min(1, v));
  const [nx, ny] = pt(v1, R - w / 2 - 6);
  svg.innerHTML =
    `<path d="${arc(0, 1, R)}" fill="none" stroke="#ebe8e4" stroke-width="${w}" stroke-linecap="round"/>` +
    `<path d="${arc(0, v1, R)}" fill="none" stroke="${col}" stroke-width="${w}" stroke-linecap="round"/>` +
    `<line x1="${cx}" y1="${cy}" x2="${nx.toFixed(1)}" y2="${ny.toFixed(1)}" stroke="#1c1917" stroke-width="2" stroke-linecap="round"/>` +
    `<circle cx="${cx}" cy="${cy}" r="4" fill="#1c1917"/>` +
    `<text x="14" y="138" fill="#777169" font-size="10" font-family="Inter,system-ui">reliable</text>` +
    `<text x="${cx}" y="14" fill="#777169" font-size="10" font-family="Inter,system-ui" text-anchor="middle">uncertain</text>` +
    `<text x="226" y="138" fill="#777169" font-size="10" font-family="Inter,system-ui" text-anchor="end">hallucinated</text>`;
}

function renderMetrics() {
  $("metrics").hidden = false;
  const ms = data.layer_metrics;
  drawChart($("m-acc"), ms.map((m) => m.next_token_acc), 0, 1);
  drawChart($("m-kurt"), ms.map((m) => m.mean_kurtosis), 0, null, true);
  drawChart($("m-auto"), ms.map((m) => m.top1_autocorr), 0, 1);
}

// Minimal SVG line chart; optional shaded "workspace band" where the series
// exceeds half its max (heuristic, only drawn when shadeBand is set).
function drawChart(svg, ys, ymin, ymax, shadeBand = false) {
  const W = svg.clientWidth || 280, H = 110, padL = 34, padB = 16, padT = 6;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const lo = ymin ?? Math.min(...ys), hi = (ymax ?? Math.max(...ys)) || 1;
  const x = (i) => padL + (i * (W - padL - 6)) / Math.max(1, ys.length - 1);
  const y = (v) => padT + (H - padB - padT) * (1 - (v - lo) / (hi - lo || 1));
  let out = "";
  if (shadeBand) {
    const cut = Math.max(...ys) / 2;
    let start = null;
    for (let i = 0; i <= ys.length; i++) {
      const inBand = i < ys.length && ys[i] >= cut;
      if (inBand && start === null) start = i;
      if (!inBand && start !== null) {
        out += `<rect x="${x(start)}" y="${padT}" width="${x(i - 1) - x(start)}" height="${H - padB - padT}" fill="rgba(31,79,128,.08)"/>`;
        start = null;
      }
    }
  }
  out += `<line x1="${padL}" y1="${H - padB}" x2="${W - 6}" y2="${H - padB}" stroke="#e7e5e4"/>`;
  out += `<text x="${padL - 4}" y="${y(hi) + 4}" fill="#777169" font-size="9" text-anchor="end">${hi.toFixed(hi >= 10 ? 0 : 2)}</text>`;
  out += `<text x="${padL - 4}" y="${H - padB}" fill="#777169" font-size="9" text-anchor="end">${lo.toFixed(lo >= 10 ? 0 : 2)}</text>`;
  const lastFitted = ys.length - 2; // final entry is the output row (J = I)
  out += `<polyline fill="none" stroke="#1f4f80" stroke-width="1.75" points="${ys
    .slice(0, lastFitted + 1).map((v, i) => `${x(i)},${y(v)}`).join(" ")}"/>`;
  out += `<circle cx="${x(ys.length - 1)}" cy="${y(ys[ys.length - 1])}" r="2.5" fill="#c2703d"/>`;
  const L = data.layers;
  out += `<text x="${padL}" y="${H - 4}" fill="#777169" font-size="9">L${L[0]}</text>`;
  out += `<text x="${W - 6}" y="${H - 4}" fill="#777169" font-size="9" text-anchor="end">out</text>`;
  svg.innerHTML = out;
}

// ---- tooltip -------------------------------------------------------------
const tip = $("tooltip");
$("grid").addEventListener("mousemove", (e) => {
  const cell = e.target.closest(".cell");
  if (!cell || !data) { tip.hidden = true; return; }
  const li = +cell.dataset.l, t = +cell.dataset.t;
  const row = data.grid[li];
  const rows = row.top_ids[t]
    .map((id, i) =>
      `<tr><td>${esc(tokStr(data.vocab[id]))}${GLOSS.has((data.vocab[id] || "").trim()) ? ` <span class=\"gloss\">≈ ${esc(GLOSS.get((data.vocab[id] || "").trim()))}</span>` : ""}</td><td class="p">${(row.top_probs[t][i] * 100).toFixed(1)}%</td></tr>`)
    .join("");
  tip.innerHTML =
    `<table>${rows}</table>` +
    `<div class="meta">${row.is_output ? "output" : "L" + row.layer} · pos ${t} · ` +
    `H=${row.entropy[t]} · kurt=${row.kurtosis[t]}${row.jsd_prev ? ` · Δprev=${(100 * row.jsd_prev[t]).toFixed(0)}%` : ""}<br>click to pin “${esc(tokStr(data.vocab[row.top_ids[t][0]]))}”</div>`;
  tip.hidden = false;
  const w = tip.offsetWidth, h = tip.offsetHeight;
  tip.style.left = Math.min(e.clientX + 14, innerWidth - w - 8) + "px";
  tip.style.top = Math.min(e.clientY + 14, innerHeight - h - 8) + "px";
});
$("grid").addEventListener("mouseleave", () => (tip.hidden = true));

// ---- interactions ----------------------------------------------------------
$("grid").addEventListener("click", (e) => {
  const cell = e.target.closest(".cell");
  if (!cell || !data) return;
  const row = data.grid[+cell.dataset.l];
  const id = row.top_ids[+cell.dataset.t][0];
  if (!pinned.some((p) => p.id === id)) {
    if (pinned.length >= 16) pinned.shift();
    pinned.push({ id, str: data.vocab[id] ?? "?" });
  }
  rankViewId = id;
  renderPinned();
  read(); // refetch so pinned_ranks include the new token
});

$("pinned-chips").addEventListener("click", (e) => {
  const x = e.target.dataset.x, chip = e.target.closest(".pin-chip");
  if (x !== undefined) {
    pinned = pinned.filter((p) => p.id !== +x);
    if (rankViewId === +x) rankViewId = null;
    renderPinned(); read();
  } else if (chip) {
    rankViewId = +chip.dataset.id;
    renderPinned(); renderGrid();
  }
});
$("view-top1").addEventListener("click", () => { rankViewId = null; renderPinned(); renderGrid(); });

$("mode-toggle").addEventListener("click", (e) => {
  const btn = e.target.closest("button");
  if (!btn) return;
  mode = btn.dataset.mode;
  for (const b of $("mode-toggle").children) b.classList.toggle("active", b === btn);
  // the J-lens reads real concepts mid-network, often as single CJK tokens in a bilingual model
  $("english").checked = mode !== "jlens";
  read();
});
$("examples").addEventListener("change", (e) => {
  if (e.target.value === "") return;
  loadExample(+e.target.value);
});
$("read").addEventListener("click", read);
$("prompt").addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") read();
});
// Editing the prompt by hand means it's no longer the selected example.
// In live mode, re-read as you type (debounced) — watch the thoughts move.
let liveTimer = null;
$("prompt").addEventListener("input", () => {
  if (playing) return; // typewriter playback, not the user
  playToken++; // typing cancels any running playback
  $("examples").value = "";
  if (STATIC || !$("live").checked) return;
  clearTimeout(liveTimer);
  liveTimer = setTimeout(() => read({ live: true }), 650);
});

init().catch((e) => setStatus(e.message, true));

// ---- compare view ----------------------------------------------------------
// Two example runs side by side: who wins the candidate race at the answer
// onset, how foggy each layer is, and the detector's signals A vs B.
const CMP_PAIRS = [
  { label: "Known vs obscure capital", a: 0, b: 1 },
  { label: "Fact vs fabricated book", a: 0, b: 4 },
  { label: "Obscure vs fabricated", a: 2, b: 4 },
];
const cmp = { a: 0, b: 1, runs: new Map(), menu: null, grids: false, token: 0 };
const DECLINE = /no such|not aware|does ?n[o']t exist|not exist|fictional|could ?n[o']t find|unable to|not familiar|no record|not a real|can ?n[o']t find|not sure|no information|don't have/i;

function showView(v) {
  for (const b of $("view-nav").children) b.classList.toggle("active", b.dataset.view === v);
  $("toolbar").hidden = $("main").hidden = v !== "explore";
  $("compare").hidden = v !== "compare";
  if (v === "compare") renderCompare();
}
$("view-nav").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) showView(b.dataset.view); });

async function cmpRead(i) {
  const key = `${i}:${mode}`;
  if (cmp.runs.has(key)) return cmp.runs.get(key);
  const res = await fetch("api/read", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt: INFO.examples[i].prompt, mode, top_k: 8, chat: $("chat").checked, max_new_tokens: +$("newtok").value, threshold: +$("halu-thresh").value }),
  });
  if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
  const d = await res.json();
  cmp.runs.set(key, d);
  return d;
}

// gold answer: substring match; a fabricated premise is correct only if the model declines
function grade(ex, d) {
  const ans = d.continuation || "";
  if (!ex.gold) return "ungraded";
  if (ex.fabricated) return DECLINE.test(ans) ? "correct" : "wrong";
  return ans.toLowerCase().includes(ex.gold.toLowerCase()) ? "correct" : "wrong";
}
function shortAnswer(d) {
  const s = (d.continuation || "").replace(/\s+/g, " ").trim();
  return s.length > 64 ? s.slice(0, 63) + "…" : s;
}
// The answer onset: first generated word that is not a stopword or a prompt word.
function onsetPos(d) {
  const pl = d.prompt_len ?? d.seq_len;
  const tok = (t) => d.vocab[d.context_ids[t]] || "";
  const promptText = d.context_ids.slice(0, pl).map((id) => d.vocab[id] || "").join("").toLowerCase();
  for (let t = pl; t < d.seq_len; t++) {
    if (t > pl && !/^[\s▁]/.test(tok(t))) continue; // not a word start
    let w = tok(t).trim();
    for (let u = t + 1; u < d.seq_len && /^[A-Za-z0-9]/.test(tok(u)); u++) w += tok(u); // join the word's pieces
    w = w.toLowerCase();
    if (/^[a-z0-9]/.test(w) && w.length > 2 && !STOP.has(w) && !promptText.includes(w)) return t;
  }
  return Math.min(pl, d.seq_len - 1);
}
const VTXT = { correct: '<span class="v-ok">✓ Correct</span>', wrong: '<span class="v-bad">✕ Wrong</span>', ungraded: '<span class="v-na">○ Ungraded</span>' };

function renderSlots(runs) {
  $("cmp-pairs").innerHTML = CMP_PAIRS.map((p, k) => `<button class="pill${p.a === cmp.a && p.b === cmp.b ? " sel" : ""}" data-pair="${k}">${esc(p.label)}</button>`).join("");
  $("cmp-slots").innerHTML = ["a", "b"].map((s, k) => {
    const i = cmp[s], ex = INFO.examples[i], d = runs[k];
    const menu = cmp.menu === s
      ? `<div class="cmp-menu" role="menu">${INFO.examples.map((e, j) => `<button role="menuitem" data-pick="${s}:${j}" class="${j === i ? "cur" : ""}"><span>${esc(e.name)}</span><span class="hint">${esc(e.gold || "")}</span></button>`).join("")}</div>` : "";
    const meta = d
      ? `<code>→ ${esc(shortAnswer(d))}</code>${VTXT[grade(ex, d)]}<span>gold: ${esc(ex.gold || "none")}</span>`
      : `<span class="hint">${runs[k] === undefined ? "waiting" : "reading…"}</span>`;
    return `<div class="cmp-slot card"><span class="tag">${s.toUpperCase()}</span><div class="body"><span class="q" title="${esc(ex.prompt)}">${esc(ex.name)}</span><span class="meta">${meta}</span></div><button class="pill" data-menu="${s}" aria-expanded="${cmp.menu === s}">Change run</button>${menu}</div>`;
  }).join("");
}

async function renderCompare() {
  if (STATIC) { $("cmp-slots").innerHTML = '<p class="hint">Compare needs a live model — clone the repo and run it locally.</p>'; return; }
  const token = ++cmp.token;
  const runs = [null, undefined];
  renderSlots(runs);
  for (const [k, s] of [[0, "a"], [1, "b"]]) {
    runs[k] = null;
    renderSlots(runs);
    try { runs[k] = await cmpRead(cmp[s]); } catch (e) { setStatus(e.message, true); $("cmp-slots").insertAdjacentHTML("beforeend", `<p class="hint v-bad">${esc(e.message)}</p>`); return; }
    if (token !== cmp.token) return;
  }
  renderSlots(runs);
  renderCmpBody(runs);
}
$("compare").addEventListener("click", (e) => {
  const pair = e.target.closest("[data-pair]"), menu = e.target.closest("[data-menu]"), pick = e.target.closest("[data-pick]");
  if (pair) { const p = CMP_PAIRS[+pair.dataset.pair]; cmp.a = p.a; cmp.b = p.b; cmp.menu = null; renderCompare(); }
  else if (pick) { const [s, j] = pick.dataset.pick.split(":"); cmp[s] = +j; cmp.menu = null; renderCompare(); }
  else if (menu) { cmp.menu = cmp.menu === menu.dataset.menu ? null : menu.dataset.menu; renderSlots([cmp.runs.get(`${cmp.a}:${mode}`), cmp.runs.get(`${cmp.b}:${mode}`)]); }
  else if (e.target.closest("#cmp-grids-toggle")) { cmp.grids = !cmp.grids; $("cmp-grids").hidden = !cmp.grids; $("cmp-grids-toggle").textContent = cmp.grids ? "Hide slice grids" : "Show slice grids"; }
  else if (cmp.menu && !e.target.closest(".cmp-menu")) { cmp.menu = null; renderSlots([cmp.runs.get(`${cmp.a}:${mode}`), cmp.runs.get(`${cmp.b}:${mode}`)]); }
});

const fittedRows = (d) => d.grid.filter((r) => !r.is_output);
const bandOf = (rows) => [Math.floor(rows.length * 0.25), Math.ceil(rows.length * 0.85) - 1];
const axisX = (W, pl, pr, n) => (i) => pl + (i * (W - pl - pr)) / Math.max(1, n - 1);

function renderCmpBody([A, B]) {
  $("cmp-race").innerHTML = [["A", cmp.a, A], ["B", cmp.b, B]].map(([tag, i, d]) => raceChart(tag, INFO.examples[i], d)).join("");
  $("cmp-fog").innerHTML = fogChart(A, B);
  $("cmp-sig").innerHTML = sigTable(A, B);
  $("cmp-grids").innerHTML = [["A", cmp.a, A], ["B", cmp.b, B]].map(([tag, i, d]) => miniGrid(tag, INFO.examples[i], d)).join("");
}

// Which candidates the lens ranked first at the onset position, layer by layer.
function raceChart(tag, ex, d) {
  const t = onsetPos(d), c = t - 1, rows = fittedRows(d), emitted = d.context_ids[t];
  const goldW = ex.gold && !ex.fabricated ? ex.gold.toLowerCase().split(/\s+/)[0] : null;
  const peak = new Map();
  rows.forEach((r) => r.top_ids[c].forEach((id, k) => { if (r.top_probs[c][k] > (peak.get(id) || 0)) peak.set(id, r.top_probs[c][k]); }));
  const others = [...peak.entries()].sort((x, y) => y[1] - x[1]).map(([id]) => id).filter((id) => id !== emitted).slice(0, 5);
  const goldId = [...peak.keys()].find((id) => (d.vocab[id] || "").trim().toLowerCase() === goldW);
  const W = 460, H = 200, pl = 30, pr = 12, pt = 12, pb = 20;
  const x = axisX(W, pl, pr, rows.length), y = (p) => pt + (H - pt - pb) * (1 - p);
  const prob = (r, id) => { const k = r.top_ids[c].indexOf(id); return k < 0 ? 0 : r.top_probs[c][k]; };
  const [lo, hi] = bandOf(rows);
  let out = `<rect x="${x(lo)}" y="${pt}" width="${x(hi) - x(lo)}" height="${H - pt - pb}" fill="#eef3f8"/>`;
  for (const g of [0, 0.5, 1]) out += `<line x1="${pl}" x2="${W - pr}" y1="${y(g)}" y2="${y(g)}" stroke="#e7e5e4"/><text x="${pl - 5}" y="${y(g) + 3}" text-anchor="end" font-size="9" fill="#777169">${g}</text>`;
  const greys = ["#777169", "#8f8983", "#a59f97", "#b6b0a8", "#c9c4bd"];
  [...others.reverse(), emitted].forEach((id, k) => {
    const isE = id === emitted, isG = id === goldId;
    const ps = rows.map((r) => prob(r, id));
    const stroke = isE ? "#c2703d" : isG ? "#1c1917" : greys[others.length - 1 - k] || "#c9c4bd";
    out += `<polyline fill="none" points="${ps.map((p, i) => `${x(i).toFixed(1)},${y(p).toFixed(1)}`).join(" ")}" stroke="${stroke}" stroke-width="${isE ? 3 : isG ? 1.5 : 1.25}"${isG && !isE ? ' stroke-dasharray="6 4"' : ""} stroke-linejoin="round"/>`;
    const pi = ps.indexOf(Math.max(...ps));
    out += `<text x="${(x(pi) + (pi > rows.length / 2 ? -4 : 4)).toFixed(1)}" y="${(y(ps[pi]) - 4).toFixed(1)}" text-anchor="${pi > rows.length / 2 ? "end" : "start"}" font-size="10" fill="${stroke}" font-family="Geist Mono,ui-monospace,monospace">${esc(tokStr(d.vocab[id]))}</text>`;
  });
  const commit = rows.findIndex((r, i) => rows.slice(i).every((rr) => rr.top_ids[c][0] === emitted));
  if (commit > 0) { const late = commit > rows.length / 2; out += `<line x1="${x(commit)}" x2="${x(commit)}" y1="${pt}" y2="${H - pb}" stroke="#1c1917" stroke-dasharray="2 3"/><text x="${x(commit) + (late ? -3 : 3)}" y="${H - pb - 4}" text-anchor="${late ? "end" : "start"}" font-size="9" fill="#1c1917">commit L${rows[commit].layer}</text>`; }
  out += `<text x="${pl}" y="${H - 6}" font-size="9" fill="#777169">L${rows[0].layer}</text><text x="${W - pr}" y="${H - 6}" font-size="9" fill="#777169" text-anchor="end">L${rows[rows.length - 1].layer}</text>`;
  const note = goldW ? (goldId ? " · dashed: gold answer" : " · gold never in the top-8") : "";
  return `<div class="race"><div class="race-cap"><span class="tag sm">${tag}</span><span>p(<b>${esc(tokStr(d.vocab[emitted]))}</b>) at position ${t}, emitted in orange${note}</span></div><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Candidate race, run ${tag}">${out}</svg></div>`;
}

function fogChart(A, B) {
  const sa = fittedRows(A).map((r) => r.entropy[onsetPos(A) - 1]), sb = fittedRows(B).map((r) => r.entropy[onsetPos(B) - 1]);
  const W = 460, H = 150, pl = 30, pr = 12, pt = 10, pb = 20, n = Math.max(sa.length, sb.length);
  const top = Math.max(1, ...sa, ...sb);
  const x = axisX(W, pl, pr, n), y = (v) => pt + (H - pt - pb) * (1 - v / top);
  const [lo, hi] = bandOf(fittedRows(A));
  let out = `<rect x="${x(lo)}" y="${pt}" width="${x(hi) - x(lo)}" height="${H - pt - pb}" fill="#f6f8fb"/>`;
  for (const g of [0, top / 2, top]) out += `<line x1="${pl}" x2="${W - pr}" y1="${y(g)}" y2="${y(g)}" stroke="#e7e5e4"/><text x="${pl - 5}" y="${y(g) + 3}" text-anchor="end" font-size="9" fill="#777169">${g.toFixed(0)}</text>`;
  for (const [s, col] of [[sa, "#1f4f80"], [sb, "#c2703d"]]) out += `<polyline fill="none" stroke="${col}" stroke-width="2" stroke-linejoin="round" points="${s.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ")}"/>`;
  out += `<text x="${pl}" y="${H - 6}" font-size="9" fill="#777169">L0</text><text x="${W - pr}" y="${H - 6}" font-size="9" fill="#777169" text-anchor="end">L${n - 1}</text>`;
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Entropy by layer, A and B">${out}</svg>` +
    `<div class="legend"><i class="sw" style="background:#1f4f80"></i>A<i class="sw" style="background:#c2703d"></i>B<i class="sw" style="background:#eef3f8"></i>workspace band</div>`;
}

function sigTable(A, B) {
  const sig = (d) => {
    const H = d.hallucination, t = onsetPos(d), tok = H && H.tokens.find((k) => k.pos === t);
    const rows = fittedRows(d), [lo, hi] = bandOf(rows);
    const ent = rows.slice(lo, hi + 1).reduce((s, r) => s + r.entropy[t - 1], 0) / (hi - lo + 1);
    return { pw: H ? verdictFor(H, +$("halu-thresh").value).pWrong : null, max: H ? H.max_score : null, p: tok ? tok.p_final : null, d: tok ? tok.commit_depth : null, s: tok ? tok.support : null, i: tok ? (tok.instability ?? 0) : null, ent };
  };
  const a = sig(A), b = sig(B);
  const pct = (v) => (v == null ? "—" : `${Math.round(v * 100)}%`);
  const rows = [
    ["P(wrong), calibrated", a.pw, b.pw, pct, 1, true],
    ["Max hallucination score", a.max, b.max, (v) => (v == null ? "—" : v.toFixed(2)), 1, false],
    ["Confidence at onset", a.p, b.p, pct, 1, false],
    ["Decided at (depth)", a.d, b.d, pct, 1, false],
    ["Workspace support", a.s, b.s, pct, 1, false],
    ["Instability", a.i, b.i, pct, 1, false],
    ["Band entropy at onset, bits", a.ent, b.ent, (v) => v.toFixed(2), Math.max(a.ent, b.ent, 1), false],
  ];
  return `<div class="sig-head"><span>Signal</span><span>A</span><span>B</span><span>Gap, scaled to signal range</span></div>` + rows.map(([label, va, vb, fmt, range, bold]) => {
    const gap = va == null || vb == null ? 0 : Math.abs(va - vb) / range;
    const tag = va == null || vb == null ? "" : Math.abs(va - vb) < 1e-3 ? "same" : va > vb ? "A higher" : "B higher";
    return `<div class="sig-line${bold ? " bold" : ""}"><span>${label}</span><span>${fmt(va)}</span><span>${fmt(vb)}</span><span class="gap"><i><b style="width:${Math.round(gap * 100)}%;background:${va > vb ? "#1f4f80" : "#c2703d"}"></b></i><em>${tag}</em></span></div>`;
  }).join("");
}

// Compact slice grid: the last 8 prompt tokens plus the answer, every layer.
function miniGrid(tag, ex, d) {
  const pl = d.prompt_len ?? d.seq_len, from = Math.max(0, pl - 8), eng = $("english").checked;
  const cols = []; for (let t = from; t < d.seq_len; t++) cols.push(t);
  const head = `<div class="cg-row cg-head" style="grid-template-columns:28px repeat(${cols.length},56px)"><span></span>${cols.map((t) => `<span class="${t >= pl ? "gen" : ""}" title="position ${t}">${esc(tokStr(d.vocab[d.context_ids[t]]))}</span>`).join("")}</div>`;
  const body = d.grid.map((row) => `<div class="cg-row${row.is_output ? " out" : ""}" style="grid-template-columns:28px repeat(${cols.length},56px)"><span class="lab">${row.is_output ? "out" : "L" + row.layer}</span>${cols.map((t) => {
    let i = 0; if (eng) i = row.top_ids[t].findIndex((id) => ENGLISH.test(d.vocab[id] || ""));
    if (i < 0) return `<span style="background:transparent">·</span>`;
    const [bg, dk] = probColor(row.top_probs[t][i]);
    return `<span class="${dk ? "dk" : ""}" style="background:${bg}" title="${(row.top_probs[t][i] * 100).toFixed(0)}%">${esc(tokStr(d.vocab[row.top_ids[t][i]]))}</span>`;
  }).join("")}</div>`).join("");
  return `<div class="cg"><span class="cg-title"><span class="tag sm">${tag}</span>${esc(ex.name)}</span><div class="cg-scroll">${head}${body}</div></div>`;
}
