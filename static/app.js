// Voice shell for CBRE dispatch agent.
// Captures live conversation via Web Speech API, accumulates turns in the
// canonical [{speaker, text}] shape, and hands them to the same classify()
// the batch harness uses. TTS reads back the clarification question or
// final dispatch summary.

const $ = (id) => document.getElementById(id);
const els = {
  status: $("status"),
  phone: $("phone"),
  callerInfo: $("callerInfo"),
  micBtn: $("micBtn"),
  micLabel: document.querySelector(".mic-label"),
  resetBtn: $("resetBtn"),
  ttsToggle: $("ttsToggle"),
  transcript: $("transcript"),
  result: $("result"),
  resultBody: $("resultBody"),
  summary: $("summary"),
  hint: $("hint"),
  composer: $("composer"),
  textInput: $("textInput"),
  sendBtn: $("sendBtn"),
};

const SpeechRecognitionImpl =
  window.SpeechRecognition || window.webkitSpeechRecognition;

const state = {
  turns: [],
  recognition: null,
  listening: false,
  ttsOn: true,
  // Once we have a finalized classification (no more clarification needed)
  // we stop auto-classifying so the user can review.
  callClosed: false,
  // We block the mic while the backend is thinking or the agent is speaking.
  busy: false,
  pendingInterim: null,
};

// ── UI helpers ──────────────────────────────────────────────────────────────

function setStatus(text, kind = "idle") {
  els.status.textContent = text;
  els.status.dataset.state = kind;
}

function setMicState(active) {
  state.listening = active;
  els.micBtn.setAttribute("aria-pressed", active ? "true" : "false");
  els.micLabel.textContent = active ? "Stop listening" : "Start call";
}

function setBusy(busy) {
  state.busy = busy;
  els.micBtn.disabled = busy;
  els.textInput.disabled = busy;
  els.sendBtn.disabled = busy;
}

function renderTurns() {
  // Re-render keeps DOM in sync with state.turns. Cheap enough for a demo.
  els.transcript.innerHTML = "";
  for (const t of state.turns) {
    appendTurnNode(t.speaker, t.text, false);
  }
  if (state.pendingInterim) {
    appendTurnNode("caller", state.pendingInterim, true);
  }
  els.transcript.scrollTop = els.transcript.scrollHeight;
}

function appendTurnNode(speaker, text, interim) {
  const div = document.createElement("div");
  div.className = `turn ${speaker}${interim ? " interim" : ""}`;
  const who = document.createElement("span");
  who.className = "who";
  who.textContent = speaker;
  const body = document.createElement("span");
  body.textContent = text;
  div.appendChild(who);
  div.appendChild(body);
  els.transcript.appendChild(div);
}

function pushTurn(speaker, text) {
  const trimmed = (text || "").trim();
  if (!trimmed) return;
  state.turns.push({ speaker, text: trimmed });
  state.pendingInterim = null;
  renderTurns();
}

function showResult(pred) {
  els.result.hidden = false;
  const rows = [
    ["Category", pred.category || "—"],
    ["Subcategory", pred.subcategory || "—"],
    [
      "Risk",
      `<span class="pill ${
        (pred.risk_level || "medium").toLowerCase()
      }">${pred.risk_level || "—"}</span>`,
    ],
    ["Building", pred.building_name || "—"],
    ["Floor", pred.floor || "—"],
    ["Address", pred.address || "—"],
    ["Vendor", pred.dispatched_vendor_id || "—"],
    [
      "911 dispatched",
      pred.dispatched_emergency_services ? "YES" : "no",
    ],
    [
      "Needs human review",
      pred.needs_human_review ? "yes" : "no",
    ],
    [
      "Needs clarification",
      pred.needs_clarification ? "yes" : "no",
    ],
  ];
  els.resultBody.innerHTML = rows
    .map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`)
    .join("");
  els.summary.textContent = pred.call_summary || "";
}

function showCallerProfile(profile) {
  if (!profile) {
    els.callerInfo.hidden = true;
    els.callerInfo.innerHTML = "";
    return;
  }
  els.callerInfo.hidden = false;
  const bldg = profile.primary_building_name || profile.primary_building_id || "—";
  const name = profile.caller_name || profile.name || "unnamed";
  const role = profile.contact_role || profile.role;
  els.callerInfo.innerHTML =
    `Known caller — <b>${name}</b>` +
    (role ? `, ${role}` : "") +
    ` · primary building <b>${bldg}</b>`;
}

// ── Speech recognition (STT) ────────────────────────────────────────────────

function buildRecognizer() {
  if (!SpeechRecognitionImpl) return null;
  const rec = new SpeechRecognitionImpl();
  rec.lang = "en-US";
  rec.continuous = false;
  rec.interimResults = true;
  rec.maxAlternatives = 1;

  rec.onresult = (event) => {
    let interim = "";
    let final = "";
    for (let i = event.resultIndex; i < event.results.length; i++) {
      const r = event.results[i];
      if (r.isFinal) final += r[0].transcript;
      else interim += r[0].transcript;
    }
    if (interim) {
      state.pendingInterim = interim;
      renderTurns();
    }
    if (final) {
      pushTurn("caller", final);
    }
  };

  rec.onerror = (event) => {
    console.warn("speech error", event.error);
    if (event.error === "not-allowed" || event.error === "service-not-allowed") {
      setStatus("mic blocked", "error");
      setMicState(false);
    } else if (event.error === "no-speech") {
      // benign — onend will handle restart logic
    } else {
      setStatus(event.error, "error");
    }
  };

  rec.onend = () => {
    // Triggered after each utterance (since continuous=false). If we have a
    // pending caller turn, classify; otherwise restart listening if still on.
    if (state.callClosed) {
      setMicState(false);
      setStatus("call closed", "idle");
      return;
    }
    const newCallerTurn = state.turns.length > 0 &&
      state.turns[state.turns.length - 1].speaker === "caller";
    if (newCallerTurn && !state.busy) {
      classifyAndRespond();
    } else if (state.listening && !state.busy) {
      // user paused without saying anything — restart
      safeStart();
    }
  };

  return rec;
}

function safeStart() {
  if (!state.recognition) return;
  try {
    state.recognition.start();
    setStatus("listening", "listening");
  } catch (e) {
    // start() throws if recognition is already running — ignore
  }
}

function startListening() {
  if (!state.recognition) {
    state.recognition = buildRecognizer();
    if (!state.recognition) {
      setStatus("STT unsupported", "error");
      alert(
        "Web Speech API not supported in this browser. Use Chrome or Edge."
      );
      return;
    }
  }

  // Open with an agent greeting on the very first turn so the loop feels
  // like a real call. Skip if the user already has turns (resumed).
  if (state.turns.length === 0) {
    const greeting = "CBRE maintenance, what's going on?";
    pushTurn("agent", greeting);
    speak(greeting, () => {
      setMicState(true);
      safeStart();
    });
    return;
  }

  setMicState(true);
  safeStart();
}

function stopListening() {
  setMicState(false);
  setStatus("idle", "idle");
  try {
    state.recognition && state.recognition.stop();
  } catch (e) {
    /* ignore */
  }
}

// ── TTS ─────────────────────────────────────────────────────────────────────

// Voices load asynchronously in some browsers. Resolve the best available
// once and cache it. We prefer natural-sounding neural voices when present.
const VOICE_PREFERENCES = [
  // Chrome on macOS / desktop — high-quality cloud voices
  "Google US English",
  "Google UK English Female",
  "Google UK English Male",
  // macOS / Safari — neural "Siri" voices are the most natural
  "Samantha (Enhanced)",
  "Alex (Enhanced)",
  "Ava (Premium)",
  "Ava (Enhanced)",
  "Allison (Enhanced)",
  "Samantha",
  "Alex",
  // Edge / Windows — neural "Online" voices
  "Microsoft Aria Online (Natural) - English (United States)",
  "Microsoft Jenny Online (Natural) - English (United States)",
  "Microsoft Guy Online (Natural) - English (United States)",
  "Microsoft Aria - English (United States)",
];

let cachedVoice = null;
let voicesReady = false;

function pickVoice() {
  if (!("speechSynthesis" in window)) return null;
  const voices = window.speechSynthesis.getVoices();
  if (!voices || voices.length === 0) return null;

  // Exact-name match against our preference list
  for (const name of VOICE_PREFERENCES) {
    const v = voices.find((x) => x.name === name);
    if (v) return v;
  }
  // Loose match: any en-US voice flagged as a "natural"/"neural"/"premium"
  const enUS = voices.filter((v) => /^en[-_]US/i.test(v.lang));
  const natural = enUS.find((v) =>
    /natural|neural|premium|enhanced|online/i.test(v.name)
  );
  if (natural) return natural;
  // Last resort: default en-US, then default voice
  return (
    enUS.find((v) => v.default) ||
    enUS[0] ||
    voices.find((v) => v.default) ||
    voices[0]
  );
}

function refreshVoice() {
  cachedVoice = pickVoice();
  voicesReady = !!cachedVoice;
}

if ("speechSynthesis" in window) {
  refreshVoice();
  // Chrome fires voiceschanged once voices are actually loaded.
  window.speechSynthesis.addEventListener("voiceschanged", refreshVoice);
}

function speak(text, onDone) {
  if (!text) {
    onDone && onDone();
    return;
  }
  if (!state.ttsOn || !("speechSynthesis" in window)) {
    onDone && onDone();
    return;
  }
  setStatus("speaking", "speaking");
  window.speechSynthesis.cancel();

  if (!cachedVoice) refreshVoice();

  const u = new SpeechSynthesisUtterance(text);
  if (cachedVoice) {
    u.voice = cachedVoice;
    u.lang = cachedVoice.lang || "en-US";
  } else {
    u.lang = "en-US";
  }
  u.rate = 1.0;
  u.pitch = 1.0;
  u.volume = 1.0;
  u.onend = () => {
    setStatus("idle", "idle");
    onDone && onDone();
  };
  u.onerror = () => {
    setStatus("idle", "idle");
    onDone && onDone();
  };
  window.speechSynthesis.speak(u);
}

// ── Backend round-trip ──────────────────────────────────────────────────────

async function classifyAndRespond() {
  setBusy(true);
  setStatus("thinking", "thinking");

  // Pause STT during the round-trip; we'll restart based on the result.
  try { state.recognition && state.recognition.stop(); } catch (e) {}

  const phone = els.phone.value.trim() || null;
  let pred;
  try {
    const resp = await fetch("/api/classify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        turns: state.turns,
        caller_phone: phone,
      }),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ error: resp.statusText }));
      throw new Error(err.error || `HTTP ${resp.status}`);
    }
    pred = await resp.json();
  } catch (e) {
    console.error(e);
    setStatus("backend error", "error");
    setBusy(false);
    return;
  }

  showCallerProfile(pred.caller_profile);
  showResult(pred);

  const needsMore =
    pred.needs_clarification && pred.clarification_question;

  if (needsMore) {
    const question = pred.clarification_question;
    pushTurn("agent", question);
    setBusy(false);
    speak(question, () => {
      // Resume listening for the caller's answer.
      if (state.listening) safeStart();
    });
  } else {
    // Final response: read back the dispatch summary and close the call.
    state.callClosed = true;
    setMicState(false);
    const dispatched = pred.dispatched_vendor_id
      ? ` Dispatched ${pred.dispatched_vendor_id}.`
      : pred.needs_human_review
      ? " Routing this to a human dispatcher."
      : "";
    const spoken = (pred.call_summary || "Got it.") + dispatched;
    pushTurn("agent", spoken);
    setBusy(false);
    speak(spoken, () => {
      setStatus("call closed", "idle");
    });
  }
}

// ── Wiring ──────────────────────────────────────────────────────────────────

els.micBtn.addEventListener("click", () => {
  if (state.callClosed) {
    // start a fresh call automatically
    resetAll();
  }
  if (state.listening) {
    stopListening();
  } else {
    startListening();
  }
});

els.resetBtn.addEventListener("click", () => resetAll());

els.ttsToggle.addEventListener("click", () => {
  state.ttsOn = !state.ttsOn;
  els.ttsToggle.setAttribute("aria-pressed", state.ttsOn ? "true" : "false");
  els.ttsToggle.textContent = `TTS: ${state.ttsOn ? "on" : "off"}`;
  if (!state.ttsOn) window.speechSynthesis.cancel();
});

// Text composer — typed messages share the same classify pipeline as voice.
els.composer.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = els.textInput.value.trim();
  if (!text || state.busy) return;

  // If a prior call closed, start a fresh session on send.
  if (state.callClosed) {
    resetAll();
  }

  // Stop the mic loop if it was running — we don't want the recognizer
  // to grab the user's typed turn and re-classify mid-flight.
  if (state.listening) stopListening();

  // Seed an agent greeting on the very first turn so the LLM has the
  // canonical [agent, caller, ...] shape the agent expects.
  if (state.turns.length === 0) {
    pushTurn("agent", "CBRE maintenance, what's going on?");
  }

  pushTurn("caller", text);
  els.textInput.value = "";
  classifyAndRespond();
});

function resetAll() {
  stopListening();
  window.speechSynthesis && window.speechSynthesis.cancel();
  state.turns = [];
  state.pendingInterim = null;
  state.callClosed = false;
  state.busy = false;
  setBusy(false);
  renderTurns();
  els.result.hidden = true;
  els.summary.textContent = "";
  els.resultBody.innerHTML = "";
  els.textInput.value = "";
  showCallerProfile(null);
  setStatus("idle", "idle");
}

// First-run feature check — voice is best-effort, typing always works.
if (!SpeechRecognitionImpl) {
  els.micBtn.disabled = true;
  els.micLabel.textContent = "Voice unavailable";
  els.hint.innerHTML =
    "Voice not supported in this browser — you can still type below. " +
    "For voice, use <b>Chrome</b> or <b>Edge</b>.";
}

setStatus("idle", "idle");
