// Voice shell for the CBRE dispatch agent — ElevenLabs edition.
//
// STT: ElevenLabs Scribe v2 realtime, browser-side WebSocket via a single-use
// token minted server-side. VAD auto-commits transcript segments on silence.
// TTS: ElevenLabs Flash v2.5 streaming, fetched as MP3 from our /api/tts/stream
// proxy and played through an <audio> element so the first byte plays in ~75ms.
//
// The classify pipeline (POST /api/classify) is unchanged; this file only owns
// the audio layer.

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

const EL = window.ElevenLabsClient;
const Scribe = EL && EL.Scribe;
const RealtimeEvents = EL && EL.RealtimeEvents;

const state = {
  turns: [],
  scribe: null,          // active Scribe connection (or null when idle)
  scribeOpenedAt: 0,     // wall-clock ms when the current Scribe socket opened
  consecutiveQuickCloses: 0, // reconnect-loop guard (resets on healthy open)
  stopRequested: false,  // set true by intentional stops so CLOSE doesn't reconnect
  keyterms: [],          // cached domain-bias list from /api/keyterms
  listening: false,
  ttsOn: true,
  callClosed: false,
  busy: false,           // backend round-trip in flight
  pendingInterim: "",    // last partial transcript fragment for live display
  currentAudio: null,    // <audio> element for in-flight TTS (for barge-in)
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
  els.transcript.replaceChildren();
  for (const t of state.turns) els.transcript.appendChild(turnNode(t.speaker, t.text, false));
  if (state.pendingInterim) els.transcript.appendChild(turnNode("caller", state.pendingInterim, true));
  els.transcript.scrollTop = els.transcript.scrollHeight;
}

function turnNode(speaker, text, interim) {
  const div = document.createElement("div");
  div.className = `turn ${speaker}${interim ? " interim" : ""}`;
  const who = document.createElement("span");
  who.className = "who";
  who.textContent = speaker;
  const body = document.createElement("span");
  body.textContent = text;
  div.appendChild(who);
  div.appendChild(body);
  return div;
}

function pushTurn(speaker, text) {
  const trimmed = (text || "").trim();
  if (!trimmed) return;
  state.turns.push({ speaker, text: trimmed });
  state.pendingInterim = "";
  renderTurns();
}

function appendDtDd(parent, dtText, ddContent) {
  const dt = document.createElement("dt");
  dt.textContent = dtText;
  const dd = document.createElement("dd");
  if (ddContent instanceof Node) dd.appendChild(ddContent);
  else dd.textContent = ddContent == null ? "—" : String(ddContent);
  parent.appendChild(dt);
  parent.appendChild(dd);
}

function riskPill(level) {
  const span = document.createElement("span");
  span.className = `pill ${(level || "medium").toLowerCase()}`;
  span.textContent = level || "—";
  return span;
}

function showResult(pred) {
  els.result.hidden = false;
  els.resultBody.replaceChildren();
  appendDtDd(els.resultBody, "Category", pred.category);
  appendDtDd(els.resultBody, "Subcategory", pred.subcategory);
  appendDtDd(els.resultBody, "Risk", riskPill(pred.risk_level));
  appendDtDd(els.resultBody, "Building", pred.building_name);
  appendDtDd(els.resultBody, "Floor", pred.floor);
  appendDtDd(els.resultBody, "Address", pred.address);
  appendDtDd(els.resultBody, "Vendor", pred.dispatched_vendor_id);
  appendDtDd(els.resultBody, "911 dispatched", pred.dispatched_emergency_services ? "YES" : "no");
  appendDtDd(els.resultBody, "Needs human review", pred.needs_human_review ? "yes" : "no");
  appendDtDd(els.resultBody, "Needs clarification", pred.needs_clarification ? "yes" : "no");
  els.summary.textContent = pred.call_summary || "";
}

function showCallerProfile(profile) {
  if (!profile) {
    els.callerInfo.hidden = true;
    els.callerInfo.replaceChildren();
    return;
  }
  els.callerInfo.hidden = false;
  els.callerInfo.replaceChildren();
  const bldg = profile.primary_building_name || profile.primary_building_id || "—";
  const name = profile.caller_name || profile.name || "unnamed";
  const role = profile.contact_role || profile.role;

  els.callerInfo.appendChild(document.createTextNode("Known caller — "));
  const nameEl = document.createElement("b");
  nameEl.textContent = name;
  els.callerInfo.appendChild(nameEl);
  if (role) els.callerInfo.appendChild(document.createTextNode(`, ${role}`));
  els.callerInfo.appendChild(document.createTextNode(" · primary building "));
  const bldgEl = document.createElement("b");
  bldgEl.textContent = bldg;
  els.callerInfo.appendChild(bldgEl);
}

// ── ElevenLabs Scribe (STT) ─────────────────────────────────────────────────

async function fetchKeyterms() {
  if (state.keyterms.length) return state.keyterms;
  try {
    const res = await fetch("/api/keyterms");
    if (!res.ok) return [];
    const data = await res.json();
    state.keyterms = data.keyterms || [];
  } catch (_) {
    state.keyterms = [];
  }
  return state.keyterms;
}

async function mintScribeToken() {
  const res = await fetch("/api/scribe-token");
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error || `token endpoint ${res.status}`);
  }
  const { token } = await res.json();
  return token;
}

async function startScribe() {
  if (!Scribe) throw new Error("ElevenLabs client failed to load");
  const [token, allKeyterms] = await Promise.all([mintScribeToken(), fetchKeyterms()]);
  // Cap at 25 to keep the upgrade URL under ~3 KB; 100 keyterms turned the URL
  // into 5-10 KB and the server kept rejecting it with an immediate close.
  const keyterms = allKeyterms.slice(0, 25);

  console.log("[scribe] connecting", { keyterms: keyterms.length });
  state.scribeOpenedAt = Date.now();
  state.stopRequested = false; // fresh session — allow CLOSE-driven reconnects

  const conn = Scribe.connect({
    token,
    modelId: "scribe_v2_realtime",
    commitStrategy: "vad",
    vadSilenceThresholdSecs: 1.2,
    keyterms,
    noVerbatim: true,
    microphone: {
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  });

  // Subscribe to every diagnostic event so a real failure surfaces in console
  // instead of being silently swallowed by the generic ERROR handler.
  const noisyEvents = [
    "AUTH_ERROR", "UNACCEPTED_TERMS", "QUOTA_EXCEEDED", "RATE_LIMITED",
    "TRANSCRIBER_ERROR", "INPUT_ERROR", "RESOURCE_EXHAUSTED",
    "INSUFFICIENT_AUDIO_ACTIVITY", "SESSION_TIME_LIMIT_EXCEEDED",
    "COMMIT_THROTTLED", "QUEUE_OVERFLOW", "CHUNK_SIZE_EXCEEDED",
    "SESSION_STARTED",
  ];
  for (const name of noisyEvents) {
    const ev = RealtimeEvents[name];
    if (ev !== undefined) {
      conn.on(ev, (data) => console.log(`[scribe] ${name}`, data));
    }
  }

  conn.on(RealtimeEvents.OPEN, () => {
    console.log("[scribe] OPEN");
    // NOTE: do NOT reset consecutiveQuickCloses here. With invalid options the
    // server accepts the WS upgrade (firing OPEN) then closes 100-200ms later;
    // a reset on OPEN would clear the counter on every cycle and the guard
    // would never trip. The CLOSE handler resets it only when the connection
    // survived past 3s — a true healthy-session signal.
    setStatus("listening", "listening");
  });
  conn.on(RealtimeEvents.PARTIAL_TRANSCRIPT, (data) => {
    // Barge-in: caller speaking cancels in-flight TTS so the agent stops mid-word.
    if (state.currentAudio && !state.currentAudio.paused) {
      try { state.currentAudio.pause(); } catch (_) {}
      state.currentAudio = null;
    }
    state.pendingInterim = data.text || "";
    renderTurns();
  });
  conn.on(RealtimeEvents.COMMITTED_TRANSCRIPT, (data) => {
    const text = (data.text || "").trim();
    if (!text) return;
    pushTurn("caller", text);
    if (!state.busy) classifyAndRespond();
  });
  conn.on(RealtimeEvents.ERROR, (err) => {
    console.warn("[scribe] ERROR", err);
    setStatus("stt error", "error");
  });
  conn.on(RealtimeEvents.CLOSE, (info) => {
    const liveMs = Date.now() - (state.scribeOpenedAt || Date.now());
    console.log(`[scribe] CLOSE after ${liveMs}ms`, info);
    // Intentional stop (Reset / Stop listening / end-of-call) — bail without
    // reconnecting. Without this guard the auto-reconnect would re-open the
    // mic the moment the user tries to stop.
    if (state.stopRequested || !state.listening || state.callClosed) return;

    // Reconnect loop guard: if the socket dies within 3s, count it as a quick
    // failure. After 3 in a row, give up — something is broken (likely mic
    // permission, server-side rejection, or a config mismatch) and reconnecting
    // forever just makes the UI flap.
    if (liveMs < 3000) {
      state.consecutiveQuickCloses = (state.consecutiveQuickCloses || 0) + 1;
    } else {
      state.consecutiveQuickCloses = 0;
    }
    if (state.consecutiveQuickCloses >= 3) {
      console.error("[scribe] giving up — 3 consecutive quick closes. " +
        "Open devtools network tab + check the [scribe] log lines above.");
      setStatus("stt offline", "error");
      setMicState(false);
      state.scribe = null;
      return;
    }

    setStatus("reconnecting", "thinking");
    setTimeout(() => state.listening && reconnectScribe(), 800);
  });

  return conn;
}

async function reconnectScribe() {
  try {
    await stopScribe(false);
    state.scribe = await startScribe();
  } catch (e) {
    console.error("reconnect failed", e);
    setStatus("stt offline", "error");
    setMicState(false);
  }
}

async function stopScribe(updateUi = true) {
  if (!state.scribe) return;
  const conn = state.scribe;
  state.scribe = null;
  // Tell the CLOSE handler this was a deliberate stop so it doesn't reconnect.
  // Must be set BEFORE close() because close() fires CLOSE synchronously via
  // the WebSocket onclose handler.
  if (updateUi) {
    state.stopRequested = true;
    setMicState(false);
    setStatus("idle", "idle");
  }
  // close() is the SDK's real teardown — it calls _audioCleanup() to stop the
  // microphone stream and closes the WebSocket. disconnect() is a no-op
  // callback trigger that leaves both the mic and socket open.
  try { conn.close(); } catch (_) {}
}

// The Scribe SDK doesn't expose a public mute API. The media track lives at
// _mediaStreamTrack (set in Scribe.connect — see the IIFE bundle). Toggling
// track.enabled keeps the track alive (no permission re-prompt, no audio
// pipeline reset) but stops feeding samples to consumers — the standard
// WebRTC mute pattern. Without this, Scribe transcribes the agent's own
// TTS playback as a caller turn.
function _scribeMicTrack() {
  return state.scribe && state.scribe._mediaStreamTrack;
}

function pauseScribe() {
  const track = _scribeMicTrack();
  if (track) track.enabled = false;
}

function resumeScribe() {
  const track = _scribeMicTrack();
  if (track) track.enabled = true;
}

// ── ElevenLabs streaming TTS ────────────────────────────────────────────────

async function speak(text, onDone) {
  if (!text) { onDone && onDone(); return; }
  if (!state.ttsOn) { onDone && onDone(); return; }

  // Cancel any in-flight TTS first.
  if (state.currentAudio) {
    try { state.currentAudio.pause(); } catch (_) {}
    state.currentAudio = null;
  }

  setStatus("speaking", "speaking");
  await pauseScribe();

  let audio;
  try {
    const res = await fetch("/api/tts/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: res.statusText }));
      throw new Error(err.error || `tts ${res.status}`);
    }
    const blob = await res.blob();
    audio = new Audio(URL.createObjectURL(blob));
    state.currentAudio = audio;

    audio.onended = async () => {
      state.currentAudio = null;
      setStatus("idle", "idle");
      await resumeScribe();
      onDone && onDone();
    };
    audio.onerror = async () => {
      state.currentAudio = null;
      setStatus("idle", "idle");
      await resumeScribe();
      onDone && onDone();
    };
    audio.play().catch(async (e) => {
      console.warn("audio play blocked", e);
      state.currentAudio = null;
      setStatus("idle", "idle");
      await resumeScribe();
      onDone && onDone();
    });
  } catch (e) {
    console.error("tts failed", e);
    setStatus("tts error", "error");
    await resumeScribe();
    onDone && onDone();
  }
}

// ── Backend round-trip ──────────────────────────────────────────────────────

async function classifyAndRespond() {
  setBusy(true);
  setStatus("thinking", "thinking");

  const phone = els.phone.value.trim() || null;
  let pred;
  try {
    const resp = await fetch("/api/classify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ turns: state.turns, caller_phone: phone }),
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

  const needsMore = pred.needs_clarification && pred.clarification_question;

  if (needsMore) {
    const question = pred.clarification_question;
    pushTurn("agent", question);
    setBusy(false);
    speak(question);
  } else {
    state.callClosed = true;
    const dispatched = pred.dispatched_vendor_id
      ? ` Dispatched ${pred.dispatched_vendor_id}.`
      : pred.needs_human_review
      ? " Routing this to a human dispatcher."
      : "";
    const spoken = (pred.call_summary || "Got it.") + dispatched;
    pushTurn("agent", spoken);
    setBusy(false);
    speak(spoken, async () => {
      await stopScribe();
      setStatus("call closed", "idle");
    });
  }
}

// ── Wiring ──────────────────────────────────────────────────────────────────

async function startListening() {
  if (!Scribe) {
    setStatus("voice unavailable", "error");
    alert("ElevenLabs client failed to load. Check network or use the text input.");
    return;
  }

  // Open with an agent greeting so the conversation has an anchor turn.
  if (state.turns.length === 0) {
    const greeting = "CBRE maintenance, what's going on?";
    pushTurn("agent", greeting);
    setMicState(true);
    try {
      state.scribe = await startScribe();
    } catch (e) {
      console.error("scribe start failed", e);
      setStatus("voice unavailable", "error");
      setMicState(false);
      return;
    }
    speak(greeting);
    return;
  }

  setMicState(true);
  try {
    state.scribe = await startScribe();
  } catch (e) {
    console.error("scribe start failed", e);
    setStatus("voice unavailable", "error");
    setMicState(false);
  }
}

els.micBtn.addEventListener("click", async () => {
  if (state.callClosed) resetAll();
  if (state.listening) {
    await stopScribe();
  } else {
    await startListening();
  }
});

els.resetBtn.addEventListener("click", () => resetAll());

els.ttsToggle.addEventListener("click", () => {
  state.ttsOn = !state.ttsOn;
  els.ttsToggle.setAttribute("aria-pressed", state.ttsOn ? "true" : "false");
  els.ttsToggle.textContent = `TTS: ${state.ttsOn ? "on" : "off"}`;
  if (!state.ttsOn && state.currentAudio) {
    try { state.currentAudio.pause(); } catch (_) {}
    state.currentAudio = null;
  }
});

// Text composer — typed messages share the same classify pipeline as voice.
els.composer.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = els.textInput.value.trim();
  if (!text || state.busy) return;

  if (state.callClosed) resetAll();
  if (state.listening) await stopScribe();

  if (state.turns.length === 0) pushTurn("agent", "CBRE maintenance, what's going on?");

  pushTurn("caller", text);
  els.textInput.value = "";
  classifyAndRespond();
});

async function resetAll() {
  await stopScribe();
  if (state.currentAudio) {
    try { state.currentAudio.pause(); } catch (_) {}
    state.currentAudio = null;
  }
  state.turns = [];
  state.pendingInterim = "";
  state.callClosed = false;
  state.busy = false;
  setBusy(false);
  renderTurns();
  els.result.hidden = true;
  els.summary.textContent = "";
  els.resultBody.replaceChildren();
  els.textInput.value = "";
  showCallerProfile(null);
  setStatus("idle", "idle");
}

// First-run feature check.
if (!Scribe) {
  els.micBtn.disabled = true;
  els.micLabel.textContent = "Voice unavailable";
  els.hint.textContent =
    "ElevenLabs client failed to load — you can still type below.";
}
