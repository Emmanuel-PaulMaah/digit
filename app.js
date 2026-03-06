import {
  HandLandmarker,
  FilesetResolver,
  DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

// =====================
// DOM
// =====================
const startBtn = document.getElementById("startBtn");
const resetBtn = document.getElementById("resetBtn");
const statusEl = document.getElementById("status");

const video = document.getElementById("webcam");
const stage = document.getElementById("stage");
const desktop = document.getElementById("desktop");
const overlay = document.getElementById("overlay");
const logs = document.getElementById("logs");

const cursorEl = document.getElementById("cursor");
const viewerEl = document.getElementById("viewer");
const viewerImg = document.getElementById("viewerImg");
const viewerTitle = document.getElementById("viewerTitle");
const closeViewerBtn = document.getElementById("closeViewerBtn");

const contextMenu = document.getElementById("contextMenu");
const scrollbox = document.getElementById("scrollbox");
const files = Array.from(document.querySelectorAll(".file"));

const hudResolution = document.getElementById("hudResolution");
const hudFps = document.getElementById("hudFps");
const hudBrightness = document.getElementById("hudBrightness");
const hudContrast = document.getElementById("hudContrast");
const hudSharpness = document.getElementById("hudSharpness");
const hudJitter = document.getElementById("hudJitter");
const hudLighting = document.getElementById("hudLighting");
const hudProfile = document.getElementById("hudProfile");

const overlayCtx = overlay.getContext("2d");
const drawingUtils = new DrawingUtils(overlayCtx);

// =====================
// Hidden analysis canvas
// =====================
const analysisCanvas = document.createElement("canvas");
analysisCanvas.width = 160;
analysisCanvas.height = 120;
const analysisCtx = analysisCanvas.getContext("2d", { willReadFrequently: true });

// =====================
// Logging / status
// =====================
function log(msg) {
  const ts = new Date().toISOString().split("T")[1].split(".")[0];
  logs.value = `[${ts}] ${msg}\n` + logs.value;
}

function setStatus(msg) {
  statusEl.textContent = msg;
  log(msg);
}

// =====================
// Utils
// =====================
function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v));
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function easeOutCubic(t) {
  return 1 - Math.pow(1 - t, 3);
}

function distance(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.hypot(dx, dy);
}

function avg(nums) {
  if (!nums.length) return 0;
  return nums.reduce((s, n) => s + n, 0) / nums.length;
}

function dispatchMouse(type, clientX, clientY, button = 0, target = null, extra = {}) {
  const el = target || document.elementFromPoint(clientX, clientY);
  if (!el) return null;

  const evt = new MouseEvent(type, {
    bubbles: true,
    cancelable: true,
    clientX,
    clientY,
    button,
    buttons: button === 2 ? 2 : button === 0 ? 1 : 0,
    ...extra
  });

  el.dispatchEvent(evt);
  return el;
}

function dispatchWheel(target, clientX, clientY, deltaY) {
  if (!target) return;
  const evt = new WheelEvent("wheel", {
    bubbles: true,
    cancelable: true,
    clientX,
    clientY,
    deltaY
  });
  target.dispatchEvent(evt);
}

// =====================
// One Euro filter
// =====================
class OneEuroFilter {
  constructor(freq = 60, minCutoff = 1.0, beta = 0.03, dCutoff = 1.0) {
    this.freq = freq;
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this.xPrev = null;
    this.dxPrev = 0;
    this.tPrev = null;
  }

  alpha(cutoff) {
    const te = 1 / this.freq;
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / te);
  }

  lowpass(prev, curr, a) {
    return prev + a * (curr - prev);
  }

  filter(x, tMs) {
    if (this.tPrev == null) {
      this.tPrev = tMs;
      this.xPrev = x;
      this.dxPrev = 0;
      return x;
    }

    const dt = Math.max(1e-6, (tMs - this.tPrev) / 1000);
    this.freq = 1 / dt;

    const dx = (x - this.xPrev) / dt;
    const ad = this.alpha(this.dCutoff);
    const dxHat = this.lowpass(this.dxPrev, dx, ad);

    const cutoff = this.minCutoff + this.beta * Math.abs(dxHat);
    const a = this.alpha(cutoff);
    const xHat = this.lowpass(this.xPrev, x, a);

    this.tPrev = tMs;
    this.xPrev = xHat;
    this.dxPrev = dxHat;

    return xHat;
  }

  reset() {
    this.xPrev = null;
    this.dxPrev = 0;
    this.tPrev = null;
  }
}

const xFilter = new OneEuroFilter();
const yFilter = new OneEuroFilter();
const scaleFilter = new OneEuroFilter(60, 0.9, 0.01, 1.0);

// =====================
// State
// =====================
let handLandmarker = null;
let running = false;
let lastVideoTime = -1;
let lastSeenMs = 0;
let cameraInfoLogged = false;

let hoverCandidate = null;
let hoverCandidateSince = 0;
let hoveredEl = null;

let lastCursorX = 0;
let lastCursorY = 0;
let lastT = 0;

let leftPinchActive = false;
let rightPinchActive = false;
let scrollActive = false;
let thumbsUpActive = false;

let wasLeftPinch = false;
let wasRightPinch = false;
let wasScroll = false;
let wasThumbsUp = false;

let isMouseDown = false;
let dragTarget = null;
let dragOffsetX = 0;
let dragOffsetY = 0;
let dragMoved = false;
let pressTarget = null;
let pressStartX = 0;
let pressStartY = 0;

let scrollTarget = null;
let scrollAnchorY = 0;
let scrollAccumulator = 0;
let contextTarget = null;

let leftOnCount = 0;
let leftOffCount = 0;
let rightOnCount = 0;
let rightOffCount = 0;
let scrollOnCount = 0;
let scrollOffCount = 0;
let thumbsOnCount = 0;
let thumbsOffCount = 0;

// =====================
// Runtime diagnostics
// =====================
const diagnostics = {
  actualWidth: 0,
  actualHeight: 0,
  fps: 0,
  brightness: 0,
  contrast: 0,
  sharpness: 0,
  exposureBlackPct: 0,
  exposureWhitePct: 0,
  landmarkJitter: 0,
  lighting: "unknown",
  profile: "medium"
};

const frameTimes = [];
const jitterSamples = [];

let lastDiagnosticsAt = 0;
const DIAGNOSTICS_INTERVAL_MS = 300;

let lastLandmarksForJitter = null;

// =====================
// Adaptive profiles
// =====================
const activeRegionBase = {
  left: 0.26,
  right: 0.74,
  top: 0.20,
  bottom: 0.82
};

const adaptive = {
  profileName: "medium",
  xMinCutoff: 1.7,
  yMinCutoff: 1.7,
  xBeta: 0.08,
  yBeta: 0.08,
  gainX: 1.25,
  gainY: 1.22,
  leftPinchOn: 0.30,
  leftPinchOff: 0.56,
  rightPinchOn: 0.34,
  rightPinchOff: 0.58,
  onFrames: 2,
  offFrames: 3,
  scrollOnFrames: 3,
  scrollOffFrames: 3,
  thumbsOnFrames: 4,
  thumbsOffFrames: 4,
  hoverDwellMs: 45,
  handLostGraceMs: 300,
  scrollMultiplier: 1.7
};

function applyAdaptiveProfile(name) {
  adaptive.profileName = name;

  if (name === "good") {
    adaptive.xMinCutoff = 2.8;
    adaptive.yMinCutoff = 2.8;
    adaptive.xBeta = 0.12;
    adaptive.yBeta = 0.12;
    adaptive.gainX = 1.48;
    adaptive.gainY = 1.42;
    adaptive.leftPinchOn = 0.29;
    adaptive.leftPinchOff = 0.54;
    adaptive.rightPinchOn = 0.33;
    adaptive.rightPinchOff = 0.56;
    adaptive.onFrames = 1;
    adaptive.offFrames = 2;
    adaptive.scrollOnFrames = 2;
    adaptive.scrollOffFrames = 2;
    adaptive.thumbsOnFrames = 3;
    adaptive.thumbsOffFrames = 3;
    adaptive.hoverDwellMs = 28;
    adaptive.handLostGraceMs = 220;
    adaptive.scrollMultiplier = 2.1;
  } else if (name === "medium") {
    adaptive.xMinCutoff = 1.9;
    adaptive.yMinCutoff = 1.9;
    adaptive.xBeta = 0.09;
    adaptive.yBeta = 0.09;
    adaptive.gainX = 1.34;
    adaptive.gainY = 1.30;
    adaptive.leftPinchOn = 0.30;
    adaptive.leftPinchOff = 0.56;
    adaptive.rightPinchOn = 0.34;
    adaptive.rightPinchOff = 0.58;
    adaptive.onFrames = 2;
    adaptive.offFrames = 3;
    adaptive.scrollOnFrames = 3;
    adaptive.scrollOffFrames = 3;
    adaptive.thumbsOnFrames = 4;
    adaptive.thumbsOffFrames = 4;
    adaptive.hoverDwellMs = 42;
    adaptive.handLostGraceMs = 320;
    adaptive.scrollMultiplier = 1.7;
  } else {
    adaptive.xMinCutoff = 1.1;
    adaptive.yMinCutoff = 1.1;
    adaptive.xBeta = 0.05;
    adaptive.yBeta = 0.05;
    adaptive.gainX = 1.16;
    adaptive.gainY = 1.14;
    adaptive.leftPinchOn = 0.32;
    adaptive.leftPinchOff = 0.61;
    adaptive.rightPinchOn = 0.36;
    adaptive.rightPinchOff = 0.62;
    adaptive.onFrames = 3;
    adaptive.offFrames = 4;
    adaptive.scrollOnFrames = 4;
    adaptive.scrollOffFrames = 4;
    adaptive.thumbsOnFrames = 5;
    adaptive.thumbsOffFrames = 5;
    adaptive.hoverDwellMs = 65;
    adaptive.handLostGraceMs = 460;
    adaptive.scrollMultiplier = 1.2;
  }

  xFilter.minCutoff = adaptive.xMinCutoff;
  yFilter.minCutoff = adaptive.yMinCutoff;
  xFilter.beta = adaptive.xBeta;
  yFilter.beta = adaptive.yBeta;

  diagnostics.profile = name;
}

applyAdaptiveProfile("medium");

// =====================
// Mapping
// =====================
function resizeOverlay() {
  const rect = stage.getBoundingClientRect();
  overlay.width = Math.round(rect.width);
  overlay.height = Math.round(rect.height);
}

function getAdaptiveActiveRegion() {
  const cx = 0.5;
  const cy = 0.5;

  const widthBase = activeRegionBase.right - activeRegionBase.left;
  const heightBase = activeRegionBase.bottom - activeRegionBase.top;

  const width = clamp(widthBase / adaptive.gainX, 0.28, 0.52);
  const height = clamp(heightBase / adaptive.gainY, 0.28, 0.56);

  return {
    left: cx - width / 2,
    right: cx + width / 2,
    top: cy - height / 2,
    bottom: cy + height / 2
  };
}

function mapNormToStage(landmark) {
  const rect = stage.getBoundingClientRect();
  const region = getAdaptiveActiveRegion();

  const mx = 1 - landmark.x;
  const my = landmark.y;

  const nx = clamp((mx - region.left) / (region.right - region.left), 0, 1);
  const ny = clamp((my - region.top) / (region.bottom - region.top), 0, 1);

  return {
    x: nx * rect.width,
    y: ny * rect.height
  };
}

function stageToClient(x, y) {
  const rect = stage.getBoundingClientRect();
  return {
    clientX: rect.left + x,
    clientY: rect.top + y
  };
}

// =====================
// Hand geometry
// =====================
function getHandScaleNorm(landmarks) {
  const wrist = landmarks[0];
  const indexMcp = landmarks[5];
  return Math.max(distance(wrist, indexMcp), 0.0001);
}

function getHandScalePx(landmarks) {
  const rect = stage.getBoundingClientRect();
  const minDim = Math.min(rect.width, rect.height);
  return getHandScaleNorm(landmarks) * minDim;
}

function handScaleToT(handScalePx) {
  const FAR_PX = 40;
  const NEAR_PX = 260;
  const lin = clamp((handScalePx - FAR_PX) / (NEAR_PX - FAR_PX), 0, 1);
  return easeOutCubic(lin);
}

function updateCursorSize(t) {
  const d = lerp(16, 34, t);
  cursorEl.style.width = `${d}px`;
  cursorEl.style.height = `${d}px`;
}

function updateCursorPose(x, y) {
  lastCursorX = x;
  lastCursorY = y;
  cursorEl.style.left = `${x}px`;
  cursorEl.style.top = `${y}px`;
}

// =====================
// Overlay drawing
// =====================
function clearOverlay() {
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
}

function drawPoint(x, y, r, fill, stroke) {
  overlayCtx.beginPath();
  overlayCtx.arc(x, y, r, 0, Math.PI * 2);
  overlayCtx.fillStyle = fill;
  overlayCtx.fill();
  overlayCtx.lineWidth = 2;
  overlayCtx.strokeStyle = stroke;
  overlayCtx.stroke();
}

function drawLandmarks(landmarks, t) {
  clearOverlay();

  const mirrored = landmarks.map((p) => ({ ...p, x: 1 - p.x }));
  drawingUtils.drawConnectors(mirrored, HandLandmarker.HAND_CONNECTIONS, {
    color: "rgba(255,255,255,0.35)",
    lineWidth: 2
  });
  drawingUtils.drawLandmarks(mirrored, {
    color: "rgba(255,255,255,0.92)",
    radius: 3
  });

  const index = mapNormToStage(landmarks[8]);
  const thumb = mapNormToStage(landmarks[4]);
  const middle = mapNormToStage(landmarks[12]);

  drawPoint(index.x, index.y, lerp(8, 20, t), "rgba(122,184,255,0.28)", "rgba(122,184,255,0.95)");
  drawPoint(thumb.x, thumb.y, lerp(5, 14, t), "rgba(255,124,67,0.22)", "rgba(255,124,67,0.95)");
  drawPoint(middle.x, middle.y, lerp(5, 12, t), "rgba(255,92,122,0.22)", "rgba(255,92,122,0.95)");
}

// =====================
// Diagnostics
// =====================
function updateFps(nowMs) {
  frameTimes.push(nowMs);
  while (frameTimes.length && nowMs - frameTimes[0] > 1000) {
    frameTimes.shift();
  }
  diagnostics.fps = frameTimes.length;
}

function computeVideoMetrics() {
  const vw = video.videoWidth || 0;
  const vh = video.videoHeight || 0;
  diagnostics.actualWidth = vw;
  diagnostics.actualHeight = vh;

  if (!vw || !vh) return;

  analysisCtx.drawImage(video, 0, 0, analysisCanvas.width, analysisCanvas.height);
  const img = analysisCtx.getImageData(0, 0, analysisCanvas.width, analysisCanvas.height);
  const data = img.data;

  let sum = 0;
  let sumSq = 0;
  let blacks = 0;
  let whites = 0;

  const gray = new Float32Array(analysisCanvas.width * analysisCanvas.height);

  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    const y = 0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2];
    gray[p] = y;
    sum += y;
    sumSq += y * y;
    if (y < 20) blacks++;
    if (y > 235) whites++;
  }

  const n = gray.length;
  const mean = sum / n;
  const variance = Math.max(0, sumSq / n - mean * mean);
  const stdev = Math.sqrt(variance);

  let edgeSum = 0;
  const w = analysisCanvas.width;
  const h = analysisCanvas.height;

  for (let y = 1; y < h - 1; y += 2) {
    for (let x = 1; x < w - 1; x += 2) {
      const idx = y * w + x;
      const gx = gray[idx + 1] - gray[idx - 1];
      const gy = gray[idx + w] - gray[idx - w];
      edgeSum += Math.abs(gx) + Math.abs(gy);
    }
  }

  const sharpness = edgeSum / ((w * h) / 4);

  diagnostics.brightness = mean;
  diagnostics.contrast = stdev;
  diagnostics.sharpness = sharpness;
  diagnostics.exposureBlackPct = blacks / n;
  diagnostics.exposureWhitePct = whites / n;

  const brightOk = mean >= 65 && mean <= 195;
  const contrastOk = stdev >= 28;
  const sharpOk = sharpness >= 22;
  const clippingBad = diagnostics.exposureBlackPct > 0.35 || diagnostics.exposureWhitePct > 0.20;

  if (brightOk && contrastOk && sharpOk && !clippingBad) {
    diagnostics.lighting = "good";
  } else if ((mean >= 45 && mean <= 220) && stdev >= 18 && sharpness >= 14) {
    diagnostics.lighting = "fair";
  } else {
    diagnostics.lighting = "poor";
  }
}

function computeLandmarkJitter(landmarks) {
  if (!lastLandmarksForJitter) {
    lastLandmarksForJitter = landmarks.map((p) => ({ x: p.x, y: p.y }));
    return;
  }

  let total = 0;
  for (let i = 0; i < landmarks.length; i++) {
    total += distance(landmarks[i], lastLandmarksForJitter[i]);
  }
  const meanDelta = total / landmarks.length;
  jitterSamples.push(meanDelta);

  while (jitterSamples.length > 24) jitterSamples.shift();
  diagnostics.landmarkJitter = avg(jitterSamples);

  lastLandmarksForJitter = landmarks.map((p) => ({ x: p.x, y: p.y }));
}

function chooseAdaptiveProfile() {
  const fpsGood = diagnostics.fps >= 24;
  const fpsPoor = diagnostics.fps < 16;

  const resGood = diagnostics.actualWidth >= 1280;
  const resPoor = diagnostics.actualWidth > 0 && diagnostics.actualWidth < 960;

  const jitterGood = diagnostics.landmarkJitter > 0 && diagnostics.landmarkJitter < 0.0065;
  const jitterPoor = diagnostics.landmarkJitter >= 0.014;

  const lightingGood = diagnostics.lighting === "good";
  const lightingPoor = diagnostics.lighting === "poor";

  let next = "medium";

  if ((lightingGood || diagnostics.lighting === "fair") && fpsGood && !jitterPoor && !resPoor && jitterGood) {
    next = "good";
  }

  if (lightingPoor || fpsPoor || jitterPoor || resPoor) {
    next = "poor";
  }

  if (next !== adaptive.profileName) {
    applyAdaptiveProfile(next);
    log(`PROFILE -> ${next}`);
  }
}

function updateHud() {
  hudResolution.textContent = diagnostics.actualWidth
    ? `${diagnostics.actualWidth}×${diagnostics.actualHeight}`
    : "—";
  hudFps.textContent = diagnostics.fps ? `${diagnostics.fps}` : "—";
  hudBrightness.textContent = `${diagnostics.brightness.toFixed(0)}`;
  hudContrast.textContent = `${diagnostics.contrast.toFixed(0)}`;
  hudSharpness.textContent = `${diagnostics.sharpness.toFixed(0)}`;
  hudJitter.textContent = `${diagnostics.landmarkJitter.toFixed(4)}`;
  hudLighting.textContent = diagnostics.lighting;
  hudProfile.textContent = diagnostics.profile;
}

// =====================
// Hover / target finding
// =====================
function getInteractiveTarget(clientX, clientY) {
  const el = document.elementFromPoint(clientX, clientY);
  if (!el) return null;

  if (viewerEl.classList.contains("visible")) {
    if (el === closeViewerBtn || closeViewerBtn.contains(el)) return closeViewerBtn;
  }

  const file = el.closest?.(".file");
  if (file) return file;

  const contextAction = el.closest?.(".context-menu button");
  if (contextAction) return contextAction;

  if (scrollbox.contains(el)) return scrollbox;

  return el;
}

function applyHovered(next) {
  if (hoveredEl === next) return;

  if (hoveredEl?.classList?.contains("hovered")) {
    hoveredEl.classList.remove("hovered");
  }

  hoveredEl = next;

  if (hoveredEl?.classList) {
    hoveredEl.classList.add("hovered");
  }
}

function stabilizeHover(next, nowMs) {
  if (next !== hoverCandidate) {
    hoverCandidate = next;
    hoverCandidateSince = nowMs;
    return;
  }

  if (hoveredEl !== hoverCandidate && nowMs - hoverCandidateSince >= adaptive.hoverDwellMs) {
    applyHovered(hoverCandidate);
  }
}

// =====================
// Viewer / context menu
// =====================
function openViewer(fileEl) {
  const img = fileEl.querySelector("img");
  const title = fileEl.querySelector(".title")?.textContent || "Preview";
  if (!img) return;

  viewerImg.src = img.src;
  viewerTitle.textContent = title;
  viewerEl.classList.add("visible");
  viewerEl.setAttribute("aria-hidden", "false");
  hideContextMenu();

  log(`OPEN ${fileEl.dataset.id}`);
}

function closeViewer() {
  viewerEl.classList.remove("visible");
  viewerEl.setAttribute("aria-hidden", "true");
  log("CLOSE viewer");
}

function showContextMenu(x, y, targetEl) {
  contextTarget = targetEl;
  hideContextMenu(false);

  const rect = desktop.getBoundingClientRect();
  const menuWidth = 190;
  const menuHeight = 130;

  const left = clamp(x - rect.left, 12, rect.width - menuWidth - 12);
  const top = clamp(y - rect.top, 12, rect.height - menuHeight - 12);

  contextMenu.style.left = `${left}px`;
  contextMenu.style.top = `${top}px`;
  contextMenu.classList.add("visible");
  contextMenu.setAttribute("aria-hidden", "false");

  log(`CONTEXT menu on ${targetEl?.dataset?.id || targetEl?.id || targetEl?.tagName}`);
}

function hideContextMenu(clearTarget = true) {
  contextMenu.classList.remove("visible");
  contextMenu.setAttribute("aria-hidden", "true");
  if (clearTarget) contextTarget = null;
}

// =====================
// DOM behavior
// =====================
files.forEach((file) => {
  file.addEventListener("click", () => openViewer(file));
  file.addEventListener("contextmenu", (e) => {
    e.preventDefault();
    showContextMenu(e.clientX, e.clientY, file);
  });
});

contextMenu.addEventListener("click", (e) => {
  const btn = e.target.closest("button");
  if (!btn) return;

  const action = btn.dataset.action;
  const id = contextTarget?.dataset?.id || "unknown";

  log(`MENU ${action} on ${id}`);

  if (action === "open" && contextTarget?.classList.contains("file")) {
    openViewer(contextTarget);
  }

  if (action === "rename") {
    alert(`Rename action for file ${id}`);
  }

  if (action === "details") {
    alert(`Details action for file ${id}`);
  }

  hideContextMenu();
});

desktop.addEventListener("click", (e) => {
  if (!contextMenu.contains(e.target)) {
    hideContextMenu();
  }
});

closeViewerBtn.addEventListener("click", closeViewer);

// =====================
// Gesture detectors
// =====================
function debouncedBinary(raw, current, onRef, offRef, onFrames, offFrames) {
  if (!current) {
    if (raw) onRef.count++;
    else onRef.count = 0;

    if (onRef.count >= onFrames) {
      onRef.count = 0;
      offRef.count = 0;
      return true;
    }
    return false;
  }

  if (!raw) offRef.count++;
  else offRef.count = 0;

  if (offRef.count >= offFrames) {
    onRef.count = 0;
    offRef.count = 0;
    return false;
  }

  return true;
}

function updateLeftPinch(landmarks) {
  const hs = getHandScaleNorm(landmarks);
  const d = distance(landmarks[4], landmarks[8]) / hs;
  const raw = leftPinchActive ? d < adaptive.leftPinchOff : d < adaptive.leftPinchOn;

  leftPinchActive = debouncedBinary(
    raw,
    leftPinchActive,
    { get count() { return leftOnCount; }, set count(v) { leftOnCount = v; } },
    { get count() { return leftOffCount; }, set count(v) { leftOffCount = v; } },
    adaptive.onFrames,
    adaptive.offFrames
  );

  return leftPinchActive;
}

function updateRightPinch(landmarks) {
  const hs = getHandScaleNorm(landmarks);
  const d = distance(landmarks[4], landmarks[12]) / hs;
  const raw = rightPinchActive ? d < adaptive.rightPinchOff : d < adaptive.rightPinchOn;

  rightPinchActive = debouncedBinary(
    raw,
    rightPinchActive,
    { get count() { return rightOnCount; }, set count(v) { rightOnCount = v; } },
    { get count() { return rightOffCount; }, set count(v) { rightOffCount = v; } },
    adaptive.onFrames,
    adaptive.offFrames
  );

  return rightPinchActive;
}

function isFingerExtended(tip, pip) {
  return tip.y < pip.y;
}

function updateScrollGesture(landmarks) {
  const indexUp = isFingerExtended(landmarks[8], landmarks[6]);
  const middleUp = isFingerExtended(landmarks[12], landmarks[10]);
  const ringDown = landmarks[16].y > landmarks[14].y;
  const pinkyDown = landmarks[20].y > landmarks[18].y;

  const hs = getHandScaleNorm(landmarks);
  const thumbIndexDist = distance(landmarks[4], landmarks[8]) / hs;
  const thumbMiddleDist = distance(landmarks[4], landmarks[12]) / hs;

  const raw = indexUp && middleUp && ringDown && pinkyDown && thumbIndexDist > 0.55 && thumbMiddleDist > 0.55;

  scrollActive = debouncedBinary(
    raw,
    scrollActive,
    { get count() { return scrollOnCount; }, set count(v) { scrollOnCount = v; } },
    { get count() { return scrollOffCount; }, set count(v) { scrollOffCount = v; } },
    adaptive.scrollOnFrames,
    adaptive.scrollOffFrames
  );

  return scrollActive;
}

function updateThumbsUp(landmarks) {
  const wrist = landmarks[0];
  const thumbTip = landmarks[4];
  const thumbIp = landmarks[3];
  const thumbMcp = landmarks[2];

  const thumbUp =
    thumbTip.y < thumbIp.y &&
    thumbIp.y < thumbMcp.y &&
    thumbTip.y < wrist.y;

  const othersFolded =
    landmarks[8].y > landmarks[6].y &&
    landmarks[12].y > landmarks[10].y &&
    landmarks[16].y > landmarks[14].y &&
    landmarks[20].y > landmarks[18].y;

  const raw = thumbUp && othersFolded;

  thumbsUpActive = debouncedBinary(
    raw,
    thumbsUpActive,
    { get count() { return thumbsOnCount; }, set count(v) { thumbsOnCount = v; } },
    { get count() { return thumbsOffCount; }, set count(v) { thumbsOffCount = v; } },
    adaptive.thumbsOnFrames,
    adaptive.thumbsOffFrames
  );

  return thumbsUpActive;
}

// =====================
// Dragging
// =====================
function beginLeftPress(target, clientX, clientY) {
  if (!target) return;

  pressTarget = target;
  pressStartX = clientX;
  pressStartY = clientY;
  dragMoved = false;
  isMouseDown = true;

  dispatchMouse("mousedown", clientX, clientY, 0, target);
  cursorEl.classList.add("left-down");

  if (target.classList?.contains("file")) {
    const desktopRect = desktop.getBoundingClientRect();
    const rect = target.getBoundingClientRect();
    dragTarget = target;
    dragTarget.classList.add("dragging", "pressed");
    dragOffsetX = clientX - rect.left;
    dragOffsetY = clientY - rect.top;

    target.style.position = "absolute";
    target.style.left = `${rect.left - desktopRect.left}px`;
    target.style.top = `${rect.top - desktopRect.top}px`;
    target.style.width = `${rect.width}px`;
  } else {
    dragTarget = null;
  }
}

function updateDrag(clientX, clientY) {
  if (!dragTarget) return;

  const desktopRect = desktop.getBoundingClientRect();
  const rect = dragTarget.getBoundingClientRect();

  let left = clientX - desktopRect.left - dragOffsetX;
  let top = clientY - desktopRect.top - dragOffsetY;

  left = clamp(left, 0, desktopRect.width - rect.width);
  top = clamp(top, 0, desktopRect.height - rect.height);

  dragTarget.style.left = `${left}px`;
  dragTarget.style.top = `${top}px`;

  if (Math.abs(clientX - pressStartX) > 8 || Math.abs(clientY - pressStartY) > 8) {
    dragMoved = true;
  }
}

function endLeftPress(clientX, clientY) {
  if (!isMouseDown) return;

  const target = pressTarget || document.elementFromPoint(clientX, clientY);
  dispatchMouse("mouseup", clientX, clientY, 0, target);

  if (!dragMoved) {
    dispatchMouse("click", clientX, clientY, 0, target);
  }

  if (dragTarget) {
    dragTarget.classList.remove("dragging", "pressed");
  }

  cursorEl.classList.remove("left-down");
  isMouseDown = false;
  dragTarget = null;
  pressTarget = null;
  dragMoved = false;
}

// =====================
// Reset
// =====================
function resetGestureStates() {
  leftPinchActive = false;
  rightPinchActive = false;
  scrollActive = false;
  thumbsUpActive = false;

  wasLeftPinch = false;
  wasRightPinch = false;
  wasScroll = false;
  wasThumbsUp = false;

  leftOnCount = 0;
  leftOffCount = 0;
  rightOnCount = 0;
  rightOffCount = 0;
  scrollOnCount = 0;
  scrollOffCount = 0;
  thumbsOnCount = 0;
  thumbsOffCount = 0;

  xFilter.reset();
  yFilter.reset();
  scaleFilter.reset();

  cursorEl.classList.remove("left-down", "right-down", "scrolling");

  if (dragTarget) dragTarget.classList.remove("dragging", "pressed");
  if (isMouseDown) {
    const { clientX, clientY } = stageToClient(lastCursorX, lastCursorY);
    endLeftPress(clientX, clientY);
  }

  hoverCandidate = null;
  hoverCandidateSince = 0;
  applyHovered(null);
  lastLandmarksForJitter = null;
  jitterSamples.length = 0;
}

function hardReset() {
  resetGestureStates();
  hideContextMenu();
  clearOverlay();
  log("STATE RESET");
}

// =====================
// MediaPipe init/start
// =====================
async function initHandLandmarker() {
  setStatus("Loading MediaPipe hand model…");

  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
  );

  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task"
    },
    numHands: 1,
    runningMode: "VIDEO"
  });

  setStatus("Model loaded. Click Start Camera.");
}

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: 1280 },
      height: { ideal: 720 },
      facingMode: "user"
    }
  });

  video.srcObject = stream;

  await new Promise((resolve) => {
    video.onloadedmetadata = () => {
      video.play();
      resolve();
    };
  });

  resizeOverlay();
  window.addEventListener("resize", resizeOverlay);

  running = true;
  cursorEl.classList.add("visible");

  const track = stream.getVideoTracks()[0];
  if (track && !cameraInfoLogged) {
    const settings = track.getSettings?.() || {};
    log(`CAMERA settings requested/actual: ${JSON.stringify(settings)}`);
    cameraInfoLogged = true;
  }

  setStatus("Camera started. Adaptive hand mouse is live.");
  log("Ready.");

  requestAnimationFrame(loop);
}

// =====================
// Main loop
// =====================
let lastScaleLogAt = 0;

function loop(nowMs) {
  if (!running || !handLandmarker) return;

  const videoTime = video.currentTime;
  if (videoTime === lastVideoTime) {
    requestAnimationFrame(loop);
    return;
  }
  lastVideoTime = videoTime;

  updateFps(nowMs);

  if (nowMs - lastDiagnosticsAt > DIAGNOSTICS_INTERVAL_MS) {
    computeVideoMetrics();
    chooseAdaptiveProfile();
    updateHud();
    lastDiagnosticsAt = nowMs;
  }

  const results = handLandmarker.detectForVideo(video, nowMs);

  if (results?.landmarks?.length) {
    lastSeenMs = nowMs;

    const landmarks = results.landmarks[0];
    computeLandmarkJitter(landmarks);

    let { x, y } = mapNormToStage(landmarks[8]);
    x = xFilter.filter(x, nowMs);
    y = yFilter.filter(y, nowMs);

    updateCursorPose(x, y);

    const handScaleRaw = getHandScalePx(landmarks);
    const handScaleSmooth = scaleFilter.filter(handScaleRaw, nowMs);
    const t = handScaleToT(handScaleSmooth);
    lastT = t;
    updateCursorSize(t);

    drawLandmarks(landmarks, t);

    const { clientX, clientY } = stageToClient(x, y);
    const target = getInteractiveTarget(clientX, clientY);
    stabilizeHover(target, nowMs);

    const left = updateLeftPinch(landmarks);
    const right = updateRightPinch(landmarks);
    const scroll = updateScrollGesture(landmarks);
    const thumbs = updateThumbsUp(landmarks);

    if (thumbs && !wasThumbsUp) {
      hardReset();
      if (viewerEl.classList.contains("visible")) closeViewer();
      log("THUMBS-UP reset/release");
    }

    if (!thumbs) {
      if (right && !wasRightPinch && !left && !scroll) {
        cursorEl.classList.add("right-down");
        if (target?.classList?.contains("file")) {
          dispatchMouse("contextmenu", clientX, clientY, 2, target);
        } else if (target === closeViewerBtn) {
          closeViewer();
        } else {
          hideContextMenu();
        }
        setTimeout(() => cursorEl.classList.remove("right-down"), 120);
      }

      if (left && !wasLeftPinch && !right && !scroll) {
        beginLeftPress(target, clientX, clientY);
      }

      if (left && isMouseDown) {
        updateDrag(clientX, clientY);
      }

      if (!left && wasLeftPinch && isMouseDown) {
        endLeftPress(clientX, clientY);
      }

      if (scroll && !left && !right) {
        cursorEl.classList.add("scrolling");

        if (!wasScroll) {
          scrollTarget = target === scrollbox || scrollbox.contains(target) ? scrollbox : scrollbox;
          scrollAnchorY = y;
          scrollAccumulator = 0;
          log("SCROLL start");
        } else {
          const dy = y - scrollAnchorY;
          scrollAccumulator += dy * adaptive.scrollMultiplier;

          if (Math.abs(scrollAccumulator) >= 8) {
            const deltaY = scrollAccumulator;
            dispatchWheel(scrollTarget, clientX, clientY, deltaY);
            scrollTarget.scrollTop += deltaY;
            scrollAccumulator = 0;
            scrollAnchorY = y;
          }
        }
      } else {
        cursorEl.classList.remove("scrolling");
        if (wasScroll) {
          log("SCROLL end");
        }
      }
    }

    wasLeftPinch = left;
    wasRightPinch = right;
    wasScroll = scroll;
    wasThumbsUp = thumbs;

    if (nowMs - lastScaleLogAt > 1200) {
      lastScaleLogAt = nowMs;
      log(
        `profile=${adaptive.profileName} fps=${diagnostics.fps} light=${diagnostics.lighting} jitter=${diagnostics.landmarkJitter.toFixed(4)} rawScale=${handScaleRaw.toFixed(1)}`
      );
    }
  } else {
    const lostFor = nowMs - lastSeenMs;

    if (lostFor <= adaptive.handLostGraceMs) {
      updateCursorPose(lastCursorX, lastCursorY);
      updateCursorSize(lastT);
    } else {
      clearOverlay();
      applyHovered(null);

      if (isMouseDown) {
        const { clientX, clientY } = stageToClient(lastCursorX, lastCursorY);
        endLeftPress(clientX, clientY);
      }

      resetGestureStates();
    }
  }

  requestAnimationFrame(loop);
}

// =====================
// Wire up
// =====================
startBtn.addEventListener("click", async () => {
  if (!handLandmarker) {
    setStatus("Still loading model…");
    return;
  }

  if (!running) {
    try {
      await startCamera();
    } catch (err) {
      setStatus(`Error starting camera: ${err?.message || err}`);
    }
  }
});

resetBtn.addEventListener("click", hardReset);

if (!("mediaDevices" in navigator && "getUserMedia" in navigator.mediaDevices)) {
  setStatus("getUserMedia is not supported in this browser.");
} else {
  initHandLandmarker().catch((err) => {
    setStatus(`Failed to init MediaPipe: ${err?.message || err}`);
  });
}
