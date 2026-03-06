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

const overlayCtx = overlay.getContext("2d");
const drawingUtils = new DrawingUtils(overlayCtx);

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

const xFilter = new OneEuroFilter(60, 0.85, 0.03, 1.0);
const yFilter = new OneEuroFilter(60, 0.85, 0.03, 1.0);
const scaleFilter = new OneEuroFilter(60, 0.9, 0.01, 1.0);

// =====================
// State
// =====================
let handLandmarker = null;
let running = false;
let lastVideoTime = -1;
let lastSeenMs = 0;

const HAND_LOST_GRACE_MS = 300;
const HOVER_DWELL_MS = 50;

let hoverCandidate = null;
let hoverCandidateSince = 0;
let hoveredEl = null;

let lastCursorX = 0;
let lastCursorY = 0;
let lastT = 0;

// Press / drag / right click state
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

// context menu state
let contextTarget = null;

// =====================
// Gesture debounce state
// =====================
let leftOnCount = 0;
let leftOffCount = 0;

let rightOnCount = 0;
let rightOffCount = 0;

let scrollOnCount = 0;
let scrollOffCount = 0;

let thumbsOnCount = 0;
let thumbsOffCount = 0;

const LEFT_PINCH_ON = 0.30;
const LEFT_PINCH_OFF = 0.58;
const RIGHT_PINCH_ON = 0.34;
const RIGHT_PINCH_OFF = 0.60;

const ON_FRAMES = 2;
const OFF_FRAMES = 3;

const SCROLL_ON_FRAMES = 3;
const SCROLL_OFF_FRAMES = 3;

const THUMBS_ON_FRAMES = 4;
const THUMBS_OFF_FRAMES = 4;

// =====================
// Pointer mapping
// Small real movement => larger on-screen motion
// =====================
const ACTIVE_REGION = {
  left: 0.22,
  right: 0.78,
  top: 0.18,
  bottom: 0.84
};

function mapNormToStage(landmark) {
  const rect = stage.getBoundingClientRect();

  const mx = 1 - landmark.x; // mirror X
  const my = landmark.y;

  const nx = clamp((mx - ACTIVE_REGION.left) / (ACTIVE_REGION.right - ACTIVE_REGION.left), 0, 1);
  const ny = clamp((my - ACTIVE_REGION.top) / (ACTIVE_REGION.bottom - ACTIVE_REGION.top), 0, 1);

  const x = nx * rect.width;
  const y = ny * rect.height;

  return { x, y };
}

function resizeOverlay() {
  const rect = stage.getBoundingClientRect();
  overlay.width = Math.round(rect.width);
  overlay.height = Math.round(rect.height);
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

function stageToClient(x, y) {
  const rect = stage.getBoundingClientRect();
  return {
    clientX: rect.left + x,
    clientY: rect.top + y
  };
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

  if (hoveredEl !== hoverCandidate && nowMs - hoverCandidateSince >= HOVER_DWELL_MS) {
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
// DOM app behavior
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

function leftPinchRaw(landmarks) {
  const thumbTip = landmarks[4];
  const indexTip = landmarks[8];
  const hs = getHandScaleNorm(landmarks);
  return distance(thumbTip, indexTip) / hs < LEFT_PINCH_ON;
}

function rightPinchRaw(landmarks) {
  const thumbTip = landmarks[4];
  const middleTip = landmarks[12];
  const hs = getHandScaleNorm(landmarks);
  return distance(thumbTip, middleTip) / hs < RIGHT_PINCH_ON;
}

function updateLeftPinch(landmarks) {
  const thumbTip = landmarks[4];
  const indexTip = landmarks[8];
  const hs = getHandScaleNorm(landmarks);
  const d = distance(thumbTip, indexTip) / hs;

  const raw = leftPinchActive ? d < LEFT_PINCH_OFF : d < LEFT_PINCH_ON;

  leftPinchActive = debouncedBinary(
    raw,
    leftPinchActive,
    { get count() { return leftOnCount; }, set count(v) { leftOnCount = v; } },
    { get count() { return leftOffCount; }, set count(v) { leftOffCount = v; } },
    ON_FRAMES,
    OFF_FRAMES
  );

  return leftPinchActive;
}

function updateRightPinch(landmarks) {
  const thumbTip = landmarks[4];
  const middleTip = landmarks[12];
  const hs = getHandScaleNorm(landmarks);
  const d = distance(thumbTip, middleTip) / hs;

  const raw = rightPinchActive ? d < RIGHT_PINCH_OFF : d < RIGHT_PINCH_ON;

  rightPinchActive = debouncedBinary(
    raw,
    rightPinchActive,
    { get count() { return rightOnCount; }, set count(v) { rightOnCount = v; } },
    { get count() { return rightOffCount; }, set count(v) { rightOffCount = v; } },
    ON_FRAMES,
    OFF_FRAMES
  );

  return rightPinchActive;
}

function isFingerExtended(tip, pip) {
  return tip.y < pip.y;
}

function scrollRaw(landmarks) {
  const indexTip = landmarks[8];
  const indexPip = landmarks[6];
  const middleTip = landmarks[12];
  const middlePip = landmarks[10];
  const ringTip = landmarks[16];
  const ringPip = landmarks[14];
  const pinkyTip = landmarks[20];
  const pinkyPip = landmarks[18];

  const indexUp = isFingerExtended(indexTip, indexPip);
  const middleUp = isFingerExtended(middleTip, middlePip);
  const ringDown = ringTip.y > ringPip.y;
  const pinkyDown = pinkyTip.y > pinkyPip.y;

  const hs = getHandScaleNorm(landmarks);
  const thumbIndexDist = distance(landmarks[4], landmarks[8]) / hs;
  const thumbMiddleDist = distance(landmarks[4], landmarks[12]) / hs;

  return indexUp && middleUp && ringDown && pinkyDown && thumbIndexDist > 0.55 && thumbMiddleDist > 0.55;
}

function updateScrollGesture(landmarks) {
  const raw = scrollRaw(landmarks);

  scrollActive = debouncedBinary(
    raw,
    scrollActive,
    { get count() { return scrollOnCount; }, set count(v) { scrollOnCount = v; } },
    { get count() { return scrollOffCount; }, set count(v) { scrollOffCount = v; } },
    SCROLL_ON_FRAMES,
    SCROLL_OFF_FRAMES
  );

  return scrollActive;
}

function thumbsUpRaw(landmarks) {
  const wrist = landmarks[0];
  const thumbTip = landmarks[4];
  const thumbIp = landmarks[3];
  const thumbMcp = landmarks[2];

  const indexTip = landmarks[8];
  const indexPip = landmarks[6];
  const middleTip = landmarks[12];
  const middlePip = landmarks[10];
  const ringTip = landmarks[16];
  const ringPip = landmarks[14];
  const pinkyTip = landmarks[20];
  const pinkyPip = landmarks[18];

  const thumbUp =
    thumbTip.y < thumbIp.y &&
    thumbIp.y < thumbMcp.y &&
    thumbTip.y < wrist.y;

  const othersFolded =
    indexTip.y > indexPip.y &&
    middleTip.y > middlePip.y &&
    ringTip.y > ringPip.y &&
    pinkyTip.y > pinkyPip.y;

  return thumbUp && othersFolded;
}

function updateThumbsUp(landmarks) {
  const raw = thumbsUpRaw(landmarks);

  thumbsUpActive = debouncedBinary(
    raw,
    thumbsUpActive,
    { get count() { return thumbsOnCount; }, set count(v) { thumbsOnCount = v; } },
    { get count() { return thumbsOffCount; }, set count(v) { thumbsOffCount = v; } },
    THUMBS_ON_FRAMES,
    THUMBS_OFF_FRAMES
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
      width: 1280,
      height: 720,
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

  setStatus("Camera started. Hand mouse is live.");
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

  const results = handLandmarker.detectForVideo(video, nowMs);

  if (results?.landmarks?.length) {
    lastSeenMs = nowMs;

    const landmarks = results.landmarks[0];

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
          scrollAccumulator += dy * 1.5;

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
        `handScalePx(raw=${handScaleRaw.toFixed(1)} smooth=${handScaleSmooth.toFixed(1)}) left=${left} right=${right} scroll=${scroll} thumbs=${thumbs}`
      );
    }
  } else {
    const lostFor = nowMs - lastSeenMs;

    if (lostFor <= HAND_LOST_GRACE_MS) {
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
