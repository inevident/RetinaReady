const elements = {
  dropzone: document.querySelector("#dropzone"),
  fileInput: document.querySelector("#fileInput"),
  dropEmpty: document.querySelector("#dropEmpty"),
  dropTitle: document.querySelector("#dropTitle"),
  dropBrowse: document.querySelector("#dropBrowse"),
  dropFormat: document.querySelector("#dropFormat"),
  previewWrap: document.querySelector("#previewWrap"),
  previewImage: document.querySelector("#previewImage"),
  replaceButton: document.querySelector("#replaceButton"),
  fileName: document.querySelector("#fileName"),
  fileSize: document.querySelector("#fileSize"),
  guidanceLaunch: document.querySelector("#guidanceLaunch"),
  startGuidanceButton: document.querySelector("#startGuidanceButton"),
  loadVideoButton: document.querySelector("#loadVideoButton"),
  videoInput: document.querySelector("#videoInput"),
  stopGuidanceButton: document.querySelector("#stopGuidanceButton"),
  guidanceStage: document.querySelector("#guidanceStage"),
  guidanceViewport: document.querySelector("#guidanceViewport"),
  guidanceCanvas: document.querySelector("#guidanceCanvas"),
  guidanceVideo: document.querySelector("#guidanceVideo"),
  captureReadyChime: document.querySelector("#captureReadyChime"),
  guidanceSource: document.querySelector("#guidanceSource"),
  guidanceRate: document.querySelector("#guidanceRate"),
  guidanceDisclosure: document.querySelector("#guidanceDisclosure"),
  guidanceReticle: document.querySelector("#guidanceReticle"),
  guidanceFlash: document.querySelector("#guidanceFlash"),
  guidancePhase: document.querySelector("#guidancePhase"),
  guidanceInstruction: document.querySelector("#guidanceInstruction"),
  guidanceScoreRing: document.querySelector("#guidanceScoreRing"),
  guidanceScore: document.querySelector("#guidanceScore"),
  guidanceFieldBar: document.querySelector("#guidanceFieldBar"),
  guidanceField: document.querySelector("#guidanceField"),
  guidanceClarityBar: document.querySelector("#guidanceClarityBar"),
  guidanceClarity: document.querySelector("#guidanceClarity"),
  guidanceLightBar: document.querySelector("#guidanceLightBar"),
  guidanceLight: document.querySelector("#guidanceLight"),
  guidanceStabilityBar: document.querySelector("#guidanceStabilityBar"),
  guidanceStability: document.querySelector("#guidanceStability"),
  guidanceLockBar: document.querySelector("#guidanceLockBar"),
  analyzeButton: document.querySelector("#analyzeButton"),
  analyzeButtonText: document.querySelector("#analyzeButtonText"),
  resultCard: document.querySelector("#resultCard"),
  resultEmpty: document.querySelector("#resultEmpty"),
  loadingState: document.querySelector("#loadingState"),
  resultContent: document.querySelector("#resultContent"),
  statusBadge: document.querySelector("#statusBadge"),
  resultIcon: document.querySelector("#resultIcon"),
  resultEyebrow: document.querySelector("#resultEyebrow"),
  resultSummary: document.querySelector("#resultSummary"),
  decisionTrace: document.querySelector("#decisionTrace"),
  traceSpecialist: document.querySelector("#traceSpecialist"),
  traceGemmaLabel: document.querySelector("#traceGemmaLabel"),
  traceGemma: document.querySelector("#traceGemma"),
  tracePolicy: document.querySelector("#tracePolicy"),
  issuesSection: document.querySelector("#issuesSection"),
  issueChips: document.querySelector("#issueChips"),
  scoresSection: document.querySelector("#scoresSection"),
  scoreRows: document.querySelector("#scoreRows"),
  qualityAttention: document.querySelector("#qualityAttention"),
  attentionLabel: document.querySelector("#attentionLabel"),
  attentionMethod: document.querySelector("#attentionMethod"),
  attentionImage: document.querySelector("#attentionImage"),
  attentionFactor: document.querySelector("#attentionFactor"),
  instructionText: document.querySelector("#instructionText"),
  privacyMeta: document.querySelector("#privacyMeta"),
  modelMeta: document.querySelector("#modelMeta"),
  latencyMeta: document.querySelector("#latencyMeta"),
  runtimeLabel: document.querySelector("#runtimeLabel"),
  enginePill: document.querySelector("#enginePill"),
  sampleRow: document.querySelector(".sample-row"),
  modeKicker: document.querySelector("#modeKicker"),
  pageTitle: document.querySelector("#pageTitle"),
  modeDescription: document.querySelector("#modeDescription"),
  modeButtons: document.querySelectorAll("[data-product-mode]"),
  adapterState: document.querySelector("#adapterState"),
  resultStepLabel: document.querySelector("#resultStepLabel"),
  resultTitle: document.querySelector("#resultTitle"),
  resultEmptyTitle: document.querySelector("#resultEmptyTitle"),
  resultEmptyCopy: document.querySelector("#resultEmptyCopy"),
  loadingTitle: document.querySelector("#loadingTitle"),
  issuesLabel: document.querySelector("#issuesLabel"),
  workflowTrace: document.querySelector("#workflowTrace"),
  workflowModeLabel: document.querySelector("#workflowModeLabel"),
  workflowTraceList: document.querySelector("#workflowTraceList"),
  toast: document.querySelector("#toast"),
};

const state = {
  file: null,
  previewUrl: null,
  scenario: "",
  inputOrigin: "",
  processing: false,
  useDatasetSamples: false,
  datasetOnly: false,
  runtimeMode: "unknown",
  gemmaEscalation: false,
  productMode: "COMBINED",
  sampleRowAvailable: true,
  guidanceRunning: false,
  guidanceCaptured: false,
  guidanceAnimation: 0,
  guidanceFinishTimer: 0,
  guidanceStartedAt: 0,
  guidanceLastAnalysisAt: 0,
  guidanceOriginalFile: null,
  guidanceSourceImage: null,
  guidancePreviousGray: null,
  guidanceController: null,
  guidanceMode: "",
  guidanceVideoUrl: "",
  guidanceVideoFile: null,
  guidanceLastVideoTime: -1,
  videoCandidateWorkflowEnabled: false,
};

let captureReadyChimeAudioContext = null;
let captureReadyChimeBufferPromise = null;

const MAX_FILE_SIZE = 16 * 1024 * 1024;
const MAX_VIDEO_SIZE = 128 * 1024 * 1024;
const VIDEO_DECODE_TIMEOUT_MS = 30000;
const WORKFLOW_TIMEOUT_MS = 120000;
const QUALITY_ATTENTION_LABEL =
  "Model quality attention \u2014 not pathology localization.";
const ACCEPTED_TYPES = new Set([
  "image/jpeg",
  "image/png",
  "image/webp",
]);
const ACCEPTED_VIDEO_TYPES = new Set([
  "video/mp4",
  "video/webm",
  "video/quicktime",
]);

const MODE_COPY = {
  QUALITY_ONLY: {
    label: "Quality only",
    kicker: "<span>Gemma 4</span> · On-device capture QA",
    specialistKicker: "<span>Local quality specialist</span> · Dataset capture QA",
    demoKicker: "<span>Local demo</span> · On-device capture QA",
    title: "Know when to retake.<br /><em>Before the patient leaves.</em>",
    description: "RetinaReady checks only technical capture quality; it does not prioritize pathology review.",
    step: "02 · Quality gate",
    result: "Technical assessment",
    action: "Check capture quality",
    loading: "Checking capture quality",
    emptyTitle: "Your quality result will appear here",
    emptyCopy: "RetinaReady checks focus, illumination, and field coverage for conventional central-field color fundus photos.",
    issuesLabel: "Detected capture issues",
  },
  ESCALATION_ONLY: {
    label: "Escalation only",
    kicker: "<span>Gemma 4</span> · Offline review prioritization",
    specialistKicker: "<span>Local priority specialist</span> · Offline review prioritization",
    demoKicker: "<span>Local demo</span> · Offline review prioritization",
    title: "Sort the queue.<br /><em>Keep clinicians in control.</em>",
    description: "RetinaPriority suggests review order only. It does not diagnose disease or recommend treatment.",
    step: "02 · Priority stage",
    result: "Review-priority suggestion",
    action: "Assess review priority",
    loading: "Assessing review priority",
    emptyTitle: "Your priority suggestion will appear here",
    emptyCopy: "RetinaPriority suggests clinician review order without diagnosing disease or recommending treatment.",
    issuesLabel: "Review-priority signals",
  },
  COMBINED: {
    label: "Combined",
    kicker: "<span>Gemma 4</span> · On-device retinal workflow",
    specialistKicker: "<span>Local specialists</span> · On-device retinal workflow",
    demoKicker: "<span>Local demo</span> · On-device retinal workflow",
    title: "Quality first.<br /><em>Then prioritize review.</em>",
    description: "Quality runs first; only READY images can reach non-diagnostic review prioritization.",
    step: "02 · Workflow result",
    result: "Quality + review priority",
    action: "Run local workflow",
    loading: "Running quality-first workflow",
    emptyTitle: "Your workflow result will appear here",
    emptyCopy: "RetinaReady checks technical quality before the optional, non-diagnostic review-priority stage.",
    issuesLabel: "Workflow signals",
  },
};

function kickerFor(copy) {
  if (
    state.gemmaEscalation &&
    state.productMode !== "QUALITY_ONLY"
  ) return copy.kicker;
  if (state.runtimeMode === "specialist-local") return copy.specialistKicker;
  if (["local-model", "hybrid-local"].includes(state.runtimeMode)) {
    return copy.kicker;
  }
  return copy.demoKicker;
}

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.hidden = false;
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => {
    elements.toast.hidden = true;
  }, 3800);
}

function resetResult() {
  elements.resultEmpty.hidden = false;
  elements.loadingState.hidden = true;
  elements.resultContent.hidden = true;
  elements.resultCard.setAttribute("aria-busy", "false");
  elements.statusBadge.textContent = "Waiting";
  elements.statusBadge.className = "status-badge status-waiting";
  elements.qualityAttention.hidden = true;
  elements.attentionImage.removeAttribute("src");
  elements.decisionTrace.hidden = true;
  elements.workflowTrace.hidden = true;
}

function applyProductMode(mode) {
  if (!MODE_COPY[mode] || state.processing) return;
  if (state.guidanceRunning) {
    showToast("Stop the guided preview before changing workflow mode.");
    return;
  }
  state.productMode = mode;
  const copy = MODE_COPY[mode];
  elements.modeKicker.innerHTML = kickerFor(copy);
  elements.pageTitle.innerHTML = copy.title;
  elements.modeDescription.textContent = copy.description;
  elements.resultStepLabel.textContent = copy.step;
  elements.resultTitle.textContent = copy.result;
  elements.resultEmptyTitle.textContent = copy.emptyTitle;
  elements.resultEmptyCopy.textContent = copy.emptyCopy;
  elements.loadingTitle.textContent = copy.loading;
  elements.issuesLabel.textContent = copy.issuesLabel;
  elements.adapterState.hidden = mode === "QUALITY_ONLY";
  elements.modeButtons.forEach((button) => {
    const active = button.dataset.productMode === mode;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  if (state.file) elements.analyzeButtonText.textContent = copy.action;
  resetResult();
}

function setFile(file, scenario = "", inputOrigin = "") {
  if (!file) return;
  if (!ACCEPTED_TYPES.has(file.type)) {
    showToast("Choose a JPG, PNG, or WEBP image.");
    return;
  }
  if (file.size > MAX_FILE_SIZE) {
    showToast("That image is larger than 16 MB.");
    return;
  }

  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.file = file;
  state.scenario = scenario;
  state.inputOrigin = inputOrigin;
  state.previewUrl = URL.createObjectURL(file);

  elements.previewImage.src = state.previewUrl;
  elements.fileName.textContent = file.name;
  elements.fileSize.textContent = formatBytes(file.size);
  elements.dropEmpty.hidden = true;
  elements.previewWrap.hidden = false;
  elements.analyzeButton.disabled = false;
  elements.analyzeButtonText.textContent = MODE_COPY[state.productMode].action;
  resetResult();
}

function openFilePicker() {
  if (state.datasetOnly) {
    showToast("Use one of the fixed DeepDRiD sample buttons in this safety profile.");
    return;
  }
  if (!state.processing) elements.fileInput.click();
}

elements.dropzone.addEventListener("click", openFilePicker);
elements.dropzone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    openFilePicker();
  }
});
elements.replaceButton.addEventListener("click", (event) => {
  event.stopPropagation();
  openFilePicker();
});
elements.fileInput.addEventListener("change", () => {
  setFile(elements.fileInput.files?.[0]);
  elements.fileInput.value = "";
});

elements.modeButtons.forEach((button) => {
  button.addEventListener("click", () => applyProductMode(button.dataset.productMode));
});

["dragenter", "dragover"].forEach((eventName) => {
  elements.dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropzone.classList.add("is-dragging");
  });
});
["dragleave", "drop"].forEach((eventName) => {
  elements.dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropzone.classList.remove("is-dragging");
  });
});
elements.dropzone.addEventListener("drop", (event) => {
  if (state.datasetOnly) {
    showToast("Use one of the fixed DeepDRiD sample buttons in this safety profile.");
    return;
  }
  setFile(event.dataTransfer.files?.[0]);
});

function seededRandom(seed) {
  let value = seed >>> 0;
  return () => {
    value = (value * 1664525 + 1013904223) >>> 0;
    return value / 4294967296;
  };
}

function drawFundusSample(context, width, height, scenario) {
  const random = seededRandom(
    { READY: 17, LIMITED: 31, RETAKE: 53, UNSUPPORTED: 79 }[scenario] || 17,
  );
  context.fillStyle = "#07181b";
  context.fillRect(0, 0, width, height);

  if (scenario === "UNSUPPORTED") {
    const scan = context.createLinearGradient(0, 0, 0, height);
    scan.addColorStop(0, "#101a1e");
    scan.addColorStop(0.55, "#69767b");
    scan.addColorStop(0.58, "#f0f2e8");
    scan.addColorStop(0.64, "#9da68d");
    scan.addColorStop(1, "#10171b");
    context.fillStyle = scan;
    context.fillRect(0, 80, width, height - 160);
    context.strokeStyle = "rgba(232, 245, 224, .72)";
    context.lineWidth = 7;
    context.beginPath();
    for (let x = 0; x <= width; x += 12) {
      const y = height * 0.58 + Math.sin(x / 54) * 22 + Math.sin(x / 17) * 6;
      if (x === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    }
    context.stroke();
    return;
  }

  const cx = width * 0.5;
  const cy = height * 0.49;
  const radius = width * (scenario === "LIMITED" ? 0.39 : 0.42);
  const retina = context.createRadialGradient(
    cx - radius * 0.25,
    cy - radius * 0.25,
    radius * 0.06,
    cx,
    cy,
    radius,
  );
  retina.addColorStop(0, scenario === "RETAKE" ? "#b45e3d" : "#f48a54");
  retina.addColorStop(0.42, scenario === "RETAKE" ? "#75351f" : "#b5452e");
  retina.addColorStop(0.88, "#451a18");
  retina.addColorStop(1, "#171018");
  context.fillStyle = retina;
  context.beginPath();
  context.arc(cx, cy, radius, 0, Math.PI * 2);
  context.fill();

  const discX = scenario === "LIMITED" ? cx + radius * 0.72 : cx + radius * 0.33;
  const discY = cy - radius * 0.06;
  const disc = context.createRadialGradient(
    discX,
    discY,
    0,
    discX,
    discY,
    radius * 0.17,
  );
  disc.addColorStop(0, "rgba(255,240,174,.95)");
  disc.addColorStop(1, "rgba(246,182,108,.12)");
  context.fillStyle = disc;
  context.beginPath();
  context.ellipse(discX, discY, radius * 0.14, radius * 0.18, 0.2, 0, Math.PI * 2);
  context.fill();

  context.lineCap = "round";
  for (let vessel = 0; vessel < 22; vessel += 1) {
    const angle = random() * Math.PI * 2;
    const length = radius * (0.45 + random() * 0.52);
    const endX = discX + Math.cos(angle) * length;
    const endY = discY + Math.sin(angle) * length * 0.76;
    const bend = (random() - 0.5) * radius * 0.55;
    context.strokeStyle = `rgba(75, 12, 19, ${0.48 + random() * 0.3})`;
    context.lineWidth = 1.2 + random() * 3.7;
    context.beginPath();
    context.moveTo(discX, discY);
    context.bezierCurveTo(
      discX + Math.cos(angle) * length * 0.3,
      discY + Math.sin(angle) * length * 0.2 + bend,
      discX + Math.cos(angle) * length * 0.7,
      discY + Math.sin(angle) * length * 0.7 - bend * 0.4,
      endX,
      endY,
    );
    context.stroke();
  }

  if (scenario === "LIMITED") {
    const shade = context.createLinearGradient(0, 0, width, 0);
    shade.addColorStop(0, "rgba(6,15,18,.7)");
    shade.addColorStop(0.52, "rgba(6,15,18,0)");
    context.fillStyle = shade;
    context.fillRect(0, 0, width, height);
  }

  if (scenario === "RETAKE") {
    context.fillStyle = "rgba(10,18,20,.23)";
    context.fillRect(0, 0, width, height);
    context.globalAlpha = 0.34;
    context.drawImage(context.canvas, 10, 3);
    context.drawImage(context.canvas, -9, -2);
    context.globalAlpha = 1;
  }
}

async function createSampleFile(scenario) {
  const canvas = document.createElement("canvas");
  canvas.width = 760;
  canvas.height = 570;
  const context = canvas.getContext("2d");
  drawFundusSample(context, canvas.width, canvas.height, scenario);
  const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.88));
  return new File([blob], `${scenario.toLowerCase()}-demo-capture.jpg`, {
    type: "image/jpeg",
  });
}

async function loadDatasetSample(scenario) {
  const response = await fetch(`/api/demo-samples/${scenario}`);
  if (!response.ok) throw new Error("Dataset sample unavailable");
  const blob = await response.blob();
  return new File(
    [blob],
    `deepdrid-${scenario.toLowerCase()}-sample.jpg`,
    { type: "image/jpeg" },
  );
}

function loadImage(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => {
      URL.revokeObjectURL(url);
      resolve(image);
    };
    image.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("The replay image could not be decoded."));
    };
    image.src = url;
  });
}

const guidanceAnalysisCanvas = document.createElement("canvas");
guidanceAnalysisCanvas.width = 180;
guidanceAnalysisCanvas.height = 135;
const guidanceAnalysisContext = guidanceAnalysisCanvas.getContext("2d", {
  willReadFrequently: true,
});

function easeInOut(value) {
  const clamped = Math.max(0, Math.min(1, value));
  return clamped * clamped * (3 - 2 * clamped);
}

function replayTransform(elapsed) {
  if (elapsed < 2600) {
    const progress = easeInOut(elapsed / 2600);
    return {
      phase: "Framing",
      x: 158 - progress * 96,
      y: 20 - progress * 14,
      scale: 0.92 + progress * 0.04,
      blur: 0.6,
      brightness: 0.72,
    };
  }
  if (elapsed < 5000) {
    const progress = easeInOut((elapsed - 2600) / 2400);
    return {
      phase: "Aligning",
      x: 62 - progress * 54 + Math.sin(elapsed / 52) * 7,
      y: 6 + Math.cos(elapsed / 61) * 5,
      scale: 0.96 + progress * 0.04,
      blur: 1.1,
      brightness: 0.78,
    };
  }
  if (elapsed < 7200) {
    const progress = easeInOut((elapsed - 5000) / 2200);
    return {
      phase: "Focusing",
      x: 8 + Math.sin(elapsed / 88) * (4 - progress * 3),
      y: Math.cos(elapsed / 94) * (3 - progress * 2),
      scale: 1,
      blur: 5.6 - progress * 2.6,
      brightness: 0.86,
    };
  }
  if (elapsed < 9000) {
    const progress = easeInOut((elapsed - 7200) / 1800);
    return {
      phase: "Balancing light",
      x: 1,
      y: 0,
      scale: 1,
      blur: 3 - progress * 3,
      brightness: 0.7 + progress * 0.3,
    };
  }
  return {
    phase: "Stabilizing",
    x: 0,
    y: 0,
    scale: 1,
    blur: 0,
    brightness: 1,
  };
}

function drawGuidanceFrame(transform) {
  const canvas = elements.guidanceCanvas;
  const context = canvas.getContext("2d");
  const image = state.guidanceSourceImage;
  context.save();
  context.fillStyle = "#020d0f";
  context.fillRect(0, 0, canvas.width, canvas.height);
  const fitScale = Math.min(
    (canvas.width * 0.82) / image.naturalWidth,
    (canvas.height * 0.96) / image.naturalHeight,
  );
  const width = image.naturalWidth * fitScale * transform.scale;
  const height = image.naturalHeight * fitScale * transform.scale;
  context.translate(
    canvas.width / 2 + transform.x,
    canvas.height / 2 + transform.y,
  );
  context.filter = `blur(${transform.blur}px) brightness(${transform.brightness}) saturate(1.04)`;
  context.drawImage(image, -width / 2, -height / 2, width, height);
  context.restore();
}

function drawVideoGuidanceFrame() {
  const canvas = elements.guidanceCanvas;
  const context = canvas.getContext("2d");
  const video = elements.guidanceVideo;
  context.fillStyle = "#020d0f";
  context.fillRect(0, 0, canvas.width, canvas.height);
  if (!video.videoWidth || !video.videoHeight) return;
  const scale = Math.min(
    canvas.width / video.videoWidth,
    canvas.height / video.videoHeight,
  );
  const width = video.videoWidth * scale;
  const height = video.videoHeight * scale;
  context.drawImage(
    video,
    (canvas.width - width) / 2,
    (canvas.height - height) / 2,
    width,
    height,
  );
}

function prepareGuidanceVideo(file) {
  if (!file || !ACCEPTED_VIDEO_TYPES.has(file.type)) {
    throw new Error("Choose an MP4, WEBM, or MOV camera recording.");
  }
  if (file.size > MAX_VIDEO_SIZE) {
    throw new Error("That recording is larger than 128 MB.");
  }
  if (state.guidanceVideoUrl) URL.revokeObjectURL(state.guidanceVideoUrl);
  state.guidanceVideoUrl = URL.createObjectURL(file);
  state.guidanceVideoFile = file;

  return new Promise((resolve, reject) => {
    const video = elements.guidanceVideo;
    const timeout = window.setTimeout(() => {
      cleanup();
      reject(
        new Error(
          "The camera recording could not decode a first frame. Close memory-heavy apps or use the idle-sleep launcher and try again.",
        ),
      );
    }, VIDEO_DECODE_TIMEOUT_MS);
    const cleanup = () => {
      window.clearTimeout(timeout);
      video.removeEventListener("loadeddata", handleLoaded);
      video.removeEventListener("canplay", handleLoaded);
      video.removeEventListener("error", handleError);
    };
    const handleLoaded = () => {
      if (
        video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA ||
        !video.videoWidth ||
        !video.videoHeight
      ) return;
      cleanup();
      if (!Number.isFinite(video.duration) || video.duration <= 0) {
        reject(new Error("The camera recording has no playable frames."));
        return;
      }
      resolve();
    };
    const handleError = () => {
      cleanup();
      reject(new Error("The camera recording could not be decoded locally."));
    };
    video.addEventListener("loadeddata", handleLoaded);
    video.addEventListener("canplay", handleLoaded);
    video.addEventListener("error", handleError);
    video.muted = true;
    video.playsInline = true;
    video.preload = "auto";
    video.src = state.guidanceVideoUrl;
    video.load();
  });
}

function waitForPresentedVideoFrame(video) {
  return new Promise((resolve, reject) => {
    let callbackId = 0;
    const timeout = window.setTimeout(() => {
      cleanup();
      reject(
        new Error(
          "The recording loaded, but the browser could not present a decoded frame.",
        ),
      );
    }, VIDEO_DECODE_TIMEOUT_MS);
    const cleanup = () => {
      window.clearTimeout(timeout);
      video.removeEventListener("timeupdate", handleFallbackFrame);
      if (callbackId && typeof video.cancelVideoFrameCallback === "function") {
        video.cancelVideoFrameCallback(callbackId);
      }
    };
    const finish = () => {
      cleanup();
      resolve();
    };
    const handleFallbackFrame = () => {
      if (
        video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
        video.videoWidth &&
        video.videoHeight &&
        video.currentTime > 0
      ) finish();
    };

    if (typeof video.requestVideoFrameCallback === "function") {
      callbackId = video.requestVideoFrameCallback(finish);
    } else {
      video.addEventListener("timeupdate", handleFallbackFrame);
    }
  });
}

function analyzeGuidanceCanvas(previousGray) {
  guidanceAnalysisContext.clearRect(
    0,
    0,
    guidanceAnalysisCanvas.width,
    guidanceAnalysisCanvas.height,
  );
  guidanceAnalysisContext.drawImage(
    elements.guidanceCanvas,
    0,
    0,
    guidanceAnalysisCanvas.width,
    guidanceAnalysisCanvas.height,
  );
  const imageData = guidanceAnalysisContext.getImageData(
    0,
    0,
    guidanceAnalysisCanvas.width,
    guidanceAnalysisCanvas.height,
  );
  return window.RetinaCaptureGuidance.analyzeFrame(imageData, previousGray);
}

function setGuidanceMetric(bar, label, value) {
  const safeValue = Math.max(0, Math.min(100, Math.round(value)));
  bar.style.width = `${safeValue}%`;
  label.textContent = String(safeValue);
}

function resetCaptureReadyChime() {
  const chime = elements.captureReadyChime;
  if (!chime) return;
  chime.pause();
  chime.currentTime = 0;
}

function captureReadyAudioContext() {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) return null;
  if (!captureReadyChimeAudioContext) {
    captureReadyChimeAudioContext = new AudioContextClass();
  }
  return captureReadyChimeAudioContext;
}

function decodeCaptureReadyChime(context, encodedAudio) {
  return new Promise((resolve, reject) => {
    let settled = false;
    const finish = (handler) => (value) => {
      if (settled) return;
      settled = true;
      handler(value);
    };
    const decoded = context.decodeAudioData(
      encodedAudio,
      finish(resolve),
      finish(reject),
    );
    if (decoded && typeof decoded.then === "function") {
      decoded.then(finish(resolve), finish(reject));
    }
  });
}

function loadCaptureReadyChimeBuffer(context) {
  if (!captureReadyChimeBufferPromise) {
    const source =
      elements.captureReadyChime?.currentSrc ||
      elements.captureReadyChime?.getAttribute("src");
    captureReadyChimeBufferPromise = fetch(source, { cache: "force-cache" })
      .then((response) => {
        if (!response.ok) throw new Error("Capture chime asset unavailable.");
        return response.arrayBuffer();
      })
      .then((encodedAudio) => decodeCaptureReadyChime(context, encodedAudio))
      .catch((error) => {
        captureReadyChimeBufferPromise = null;
        throw error;
      });
  }
  return captureReadyChimeBufferPromise;
}

function primeCaptureReadyChime() {
  const context = captureReadyAudioContext();
  if (!context) return Promise.resolve();
  const resume =
    context.state !== "running" ? context.resume() : Promise.resolve();
  return Promise.allSettled([resume, loadCaptureReadyChimeBuffer(context)]);
}

function playCaptureReadyChimeFallback() {
  const chime = elements.captureReadyChime;
  if (!chime) return;
  try {
    chime.pause();
    chime.currentTime = 0;
    const playback = chime.play();
    if (playback && typeof playback.catch === "function") {
      playback.catch(() => {});
    }
  } catch (_error) {
    // Audio should never interrupt the local capture workflow.
  }
}

async function playCaptureReadyChime() {
  try {
    const context = captureReadyAudioContext();
    if (!context) throw new Error("Web Audio is unavailable.");
    if (context.state !== "running") await context.resume();
    const buffer = await loadCaptureReadyChimeBuffer(context);
    const source = context.createBufferSource();
    source.buffer = buffer;
    source.connect(context.destination);
    source.start(0);
  } catch (_error) {
    playCaptureReadyChimeFallback();
  }
}

function renderGuidanceEvaluation(evaluation, phase) {
  const { scores, instruction, stableFrames, requiredStableFrames } = evaluation;
  elements.guidanceScore.textContent = String(scores.overall);
  elements.guidanceScoreRing.style.setProperty("--score", String(scores.overall));
  setGuidanceMetric(elements.guidanceFieldBar, elements.guidanceField, scores.field);
  setGuidanceMetric(
    elements.guidanceClarityBar,
    elements.guidanceClarity,
    scores.clarity,
  );
  setGuidanceMetric(
    elements.guidanceLightBar,
    elements.guidanceLight,
    scores.illumination,
  );
  setGuidanceMetric(
    elements.guidanceStabilityBar,
    elements.guidanceStability,
    scores.stability,
  );
  elements.guidanceLockBar.style.width = `${Math.min(
    100,
    (stableFrames / requiredStableFrames) * 100,
  )}%`;
  elements.guidanceInstruction.textContent = instruction.label;
  elements.guidancePhase.textContent = phase;
  elements.guidanceStage.dataset.state =
    instruction.code === "CAPTURE"
      ? "captured"
      : stableFrames > 0
        ? "stabilizing"
        : "guiding";
}

function restoreCaptureControls() {
  elements.guidanceStage.hidden = true;
  elements.guidanceLaunch.hidden = false;
  elements.dropzone.hidden = false;
  elements.sampleRow.hidden = !state.sampleRowAvailable;
  elements.analyzeButton.hidden = false;
}

function setGuidanceCopy(mode) {
  if (mode === "video") {
    elements.guidanceStage.setAttribute(
      "aria-label",
      "Prerecorded clinical fundus-camera video preview",
    );
    elements.guidanceCanvas.setAttribute(
      "aria-label",
      "Locally decoded clinical fundus-camera recording",
    );
    elements.guidanceSource.textContent = "Prerecorded device video";
    elements.guidanceRate.textContent = "6 Hz technical telemetry";
    elements.guidanceDisclosure.textContent =
      "Prerecorded fundus-camera input. Raw video stays in this browser. If five consecutive frames pass the prototype gate, one JPEG still is frozen locally and handed to the experimental quality-first workflow. This is not device or clinical validation.";
    return;
  }
  elements.guidanceStage.setAttribute(
    "aria-label",
    "Simulated retinal acquisition replay",
  );
  elements.guidanceCanvas.setAttribute(
    "aria-label",
    "Retrospective color fundus acquisition replay",
  );
  elements.guidanceSource.textContent = "Retrospective DeepDRiD replay";
  elements.guidanceRate.textContent = "6 Hz guidance";
  elements.guidanceDisclosure.textContent =
    "Simulated acquisition using retrospective color fundus imagery. Frame guidance is a local technical prototype, not a clinical or device-validated model output.";
}

function releaseGuidanceVideoSource() {
  elements.guidanceVideo.pause();
  elements.guidanceVideo.removeAttribute("src");
  elements.guidanceVideo.load();
  if (state.guidanceVideoUrl) URL.revokeObjectURL(state.guidanceVideoUrl);
  state.guidanceVideoUrl = "";
  state.guidanceVideoFile = null;
  state.guidanceLastVideoTime = -1;
}

function stopGuidedCapture({ notify = false } = {}) {
  const stoppedMode = state.guidanceMode;
  window.cancelAnimationFrame(state.guidanceAnimation);
  window.clearTimeout(state.guidanceFinishTimer);
  releaseGuidanceVideoSource();
  state.guidanceAnimation = 0;
  state.guidanceFinishTimer = 0;
  state.guidanceRunning = false;
  state.guidanceCaptured = false;
  state.guidancePreviousGray = null;
  state.guidanceController = null;
  state.guidanceMode = "";
  state.guidanceOriginalFile = null;
  state.guidanceSourceImage = null;
  restoreCaptureControls();
  if (notify) {
    showToast(
      stoppedMode === "video"
        ? "Camera recording preview stopped. No frame was submitted."
        : "Guided acquisition replay stopped.",
    );
  }
}

function finishReplayCapture() {
  if (!state.guidanceRunning || state.guidanceCaptured) return;
  state.guidanceCaptured = true;
  window.cancelAnimationFrame(state.guidanceAnimation);
  drawGuidanceFrame({
    phase: "Captured",
    x: 0,
    y: 0,
    scale: 1,
    blur: 0,
    brightness: 1,
  });
  elements.guidanceStage.dataset.state = "captured";
  elements.guidancePhase.textContent = "Frame locked";
  elements.guidanceInstruction.textContent =
    "Best frame captured · Running quality-first workflow";
  elements.guidanceLockBar.style.width = "100%";
  state.guidanceFinishTimer = window.setTimeout(async () => {
    const finalFile = state.guidanceOriginalFile;
    state.guidanceRunning = false;
    restoreCaptureControls();
    setFile(finalFile, state.runtimeMode === "demo" ? "READY" : "");
    await analyzeCurrentFile();
  }, 850);
}

function replayGuidanceAnimationFrame(timestamp) {
  if (!state.guidanceRunning || state.guidanceCaptured) return;
  if (!state.guidanceStartedAt) state.guidanceStartedAt = timestamp;
  const elapsed = timestamp - state.guidanceStartedAt;
  const transform = replayTransform(elapsed);
  drawGuidanceFrame(transform);

  if (timestamp - state.guidanceLastAnalysisAt >= 165) {
    const analyzed = analyzeGuidanceCanvas(state.guidancePreviousGray);
    state.guidancePreviousGray = analyzed.gray;
    state.guidanceLastAnalysisAt = timestamp;
    const evaluation = state.guidanceController.update(analyzed.metrics);
    renderGuidanceEvaluation(evaluation, transform.phase);
    if (evaluation.captureTriggered) playCaptureReadyChime();
    if (evaluation.captureReady && elapsed >= 9000) {
      finishReplayCapture();
      return;
    }
  }
  state.guidanceAnimation = window.requestAnimationFrame(replayGuidanceAnimationFrame);
}

function captureVideoCandidateFrame() {
  const video = elements.guidanceVideo;
  if (!video.videoWidth || !video.videoHeight) {
    return Promise.reject(new Error("The candidate frame has no decoded pixels."));
  }

  const maxDimension = 2048;
  const scale = Math.min(
    1,
    maxDimension / Math.max(video.videoWidth, video.videoHeight),
  );
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
  canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
  const context = canvas.getContext("2d");
  if (!context) {
    return Promise.reject(new Error("The browser could not create a still frame."));
  }
  context.drawImage(video, 0, 0, canvas.width, canvas.height);

  const originalName = state.guidanceVideoFile?.name || "fundus-recording";
  const safeStem = originalName
    .replace(/\.[^.]+$/, "")
    .replace(/[^a-z0-9._-]+/gi, "-")
    .slice(0, 120);
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => {
        if (!blob) {
          reject(new Error("The candidate frame could not be encoded."));
          return;
        }
        resolve(
          new File([blob], `${safeStem}-candidate.jpg`, {
            type: "image/jpeg",
            lastModified: Date.now(),
          }),
        );
      },
      "image/jpeg",
      0.94,
    );
  });
}

async function finishVideoPreview({ candidateFound = false } = {}) {
  if (!state.guidanceRunning || state.guidanceCaptured) return;
  state.guidanceCaptured = true;
  window.cancelAnimationFrame(state.guidanceAnimation);
  elements.guidanceVideo.pause();
  elements.guidanceStage.dataset.state = "preview-complete";
  elements.guidancePhase.textContent = candidateFound
    ? "Candidate frame"
    : "Preview complete";
  elements.guidanceInstruction.textContent = candidateFound
    ? "Candidate frame found · Freezing local JPEG"
    : "Recording ended · No verified still was submitted";
  if (!candidateFound) return;

  elements.guidanceLockBar.style.width = "100%";
  if (!state.videoCandidateWorkflowEnabled) {
    elements.guidanceInstruction.textContent =
      "Candidate frame found · Experimental still workflow is disabled";
    showToast("Restart with the final demo launcher to enable candidate analysis.");
    return;
  }

  try {
    const candidateFile = await captureVideoCandidateFrame();
    if (!state.guidanceCaptured || state.guidanceMode !== "video") return;
    elements.guidanceInstruction.textContent =
      "Best frame captured · Running quality + review priority";
    await new Promise((resolve) => window.setTimeout(resolve, 350));
    if (!state.guidanceCaptured || state.guidanceMode !== "video") return;

    state.guidanceOriginalFile = candidateFile;
    state.guidanceRunning = false;
    state.guidanceMode = "";
    state.guidancePreviousGray = null;
    state.guidanceController = null;
    releaseGuidanceVideoSource();
    restoreCaptureControls();
    setFile(candidateFile, "", "video-candidate");
    await analyzeCurrentFile();
  } catch (error) {
    if (!state.guidanceCaptured || state.guidanceMode !== "video") return;
    elements.guidancePhase.textContent = "Frame export failed";
    elements.guidanceInstruction.textContent =
      "Candidate could not be frozen · No image was submitted";
    showToast(error.message || "The candidate frame could not be exported.");
  }
}

function videoGuidanceAnimationFrame(timestamp) {
  if (
    !state.guidanceRunning ||
    state.guidanceMode !== "video" ||
    state.guidanceCaptured
  ) return;

  const video = elements.guidanceVideo;
  if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
    drawVideoGuidanceFrame();
    if (
      timestamp - state.guidanceLastAnalysisAt >= 165 &&
      video.currentTime !== state.guidanceLastVideoTime
    ) {
      const analyzed = analyzeGuidanceCanvas(state.guidancePreviousGray);
      state.guidancePreviousGray = analyzed.gray;
      state.guidanceLastAnalysisAt = timestamp;
      state.guidanceLastVideoTime = video.currentTime;
      const evaluation = state.guidanceController.update(analyzed.metrics);
      renderGuidanceEvaluation(evaluation, "Device video");
      if (evaluation.captureTriggered) playCaptureReadyChime();
      if (evaluation.captureReady) {
        void finishVideoPreview({ candidateFound: true });
        return;
      }
    }
  }
  if (video.ended) {
    void finishVideoPreview();
    return;
  }
  state.guidanceAnimation = window.requestAnimationFrame(videoGuidanceAnimationFrame);
}

async function startGuidedCapture() {
  if (state.processing || state.guidanceRunning) return;
  if (!window.RetinaCaptureGuidance) {
    showToast("The local capture-guidance module is unavailable.");
    return;
  }
  try {
    await primeCaptureReadyChime();
    resetCaptureReadyChime();
    applyProductMode("COMBINED");
    const file = await loadDatasetSample("READY");
    const image = await loadImage(file);
    state.guidanceOriginalFile = file;
    state.guidanceSourceImage = image;
    state.guidancePreviousGray = null;
    state.guidanceCaptured = false;
    state.guidanceRunning = true;
    state.guidanceMode = "replay";
    state.guidanceStartedAt = 0;
    state.guidanceLastAnalysisAt = 0;

    drawGuidanceFrame({
      phase: "Reference",
      x: 0,
      y: 0,
      scale: 1,
      blur: 0,
      brightness: 1,
    });
    const reference = analyzeGuidanceCanvas(null).metrics;
    state.guidanceController = window.RetinaCaptureGuidance.createController(
      reference,
      { requiredStableFrames: 5 },
    );

    resetResult();
    elements.guidanceLaunch.hidden = true;
    elements.dropzone.hidden = true;
    elements.sampleRow.hidden = true;
    elements.analyzeButton.hidden = true;
    elements.guidanceStage.hidden = false;
    elements.guidanceStage.dataset.state = "guiding";
    elements.guidanceInstruction.textContent = "Finding the retinal field…";
    elements.guidancePhase.textContent = "Acquiring";
    elements.guidanceLockBar.style.width = "0%";
    setGuidanceCopy("replay");
    state.guidanceAnimation = window.requestAnimationFrame(replayGuidanceAnimationFrame);
  } catch (error) {
    stopGuidedCapture();
    showToast(error.message || "The guided acquisition replay could not start.");
  }
}

async function startVideoGuidance(file) {
  if (state.processing || state.guidanceRunning) return;
  if (!window.RetinaCaptureGuidance) {
    showToast("The local capture-guidance module is unavailable.");
    return;
  }
  try {
    await primeCaptureReadyChime();
    resetCaptureReadyChime();
    applyProductMode("COMBINED");
    const [referenceFile] = await Promise.all([
      loadDatasetSample("READY"),
      prepareGuidanceVideo(file),
    ]);
    const referenceImage = await loadImage(referenceFile);
    state.guidanceSourceImage = referenceImage;
    drawGuidanceFrame({
      phase: "Reference",
      x: 0,
      y: 0,
      scale: 1,
      blur: 0,
      brightness: 1,
    });
    const reference = analyzeGuidanceCanvas(null).metrics;

    state.guidanceOriginalFile = null;
    state.guidancePreviousGray = null;
    state.guidanceCaptured = false;
    state.guidanceRunning = true;
    state.guidanceMode = "video";
    state.guidanceStartedAt = 0;
    state.guidanceLastAnalysisAt = 0;
    state.guidanceLastVideoTime = -1;
    state.guidanceController = window.RetinaCaptureGuidance.createController(
      reference,
      { requiredStableFrames: 5 },
    );

    resetResult();
    elements.guidanceLaunch.hidden = true;
    elements.dropzone.hidden = true;
    elements.sampleRow.hidden = true;
    elements.analyzeButton.hidden = true;
    elements.guidanceStage.hidden = false;
    elements.guidanceStage.dataset.state = "guiding";
    elements.guidanceInstruction.textContent = "Looking for a color fundus field…";
    elements.guidancePhase.textContent = "Input check";
    elements.guidanceLockBar.style.width = "0%";
    setGuidanceCopy("video");

    elements.guidanceVideo.currentTime = 0;
    elements.guidancePhase.textContent = "Decoding first frame";
    const firstFrame = waitForPresentedVideoFrame(elements.guidanceVideo);
    await Promise.all([elements.guidanceVideo.play(), firstFrame]);
    drawVideoGuidanceFrame();
    elements.guidancePhase.textContent = "Input check";
    state.guidanceAnimation = window.requestAnimationFrame(videoGuidanceAnimationFrame);
  } catch (error) {
    stopGuidedCapture();
    showToast(error.message || "The camera recording preview could not start.");
  }
}

elements.startGuidanceButton.addEventListener("click", startGuidedCapture);
elements.loadVideoButton.addEventListener("click", () => {
  if (!state.processing && !state.guidanceRunning) {
    primeCaptureReadyChime();
    elements.videoInput.click();
  }
});
elements.videoInput.addEventListener("change", () => {
  const file = elements.videoInput.files?.[0];
  elements.videoInput.value = "";
  if (file) startVideoGuidance(file);
});
elements.stopGuidanceButton.addEventListener("click", () =>
  stopGuidedCapture({ notify: true }),
);

document.querySelectorAll(".sample-button").forEach((button) => {
  button.addEventListener("click", async () => {
    const scenario = button.dataset.scenario;
    if (state.guidanceRunning) stopGuidedCapture();
    if (state.useDatasetSamples && scenario !== "UNSUPPORTED") {
      try {
        const file = await loadDatasetSample(scenario);
        // A live model evaluates the actual image; no deterministic scenario
        // header is attached.
        setFile(file, "");
        return;
      } catch (error) {
        showToast("The local DeepDRiD sample could not be loaded.");
        return;
      }
    }
    const file = await createSampleFile(scenario);
    setFile(file, scenario);
  });
});

function setLoading(isLoading) {
  state.processing = isLoading;
  elements.resultCard.setAttribute("aria-busy", String(isLoading));
  elements.loadingState.hidden = !isLoading;
  elements.analyzeButton.disabled = isLoading || !state.file;
  elements.analyzeButtonText.textContent = isLoading
    ? "Analyzing locally…"
    : MODE_COPY[state.productMode].action;
  if (isLoading) {
    elements.resultEmpty.hidden = true;
    elements.resultContent.hidden = true;
    elements.statusBadge.textContent = "Processing";
    elements.statusBadge.className = "status-badge status-waiting";
  }
}

function renderWorkflow(workflow) {
  const trace = Array.isArray(workflow.workflow_trace)
    ? workflow.workflow_trace
    : [];
  elements.workflowTraceList.replaceChildren();
  const validStates = new Set([
    "COMPLETED",
    "SKIPPED",
    "BLOCKED",
    "UNAVAILABLE",
    "ABSTAINED",
    "RELEASED",
  ]);
  const safeTrace = trace.length === 3 && trace.every(
    (stage) =>
      typeof stage.stage === "string" &&
      typeof stage.detail === "string" &&
      validStates.has(stage.state),
  );
  elements.workflowTrace.hidden = !safeTrace;
  if (!safeTrace) return;
  elements.workflowModeLabel.textContent =
    MODE_COPY[workflow.product_mode]?.label || "Local workflow";
  trace.forEach((stage) => {
    const item = document.createElement("li");
    const name = document.createElement("span");
    const detail = document.createElement("strong");
    name.textContent = stage.stage;
    detail.textContent = `${stage.state} · ${stage.detail}`;
    item.append(name, detail);
    elements.workflowTraceList.append(item);
  });
}

function addIssueChip(text, isClear = false) {
  const chip = document.createElement("span");
  chip.className = `issue-chip${isClear ? " issue-clear" : ""}`;
  chip.textContent = text;
  elements.issueChips.append(chip);
}

function addScoreRow(label, value) {
  const row = document.createElement("div");
  row.className = "score-row";

  const name = document.createElement("span");
  name.textContent = label;

  const track = document.createElement("div");
  track.className = "score-track";
  const fill = document.createElement("div");
  fill.className = "score-fill";
  track.append(fill);

  const score = document.createElement("strong");
  score.textContent = `${value}%`;

  row.append(name, track, score);
  elements.scoreRows.append(row);
  window.requestAnimationFrame(() => {
    fill.style.width = `${Math.max(0, Math.min(100, value))}%`;
  });
}

function renderResult(result) {
  const status = String(result.status || "UNSUPPORTED").toUpperCase();
  const statusClass = status.toLowerCase();

  elements.loadingState.hidden = true;
  elements.resultEmpty.hidden = true;
  elements.resultContent.hidden = false;
  elements.resultCard.setAttribute("aria-busy", "false");
  elements.statusBadge.textContent = status;
  elements.statusBadge.className = `status-badge status-${statusClass}`;
  elements.resultIcon.className = `result-icon result-icon-${statusClass}`;
  elements.resultEyebrow.textContent = result.eyebrow;
  elements.resultSummary.textContent = result.summary;
  elements.instructionText.textContent = result.instruction;

  const trace = result.decision_trace;
  const validSpecialistStates = new Set([
    "READY candidate",
    "RETAKE candidate",
    "READY decision",
    "RETAKE decision",
    "Abstained",
    "Input rejected",
    "Unavailable",
    "Inference failed",
    "Invalid output",
    "Outside dataset scope",
  ]);
  const validGemmaStates = new Set([
    "Confirmed",
    "Skipped",
    "No confirmation",
    "Not used",
  ]);
  const validPolicyStates = new Set(["READY", "RETAKE", "LIMITED"]);
  const hasSafeTrace =
    trace &&
    validSpecialistStates.has(trace.specialist) &&
    validGemmaStates.has(trace.gemma) &&
    validPolicyStates.has(trace.policy);
  elements.decisionTrace.hidden = !hasSafeTrace;
  if (hasSafeTrace) {
    elements.traceSpecialist.textContent = trace.specialist;
    elements.traceGemmaLabel.textContent =
      state.gemmaEscalation && state.productMode !== "QUALITY_ONLY"
        ? "Gemma review-priority LoRA"
        : state.runtimeMode === "specialist-local"
          ? "Quality-stage Gemma verifier (not loaded)"
          : "Quality-stage Gemma verifier";
    elements.traceGemma.textContent = trace.gemma;
    elements.tracePolicy.textContent = trace.policy;
  }

  elements.issueChips.replaceChildren();
  if (result.issues?.length) {
    result.issues.forEach((issue) => addIssueChip(issue));
  } else {
    addIssueChip(
      status === "ROUTINE_REVIEW"
        ? "No priority flag released"
        : "No technical issues detected",
      true,
    );
  }

  elements.scoreRows.replaceChildren();
  const scores = result.scores;
  elements.scoresSection.hidden = !scores;
  if (scores) {
    Object.entries(scores).forEach(([label, value]) => addScoreRow(label, value));
  }

  const attention = result.quality_attention;
  const hasSafeAttention =
    attention &&
    attention.label === QUALITY_ATTENTION_LABEL &&
    typeof attention.factor_label === "string" &&
    typeof attention.image_data_url === "string" &&
    attention.image_data_url.startsWith("data:image/png;base64,");
  elements.qualityAttention.hidden = !hasSafeAttention;
  if (hasSafeAttention) {
    elements.attentionLabel.textContent = QUALITY_ATTENTION_LABEL;
    elements.attentionFactor.textContent = attention.factor_label;
    elements.attentionMethod.textContent =
      attention.method === "factor-grad-cam"
        ? "Factor Grad-CAM"
        : "Gradient sensitivity";
    elements.attentionImage.alt = `${attention.factor_label} technical quality attention map`;
    elements.attentionImage.src = attention.image_data_url;
  } else {
    elements.attentionImage.removeAttribute("src");
  }

  const meta = result.meta || {};
  elements.privacyMeta.textContent = meta.retained === false ? "Local · not retained" : "On device";
  elements.modelMeta.textContent = meta.model || "Demo engine";
  elements.latencyMeta.textContent =
    typeof meta.latency_ms === "number" ? `${meta.latency_ms} ms` : "Local";
}

elements.attentionImage.addEventListener("error", () => {
  // Explanation imagery is optional and must never interfere with the gate.
  elements.qualityAttention.hidden = true;
  elements.attentionImage.removeAttribute("src");
});

function localFallback(scenario) {
  const profiles = {
    READY: {
      eyebrow: "Capture ready",
      summary: "This image is technically ready for clinical review.",
      issues: [],
      instruction: "No retake needed. Continue to the normal review workflow.",
      scores: { Clarity: 94, Illumination: 91, Field: 95 },
    },
    LIMITED: {
      eyebrow: "Usable with limitations",
      summary: "The main retinal field is visible, but capture quality is uneven.",
      issues: ["Uneven illumination", "Minor field cutoff"],
      instruction: "If practical, recenter the eye and use more even illumination.",
      scores: { Clarity: 83, Illumination: 61, Field: 72 },
    },
    RETAKE: {
      eyebrow: "Retake recommended",
      summary: "Technical quality is too low for a dependable review.",
      issues: ["Motion blur", "Field not centered", "Low contrast"],
      instruction: "Stabilize the camera, refocus, and center the retinal field before retaking.",
      scores: { Clarity: 31, Illumination: 54, Field: 38 },
    },
    UNSUPPORTED: {
      eyebrow: "Unable to assess",
      summary: "This does not appear to be a supported color fundus photograph.",
      issues: ["Unsupported image type"],
      instruction: "Upload a color fundus photograph. OCT and angiography images are not supported.",
      scores: null,
    },
  };
  const status = profiles[scenario] ? scenario : "LIMITED";
  return {
    ...profiles[status],
    status,
    meta: {
      model: "Browser demo fallback",
      latency_ms: 0,
      retained: false,
    },
  };
}

async function analyzeCurrentFile() {
  if (!state.file || state.processing) return;
  setLoading(true);

  try {
    const headers = {
      "Content-Type": state.file.type,
      "X-Filename": encodeURIComponent(state.file.name),
      "X-Product-Mode": state.productMode,
    };
    if (state.scenario) headers["X-Demo-Scenario"] = state.scenario;
    if (state.inputOrigin) headers["X-Input-Origin"] = state.inputOrigin;

    const controller = new AbortController();
    const timeout = window.setTimeout(
      () => controller.abort(),
      WORKFLOW_TIMEOUT_MS,
    );
    const response = await fetch("/api/workflow", {
      method: "POST",
      headers,
      body: state.file,
      signal: controller.signal,
    });
    window.clearTimeout(timeout);
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.detail || "The local quality check could not run.");
    }
    const workflow = await response.json();
    renderResult(workflow.display);
    renderWorkflow(workflow);
  } catch (error) {
    if (
      state.productMode === "QUALITY_ONLY" &&
      state.runtimeMode === "demo" &&
      state.scenario
    ) {
      renderResult(localFallback(state.scenario));
      showToast("Server unavailable—showing the deterministic browser demo.");
    } else {
      const unavailable = localFallback("LIMITED");
      unavailable.status = "UNCERTAIN";
      unavailable.eyebrow = "Workflow unavailable";
      unavailable.summary = "No review-priority decision was released.";
      unavailable.issues = ["Human prioritization required"];
      unavailable.instruction = "Use the normal clinician-review queue and retry the local workflow.";
      unavailable.scores = null;
      unavailable.meta.model = "Local runtime unavailable";
      renderResult(unavailable);
      showToast(
        error.name === "AbortError"
          ? "The local quality check timed out safely."
          : error.message || "The local quality check could not run.",
      );
    }
  } finally {
    setLoading(false);
  }
}

elements.analyzeButton.addEventListener("click", analyzeCurrentFile);

fetch("/api/health")
  .then((response) => response.json())
  .then((health) => {
    const isDemo = health.mode === "demo";
    state.runtimeMode = health.mode;
    state.videoCandidateWorkflowEnabled =
      health.video_candidate_workflow_enabled === true;
    const isReady = health.status === "ready";
    state.useDatasetSamples = !isDemo && Boolean(health.dataset_samples_available);
    state.sampleRowAvailable = isDemo || state.useDatasetSamples;
    elements.startGuidanceButton.disabled = !health.dataset_samples_available;
    elements.loadVideoButton.disabled = !health.dataset_samples_available;
    if (!health.dataset_samples_available) {
      elements.startGuidanceButton.title = "The local DeepDRiD replay asset is unavailable.";
      elements.loadVideoButton.title =
        "The local DeepDRiD reference asset is unavailable.";
    }
    state.datasetOnly =
      health.input_scope === "fixed-deepdrid-demo-samples";
    if (state.datasetOnly) {
      elements.dropzone.setAttribute(
        "aria-label",
        "Use one of the fixed DeepDRiD sample buttons",
      );
      elements.dropTitle.textContent = "Use a fixed DeepDRiD sample";
      elements.dropBrowse.textContent =
        "Choose Routine, Priority, Limited, or Retake below";
      elements.dropFormat.textContent = "Dataset-only safety scope";
    }
    const escalationReady = health.escalation?.release_enabled === true;
    const experimentalGemmaPriority =
      health.escalation?.profile ===
      "gemma-lora-free-generation-uncalibrated-experimental";
    state.gemmaEscalation = escalationReady && experimentalGemmaPriority;
    elements.modeKicker.innerHTML = kickerFor(MODE_COPY[state.productMode]);
    elements.adapterState.textContent = escalationReady
      ? experimentalGemmaPriority
        ? "Experimental uncalibrated Gemma priority · exact local adapter verified"
        : "Verified local priority artifact · release policy enabled"
      : "Priority artifact not loaded · fails closed to UNCERTAIN";
    elements.adapterState.classList.toggle("is-ready", escalationReady);
    if (!isReady) {
      elements.runtimeLabel.textContent = "Local model unavailable";
      elements.enginePill.textContent = "Runtime offline";
      if (!state.guidanceRunning) elements.sampleRow.hidden = false;
      return;
    }
    elements.runtimeLabel.textContent = isDemo
      ? "Labelled demo samples ready"
      : "Local model verified";
    elements.enginePill.textContent = isDemo
      ? "Demo mode"
      : health.mode === "specialist-local"
        ? state.gemmaEscalation
          ? "Quality specialist + Gemma LoRA · Local"
          : "Quality specialist · Local"
        : health.specialist_verified
        ? "Frozen hybrid · Local"
        : health.profile === "tuned-lora"
          ? "Tuned LoRA · Local"
          : "Untuned fallback · Local";
    // Live mode uses fixed, real DeepDRiD dataset images and lets the model
    // evaluate them. Demo mode keeps the clearly labelled canvas placeholders.
    if (!state.guidanceRunning) {
      elements.sampleRow.hidden = !state.sampleRowAvailable;
    }
  })
  .catch(() => {
    state.runtimeMode = "unavailable";
    state.sampleRowAvailable = true;
    elements.runtimeLabel.textContent = "Browser demo ready";
    elements.enginePill.textContent = "Browser demo";
  });

applyProductMode(state.productMode);
