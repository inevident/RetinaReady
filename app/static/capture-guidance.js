(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.RetinaCaptureGuidance = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const clamp = (value, minimum = 0, maximum = 100) =>
    Math.max(minimum, Math.min(maximum, value));

  function analyzeFrame(imageData, previousGray = null) {
    const { data, width, height } = imageData;
    const pixelCount = width * height;
    const gray = new Uint8Array(pixelCount);
    const mask = new Uint8Array(pixelCount);
    let retinalPixels = 0;
    let centralRetinalPixels = 0;
    let sumX = 0;
    let sumY = 0;
    let luminanceSum = 0;
    let darkPixels = 0;
    let glarePixels = 0;
    let motionSum = 0;

    for (let index = 0; index < pixelCount; index += 1) {
      const offset = index * 4;
      const red = data[offset];
      const green = data[offset + 1];
      const blue = data[offset + 2];
      const maximum = Math.max(red, green, blue);
      const minimum = Math.min(red, green, blue);
      const saturation = maximum ? (maximum - minimum) / maximum : 0;
      const luminance = Math.round(0.299 * red + 0.587 * green + 0.114 * blue);
      gray[index] = luminance;

      if (previousGray && previousGray.length === pixelCount) {
        motionSum += Math.abs(luminance - previousGray[index]);
      }

      // A deliberately simple technical-field mask for the replay controller.
      // It identifies warm, saturated retinal pixels; it does not identify
      // anatomy, lesions, or disease.
      const isRetinalField =
        red > 34 &&
        red > green * 1.04 &&
        red > blue * 1.03 &&
        saturation > 0.12;
      if (!isRetinalField) continue;

      mask[index] = 1;
      retinalPixels += 1;
      const x = index % width;
      const y = Math.floor(index / width);
      const normalizedX = (x - width / 2) / (width / 2);
      const normalizedY = (y - height / 2) / (height / 2);
      if (
        normalizedX * normalizedX + normalizedY * normalizedY <= 0.68 * 0.68
      ) {
        centralRetinalPixels += 1;
      }
      sumX += x;
      sumY += y;
      luminanceSum += luminance;
      if (luminance < 38) darkPixels += 1;
      if (luminance > 236 && saturation < 0.2) glarePixels += 1;
    }

    let gradientSum = 0;
    let gradientSamples = 0;
    for (let y = 1; y < height - 1; y += 2) {
      for (let x = 1; x < width - 1; x += 2) {
        const index = y * width + x;
        if (!mask[index]) continue;
        const horizontal = gray[index + 1] - gray[index - 1];
        const vertical = gray[index + width] - gray[index - width];
        gradientSum += Math.sqrt(horizontal * horizontal + vertical * vertical);
        gradientSamples += 1;
      }
    }

    const safeRetinalPixels = Math.max(1, retinalPixels);
    const centroidX = retinalPixels ? sumX / retinalPixels : width / 2;
    const centroidY = retinalPixels ? sumY / retinalPixels : height / 2;
    const offsetX = (centroidX - width / 2) / (width / 2);
    const offsetY = (centroidY - height / 2) / (height / 2);

    return {
      gray,
      metrics: {
        coverage: retinalPixels / pixelCount,
        centralFraction: centralRetinalPixels / safeRetinalPixels,
        centerOffset: Math.sqrt(offsetX * offsetX + offsetY * offsetY),
        offsetX,
        offsetY,
        luminance: luminanceSum / safeRetinalPixels,
        darkFraction: darkPixels / safeRetinalPixels,
        glareFraction: glarePixels / safeRetinalPixels,
        sharpness: gradientSamples ? gradientSum / gradientSamples : 0,
        motion: previousGray ? motionSum / pixelCount : 0,
      },
    };
  }

  function hasColorFundusField(metrics, reference) {
    const minimumCoverage = Math.max(0.08, reference.coverage * 0.35);
    const minimumCentralFraction = Math.max(
      0.5,
      reference.centralFraction * 0.65,
    );
    return (
      metrics.coverage >= minimumCoverage &&
      metrics.centralFraction >= minimumCentralFraction
    );
  }

  function scoreMetrics(metrics, reference) {
    const coverageRatio = reference.coverage
      ? metrics.coverage / reference.coverage
      : 0;
    const coverageScore = clamp(coverageRatio * 100);
    const centeringScore = clamp(100 - metrics.centerOffset * 240);
    const field = Math.round(coverageScore * 0.56 + centeringScore * 0.44);

    const sharpnessRatio = reference.sharpness
      ? metrics.sharpness / reference.sharpness
      : 0;
    const clarity = Math.round(clamp(sharpnessRatio * 108));

    const luminanceTolerance = Math.max(24, reference.luminance * 0.42);
    const exposurePenalty =
      Math.abs(metrics.luminance - reference.luminance) / luminanceTolerance;
    const excessGlare = Math.max(
      0,
      metrics.glareFraction - reference.glareFraction - 0.012,
    );
    const illumination = Math.round(
      clamp(100 - exposurePenalty * 80 - excessGlare * 850),
    );
    const stability = Math.round(clamp(100 - metrics.motion * 8.5));
    const overall = Math.round(
      field * 0.34 + clarity * 0.28 + illumination * 0.22 + stability * 0.16,
    );

    return { field, clarity, illumination, stability, overall };
  }

  function chooseInstruction(scores, metrics, reference) {
    if (!hasColorFundusField(metrics, reference)) {
      return {
        code: "NO_FIELD",
        label: "No color fundus field detected",
      };
    }
    if (scores.field < 73) {
      const horizontal = Math.abs(metrics.offsetX) >= Math.abs(metrics.offsetY);
      const direction = horizontal
        ? metrics.offsetX > 0
          ? "left"
          : "right"
        : metrics.offsetY > 0
          ? "up"
          : "down";
      return {
        code: "CENTER",
        label: `Move ${direction} to center the retinal field`,
      };
    }
    if (scores.stability < 62) {
      return { code: "STABILIZE", label: "Hold still" };
    }
    if (scores.clarity < 74) {
      return { code: "FOCUS", label: "Adjust focus" };
    }
    if (
      metrics.glareFraction > reference.glareFraction + 0.025 ||
      scores.illumination < 72
    ) {
      const tooBright = metrics.luminance > reference.luminance;
      return {
        code: tooBright ? "GLARE" : "LIGHT",
        label: tooBright ? "Reduce glare" : "Improve illumination",
      };
    }
    return { code: "HOLD", label: "Hold position" };
  }

  function createController(reference, options = {}) {
    const requiredStableFrames = Math.max(2, options.requiredStableFrames || 5);
    let stableFrames = 0;
    let lastCode = "";
    let pendingCode = "";
    let pendingTicks = 0;

    return {
      update(metrics) {
        const scores = scoreMetrics(metrics, reference);
        const candidate = chooseInstruction(scores, metrics, reference);
        const passing =
          hasColorFundusField(metrics, reference) &&
          scores.field >= 73 &&
          scores.clarity >= 74 &&
          scores.illumination >= 72 &&
          scores.stability >= 62;
        stableFrames = passing ? stableFrames + 1 : 0;

        if (candidate.code === pendingCode) pendingTicks += 1;
        else {
          pendingCode = candidate.code;
          pendingTicks = 1;
        }
        if (!lastCode || pendingTicks >= 2) lastCode = candidate.code;

        const instruction = lastCode === "NO_FIELD"
          ? { code: "NO_FIELD", label: "No color fundus field detected" }
          : lastCode === candidate.code
            ? candidate
            : chooseInstruction(
                {
                  ...scores,
                  field: lastCode === "CENTER" ? 0 : scores.field,
                  stability: lastCode === "STABILIZE" ? 0 : scores.stability,
                  clarity: lastCode === "FOCUS" ? 0 : scores.clarity,
                  illumination:
                    lastCode === "GLARE" || lastCode === "LIGHT"
                      ? 0
                      : scores.illumination,
                },
                metrics,
                reference,
              );

        const captureReady = stableFrames >= requiredStableFrames;
        return {
          scores,
          instruction: captureReady
            ? { code: "CAPTURE", label: "Quality stable · Capturing best frame" }
            : stableFrames > 0
              ? {
                  code: "STABILIZING",
                  label: "Hold position · Quality stabilizing",
                }
              : instruction,
          stableFrames,
          requiredStableFrames,
          captureReady,
        };
      },
      reset() {
        stableFrames = 0;
        lastCode = "";
        pendingCode = "";
        pendingTicks = 0;
      },
    };
  }

  return {
    analyzeFrame,
    hasColorFundusField,
    scoreMetrics,
    chooseInstruction,
    createController,
  };
});
