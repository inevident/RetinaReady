"use strict";

const assert = require("node:assert/strict");
const guidance = require("../static/capture-guidance.js");

const reference = {
  coverage: 0.45,
  centralFraction: 0.8,
  centerOffset: 0,
  offsetX: 0,
  offsetY: 0,
  luminance: 102,
  darkFraction: 0.01,
  glareFraction: 0.002,
  sharpness: 31,
  motion: 0,
};

const centered = { ...reference, motion: 1.2 };
const offCenter = {
  ...reference,
  coverage: 0.25,
  centerOffset: 0.36,
  offsetX: 0.36,
  motion: 0,
};
const blurred = { ...reference, sharpness: 10, motion: 0 };
const dark = { ...reference, luminance: 44, motion: 0 };
const noField = {
  ...reference,
  coverage: 0.08,
  centralFraction: 0.24,
  centerOffset: 0.05,
};
const warmFullFrame = {
  ...reference,
  coverage: 0.95,
  centralFraction: 0.37,
  centerOffset: 0,
};

assert.equal(guidance.hasColorFundusField(reference, reference), true);
assert.equal(guidance.hasColorFundusField(noField, reference), false);

assert.equal(guidance.chooseInstruction(
  guidance.scoreMetrics(offCenter, reference),
  offCenter,
  reference,
).code, "CENTER");

assert.equal(guidance.chooseInstruction(
  guidance.scoreMetrics(blurred, reference),
  blurred,
  reference,
).code, "FOCUS");

assert.equal(guidance.chooseInstruction(
  guidance.scoreMetrics(dark, reference),
  dark,
  reference,
).code, "LIGHT");

assert.equal(guidance.chooseInstruction(
  guidance.scoreMetrics(noField, reference),
  noField,
  reference,
).code, "NO_FIELD");

assert.equal(guidance.chooseInstruction(
  guidance.scoreMetrics(warmFullFrame, reference),
  warmFullFrame,
  reference,
).code, "NO_FIELD");

const controller = guidance.createController(reference);
let result;
for (let frame = 0; frame < 5; frame += 1) {
  result = controller.update(centered);
  assert.equal(result.captureTriggered, frame === 4);
}
assert.equal(result.captureReady, true);
assert.equal(result.instruction.code, "CAPTURE");
assert.equal(result.stableFrames, 5);
assert.equal(result.requiredStableFrames, 5);

result = controller.update(centered);
assert.equal(result.captureReady, true);
assert.equal(result.captureTriggered, false);

controller.reset();
result = controller.update(offCenter);
assert.equal(result.captureReady, false);
assert.equal(result.captureTriggered, false);
assert.equal(result.stableFrames, 0);
assert.equal(result.instruction.code, "CENTER");

controller.reset();
for (let frame = 0; frame < 5; frame += 1) {
  result = controller.update(centered);
  assert.equal(result.captureTriggered, frame === 4);
}

for (const outOfDomainFrame of [noField, warmFullFrame]) {
  controller.reset();
  for (let frame = 0; frame < 6; frame += 1) {
    result = controller.update(outOfDomainFrame);
  }
  assert.equal(result.captureReady, false);
  assert.equal(result.captureTriggered, false);
  assert.equal(result.stableFrames, 0);
  assert.equal(result.instruction.code, "NO_FIELD");
}

console.log("capture-guidance tests passed");
