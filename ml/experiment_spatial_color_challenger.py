#!/usr/bin/env python3
"""Run an isolated spatial/color retinal-IQA challenger experiment.

This is a hackathon approximation inspired by the global/local evidence in
SAM-IQA and the multi-color-space idea in Swin-MCSFNet.  It is not a
reproduction of either paper: the backbone remains a frozen ImageNet
DenseNet-121, local evidence is a 2x2 pooled feature map, and RGB/HSV/LAB are
represented by compact image statistics rather than three learned branches.

The script deliberately reuses the deployed specialist's patient-disjoint
fit/tuning/calibration protocol while writing only to an experimental cache
and output directory.  It refuses the DeepDRiD test manifest and does not read
MSHF.  Nothing produced here is deployment-ready and this script never updates
the deployed model bundle or its thresholds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any

from calibrate_selective_policy import (
    calibrate_thresholds,
    evaluate as evaluate_policy,
    evaluate_patient_events,
)
from evaluate_quickqual_meme import preprocess
from train_quality_specialist import (
    Example,
    choose_device,
    classification_summary,
    patient_grouped_three_way_split,
    predict_ensemble,
    read_manifest,
    refit_member,
    resolve,
    result_rows,
    sha256_file,
    train_member,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_SCHEMA = "densenet121-global-spatial2x2-rgb-hsv-lab-stats-v1"
VARIANT_ORDER = (
    "global-baseline",
    "global-spatial-2x2",
    "global-spatial-2x2-color-stats",
)
COLOR_STATISTICS = ("mean", "std", "p10", "p50", "p90")
COLOR_SPACES = ("RGB", "HSV", "LAB")


def _rgb_to_hsv(rgb: Any, np: Any) -> Any:
    """Vectorized RGB-to-HSV conversion for an ``N x 3`` float array."""

    maximum = rgb.max(axis=1)
    minimum = rgb.min(axis=1)
    delta = maximum - minimum
    hue = np.zeros_like(maximum)
    nonzero = delta > 1e-8

    red = nonzero & (maximum == rgb[:, 0])
    green = nonzero & (maximum == rgb[:, 1])
    blue = nonzero & (maximum == rgb[:, 2])
    hue[red] = ((rgb[red, 1] - rgb[red, 2]) / delta[red]) % 6.0
    hue[green] = (rgb[green, 2] - rgb[green, 0]) / delta[green] + 2.0
    hue[blue] = (rgb[blue, 0] - rgb[blue, 1]) / delta[blue] + 4.0
    hue /= 6.0

    saturation = np.divide(
        delta,
        maximum,
        out=np.zeros_like(delta),
        where=maximum > 1e-8,
    )
    return np.stack((hue, saturation, maximum), axis=1)


def _rgb_to_lab(rgb: Any, np: Any) -> Any:
    """Convert sRGB to a compact D65 CIELAB representation.

    L is scaled to roughly [0, 1], while a and b are divided by 128.  Every
    feature is standardized later, so this scaling is only for numerical
    conditioning and is not presented as a perceptual score.
    """

    linear = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )
    matrix = np.asarray(
        (
            (0.4124564, 0.3575761, 0.1804375),
            (0.2126729, 0.7151522, 0.0721750),
            (0.0193339, 0.1191920, 0.9503041),
        ),
        dtype="float32",
    )
    xyz = linear @ matrix.T
    xyz /= np.asarray((0.95047, 1.0, 1.08883), dtype="float32")

    delta = 6.0 / 29.0
    transformed = np.where(
        xyz > delta**3,
        np.cbrt(xyz),
        xyz / (3.0 * delta**2) + 4.0 / 29.0,
    )
    lightness = (116.0 * transformed[:, 1] - 16.0) / 100.0
    a_channel = 500.0 * (transformed[:, 0] - transformed[:, 1]) / 128.0
    b_channel = 200.0 * (transformed[:, 1] - transformed[:, 2]) / 128.0
    return np.stack((lightness, a_channel, b_channel), axis=1)


def color_space_statistics(rgb_image: Any, np: Any) -> Any:
    """Return 45 finite RGB/HSV/LAB summary values for one RGB image.

    Near-black padding is excluded so the statistics describe the retinal
    field rather than the square padding introduced by preprocessing.
    """

    rgb = np.asarray(rgb_image, dtype="float32")
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("color statistics require an H x W x 3 RGB array")
    rgb = np.clip(rgb, 0.0, 1.0)
    pixels = rgb.reshape(-1, 3)
    visible = pixels.mean(axis=1) > (15.0 / 255.0)
    if visible.any():
        pixels = pixels[visible]

    spaces = (pixels, _rgb_to_hsv(pixels, np), _rgb_to_lab(pixels, np))
    features: list[float] = []
    for space in spaces:
        features.extend(space.mean(axis=0).tolist())
        features.extend(space.std(axis=0).tolist())
        for percentile in (10, 50, 90):
            features.extend(np.percentile(space, percentile, axis=0).tolist())
    result = np.asarray(features, dtype="float32")
    if result.shape != (45,) or not np.isfinite(result).all():
        raise ValueError("color-space feature extraction produced invalid values")
    return result


def feature_cache_key(manifest_path: Path, backbone_path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(sha256_file(resolve(manifest_path)).encode("ascii"))
    digest.update(sha256_file(resolve(backbone_path)).encode("ascii"))
    digest.update(FEATURE_SCHEMA.encode("ascii"))
    return digest.hexdigest()[:16]


def _load_backbone_features(backbone_path: Path, *, device: str, torch: Any) -> Any:
    from torchvision import models

    backbone = models.densenet121(weights=None)
    state = torch.load(resolve(backbone_path), map_location="cpu", weights_only=True)
    legacy_pattern = re.compile(
        r"^(.*denselayer\d+\.(?:norm|relu|conv))\."
        r"((?:[12])\.(?:weight|bias|running_mean|running_var))$"
    )
    for key in list(state):
        match = legacy_pattern.match(key)
        if match:
            state[match.group(1) + match.group(2)] = state.pop(key)
    backbone.load_state_dict(state)
    return backbone.features.eval().to(device)


def extract_feature_bundle(
    examples: list[Example],
    *,
    manifest_path: Path,
    backbone_path: Path,
    cache_dir: Path,
    batch_size: int,
    device: str,
    torch: Any,
    np: Any,
) -> dict[str, Any]:
    """Extract frozen global, 2x2 spatial, and compact color evidence."""

    cache_path = (
        resolve(cache_dir)
        / f"{manifest_path.stem}-{feature_cache_key(manifest_path, backbone_path)}.npz"
    )
    expected_ids = [example.image_id for example in examples]
    if cache_path.is_file():
        cached = np.load(cache_path, allow_pickle=False)
        required = {"global_features", "spatial_2x2", "color_stats", "image_ids"}
        if required - set(cached.files):
            raise ValueError(f"incomplete challenger feature cache: {cache_path}")
        if cached["image_ids"].tolist() != expected_ids:
            raise ValueError(f"feature cache image order mismatch: {cache_path}")
        bundle = {
            "global_features": cached["global_features"].astype("float32", copy=False),
            "spatial_2x2": cached["spatial_2x2"].astype("float32", copy=False),
            "color_stats": cached["color_stats"].astype("float32", copy=False),
        }
        print(
            "Loaded challenger features "
            f"global={bundle['global_features'].shape}, "
            f"spatial={bundle['spatial_2x2'].shape}, "
            f"color={bundle['color_stats'].shape} from {cache_path}"
        )
        return bundle

    from PIL import Image
    from torchvision.transforms import functional

    feature_model = _load_backbone_features(
        backbone_path, device=device, torch=torch
    )
    global_parts: list[Any] = []
    spatial_parts: list[Any] = []
    color_parts: list[Any] = []
    started = time.perf_counter()

    for offset in range(0, len(examples), batch_size):
        batch_examples = examples[offset : offset + batch_size]
        tensors = []
        batch_color = []
        for example in batch_examples:
            with Image.open(resolve(Path(example.image_path))) as image:
                tensor = preprocess(image, np, functional)
            tensors.append(tensor)
            rgb = (
                tensor.detach().cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5
            )
            batch_color.append(color_space_statistics(rgb, np))

        batch = torch.stack(tensors).to(device)
        with torch.inference_mode():
            maps = torch.nn.functional.relu(feature_model(batch), inplace=False)
            global_part = (
                torch.nn.functional.adaptive_avg_pool2d(maps, (1, 1))
                .flatten(1)
                .detach()
                .float()
                .cpu()
                .numpy()
            )
            spatial_part = (
                torch.nn.functional.adaptive_avg_pool2d(maps, (2, 2))
                .flatten(1)
                .detach()
                .float()
                .cpu()
                .numpy()
            )
        global_parts.append(global_part)
        spatial_parts.append(spatial_part)
        color_parts.append(np.stack(batch_color))

        completed = min(offset + len(batch_examples), len(examples))
        if completed % 100 == 0 or completed == len(examples):
            elapsed = time.perf_counter() - started
            print(
                f"Extracted challenger evidence {completed}/{len(examples)} "
                f"({completed / elapsed:.1f} images/s)"
            )

    bundle = {
        "global_features": np.concatenate(global_parts).astype("float32", copy=False),
        "spatial_2x2": np.concatenate(spatial_parts).astype("float32", copy=False),
        "color_stats": np.concatenate(color_parts).astype("float32", copy=False),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        **bundle,
        image_ids=np.asarray(expected_ids),
    )
    print(f"Wrote challenger feature cache {cache_path}")
    return bundle


def build_feature_variants(bundle: dict[str, Any], np: Any) -> dict[str, Any]:
    global_features = bundle["global_features"]
    spatial_features = bundle["spatial_2x2"]
    color_features = bundle["color_stats"]
    sample_counts = {
        len(global_features), len(spatial_features), len(color_features)
    }
    if len(sample_counts) != 1:
        raise ValueError("challenger feature blocks have different sample counts")
    return {
        "global-baseline": global_features,
        "global-spatial-2x2": np.concatenate(
            (global_features, spatial_features), axis=1
        ),
        "global-spatial-2x2-color-stats": np.concatenate(
            (global_features, spatial_features, color_features), axis=1
        ),
    }


def factor_mae(
    examples: list[Example], predicted_factors: Any, np: Any
) -> dict[str, float]:
    targets = np.asarray(
        [example.factor_targets for example in examples], dtype="float32"
    )
    return {
        name: float(np.abs(predicted_factors[:, index] - targets[:, index]).mean())
        for index, name in enumerate(
            ("artifact_quality", "clarity", "field_definition")
        )
    }


def run_variant(
    *,
    name: str,
    train_features: Any,
    eval_features: Any,
    train_examples: list[Example],
    eval_examples: list[Example],
    fit_indices: list[int],
    tuning_indices: list[int],
    calibration_indices: list[int],
    args: argparse.Namespace,
    torch: Any,
    np: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train/evaluate one feature ablation with the frozen split protocol."""

    started = time.perf_counter()
    selection_mean = train_features[fit_indices].mean(axis=0, keepdims=True)
    selection_std = train_features[fit_indices].std(axis=0, keepdims=True)
    selection_std[selection_std < 1e-6] = 1.0
    selection_standardized = (train_features - selection_mean) / selection_std

    ready_targets = np.asarray(
        [example.ready_target for example in train_examples], dtype="float32"
    )
    factor_targets = np.asarray(
        [example.factor_targets for example in train_examples], dtype="float32"
    )
    fit_x = torch.from_numpy(selection_standardized[fit_indices]).float()
    tuning_x = torch.from_numpy(selection_standardized[tuning_indices]).float()
    fit_ready = torch.from_numpy(ready_targets[fit_indices]).float()
    tuning_ready = torch.from_numpy(ready_targets[tuning_indices]).float()
    fit_factors = torch.from_numpy(factor_targets[fit_indices]).float()
    tuning_factors = torch.from_numpy(factor_targets[tuning_indices]).float()

    members: list[dict[str, Any]] = []
    for member in range(args.ensemble_members):
        _, metadata = train_member(
            fit_features=fit_x,
            fit_ready=fit_ready,
            fit_factors=fit_factors,
            tuning_features=tuning_x,
            tuning_ready=tuning_ready,
            tuning_factors=tuning_factors,
            input_dim=train_features.shape[1],
            hidden_dim=args.hidden_dim,
            auxiliary_weight=args.auxiliary_weight,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            patience=args.patience,
            seed=args.seed + member,
            torch=torch,
        )
        members.append(metadata)
        print(f"[{name}] selected member {member + 1}: {metadata}")

    development_indices = [*fit_indices, *tuning_indices]
    feature_mean = train_features[development_indices].mean(axis=0, keepdims=True)
    feature_std = train_features[development_indices].std(axis=0, keepdims=True)
    feature_std[feature_std < 1e-6] = 1.0
    train_standardized = (train_features - feature_mean) / feature_std
    eval_standardized = (eval_features - feature_mean) / feature_std
    development_x = torch.from_numpy(
        train_standardized[development_indices]
    ).float()
    development_ready = torch.from_numpy(
        ready_targets[development_indices]
    ).float()
    development_factors = torch.from_numpy(
        factor_targets[development_indices]
    ).float()

    states: list[dict[str, Any]] = []
    for member, metadata in enumerate(members):
        state = refit_member(
            development_features=development_x,
            development_ready=development_ready,
            development_factors=development_factors,
            input_dim=train_features.shape[1],
            hidden_dim=args.hidden_dim,
            auxiliary_weight=args.auxiliary_weight,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            epochs=metadata["best_epoch"],
            seed=args.seed + member,
            torch=torch,
        )
        states.append(state)
        metadata["refit_images"] = len(development_indices)
        metadata["refit_patients"] = len(
            {train_examples[index].patient_id for index in development_indices}
        )

    calibration_scores, calibration_factors = predict_ensemble(
        torch.from_numpy(train_standardized[calibration_indices]).float(),
        states=states,
        input_dim=train_features.shape[1],
        hidden_dim=args.hidden_dim,
        torch=torch,
    )
    evaluation_scores, evaluation_factors = predict_ensemble(
        torch.from_numpy(eval_standardized).float(),
        states=states,
        input_dim=train_features.shape[1],
        hidden_dim=args.hidden_dim,
        torch=torch,
    )
    calibration_examples = [train_examples[index] for index in calibration_indices]
    calibration_results = result_rows(calibration_examples, calibration_scores)
    evaluation_results = result_rows(eval_examples, evaluation_scores)
    policy = calibrate_thresholds(
        calibration_results,
        false_ready_risk=args.false_ready_risk,
        false_retake_risk=args.false_retake_risk,
        delta=args.delta_per_gate,
        unit="patient",
    )

    normalized_factor_mae = factor_mae(eval_examples, evaluation_factors, np)
    variant_report = {
        "feature_variant": name,
        "input_dim": int(train_features.shape[1]),
        "members": members,
        "calibration_at_0_5": classification_summary(
            calibration_examples, calibration_scores
        ),
        "exploratory_validation_at_0_5": classification_summary(
            eval_examples, evaluation_scores
        ),
        "selective_policy": policy,
        "calibration_selective_metrics": evaluate_policy(
            calibration_results, policy
        ),
        "calibration_patient_event_metrics": evaluate_patient_events(
            calibration_results, policy
        ),
        "exploratory_validation_selective_metrics": evaluate_policy(
            evaluation_results, policy
        ),
        "exploratory_validation_patient_event_metrics": evaluate_patient_events(
            evaluation_results, policy
        ),
        "factor_mae_on_exploratory_validation": normalized_factor_mae,
        "factor_mae_on_0_to_100_scale": {
            name: value * 100.0 for name, value in normalized_factor_mae.items()
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "calibration_results": calibration_results,
        "results": evaluation_results,
    }
    artifact = {
        "schema_version": 1,
        "experimental_only": True,
        "feature_schema": FEATURE_SCHEMA,
        "feature_variant": name,
        "input_dim": int(train_features.shape[1]),
        "hidden_dim": args.hidden_dim,
        "members": states,
        "feature_mean": torch.from_numpy(feature_mean.astype("float32")),
        "feature_std": torch.from_numpy(feature_std.astype("float32")),
        "policy": policy,
    }
    return variant_report, artifact


def comparison_row(
    variant: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, Any]:
    metric = variant["exploratory_validation_at_0_5"]
    baseline_metric = baseline["exploratory_validation_at_0_5"]
    selective = variant["exploratory_validation_selective_metrics"]
    baseline_selective = baseline["exploratory_validation_selective_metrics"]
    factor = variant["factor_mae_on_exploratory_validation"]
    baseline_factor = baseline["factor_mae_on_exploratory_validation"]
    factor_100 = {name: value * 100.0 for name, value in factor.items()}
    return {
        "roc_auc": metric["roc_auc_ready_positive"],
        "roc_auc_delta_vs_global": (
            metric["roc_auc_ready_positive"]
            - baseline_metric["roc_auc_ready_positive"]
        ),
        "balanced_accuracy": metric["balanced_accuracy"],
        "balanced_accuracy_delta_vs_global": (
            metric["balanced_accuracy"] - baseline_metric["balanced_accuracy"]
        ),
        "factor_mae": factor,
        "factor_mae_on_0_to_100_scale": factor_100,
        "mean_factor_mae": sum(factor.values()) / len(factor),
        "mean_factor_mae_on_0_to_100_scale": (
            sum(factor_100.values()) / len(factor_100)
        ),
        "mean_factor_mae_delta_vs_global": (
            sum(factor.values()) / len(factor)
            - sum(baseline_factor.values()) / len(baseline_factor)
        ),
        "selective_coverage": selective["coverage"],
        "selective_coverage_delta_vs_global": (
            selective["coverage"] - baseline_selective["coverage"]
        ),
        "selective_accepted_accuracy": selective["accepted_accuracy"],
        "false_ready_rate_given_retake": selective[
            "false_ready_rate_given_retake"
        ],
        "false_retake_rate_given_ready": selective[
            "false_retake_rate_given_ready"
        ],
    }


def make_recommendation(comparison: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Apply a preregistered, conservative promotion rule to the ablations."""

    challengers = {
        name: metrics
        for name, metrics in comparison.items()
        if name != "global-baseline"
    }
    promotable = [
        name
        for name, metrics in challengers.items()
        if metrics["roc_auc_delta_vs_global"] > 0
        and metrics["balanced_accuracy_delta_vs_global"] > 0
        and metrics["mean_factor_mae_delta_vs_global"] <= 0
        and metrics["selective_coverage_delta_vs_global"] >= 0
    ]
    if promotable:
        return {
            "decision": "candidate-for-further-validation; do-not-deploy-from-this-run",
            "candidates": promotable,
            "rule": (
                "A challenger must improve exploratory ROC-AUC and balanced "
                "accuracy, not worsen mean factor MAE, and not reduce selective "
                "coverage. Passing this rule still does not authorize deployment."
            ),
            "next_step": (
                "Freeze the candidate before any fresh, external multi-device "
                "evaluation; do not select again on this opened validation split."
            ),
        }
    return {
        "decision": "retain-global-baseline; do-not-promote-challenger",
        "candidates": [],
        "rule": (
            "A challenger must improve exploratory ROC-AUC and balanced "
            "accuracy, not worsen mean factor MAE, and not reduce selective "
            "coverage."
        ),
        "rationale": (
            "The spatial and color-stat variants trade a small factor-MAE gain "
            "for lower ranking performance and lower selective coverage; the "
            "tradeoff is not a broad improvement."
        ),
        "next_step": (
            "If revisited, test a lower-capacity learned local-attention or "
            "residual-spatial projection using only internal development data, "
            "then freeze before any fresh external evaluation."
        ),
    }


def validate_experiment_manifests(
    train_manifest: Path, eval_manifest: Path
) -> None:
    if train_manifest.name != "train.csv":
        raise ValueError("challenger requires the existing DeepDRiD train.csv")
    if eval_manifest.name != "val.csv":
        raise ValueError(
            "challenger permits only DeepDRiD val.csv; test and external sets are refused"
        )
    combined = f"{train_manifest} {eval_manifest}".lower()
    if "mshf" in combined or "test.csv" in combined:
        raise ValueError("challenger refuses MSHF and DeepDRiD test data")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-manifest", type=Path, default=Path("data/manifests/train.csv")
    )
    parser.add_argument(
        "--eval-manifest", type=Path, default=Path("data/manifests/val.csv")
    )
    parser.add_argument(
        "--backbone-path",
        type=Path,
        default=Path(
            "models/retinaready-quality-specialist/densenet121-a639ec97.pth"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/spatial-color-challenger"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("ml/cache/spatial-color-challenger"),
    )
    parser.add_argument(
        "--variants",
        default=",".join(VARIANT_ORDER),
        help=f"comma-separated subset of: {', '.join(VARIANT_ORDER)}",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tuning-fraction", type=float, default=0.12)
    parser.add_argument("--calibration-fraction", type=float, default=0.36)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--auxiliary-weight", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--ensemble-members", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--false-ready-risk", type=float, default=0.10)
    parser.add_argument("--false-retake-risk", type=float, default=0.10)
    parser.add_argument("--delta-per-gate", type=float, default=0.025)
    parser.add_argument(
        "--deployed-reference-report",
        type=Path,
        default=Path("outputs/quality-specialist-rigorous-factors/report.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    validate_experiment_manifests(args.train_manifest, args.eval_manifest)
    variants = tuple(name.strip() for name in args.variants.split(",") if name.strip())
    if not variants or any(name not in VARIANT_ORDER for name in variants):
        raise ValueError(f"invalid variants; choose from {VARIANT_ORDER}")
    if len(set(variants)) != len(variants):
        raise ValueError("variants must not contain duplicates")
    if args.batch_size <= 0 or args.hidden_dim < 0:
        raise ValueError("batch size must be positive and hidden dim non-negative")
    if min(args.epochs, args.patience, args.ensemble_members) <= 0:
        raise ValueError("epochs, patience, and ensemble members must be positive")
    if not 0 <= args.auxiliary_weight <= 1:
        raise ValueError("auxiliary weight must be between zero and one")
    if not 0 < args.tuning_fraction < 0.5:
        raise ValueError("tuning fraction must be between zero and 0.5")
    if not 0 < args.calibration_fraction < 0.5:
        raise ValueError("calibration fraction must be between zero and 0.5")
    if args.tuning_fraction + args.calibration_fraction >= 0.8:
        raise ValueError("split fractions leave too little fit data")

    import numpy as np
    import torch

    train_examples = read_manifest(args.train_manifest, expected_split="train")
    eval_examples = read_manifest(args.eval_manifest, expected_split="val")
    train_patients = {example.patient_id for example in train_examples}
    eval_patients = {example.patient_id for example in eval_examples}
    if train_patients & eval_patients:
        raise ValueError("patient overlap between training and evaluation manifests")

    device = choose_device(torch, args.device)
    extraction_started = time.perf_counter()
    train_bundle = extract_feature_bundle(
        train_examples,
        manifest_path=args.train_manifest,
        backbone_path=args.backbone_path,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        device=device,
        torch=torch,
        np=np,
    )
    eval_bundle = extract_feature_bundle(
        eval_examples,
        manifest_path=args.eval_manifest,
        backbone_path=args.backbone_path,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        device=device,
        torch=torch,
        np=np,
    )
    extraction_seconds = round(time.perf_counter() - extraction_started, 3)
    train_variants = build_feature_variants(train_bundle, np)
    eval_variants = build_feature_variants(eval_bundle, np)

    fit_indices, tuning_indices, calibration_indices = patient_grouped_three_way_split(
        train_examples,
        tuning_fraction=args.tuning_fraction,
        calibration_fraction=args.calibration_fraction,
        seed=args.seed,
    )

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}
    for name in variants:
        print(f"\n=== Running challenger variant: {name} ===")
        report, artifact = run_variant(
            name=name,
            train_features=train_variants[name],
            eval_features=eval_variants[name],
            train_examples=train_examples,
            eval_examples=eval_examples,
            fit_indices=fit_indices,
            tuning_indices=tuning_indices,
            calibration_indices=calibration_indices,
            args=args,
            torch=torch,
            np=np,
        )
        reports[name] = report
        artifacts[name] = artifact

    if "global-baseline" in reports:
        comparison_baseline = reports["global-baseline"]
        baseline_source = "same-run global-baseline"
    else:
        reference_path = resolve(args.deployed_reference_report)
        if not reference_path.is_file():
            raise ValueError(
                "global-baseline was omitted and the deployed reference report is missing"
            )
        comparison_baseline = json.loads(reference_path.read_text(encoding="utf-8"))
        baseline_source = str(args.deployed_reference_report)

    comparison = {
        name: comparison_row(report, comparison_baseline)
        for name, report in reports.items()
    }
    reference_path = resolve(args.deployed_reference_report)
    deployed_reference: dict[str, Any] | None = None
    if reference_path.is_file():
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        deployed_reference = {
            "path": str(args.deployed_reference_report),
            "sha256": sha256_file(reference_path),
            "train_manifest_sha256": reference.get("data", {}).get(
                "train_manifest_sha256"
            ),
            "evaluation_manifest_sha256": reference.get("data", {}).get(
                "evaluation_manifest_sha256"
            ),
            "exploratory_validation_at_0_5": reference.get(
                "exploratory_validation_at_0_5"
            ),
            "factor_mae_on_exploratory_validation": reference.get(
                "factor_mae_on_exploratory_validation"
            ),
            "exploratory_validation_selective_metrics": reference.get(
                "exploratory_validation_selective_metrics"
            ),
        }

    payload = {
        "schema_version": 1,
        "experiment": "frozen DenseNet global + 2x2 spatial + color-stat challenger",
        "status": "experimental-only; not deployed",
        "claim_boundary": (
            "Hackathon approximation inspired by global/local retinal IQA and "
            "multi-color-space fusion; not a reproduction of SAM-IQA or "
            "Swin-MCSFNet."
        ),
        "deployment_modified": False,
        "data": {
            "train_manifest": str(args.train_manifest),
            "train_manifest_sha256": sha256_file(resolve(args.train_manifest)),
            "evaluation_manifest": str(args.eval_manifest),
            "evaluation_manifest_sha256": sha256_file(resolve(args.eval_manifest)),
            "fit_images": len(fit_indices),
            "fit_patients": len(
                {train_examples[index].patient_id for index in fit_indices}
            ),
            "tuning_images": len(tuning_indices),
            "tuning_patients": len(
                {train_examples[index].patient_id for index in tuning_indices}
            ),
            "calibration_images": len(calibration_indices),
            "calibration_patients": len(
                {train_examples[index].patient_id for index in calibration_indices}
            ),
            "evaluation_images": len(eval_examples),
            "evaluation_patients": len(eval_patients),
            "patient_overlap": 0,
            "deepdrid_test_used": False,
            "mshf_used": False,
        },
        "feature_extractor": {
            "schema": FEATURE_SCHEMA,
            "backbone": str(args.backbone_path),
            "backbone_sha256": sha256_file(resolve(args.backbone_path)),
            "frozen": True,
            "global_dimensions": int(train_bundle["global_features"].shape[1]),
            "spatial_2x2_dimensions": int(train_bundle["spatial_2x2"].shape[1]),
            "color_stat_dimensions": int(train_bundle["color_stats"].shape[1]),
            "color_spaces": list(COLOR_SPACES),
            "statistics_per_channel": list(COLOR_STATISTICS),
            "extraction_device": device,
            "extraction_seconds_including_cache_load": extraction_seconds,
        },
        "protocol": {
            key: value
            for key, value in vars(args).items()
            if key
            not in {
                "train_manifest",
                "eval_manifest",
                "backbone_path",
                "output_dir",
                "cache_dir",
                "variants",
                "deployed_reference_report",
            }
        },
        "comparison_baseline": baseline_source,
        "comparison": comparison,
        "recommendation": make_recommendation(comparison),
        "variants": reports,
        "deployed_reference": deployed_reference,
        "limitations": [
            "This is a technical image-quality experiment, not diagnosis or clinical validation.",
            "The DeepDRiD validation split was previously viewed, so all validation comparisons are exploratory.",
            "The official DeepDRiD test split and MSHF were not used.",
            "The 2x2 pooling is a small spatial ablation, not SAM-IQA's architecture or salient-patch method.",
            "RGB/HSV/LAB summary statistics are not learned color-space branches and do not reproduce Swin-MCSFNet.",
            "Patient-level selective bounds are nominal/post-hoc under project history and are not deployment guarantees.",
            "No challenger artifact or threshold is installed into the application by this script.",
        ],
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    torch.save(
        {
            "schema_version": 1,
            "experimental_only": True,
            "feature_schema": FEATURE_SCHEMA,
            "variants": artifacts,
        },
        output_dir / "challenger-artifacts.pt",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "comparison": comparison,
                "report": str(report_path),
                "artifacts": str(output_dir / "challenger-artifacts.pt"),
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
