"""Evaluate Joint CBM Class Accuracy and SL-CBM Top-K Concept Accuracy.

For each image, K is its number of positive ground-truth concepts. Concept
Accuracy is the total number of positive labels among the predicted Top-K
concepts divided by the total number of positive labels across the dataset.
Class Accuracy is Top-1 accuracy. Both accuracies are reported as percentages.
"""

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from data.dataload import load_data
from model.cbm import JointCBM


PROJECT_DIR = Path(__file__).resolve().parent


def topk_concept_counts(concept_scores, concept_labels):
    """Count Top-K positive concept hits and ground-truth positive concepts."""
    if concept_scores.ndim != 2:
        raise ValueError(
            f"concept_scores must be [B, C], got {tuple(concept_scores.shape)}"
        )
    if concept_scores.shape != concept_labels.shape:
        raise ValueError(
            f"Shape mismatch: {tuple(concept_scores.shape)} vs {tuple(concept_labels.shape)}"
        )

    positive_labels = concept_labels > 0.5
    correct_count = 0
    total_count = 0

    for sample_scores, sample_labels in zip(concept_scores, positive_labels):
        k = int(sample_labels.sum().item())
        if k == 0:
            continue

        topk_indices = torch.topk(sample_scores, k=k, dim=-1).indices
        correct_count += int(sample_labels[topk_indices].sum().item())
        total_count += k

    return correct_count, total_count


def load_model(checkpoint_path, device):
    """Load either best_model_<seed>.pth or the training state-dict checkpoint."""
    # Existing best-model files contain a complete JointCBM module.
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    training_args = {}

    if isinstance(checkpoint, nn.Module):
        model = checkpoint
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        training_args = checkpoint.get("args", {})
        if isinstance(training_args, argparse.Namespace):
            training_args = vars(training_args)
        if not isinstance(training_args, dict) or "backbone" not in training_args:
            raise ValueError("State-dict checkpoints must include args.backbone")

        # All weights come from the checkpoint, so no pretrained download is needed.
        model = JointCBM(
            backbone_name=training_args["backbone"],
            pretrained=False,
            num_concepts=112,
            num_classes=200,
            concept_activation=training_args.get("concept_activation", "none"),
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    else:
        raise ValueError(
            "Expected a complete model or a checkpoint containing model_state_dict and args"
        )

    model = model.to(device)
    model.eval()
    return model, training_args


@torch.inference_mode()
def evaluate(model, loader, device):
    """Accumulate dataset-level counts without averaging batch accuracies."""
    model.eval()
    class_correct = 0
    sample_count = 0
    concept_correct = 0
    concept_total = 0

    for images, class_labels, concept_labels in loader:
        images = images.to(device, non_blocking=True)
        class_labels = class_labels.to(device, non_blocking=True)

        # The existing CUB loader collates labels into 112 tensors shaped [B].
        if isinstance(concept_labels, (list, tuple)):
            concept_labels = torch.stack(concept_labels, dim=1)
        concept_labels = concept_labels.to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )

        outputs = model(images)
        if not isinstance(outputs, (list, tuple)) or len(outputs) != 113:
            raise ValueError("Expected JointCBM outputs [class_logits, *112 concept_logits]")

        class_logits = outputs[0]
        concept_logits = torch.cat(outputs[1:], dim=1)
        if class_logits.shape != (images.size(0), 200):
            raise ValueError("Expected class logits shaped [B, 200]")

        predictions = class_logits.argmax(dim=1)
        class_correct += int((predictions == class_labels).sum().item())
        sample_count += images.size(0)

        # Rank raw logits directly; sigmoid saturation could introduce extra ties.
        correct, total = topk_concept_counts(concept_logits, concept_labels)
        concept_correct += correct
        concept_total += total

    if sample_count == 0:
        raise ValueError("Cannot evaluate an empty dataset")

    return {
        "samples": sample_count,
        "class_correct": class_correct,
        "class_accuracy": 100.0 * class_correct / sample_count,
        "concept_correct": concept_correct,
        "concept_total": concept_total,
        # A dataset with no positive labels has an undefined Concept Accuracy.
        "concept_accuracy": 100.0 * concept_correct / concept_total if concept_total else None,
        "concept_metric": "topk_positive_hits_over_total_positive_labels",
        "accuracy_unit": "percent",
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", "--model",
        required=True,
        help="Path to best_model_<seed>.pth or last_checkpoint.pth",
    )
    parser.add_argument(
        "--data-dir",
        default=str(PROJECT_DIR.parent / "cbm/data/cub_processed/class_attr_data_10"),
        help="Directory containing the evaluation annotation pickle",
    )
    parser.add_argument(
        "--image-dir",
        default=str(PROJECT_DIR.parent / "cbm/data/CUB_200_2011/images"),
        help="Root directory containing the CUB image class folders",
    )
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--normalization",
        choices=["imagenet", "upstream"],
        default=None,
        help="Default: checkpoint training setting, or imagenet for complete-model files",
    )
    parser.add_argument("--output", help="Optional JSON file for evaluation results")

    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch-size must be positive and num-workers must be nonnegative")
    return args


def main(argv=None):
    args = parse_args(argv)
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    model, training_args = load_model(checkpoint_path, device)
    normalization = args.normalization or training_args.get("normalization", "imagenet")
    data_dir = Path(args.data_dir).expanduser().resolve()
    image_dir = Path(args.image_dir).expanduser().resolve()

    loader = load_data(
        [str(data_dir / f"{args.split}.pkl")],
        use_attr=True,
        no_img=False,
        batch_size=args.batch_size,
        image_dir=str(image_dir),
        resol=224,
        normalization=normalization,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    print(f"Checkpoint: {checkpoint_path}", flush=True)
    print(f"Evaluating {args.split} on {device}; normalization={normalization}", flush=True)
    results = evaluate(model, loader, device)
    results.update({
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "data_dir": str(data_dir),
        "image_dir": str(image_dir),
        "normalization": normalization,
    })

    print(f"Samples: {results['samples']}")
    print(
        f"Class accuracy (Top-1): {results['class_accuracy']:.4f}% "
        f"({results['class_correct']}/{results['samples']})"
    )
    if results["concept_accuracy"] is None:
        print("Concept accuracy (Top-K): N/A (no positive concept labels)")
    else:
        print(
            f"Concept accuracy (Top-K): {results['concept_accuracy']:.4f}% "
            f"({results['concept_correct']}/{results['concept_total']})"
        )

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(results, file, indent=2)
        print(f"Saved results: {output_path}")

    return results


if __name__ == "__main__":
    main()
