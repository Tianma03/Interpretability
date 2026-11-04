"""Train the ResNet18 Joint CBM on the original binary CUB112 annotations.

Reference: https://github.com/yewsiang/ConceptBottleneck/blob/master/CUB/train.py

The objective, optimizers, learning-rate schedule, model selection, early stopping,
and epoch summary follow the original Joint training code. ResNet18 uses 224-pixel
inputs and has no Inception auxiliary branch. ImageNet normalization is used for
the ResNet backbones by default; --normalization upstream selects the original
dataset.py normalization instead. No test data is used for model selection.
Defaults follow the upstream Joint0.01 seed-1 recipe: SGD, lr=0.001,
weight_decay=0.0004, batch_size=64, and scheduler_step=1000.

Example:
    python train.py --data-dir /path/to/class_attr_data_10 \
        --image-dir /path/to/CUB_200_2011/images --seed 1

Dataset loading, transforms, and imbalance weights are provided by data.dataload.
"""

import argparse
import json
import math
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from data.dataload import find_class_imbalance, load_data

from model.cbm import JointCBM
from utils.loss import JointCBMLoss


NUM_CONCEPTS = 112
NUM_CLASSES = 200
MIN_LR = 0.0001
LR_DECAY_SIZE = 0.1


class Logger:
    """Write the same messages to the terminal and log.txt, without closing stdout."""

    def __init__(self, path):
        self.file = Path(path).open("a", encoding="utf-8")

    def write(self, message):
        sys.stdout.write(message)
        sys.stdout.flush()
        self.file.write(message)
        self.file.flush()

    def close(self):
        self.file.close()


def seed_worker(worker_id):
    """Seed Python and NumPy inside workers using PyTorch's worker seed."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_loaders(args, device):
    """Reuse the existing CUB loader for train/val splits and preprocessing."""
    data_dir = Path(args.data_dir).expanduser().resolve()
    train_path = str(data_dir / "train.pkl")
    val_path = str(data_dir / "val.pkl")

    loader_options = {
        "use_attr": True,
        "no_img": False,
        "batch_size": args.batch_size,
        "image_dir": args.image_dir,
        "resol": 224,
        "normalization": args.normalization,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
    }

    # Upstream --ckpt means retraining on train + val, not resuming a checkpoint.
    training_paths = [train_path, val_path] if args.ckpt else [train_path]
    train_loader = load_data(training_paths, **loader_options)
    if len(train_loader) == 0:
        raise ValueError("Training set must contain at least batch_size samples because drop_last=True")

    val_loader = None
    if not args.ckpt:
        val_loader = load_data([val_path], **loader_options)

    for loader in (train_loader, val_loader):
        if loader is not None:
            if not loader.dataset.data:
                raise ValueError("Annotation files must contain at least one sample")
            if len(loader.dataset.data[0]["attribute_label"]) != NUM_CONCEPTS:
                raise ValueError("Expected preprocessed CUB112 labels with 112 concepts")

    # Even in train+val retraining, upstream computes weights from train.pkl only.
    weights = None
    if args.weighted_loss != "none":
        weights = find_class_imbalance(
            train_path,
            multiple_attr=args.weighted_loss == "multiple",
        )

    return train_loader, val_loader, weights


def build_optimizer(model, args):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    options = {"lr": args.lr, "weight_decay": args.weight_decay}

    if args.optimizer == "adam":
        return torch.optim.Adam(parameters, **options)
    if args.optimizer == "rmsprop":
        return torch.optim.RMSprop(parameters, momentum=0.9, **options)
    return torch.optim.SGD(parameters, momentum=0.9, **options)


def run_epoch(model, loader, criterion, device, optimizer=None):
    """Train or validate with sample-weighted means, matching upstream AverageMeter."""
    is_training = optimizer is not None
    model.train(is_training)

    totals = {
        "loss": 0.0,
        "class_loss": 0.0,
        "concept_loss": 0.0,
        "accuracy": 0.0,
        "concept_accuracy": 0.0,
    }
    sample_count = 0

    with torch.set_grad_enabled(is_training):
        for images, class_labels, concept_labels in loader:
            images = images.to(device, non_blocking=True)
            class_labels = class_labels.to(device, non_blocking=True)
            # The existing CUB loader returns one [B] label tensor per concept.
            if isinstance(concept_labels, (list, tuple)):
                concept_labels = torch.stack(concept_labels, dim=1)
            concept_labels = concept_labels.to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )

            outputs = model(images)
            losses = criterion(
                outputs,
                class_labels,
                concept_labels,
                return_components=True,
            )

            if not torch.isfinite(losses["loss"]):
                raise FloatingPointError("Non-finite Joint CBM loss")

            if is_training:
                optimizer.zero_grad()
                losses["loss"].backward()
                optimizer.step()

            batch_size = images.size(0)
            sample_count += batch_size
            for name in ("loss", "class_loss", "concept_loss"):
                totals[name] += losses[name].detach().item() * batch_size

            # Concept accuracy is diagnostic only; it never contributes to the loss.
            with torch.no_grad():
                class_predictions = outputs[0].topk(1, dim=1).indices[:, 0]
                correct_classes = (class_predictions == class_labels).sum().item()
                concept_logits = torch.cat(outputs[1:], dim=1)
                concept_predictions = concept_logits.sigmoid() >= 0.5
                correct_concepts = (concept_predictions == concept_labels.bool()).sum().item()

            totals["accuracy"] += correct_classes * 100.0
            totals["concept_accuracy"] += correct_concepts * 100.0 / NUM_CONCEPTS

    if sample_count == 0:
        raise ValueError("Cannot evaluate an empty DataLoader")

    return {name: value / sample_count for name, value in totals.items()}


def update_learning_rate(scheduler, epoch, stop_epoch):
    """Preserve upstream's epoch-indexed StepLR timing on current PyTorch versions."""
    if epoch <= stop_epoch:
        # step(epoch) preserves the original decay boundary; plain step() shifts it.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The epoch parameter in `scheduler.step\\(\\)` was not necessary.*",
                category=UserWarning,
            )
            scheduler.step(epoch)


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    if args.log_dir is None:
        args.log_dir = str(Path(__file__).parent / "runs" / f"joint_cbm_seed{args.seed}")
    log_dir = Path(args.log_dir).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, imbalance = build_loaders(args, device)
    model = JointCBM(
        backbone_name=args.backbone,
        pretrained=args.pretrained,
        num_concepts=NUM_CONCEPTS,
        num_classes=NUM_CLASSES,
        concept_activation=args.concept_activation,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    criterion = JointCBMLoss(
        attr_loss_weight=args.attr_loss_weight,
        num_concepts=NUM_CONCEPTS,
        concept_weights=imbalance,
        normalize_loss=args.normalize_loss,
    ).to(device)

    optimizer = build_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.scheduler_step,
        gamma=LR_DECAY_SIZE,
    )
    stop_epoch = int(math.log(MIN_LR / args.lr) / math.log(LR_DECAY_SIZE)) * args.scheduler_step

    best_val_epoch = -1
    best_val_acc = 0.0
    logger = Logger(log_dir / "log.txt")

    try:
        logger.write(str(args) + "\n")
        logger.write(str(imbalance) + "\n")
        logger.write(f"Device: {device}\n")
        logger.write(f"Stop epoch:  {stop_epoch}\n")
        logger.write(f"train data path: {Path(args.data_dir).expanduser().resolve() / 'train.pkl'}\n")
        if args.ckpt:
            logger.write("Retraining on train + val; Val metrics below reuse Train metrics (upstream ckpt mode).\n")

        with (log_dir / "metrics.jsonl").open("a", encoding="utf-8") as metrics_file:
            for epoch in range(args.epochs):
                learning_rate = optimizer.param_groups[0]["lr"]
                train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)

                if val_loader is not None:
                    val_metrics = run_epoch(model, val_loader, criterion, device)
                else:
                    val_metrics = dict(train_metrics)

                if val_metrics["accuracy"] > best_val_acc:
                    best_val_epoch = epoch
                    best_val_acc = val_metrics["accuracy"]
                    logger.write("New model best model at epoch %d\n" % epoch)
                    # Preserve the upstream whole-model artifact and filename.
                    torch.save(model, log_dir / f"best_model_{args.seed}.pth")

                logger.write(
                    "Epoch [%d]:\tTrain loss: %.4f\tTrain accuracy: %.4f\t"
                    "Val loss: %.4f\tVal acc: %.4f\tBest val epoch: %d\n"
                    % (
                        epoch,
                        train_metrics["loss"],
                        train_metrics["accuracy"],
                        val_metrics["loss"],
                        val_metrics["accuracy"],
                        best_val_epoch,
                    )
                )

                logger.write(
                    "Train class loss: %.4f\tTrain concept loss: %.4f\tTrain concept acc: %.4f\t"
                    "Val class loss: %.4f\tVal concept loss: %.4f\tVal concept acc: %.4f\n"
                    % (
                        train_metrics["class_loss"],
                        train_metrics["concept_loss"],
                        train_metrics["concept_accuracy"],
                        val_metrics["class_loss"],
                        val_metrics["concept_loss"],
                        val_metrics["concept_accuracy"],
                    )
                )

                update_learning_rate(scheduler, epoch, stop_epoch)
                if epoch % 10 == 0:
                    logger.write(f"Current lr: {[group['lr'] for group in optimizer.param_groups]}\n")

                metrics_file.write(json.dumps({
                    "epoch": epoch,
                    "lr": learning_rate,
                    "train": train_metrics,
                    "val": val_metrics,
                    "validation_is_training": bool(args.ckpt),
                    "best_val_epoch": best_val_epoch,
                    "best_val_accuracy": best_val_acc,
                }) + "\n")
                metrics_file.flush()

                # An additional state-dict artifact does not change best-model selection.
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "args": vars(args),
                    "concept_weights": imbalance,
                    "best_val_epoch": best_val_epoch,
                    "best_val_accuracy": best_val_acc,
                }, log_dir / "last_checkpoint.pth")

                if epoch >= 100 and val_metrics["accuracy"] < 3:
                    logger.write("Early stopping because of low accuracy\n")
                    break
                if epoch - best_val_epoch >= 100:
                    logger.write("Early stopping because acc hasn't improved for a long time\n")
                    break
    finally:
        logger.close()

    return {"best_val_epoch": best_val_epoch, "best_val_accuracy": best_val_acc}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir", "-data_dir", required=True, help="Directory containing CUB112 train.pkl and val.pkl")
    parser.add_argument("--image-dir", "-image_dir", required=True, help="Root of CUB_200_2011/images")
    parser.add_argument("--log-dir", "-log_dir", default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--backbone", choices=["resnet18", "resnet18_ptcv_cub"], default="resnet18")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--concept-activation", choices=["none", "sigmoid"], default="none")
    parser.add_argument("--epochs", "-e", type=int, default=1000)
    parser.add_argument("--batch-size", "-b", type=int, default=64)
    parser.add_argument("--optimizer", "-optimizer", type=str.lower, choices=["sgd", "adam", "rmsprop"], default="sgd")
    parser.add_argument("--lr", "-lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", "-weight_decay", type=float, default=0.0004)
    parser.add_argument("--scheduler-step", "-scheduler_step", type=int, default=1000)
    parser.add_argument("--attr-loss-weight", "-attr_loss_weight", type=float, default=0.01)
    parser.add_argument("--weighted-loss", "-weighted_loss", choices=["multiple", "single", "none"], default="multiple")
    parser.add_argument("--normalize-loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--normalization", choices=["imagenet", "upstream"], default="imagenet")
    parser.add_argument("--ckpt", "-ckpt", type=int, choices=[0, 1], default=0, help="1: retrain on train+val, as upstream; not checkpoint resume")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")

    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.scheduler_step < 1:
        parser.error("epochs, batch-size, and scheduler-step must be positive")
    if not 0 < args.lr < float("inf"):
        parser.error("lr must be finite and positive")
    if not 0 <= args.weight_decay < float("inf"):
        parser.error("weight-decay must be finite and nonnegative")
    if not 0 <= args.attr_loss_weight < float("inf"):
        parser.error("attr-loss-weight must be finite and nonnegative")
    if args.num_workers < 0:
        parser.error("num-workers must be nonnegative")
    return args


if __name__ == "__main__":
    train(parse_args())
