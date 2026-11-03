"""Original Joint CBM losses for ResNet18 models without an auxiliary branch.

Reference: https://github.com/yewsiang/ConceptBottleneck/blob/master/CUB/train.py

Class Loss: class cross-entropy averaged over the batch.
Concept Loss: batch-mean BCE for each concept, summed over all concepts.
Joint Loss: Class Loss + attr_loss_weight * Concept Loss.

Concept imbalance weights follow upstream BCE weight, rather than pos_weight.
"""

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["ClassLoss", "ConceptLoss", "JointCBMLoss"]


class ClassLoss(nn.Module):
    """Class classification loss.

    class_logits: raw class logits with shape [B, num_classes].
    class_labels: torch.long class indices with shape [B], ranging from 0 to 199 for CUB.
    Returns a scalar cross-entropy loss averaged over the batch.
    """

    def forward(self, class_logits, class_labels):
        # cross_entropy includes log_softmax; pass logits without applying softmax.
        return F.cross_entropy(class_logits, class_labels)


class ConceptLoss(nn.Module):
    """Binary concept prediction loss, defaulting to the 112 concepts in CUB112.

    concept_logits: a [B, C] tensor or a list of C tensors with shape [B, 1].
    concept_labels: a [B, C] tensor or C tensors of shape [B] from the DataLoader.
    Targets may be binary labels or soft labels in [0, 1].

    concept_weights: optional [C] weights; upstream uses N_negative / N_positive.
    Each weight scales both positive and negative losses for its concept.
    """

    def __init__(self, num_concepts=112, concept_weights=None):
        super().__init__()

        if num_concepts < 1:
            raise ValueError("num_concepts must be positive")

        self.num_concepts = num_concepts

        weights = None
        if concept_weights is not None:
            weights = torch.as_tensor(
                concept_weights,
                dtype=torch.float32,
            )
            # Weights are fixed statistics with no gradients or shared storage with the input.
            weights = weights.detach().clone()

            if weights.shape != (num_concepts,):
                raise ValueError(
                    f"concept_weights must have shape [{num_concepts}]"
                )

            if not torch.isfinite(weights).all() or (weights < 0).any():
                raise ValueError("concept_weights must be finite and nonnegative")

        # Buffers follow device transfers and enter state_dict, but are not trainable parameters.
        self.register_buffer("concept_weights", weights)

    def forward(self, concept_logits, concept_labels):
        # Concatenate C predictions of shape [B, 1] into [B, C], preserving gradients.
        if isinstance(concept_logits, (list, tuple)):
            if len(concept_logits) != self.num_concepts:
                raise ValueError(
                    f"Expected {self.num_concepts} concept logit tensors"
                )

            concept_logits = torch.cat(concept_logits, dim=1)

        if concept_logits.ndim != 2 or concept_logits.shape[1] != self.num_concepts:
            raise ValueError(
                f"concept_logits must have shape [B, {self.num_concepts}]"
            )

        # Stack the original CUB DataLoader labels by concept to match [B, C] predictions.
        if isinstance(concept_labels, (list, tuple)):
            concept_labels = torch.stack(concept_labels, dim=1)

        targets = torch.as_tensor(
            concept_labels,
            device=concept_logits.device,
            dtype=concept_logits.dtype,
        )

        if targets.shape != concept_logits.shape:
            raise ValueError(
                "concept_labels and concept_logits must have the same shape"
            )

        weights = None
        if self.concept_weights is not None:
            weights = self.concept_weights.to(concept_logits)

        # BCEWithLogits handles sigmoid internally; always pass raw concept logits.
        # Sigmoid on the Concepts -> Class path does not change the input to this loss.
        losses = F.binary_cross_entropy_with_logits(
            concept_logits,
            targets,
            weight=weights,
            reduction="none",
        )

        # [B, C] -> [C] -> scalar: average over the batch, then sum over concepts.
        # losses.mean() would also divide by the concept count, changing upstream lambda scaling.
        return losses.mean(dim=0).sum()


class JointCBMLoss(nn.Module):
    """Combine class and concept losses into the original Joint CBM training objective.

    attr_loss_weight: concept loss coefficient lambda.
    num_concepts: number of concepts, defaulting to 112 for CUB112.
    concept_weights: per-concept imbalance weights passed to ConceptLoss.
    normalize_loss: divide total loss by 1 + lambda * num_concepts, as upstream.

    forward accepts JointCBM outputs [class_logits, *concept_logits].
    By default, returns the scalar total loss for backward.
    return_components=True also returns both component losses for logging.
    Components are unweighted and unnormalized; lambda=0 skips concept supervision and reports zero concept loss.
    """

    def __init__(
        self,
        attr_loss_weight=0.01,
        num_concepts=112,
        concept_weights=None,
        normalize_loss=False,
    ):
        super().__init__()

        if not 0 <= attr_loss_weight < float("inf"):
            raise ValueError("attr_loss_weight must be finite and nonnegative")

        self.attr_loss_weight = attr_loss_weight
        self.normalize_loss = normalize_loss

        self.class_loss = ClassLoss()
        self.concept_loss = ConceptLoss(
            num_concepts=num_concepts,
            concept_weights=concept_weights,
        )

    def forward(
        self,
        outputs,
        class_labels,
        concept_labels=None,
        return_components=False,
    ):
        # outputs[0] contains class logits; outputs[1:] contains raw concept logits.
        expected_outputs = self.concept_loss.num_concepts + 1
        if not isinstance(outputs, (list, tuple)) or len(outputs) != expected_outputs:
            raise ValueError(
                "Expected JointCBM outputs [class_logits, *concept_logits]"
            )

        class_loss = self.class_loss(outputs[0], class_labels)

        # For lambda=0, skip concept labels and use a zero scalar with matching device and dtype.
        concept_loss = class_loss.new_zeros(())
        if self.attr_loss_weight > 0:
            if concept_labels is None:
                raise ValueError(
                    "concept_labels are required when attr_loss_weight > 0"
                )

            concept_loss = self.concept_loss(outputs[1:], concept_labels)

        loss = class_loss + self.attr_loss_weight * concept_loss

        # Normalize only the training objective; keep raw component values for logging.
        if self.normalize_loss:
            normalizer = 1 + self.attr_loss_weight * self.concept_loss.num_concepts
            loss = loss / normalizer

        if return_components:
            return {
                "loss": loss,
                "class_loss": class_loss,
                "concept_loss": concept_loss,
            }

        return loss
