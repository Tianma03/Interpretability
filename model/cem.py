"""Concept Embedding Model with CUB-ready backbone and intervention support."""

import torch
from torch import nn

from .backbone import get_backbone


class ConceptEmbeddingModel(nn.Module):
    """Predict concepts as probabilities and route class prediction through embeddings."""

    def __init__(
        self,
        backbone="resnet18",
        pretrained=True,
        latent_dim=None,
        n_concepts=112,
        n_tasks=200,
        emb_size=16,
        training_intervention_prob=0.25,
        embedding_activation="leakyrelu",
        shared_prob_gen=True,
        c2y_model=None,
        c2y_layers=None,
    ):
        super().__init__()
        if isinstance(backbone, str):
            backbone = get_backbone(backbone, pretrained=pretrained)
        self.backbone = backbone
        self.n_concepts = n_concepts
        self.n_tasks = n_tasks
        self.emb_size = emb_size
        self.training_intervention_prob = training_intervention_prob
        self.shared_prob_gen = shared_prob_gen
        self.feature_dim = latent_dim or getattr(backbone, "feature_dim", None)

        if self.feature_dim is None:
            raise ValueError("Pass latent_dim when the backbone has no feature_dim")
        if n_concepts < 1 or n_tasks < 1 or emb_size < 1:
            raise ValueError("n_concepts, n_tasks, and emb_size must be positive")
        if not 0 <= training_intervention_prob <= 1:
            raise ValueError("training_intervention_prob must be in [0, 1]")

        activation = {
            None: None,
            "sigmoid": nn.Sigmoid,
            "relu": nn.ReLU,
            "leakyrelu": nn.LeakyReLU,
        }
        if embedding_activation not in activation:
            raise ValueError(f"Unsupported embedding_activation: {embedding_activation}")
        self.concept_context_generators = nn.ModuleList([
            nn.Sequential(*([nn.Linear(self.feature_dim, 2 * emb_size)] +
                            ([activation[embedding_activation]()] if embedding_activation else [])))
            for _ in range(n_concepts)
        ])
        prob_gen_count = 1 if shared_prob_gen else n_concepts
        self.concept_prob_generators = nn.ModuleList([
            nn.Linear(2 * emb_size, 1) for _ in range(prob_gen_count)
        ])

        if c2y_model is not None:
            self.c2y_model = c2y_model
        else:
            dims = [n_concepts * emb_size, *(c2y_layers or []), n_tasks]
            if any(dim < 1 for dim in dims):
                raise ValueError("c2y layer dimensions must be positive")
            layers = []
            for index, (in_dim, out_dim) in enumerate(zip(dims, dims[1:])):
                layers.append(nn.Linear(in_dim, out_dim))
                if index < len(dims) - 2:
                    layers.append(nn.LeakyReLU())
            self.c2y_model = nn.Sequential(*layers)

    def _concept_targets(self, concepts, batch_size, device):
        if isinstance(concepts, (list, tuple)):
            concepts = torch.stack(concepts, dim=1)
        concepts = torch.as_tensor(concepts, device=device, dtype=torch.float32)
        if concepts.shape != (batch_size, self.n_concepts):
            raise ValueError(f"concepts must have shape [{batch_size}, {self.n_concepts}]")
        return concepts

    def _intervention_mask(self, intervention_idxs, batch_size, device):
        if intervention_idxs is None:
            return None
        mask = torch.as_tensor(intervention_idxs, device=device, dtype=torch.bool)
        if mask.ndim == 1:
            if mask.shape[0] != self.n_concepts:
                raise ValueError(f"intervention_idxs must have {self.n_concepts} concepts")
            mask = mask.unsqueeze(0).expand(batch_size, -1)
        if mask.shape != (batch_size, self.n_concepts):
            raise ValueError(f"intervention_idxs must have shape [{batch_size}, {self.n_concepts}]")
        return mask

    def forward(
        self,
        images,
        concepts=None,
        intervention_idxs=None,
        return_details=False,
    ):
        features = self.backbone(images)
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(f"backbone must return [B, {self.feature_dim}] features")

        contexts, concept_logits = [], []
        for index, context_generator in enumerate(self.concept_context_generators):
            context = context_generator(features)
            prob_generator = self.concept_prob_generators[0 if self.shared_prob_gen else index]
            contexts.append(context)
            concept_logits.append(prob_generator(context))

        contexts = torch.stack(contexts, dim=1)  # [B, C, 2E]
        concept_logits = torch.cat(concept_logits, dim=1)  # [B, C]
        concept_probs = concept_logits.sigmoid()
        pos_embeddings, neg_embeddings = contexts.split(self.emb_size, dim=-1)

        batch_size = images.shape[0]
        concept_targets = None
        if concepts is not None:
            concept_targets = self._concept_targets(concepts, batch_size, images.device)
        intervention_mask = self._intervention_mask(
            intervention_idxs, batch_size, images.device
        )
        if (intervention_mask is None and self.training and concept_targets is not None
                and self.training_intervention_prob > 0):
            # Match the reference CEM: sample concepts once and share the mask across the batch.
            mask = torch.bernoulli(torch.full(
                (self.n_concepts,), self.training_intervention_prob, device=images.device
            )).bool()
            intervention_mask = mask.unsqueeze(0).expand(batch_size, -1)
        if intervention_mask is not None and concept_targets is None:
            raise ValueError("concepts are required when intervention_idxs are provided")

        mixed_probs = concept_probs
        if intervention_mask is not None:
            mixed_probs = torch.where(intervention_mask, concept_targets, concept_probs)
        concept_embeddings = (
            pos_embeddings * mixed_probs.unsqueeze(-1) +
            neg_embeddings * (1 - mixed_probs.unsqueeze(-1))
        )  # [B, C, E]
        class_logits = self.c2y_model(concept_embeddings.flatten(start_dim=1))

        if return_details:
            return {
                "class_logits": class_logits,
                "concept_logits": concept_logits,
                "concept_probs": concept_probs,
                "mixed_probs": mixed_probs,
                "concept_embeddings": concept_embeddings,
                "intervention_mask": intervention_mask,
            }
        return [class_logits, *concept_logits.split(1, dim=1)]
