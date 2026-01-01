"""Stage C: gated context fusion, auxiliary heads, and novelty prediction.

Development-set mixture tuning and moderate-to-high refinement are applied
by the evaluation routines in egrd.py. Stages B and C train jointly.
"""
from __future__ import annotations

import math
from typing import Any

import torch

try:
    from .features import GLOBAL_FEATURE_NAMES, JOINT_GLOBAL_FEATURES, PAIR_GLOBAL_FEATURES, JOINT_LABELS, NOVELTY_LABELS
except ImportError:
    from features import GLOBAL_FEATURE_NAMES, JOINT_GLOBAL_FEATURES, PAIR_GLOBAL_FEATURES, JOINT_LABELS, NOVELTY_LABELS


class NoveltyPredictionMixin:
    def predict_novelty(
        self,
        decomposition: dict[str, Any],
        *,
        pair_features: torch.Tensor,
        global_features: torch.Tensor,
        aspect_id: torch.Tensor,
        contribution_id: torch.Tensor,
        task_id: torch.Tensor,
        joint_normalized_entropy: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Fuse Stage B residuals with evidence context and evaluate prediction heads."""
        alignment_scores = decomposition["alignment_scores"]
        batch_size = decomposition["batch_size"]
        coverage_pool_weights = decomposition["coverage_pool_weights"]
        coverage_scores = decomposition["coverage_scores"]
        covered_representation = decomposition["covered_representation"]
        estimated_joint_coverage = decomposition["estimated_joint_coverage"]
        historical_co_support = decomposition["historical_co_support"]
        interaction_attention = decomposition["interaction_attention"]
        interaction_residual_representation = decomposition["interaction_residual_representation"]
        interaction_residual_scores = decomposition["interaction_residual_scores"]
        interaction_strength = decomposition["interaction_strength"]
        pair_weights = decomposition["pair_weights"]
        raw_token_slot_assignment = decomposition["raw_token_slot_assignment"]
        residual_scores = decomposition["residual_scores"]
        semantic_attention = decomposition["semantic_attention"]
        semantic_slot_mass = decomposition["semantic_slot_mass"]
        semantic_target_mask = decomposition["semantic_target_mask"]
        semantic_units = decomposition["semantic_units"]
        target_representation = decomposition["target_representation"]
        token_slot_assignment = decomposition["token_slot_assignment"]
        unit_active_probability = decomposition["unit_active_probability"]
        unit_residual_representation = decomposition["unit_residual_representation"]
        weighted_support = decomposition["weighted_support"]

        metadata = torch.cat(
            [
                self.aspect_embedding(aspect_id),
                self.contribution_embedding(contribution_id),
                self.task_embedding(task_id),
            ],
            dim=-1,
        )
        ablated = set(self.config.ablate_features)
        if ablated & {"joint", "pair"}:
            global_features = global_features.clone()
            names = list(JOINT_GLOBAL_FEATURES) if "joint" in ablated else []
            names += list(PAIR_GLOBAL_FEATURES) if "pair" in ablated else []
            for name in names:
                global_features[:, GLOBAL_FEATURE_NAMES.index(name)] = 0.0
        if "pair" in ablated:
            pair_features = pair_features * 0.0
        global_features_for_model = global_features
        if self.training and self.config.coverage_feature_noise_std > 0:
            global_features_for_model = global_features.clone()
            noisy_feature_names = [
                "joint_evidence_score",
                "joint_expected_score",
                "joint_normalized_entropy",
                "p_not_covered",
                "p_weakly_covered",
                "p_partially_covered",
                "p_mostly_covered",
            ]
            noisy_indices = [GLOBAL_FEATURE_NAMES.index(name) for name in noisy_feature_names]
            noise = torch.randn_like(global_features_for_model[:, noisy_indices]) * float(self.config.coverage_feature_noise_std)
            global_features_for_model[:, noisy_indices] = (global_features_for_model[:, noisy_indices] + noise).clamp(0.0, 1.0)
        if self.training and self.config.coverage_feature_dropout > 0:
            global_features_for_model = global_features_for_model.clone()
            dropout_feature_names = [
                "joint_evidence_score",
                "joint_expected_score",
                "joint_normalized_entropy",
                "joint_minus_pair_union",
                "p_not_covered",
                "p_weakly_covered",
                "p_partially_covered",
                "p_mostly_covered",
            ]
            dropout_indices = [GLOBAL_FEATURE_NAMES.index(name) for name in dropout_feature_names]
            keep = (
                torch.rand_like(global_features_for_model[:, dropout_indices])
                >= float(self.config.coverage_feature_dropout)
            ).to(dtype=global_features_for_model.dtype)
            global_features_for_model[:, dropout_indices] = global_features_for_model[:, dropout_indices] * keep
        global_context = self.global_projection(torch.cat([global_features_for_model, metadata], dim=-1))
        coverage_gate = torch.sigmoid(
            self.coverage_gate(
                torch.cat(
                    [
                        unit_residual_representation,
                        interaction_residual_representation,
                        global_context,
                        joint_normalized_entropy.clamp(0.0, 1.0).unsqueeze(-1),
                    ],
                    dim=-1,
                )
            )
        )
        final_residual_representation = self.residual_fusion(
            torch.cat(
                [
                    unit_residual_representation,
                    interaction_residual_representation,
                    coverage_gate * global_context,
                ],
                dim=-1,
            )
        )
        zero = lambda name, tensor: tensor * 0.0 if name in ablated else tensor
        zero_any = lambda names, tensor: (tensor * 0.0 if ablated.intersection(names)
                                          else tensor)
        head_input = torch.cat(
            [
                zero_any(("egrd", "egrd_residual"), final_residual_representation),
                zero_any(("egrd", "egrd_covered"), covered_representation),
                zero("target", target_representation),
                zero("global", global_context),
            ],
            dim=-1,
        )
        head_input = self.head_dropout(self.head_norm(head_input))

        substantive_score = torch.sigmoid(self.substantive_head(head_input).squeeze(-1))
        surface_score = torch.sigmoid(self.surface_head(head_input).squeeze(-1))
        evidence_sufficiency = torch.sigmoid(self.evidence_head(head_input).squeeze(-1))
        interaction_residual_strength = (
            interaction_residual_scores.max(dim=-1).values
            if interaction_residual_scores.shape[-1] > 0
            else interaction_residual_representation.new_zeros((batch_size,))
        )
        minor_gate_features = torch.cat(
            [
                head_input,
                substantive_score.unsqueeze(-1),
                surface_score.unsqueeze(-1),
                interaction_residual_strength.unsqueeze(-1),
            ],
            dim=-1,
        )
        minor_substantive_logits = self.minor_substantive_head(minor_gate_features)
        innovation_family_logits = self.innovation_family_head(head_input)
        fine_type_logits = {
            coarse: head(head_input)
            for coarse, head in self.fine_type_heads.items()
        }
        flat_type_logits = self.flat_type_head(head_input)
        flat_type_probabilities = torch.softmax(flat_type_logits.float(), dim=-1)
        combination_logits = self.combination_head(interaction_residual_representation).squeeze(-1)
        combination_probability = torch.sigmoid(combination_logits)
        substantive_prob = torch.sigmoid(minor_substantive_logits.float()).squeeze(-1)
        innovation_family_probs = torch.softmax(innovation_family_logits.float(), dim=-1)
        coarse_type_probabilities = torch.cat(
            [
                (1.0 - substantive_prob).unsqueeze(-1),
                substantive_prob.unsqueeze(-1) * innovation_family_probs,
            ],
            dim=-1,
        )
        coarse_type_logits = torch.log(coarse_type_probabilities.clamp_min(1e-8))
        hierarchical_type_probabilities = self.hierarchical_type_probabilities(
            minor_substantive_logits,
            innovation_family_logits,
            fine_type_logits,
            combination_probability,
        )
        flat_blend = float(max(0.0, min(1.0, self.config.flat_type_blend_weight)))
        type_probabilities = (
            (1.0 - flat_blend) * hierarchical_type_probabilities
            + flat_blend * flat_type_probabilities
        )
        type_probabilities = type_probabilities / type_probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        joint_score = global_features[:, GLOBAL_FEATURE_NAMES.index("joint_evidence_score")].clamp(0.0, 1.0)
        joint_probs = global_features[:, -len(JOINT_LABELS):].clamp(0.0, 1.0)
        novelty_features = torch.cat(
            [
                head_input,
                zero("scalars", substantive_score.unsqueeze(-1)),
                zero("scalars", surface_score.unsqueeze(-1)),
                zero("scalars", evidence_sufficiency.unsqueeze(-1)),
                zero("type", type_probabilities),
                zero("joint", joint_score.unsqueeze(-1)),
                zero("joint", joint_probs),
            ],
            dim=-1,
        )
        novelty_direct_logits = self.novelty_direct_head(novelty_features)
        novelty_direct_probabilities = torch.softmax(novelty_direct_logits, dim=-1)
        novelty_ordinal_logits = self.novelty_ordinal_head(novelty_features)
        moderate_high_logits = self.moderate_high_head(novelty_features).squeeze(-1)
        moderate_high_probability = torch.sigmoid(moderate_high_logits)
        novelty_ordinal_probabilities, novelty_conditional_probabilities = self.ordinal_probabilities(
            novelty_ordinal_logits,
            moderate_high_probability,
        )
        predicted_novelty_score = torch.sigmoid(self.novelty_score_head(novelty_features).squeeze(-1))
        residual_novelty_logits = self.residual_novelty_head(novelty_features)
        residual_novelty_probabilities = torch.softmax(residual_novelty_logits.float(), dim=-1)
        novelty_fusion_gates = torch.softmax(self.novelty_fusion_gate(novelty_features).float(), dim=-1)
        novelty_dynamic_probabilities = (
            novelty_fusion_gates[:, 0:1] * novelty_direct_probabilities
            + novelty_fusion_gates[:, 1:2] * novelty_conditional_probabilities
            + novelty_fusion_gates[:, 2:3] * residual_novelty_probabilities
        )
        novelty_dynamic_probabilities = novelty_dynamic_probabilities / novelty_dynamic_probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        fixed_fusion_weights = novelty_direct_probabilities.new_tensor(
            [
                self.config.novelty_direct_weight,
                self.config.novelty_ordinal_weight,
                self.config.novelty_residual_weight,
            ]
        )
        fixed_fusion_weights = fixed_fusion_weights.clamp_min(0.0)
        fixed_fusion_weights = fixed_fusion_weights / fixed_fusion_weights.sum().clamp_min(1e-8)
        novelty_fixed_probabilities = (
            fixed_fusion_weights[0] * novelty_direct_probabilities
            + fixed_fusion_weights[1] * novelty_ordinal_probabilities
            + fixed_fusion_weights[2] * residual_novelty_probabilities
        )
        novelty_fixed_probabilities = novelty_fixed_probabilities / novelty_fixed_probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        novelty_probabilities = (
            novelty_dynamic_probabilities
            if self.config.novelty_fusion_mode == "dynamic"
            else novelty_fixed_probabilities
        )

        type_entropy = (
            -(type_probabilities * torch.log(type_probabilities.clamp_min(1e-8))
              + (1.0 - type_probabilities) * torch.log((1.0 - type_probabilities).clamp_min(1e-8))).mean(dim=-1)
            / math.log(2.0)
        )
        novelty_entropy = (
            -(novelty_probabilities * torch.log(novelty_probabilities.clamp_min(1e-8))).sum(dim=-1)
            / math.log(len(NOVELTY_LABELS))
        )
        prediction_uncertainty = (
            0.40 * joint_normalized_entropy.clamp(0.0, 1.0)
            + 0.35 * type_entropy.clamp(0.0, 1.0)
            + 0.25 * novelty_entropy.clamp(0.0, 1.0)
        ).clamp(0.0, 1.0)

        return {
            "substantive_score": substantive_score,
            "surface_score": surface_score,
            "evidence_sufficiency": evidence_sufficiency,
            "minor_substantive_logits": minor_substantive_logits,
            "substantive_probability": substantive_prob,
            "innovation_family_logits": innovation_family_logits,
            "innovation_family_probabilities": innovation_family_probs,
            "coarse_type_logits": coarse_type_logits,
            "fine_type_logits": fine_type_logits,
            "flat_type_logits": flat_type_logits,
            "flat_type_probabilities": flat_type_probabilities,
            "combination_logits": combination_logits,
            "combination_probability": combination_probability,
            "hierarchical_type_probabilities": hierarchical_type_probabilities,
            "type_probabilities": type_probabilities,
            "novelty_direct_logits": novelty_direct_logits,
            "novelty_direct_probabilities": novelty_direct_probabilities,
            "novelty_ordinal_logits": novelty_ordinal_logits,
            "novelty_ordinal_probabilities": novelty_ordinal_probabilities,
            "novelty_conditional_probabilities": novelty_conditional_probabilities,
            "residual_novelty_logits": residual_novelty_logits,
            "residual_novelty_probabilities": residual_novelty_probabilities,
            "novelty_fusion_gates": novelty_fusion_gates,
            "novelty_fixed_probabilities": novelty_fixed_probabilities,
            "novelty_dynamic_probabilities": novelty_dynamic_probabilities,
            "moderate_high_logits": moderate_high_logits,
            "moderate_high_probability": moderate_high_probability,
            "predicted_novelty_score": predicted_novelty_score,
            "novelty_probabilities": novelty_probabilities,
            "prediction_uncertainty": prediction_uncertainty,
            "semantic_unit_coverage_scores": coverage_scores,
            "semantic_unit_residual_scores": residual_scores,
            "pair_unit_alignment_scores": alignment_scores,
            "pair_weights": pair_weights,
            "weighted_support": weighted_support,
            "estimated_joint_coverage": estimated_joint_coverage,
            "coverage_pool_weights": coverage_pool_weights,
            "unit_active_probability": unit_active_probability,
            "coverage_gate": coverage_gate.squeeze(-1),
            "semantic_attention": semantic_attention,
            "semantic_token_assignment": token_slot_assignment,
            "semantic_raw_token_assignment": raw_token_slot_assignment,
            "semantic_slot_mass": semantic_slot_mass.squeeze(-1),
            "semantic_units": semantic_units,
            "semantic_target_mask": semantic_target_mask,
            "repr_unit_residual": unit_residual_representation,
            "repr_interaction_residual": interaction_residual_representation,
            "repr_covered": covered_representation,
            "repr_target": target_representation,
            "repr_global_context": global_context,
            "repr_fused": head_input,
            "interaction_residual_scores": interaction_residual_scores,
            "interaction_attention": interaction_attention,
            "interaction_strength": interaction_strength,
            "historical_co_support": historical_co_support,
        }
