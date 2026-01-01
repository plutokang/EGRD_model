"""Stage B: evidence-grounded unit and interaction residual decomposition.

Methods operate on the owning EGRD model to preserve checkpoint parameter names.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype, device=values.device).unsqueeze(-1)
    return (values * weights).sum(dim=dim) / weights.sum(dim=dim).clamp_min(1e-6)


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    shifted = logits - logits.max(dim=dim, keepdim=True).values
    sorted_logits = torch.sort(shifted, dim=dim, descending=True).values
    cumulative = sorted_logits.cumsum(dim)
    dimension = logits.shape[dim]
    support_range = torch.arange(
        1,
        dimension + 1,
        device=logits.device,
        dtype=logits.dtype,
    )
    view_shape = [1] * logits.ndim
    view_shape[dim] = dimension
    support_range = support_range.view(view_shape)
    support = 1.0 + support_range * sorted_logits > cumulative
    support_size = support.sum(dim=dim, keepdim=True).clamp_min(1)
    threshold_sum = cumulative.gather(dim, support_size - 1)
    threshold = (threshold_sum - 1.0) / support_size.to(dtype=logits.dtype)
    return torch.clamp(shifted - threshold, min=0.0)


def balance_token_slot_assignment(
    assignment: torch.Tensor,
    token_mask: torch.Tensor,
    *,
    iterations: int,
    min_slot_mass: float,
) -> torch.Tensor:
    mask = token_mask.unsqueeze(1).to(dtype=assignment.dtype)
    balanced = assignment * mask
    num_slots = max(1, assignment.shape[1])
    target_slot_mass = (
        token_mask.to(dtype=assignment.dtype).sum(dim=-1, keepdim=True).unsqueeze(1)
        / float(num_slots)
    )
    for _ in range(max(0, int(iterations))):
        slot_mass = balanced.sum(dim=-1, keepdim=True)
        balanced = balanced * (
            target_slot_mass / slot_mass.clamp_min(float(min_slot_mass))
        )
        token_mass = balanced.sum(dim=1, keepdim=True)
        balanced = balanced / token_mass.clamp_min(1e-6)
        balanced = balanced * mask
    return balanced


class ResidualDecompositionMixin:
    def decompose(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_token_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        pair_features: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Return differentiable residuals and alignment diagnostics for Stage C."""
        batch_size, top_k, seq_len = input_ids.shape
        flat_inputs = {
            "input_ids": input_ids.view(batch_size * top_k, seq_len),
            "attention_mask": attention_mask.view(batch_size * top_k, seq_len),
        }
        if token_type_ids is not None:
            flat_inputs["token_type_ids"] = token_type_ids.view(batch_size * top_k, seq_len)
        outputs = self.encoder(**flat_inputs)
        hidden = outputs.last_hidden_state.view(batch_size, top_k, seq_len, self.hidden_size)

        attention_bool = attention_mask.bool()
        pair_repr = masked_mean(hidden.view(batch_size * top_k, seq_len, self.hidden_size), attention_bool.view(batch_size * top_k, seq_len), dim=1)
        pair_repr = pair_repr.view(batch_size, top_k, self.hidden_size)
        pair_enhanced = pair_repr + self.pair_feature_projection(pair_features)

        valid_pair_mask = pair_mask.bool()
        has_valid_pair = valid_pair_mask.any(dim=1)
        if not bool(has_valid_pair.all()):
            valid_pair_mask = valid_pair_mask.clone()
            valid_pair_mask[~has_valid_pair, 0] = True
        valid_pair_tokens = valid_pair_mask.unsqueeze(-1)
        target_mask_per_pair = target_token_mask & attention_bool & valid_pair_tokens
        fallback_mask_per_pair = attention_bool & valid_pair_tokens
        has_target_tokens = target_mask_per_pair.any(dim=(1, 2))
        semantic_mask_per_pair = torch.where(
            has_target_tokens.view(batch_size, 1, 1),
            target_mask_per_pair,
            fallback_mask_per_pair,
        )
        semantic_token_count = semantic_mask_per_pair.sum(dim=1)
        semantic_target_mask = semantic_token_count > 0
        semantic_target_hidden = (
            hidden * semantic_mask_per_pair.unsqueeze(-1).to(dtype=hidden.dtype)
        ).sum(dim=1) / semantic_token_count.unsqueeze(-1).clamp_min(1).to(dtype=hidden.dtype)

        query = self.semantic_queries.unsqueeze(0).expand(batch_size, -1, -1)
        legacy_assignment = self.config.semantic_assignment_mode == "legacy"
        if legacy_assignment:
            semantic_scores = torch.einsum(
                "bmh,bnh->bmn",
                query.to(dtype=semantic_target_hidden.dtype),
                semantic_target_hidden,
            ) / math.sqrt(float(self.hidden_size))
        else:
            projected_query = F.normalize(self.unit_alignment_projection(query).float(), dim=-1)
            projected_tokens = F.normalize(
                self.prior_alignment_projection(semantic_target_hidden).float(),
                dim=-1,
            )
            semantic_scores = torch.einsum("bmh,bnh->bmn", projected_query, projected_tokens)
            semantic_scores = semantic_scores / max(1e-4, float(self.config.semantic_score_temperature))
            smoothing_kernel = max(1, int(self.config.semantic_score_smoothing_kernel))
            if smoothing_kernel > 1:
                if smoothing_kernel % 2 == 0:
                    smoothing_kernel += 1
                score_mask = semantic_target_mask.unsqueeze(1).to(dtype=semantic_scores.dtype)
                padding = smoothing_kernel // 2
                score_sum = F.avg_pool1d(
                    semantic_scores * score_mask,
                    kernel_size=smoothing_kernel,
                    stride=1,
                    padding=padding,
                )
                score_count = F.avg_pool1d(
                    score_mask,
                    kernel_size=smoothing_kernel,
                    stride=1,
                    padding=padding,
                )
                semantic_scores = score_sum / score_count.clamp_min(1e-6)
        semantic_scores = semantic_scores.masked_fill(~semantic_target_mask.unsqueeze(1), -1e4)
        if self.config.slot_competition:
            soft_assignment = torch.softmax(semantic_scores, dim=1)
            if legacy_assignment:
                raw_token_slot_assignment = soft_assignment
            elif self.config.semantic_assignment_mode == "sparsemax":
                sparse_assignment = sparsemax(semantic_scores, dim=1)
                sparse_weight = float(
                    np.clip(self.config.semantic_sparsemax_weight, 0.0, 1.0)
                )
                raw_token_slot_assignment = (
                    sparse_weight * sparse_assignment
                    + (1.0 - sparse_weight) * soft_assignment
                )
            else:
                raw_token_slot_assignment = soft_assignment
            raw_token_slot_assignment = (
                raw_token_slot_assignment
                * semantic_target_mask.unsqueeze(1).to(
                    dtype=raw_token_slot_assignment.dtype
                )
            )
            if legacy_assignment:
                token_slot_assignment = raw_token_slot_assignment
            else:
                token_slot_assignment = balance_token_slot_assignment(
                    raw_token_slot_assignment,
                    semantic_target_mask,
                    iterations=self.config.semantic_balance_iterations,
                    min_slot_mass=self.config.semantic_min_slot_mass,
                )
            semantic_slot_mass = token_slot_assignment.sum(dim=-1, keepdim=True)
            semantic_attention = token_slot_assignment / semantic_slot_mass.clamp_min(
                1e-6 if legacy_assignment else float(self.config.semantic_min_slot_mass)
            )
        else:
            semantic_attention = torch.softmax(semantic_scores, dim=-1)
            semantic_attention = semantic_attention * semantic_target_mask.unsqueeze(1).to(
                dtype=semantic_attention.dtype
            )
            semantic_attention = semantic_attention / semantic_attention.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            token_slot_assignment = semantic_attention
            raw_token_slot_assignment = token_slot_assignment
            semantic_slot_mass = token_slot_assignment.sum(dim=-1, keepdim=True)
        semantic_units = torch.einsum(
            "bmn,bnh->bmh",
            semantic_attention.to(dtype=semantic_target_hidden.dtype),
            semantic_target_hidden,
        )
        unit_active_probability = torch.sigmoid(self.unit_active_head(semantic_units).squeeze(-1))
        target_representation = masked_mean(semantic_target_hidden, semantic_target_mask, dim=1)

        unit_expanded = semantic_units.unsqueeze(1).expand(-1, top_k, -1, -1)
        prior_expanded = pair_enhanced.unsqueeze(2).expand(-1, -1, self.num_semantic_units, -1)
        pair_feature_expanded = pair_features.unsqueeze(2).expand(-1, -1, self.num_semantic_units, -1)
        support_input = torch.cat(
            [
                unit_expanded,
                prior_expanded,
                unit_expanded * prior_expanded,
                torch.abs(unit_expanded - prior_expanded),
                pair_feature_expanded,
            ],
            dim=-1,
        )
        support_logits = self.support_mlp(support_input).squeeze(-1)
        alignment_scores = torch.sigmoid(support_logits)
        weighted_support = alignment_scores * pair_mask.to(dtype=alignment_scores.dtype).unsqueeze(-1)
        weighted_support = weighted_support.clamp(0.0, 0.995)
        pair_weights = weighted_support.mean(dim=-1)
        coverage_scores = 1.0 - torch.prod(1.0 - weighted_support, dim=1)
        residual_scores = unit_active_probability * (1.0 - coverage_scores)
        active_coverage_scores = unit_active_probability * coverage_scores
        covered_representation = (active_coverage_scores.unsqueeze(-1) * semantic_units).sum(dim=1) / max(1, self.num_semantic_units)
        residual_weights = (residual_scores if self.config.residual_direction == "residual"
                            else active_coverage_scores)
        unit_residual_representation = (residual_weights.unsqueeze(-1) * semantic_units).sum(dim=1) / max(1, self.num_semantic_units)
        coverage_pool_weights = torch.softmax(self.coverage_pool_score(semantic_units).squeeze(-1), dim=-1)
        effective_pool_weights = coverage_pool_weights * unit_active_probability
        effective_pool_weights = effective_pool_weights / effective_pool_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        estimated_joint_coverage = (effective_pool_weights * coverage_scores).sum(dim=-1).clamp(0.0, 1.0)

        pair_indices = torch.combinations(
            torch.arange(self.num_semantic_units, device=semantic_units.device),
            r=2,
        )
        if pair_indices.numel() > 0 and self.config.unit_interaction:
            left_units = semantic_units[:, pair_indices[:, 0], :]
            right_units = semantic_units[:, pair_indices[:, 1], :]
            interaction_input = torch.cat(
                [
                    left_units,
                    right_units,
                    left_units * right_units,
                    torch.abs(left_units - right_units),
                ],
                dim=-1,
            )
            interaction_repr = self.interaction_mlp(interaction_input)
            interaction_strength = torch.sigmoid(self.interaction_strength_head(interaction_repr).squeeze(-1))
            left_support = weighted_support[:, :, pair_indices[:, 0]]
            right_support = weighted_support[:, :, pair_indices[:, 1]]
            historical_co_support = (left_support * right_support).max(dim=1).values
            interaction_residual_scores = interaction_strength * (1.0 - historical_co_support)
            interaction_pool_logits = self.interaction_pool_score(interaction_repr).squeeze(-1)
            interaction_attention = torch.softmax(
                interaction_pool_logits + torch.log(interaction_residual_scores.clamp_min(1e-8)),
                dim=-1,
            )
            interaction_residual_representation = (
                interaction_attention.unsqueeze(-1)
                * interaction_residual_scores.unsqueeze(-1)
                * interaction_repr
            ).sum(dim=1)
        else:
            interaction_repr = semantic_units.new_zeros((batch_size, 0, self.hidden_size))
            interaction_strength = semantic_units.new_zeros((batch_size, 0))
            historical_co_support = semantic_units.new_zeros((batch_size, 0))
            interaction_residual_scores = semantic_units.new_zeros((batch_size, 0))
            interaction_attention = semantic_units.new_zeros((batch_size, 0))
            interaction_residual_representation = semantic_units.new_zeros((batch_size, self.hidden_size))

        return {
            "alignment_scores": alignment_scores,
            "batch_size": batch_size,
            "coverage_pool_weights": coverage_pool_weights,
            "coverage_scores": coverage_scores,
            "covered_representation": covered_representation,
            "estimated_joint_coverage": estimated_joint_coverage,
            "historical_co_support": historical_co_support,
            "interaction_attention": interaction_attention,
            "interaction_residual_representation": interaction_residual_representation,
            "interaction_residual_scores": interaction_residual_scores,
            "interaction_strength": interaction_strength,
            "pair_weights": pair_weights,
            "raw_token_slot_assignment": raw_token_slot_assignment,
            "residual_scores": residual_scores,
            "semantic_attention": semantic_attention,
            "semantic_slot_mass": semantic_slot_mass,
            "semantic_target_mask": semantic_target_mask,
            "semantic_units": semantic_units,
            "target_representation": target_representation,
            "token_slot_assignment": token_slot_assignment,
            "unit_active_probability": unit_active_probability,
            "unit_residual_representation": unit_residual_representation,
            "weighted_support": weighted_support,
        }
