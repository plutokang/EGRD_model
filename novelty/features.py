"""Feature schema shared by historical evidence and novelty prediction."""

PAIR_LABELS = ["not_covering", "related_not_covering", "partial_cover", "large_cover"]
JOINT_LABELS = ["not_covered", "weakly_covered", "partially_covered", "mostly_covered"]
NOVELTY_LABELS = ["low_novelty", "weak_novelty", "moderate_novelty", "high_novelty"]

DEFAULT_RESIDUAL_TYPES = [
    "minor_variant",
    "new_mechanism",
    "new_theory_or_concept",
    "new_task_definition",
    "nontrivial_combination",
    "new_application_constraint",
    "new_training_signal",
    "new_system_design",
    "new_evaluation_protocol",
    "new_data_organization",
    "new_data_distribution",
    "new_objective",
    "new_constraint",
    "new_data_or_resource",
    "new_integration",
    "new_combination",
    "new_context",
    "surface_reformulation",
    "no_clear_residual",
]

COARSE_TYPES = [
    "minor_change",
    "conceptual_innovation",
    "mechanism_or_system",
    "task_or_application",
    "training_or_evaluation",
    "data_related",
    "combination",
]
INNOVATION_FAMILIES = COARSE_TYPES[1:]

FINE_TYPE_GROUPS = {
    "minor_change": [
        "minor_variant",
        "surface_reformulation",
        "no_clear_residual",
    ],
    "conceptual_innovation": [
        "new_theory_or_concept",
        "new_objective",
    ],
    "mechanism_or_system": [
        "new_mechanism",
        "new_system_design",
        "new_integration",
    ],
    "task_or_application": [
        "new_task_definition",
        "new_application_constraint",
        "new_constraint",
        "new_context",
    ],
    "training_or_evaluation": [
        "new_training_signal",
        "new_evaluation_protocol",
    ],
    "data_related": [
        "new_data_organization",
        "new_data_distribution",
        "new_data_or_resource",
    ],
    "combination": [
        "nontrivial_combination",
        "new_combination",
    ],
}

TYPE_TO_COARSE = {
    label: coarse
    for coarse, labels in FINE_TYPE_GROUPS.items()
    for label in labels
}

PAIR_TO_ID = {label: idx for idx, label in enumerate(PAIR_LABELS)}
JOINT_TO_ID = {label: idx for idx, label in enumerate(JOINT_LABELS)}
NOVELTY_TO_ID = {label: idx for idx, label in enumerate(NOVELTY_LABELS)}
COARSE_TO_ID = {label: idx for idx, label in enumerate(COARSE_TYPES)}

V22_REINIT_PREFIXES = (
    "semantic_queries",
    "unit_alignment_projection.",
    "prior_alignment_projection.",
    "support_mlp.",
    "coverage_pool_score.",
    "unit_active_head.",
    "interaction_mlp.",
    "interaction_strength_head.",
    "interaction_pool_score.",
    "coverage_gate.",
    "combination_head.",
    "flat_type_head.",
    "residual_novelty_head.",
    "novelty_fusion_gate.",
    "moderate_high_head.",
)

V24_SEMANTIC_REINIT_PREFIXES = (
    "semantic_queries",
    "unit_alignment_projection.",
    "prior_alignment_projection.",
    "support_mlp.",
    "coverage_pool_score.",
    "unit_active_head.",
)

PAIR_FEATURE_NAMES = [
    "pair_score",
    "rank_score",
    "p_not_covering",
    "p_related_not_covering",
    "p_partial_cover",
    "p_large_cover",
    "related_probability",
    "covering_probability",
    "large_probability_given_covering",
    "rank_fraction",
    "has_prior_text",
]

JOINT_GLOBAL_FEATURES = (
    "joint_evidence_score", "joint_expected_score", "joint_entropy", "joint_normalized_entropy",
    "has_joint_prediction", "joint_minus_pair_union",
    "p_not_covered", "p_weakly_covered", "p_partially_covered", "p_mostly_covered",
)
PAIR_GLOBAL_FEATURES = (
    "num_valid_priors_fraction", "max_pair_score", "mean_pair_score", "std_pair_score",
    "sum_pair_score", "pair_union_score", "covering_prior_fraction", "related_prior_fraction",
    "top1_top2_rank_margin", "pair_score_entropy",
)

ABLATION_TARGETS = ("joint", "pair", "pair_meta", "scalars", "target", "egrd",
                    "egrd_residual", "egrd_covered", "global", "type")

GLOBAL_FEATURE_NAMES = [
    "target_date_scaled",
    "num_valid_priors_fraction",
    "max_pair_score",
    "mean_pair_score",
    "std_pair_score",
    "sum_pair_score",
    "pair_union_score",
    "covering_prior_fraction",
    "related_prior_fraction",
    "top1_top2_rank_margin",
    "pair_score_entropy",
    "joint_evidence_score",
    "joint_expected_score",
    "joint_entropy",
    "joint_normalized_entropy",
    "has_joint_prediction",
    "joint_minus_pair_union",
    "p_not_covered",
    "p_weakly_covered",
    "p_partially_covered",
    "p_mostly_covered",
]

