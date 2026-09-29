"""Maps Jev routing features to a model tier."""

from dataclasses import dataclass
from enum import IntEnum

from jev_router.features import RoutingFeatures


class Tier(IntEnum):
    SMALL = 0
    MEDIUM = 1
    LARGE = 2


@dataclass(frozen=True)
class Policy:
    """Thresholds on the 0-3 difficulty score plus safety escalations."""

    small_max: float = 0.9
    medium_max: float = 2.0
    min_confidence: float = 0.5
    multistep_threshold: float = 0.9

    def __post_init__(self) -> None:
        if self.small_max > self.medium_max:
            raise ValueError("small_max must not exceed medium_max")


@dataclass(frozen=True)
class Decision:
    tier: Tier
    base_tier: Tier
    escalated_for_low_confidence: bool
    escalated_for_multistep: bool


def _base_tier(difficulty: float, policy: Policy) -> Tier:
    if difficulty <= policy.small_max:
        return Tier.SMALL
    if difficulty <= policy.medium_max:
        return Tier.MEDIUM
    return Tier.LARGE


def route(features: RoutingFeatures, policy: Policy) -> Decision:
    base = _base_tier(features.difficulty, policy)
    low_conf = features.difficulty_confidence < policy.min_confidence
    multistep = features.multistep >= policy.multistep_threshold
    bump = 1 if (low_conf or multistep) else 0
    tier = Tier(min(base + bump, Tier.LARGE))
    return Decision(
        tier=tier,
        base_tier=base,
        escalated_for_low_confidence=low_conf and tier > base,
        escalated_for_multistep=multistep and not low_conf and tier > base,
    )
