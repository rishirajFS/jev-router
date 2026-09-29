"""Quality-per-dollar evaluation of routing strategies, with and without Jev."""

import hashlib
import itertools
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from jev_router.anthropic_runner import MODELS, RunRecord
from jev_router.features import RoutingFeatures
from jev_router.policy import Policy, Tier, route
from jev_router.sampling import BenchItem

HAIKU, SONNET, OPUS = MODELS
TIER_MODEL = {Tier.SMALL: HAIKU, Tier.MEDIUM: SONNET, Tier.LARGE: OPUS}
_GRID = [-0.01] + [round(0.1 * k, 1) for k in range(31)]


@dataclass(frozen=True)
class ItemOutcome:
    item_id: str
    family: str
    prompt_chars: int
    correct: Mapping[str, bool]
    cost: Mapping[str, float]


@dataclass(frozen=True)
class StrategyResult:
    n: int
    correct: float
    model_cost: float
    overhead_cost: float = 0.0
    tier_counts: Mapping[str, int] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        return self.correct / self.n

    @property
    def total_cost(self) -> float:
        return self.model_cost + self.overhead_cost

    @property
    def correct_per_dollar(self) -> float:
        return self.correct / self.total_cost if self.total_cost > 0 else float("inf")


def build_outcomes(
    items: Sequence[BenchItem], records: Sequence[RunRecord]
) -> list[ItemOutcome]:
    by_key = {(r.item_id, r.model): r for r in records}
    outcomes = []
    for item in items:
        rows = {m: by_key[(item.id, m)] for m in MODELS}
        outcomes.append(
            ItemOutcome(
                item_id=item.id,
                family=item.family,
                prompt_chars=len(item.prompt),
                correct={m: rows[m].correct for m in MODELS},
                cost={m: rows[m].cost_usd for m in MODELS},
            )
        )
    return outcomes


def _tier_counts(tiers: Sequence[Tier]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for tier in tiers:
        counts[tier.name] = counts.get(tier.name, 0) + 1
    return counts


def single_model(outcomes: Sequence[ItemOutcome], model: str) -> StrategyResult:
    return StrategyResult(
        n=len(outcomes),
        correct=sum(o.correct[model] for o in outcomes),
        model_cost=sum(o.cost[model] for o in outcomes),
    )


def score_assignment(
    outcomes: Sequence[ItemOutcome],
    tiers: Sequence[Tier],
    jev_cost_per_call: float = 0.0,
) -> StrategyResult:
    models = [TIER_MODEL[t] for t in tiers]
    return StrategyResult(
        n=len(outcomes),
        correct=sum(o.correct[m] for o, m in zip(outcomes, models, strict=True)),
        model_cost=sum(o.cost[m] for o, m in zip(outcomes, models, strict=True)),
        overhead_cost=jev_cost_per_call * len(outcomes),
        tier_counts=_tier_counts(tiers),
    )


def oracle(outcomes: Sequence[ItemOutcome]) -> StrategyResult:
    """Upper bound: the cheapest model that answers each item correctly."""
    correct = cost = 0.0
    for o in outcomes:
        winners = [m for m in MODELS if o.correct[m]]
        pool = winners or list(MODELS)
        best = min(pool, key=lambda m: o.cost[m])
        correct += bool(winners)
        cost += o.cost[best]
    return StrategyResult(n=len(outcomes), correct=correct, model_cost=cost)


def random_matched(
    outcomes: Sequence[ItemOutcome], tier_counts: Mapping[str, int]
) -> StrategyResult:
    """Exact expectation of assigning tiers at random with the same tier mix."""
    n = len(outcomes)
    correct = cost = 0.0
    for name, count in tier_counts.items():
        model = TIER_MODEL[Tier[name]]
        share = count / n
        correct += share * sum(o.correct[model] for o in outcomes)
        cost += share * sum(o.cost[model] for o in outcomes)
    return StrategyResult(n=n, correct=correct, model_cost=cost, tier_counts=dict(tier_counts))


def length_tiers(
    outcomes: Sequence[ItemOutcome], tier_counts: Mapping[str, int]
) -> list[Tier]:
    """Non-Jev heuristic: same tier mix, longest prompts get the biggest model."""
    order = sorted(range(len(outcomes)), key=lambda i: (outcomes[i].prompt_chars, i))
    tiers: list[Tier | None] = [None] * len(outcomes)
    cursor = 0
    for tier in Tier:
        count = tier_counts.get(tier.name, 0)
        for index in order[cursor : cursor + count]:
            tiers[index] = tier
        cursor += count
    return [t for t in tiers if t is not None]


def length_matched(
    outcomes: Sequence[ItemOutcome], tier_counts: Mapping[str, int]
) -> StrategyResult:
    return score_assignment(outcomes, length_tiers(outcomes, tier_counts))


def correct_vector(outcomes: Sequence[ItemOutcome], tiers: Sequence[Tier]) -> list[float]:
    return [float(o.correct[TIER_MODEL[t]]) for o, t in zip(outcomes, tiers, strict=True)]


def random_correct_vector(
    outcomes: Sequence[ItemOutcome], tier_counts: Mapping[str, int]
) -> list[float]:
    """Per-item expected correctness under random tier assignment with this mix."""
    n = len(outcomes)
    return [
        sum(count / n * o.correct[TIER_MODEL[Tier[name]]] for name, count in tier_counts.items())
        for o in outcomes
    ]


def bootstrap_diff(
    a: Sequence[float], b: Sequence[float], seed: int, reps: int = 2000
) -> tuple[float, float, float]:
    """Paired bootstrap of mean(a) - mean(b): (mean, 2.5th pct, 97.5th pct)."""
    if len(a) != len(b) or not a:
        raise ValueError("bootstrap_diff needs two equal-length, non-empty sequences")
    rng = random.Random(seed)
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    return sum(diffs) / n, means[int(0.025 * reps)], means[int(0.975 * reps) - 1]


def break_even_jev_cost(jev: StrategyResult, other: StrategyResult) -> float:
    """Max Jev price per call at which Jev routing still matches `other` on correct-per-dollar."""
    return (jev.correct / other.correct_per_dollar - jev.model_cost) / jev.n


def assign_tiers(features: Sequence[RoutingFeatures], policy: Policy) -> list[Tier]:
    return [route(f, policy).tier for f in features]


def _grid_policies() -> list[Policy]:
    return [
        Policy(small_max=s, medium_max=m, min_confidence=0.0, multistep_threshold=1.01)
        for s, m in itertools.combinations_with_replacement(_GRID, 2)
    ]


def select_policy(
    outcomes: Sequence[ItemOutcome],
    features: Sequence[RoutingFeatures],
    quality_floor: float,
) -> tuple[Policy, StrategyResult]:
    """Cheapest difficulty-threshold policy whose accuracy reaches `quality_floor`."""
    best: tuple[Policy, StrategyResult] | None = None
    for policy in _grid_policies():
        result = score_assignment(outcomes, assign_tiers(features, policy))
        if result.accuracy + 1e-12 < quality_floor:
            continue
        if best is None or result.total_cost < best[1].total_cost:
            best = (policy, result)
    if best is not None:
        return best
    fallback = Policy(small_max=-0.01, medium_max=-0.01, min_confidence=0.0, multistep_threshold=1.01)
    return fallback, score_assignment(outcomes, assign_tiers(features, fallback))


def split_ids(ids: Sequence[str], seed: int) -> tuple[list[str], list[str]]:
    dev: list[str] = []
    test: list[str] = []
    for item_id in ids:
        digest = hashlib.sha256(f"{seed}:{item_id}".encode()).digest()
        (dev if digest[0] % 2 == 0 else test).append(item_id)
    return dev, test
