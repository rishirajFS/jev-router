import pytest

from jev_router.features import RoutingFeatures
from jev_router.policy import Policy, Tier
from jev_router.routing_eval import (
    HAIKU,
    OPUS,
    SONNET,
    ItemOutcome,
    assign_tiers,
    break_even_jev_cost,
    length_matched,
    oracle,
    random_matched,
    score_assignment,
    select_policy,
    single_model,
    split_ids,
)


def outcome(i, h, s, o, chars=100):
    return ItemOutcome(
        item_id=f"i{i}",
        family="mmlu",
        prompt_chars=chars,
        correct={HAIKU: h, SONNET: s, OPUS: o},
        cost={HAIKU: 0.001, SONNET: 0.004, OPUS: 0.010},
    )


OUTCOMES = [
    outcome(0, True, True, True, 50),
    outcome(1, True, True, True, 60),
    outcome(2, False, False, True, 500),
    outcome(3, False, True, True, 600),
]


def test_single_model_reports_accuracy_cost_and_qpd():
    r = single_model(OUTCOMES, HAIKU)
    assert r.accuracy == pytest.approx(0.5)
    assert r.total_cost == pytest.approx(0.004)
    assert r.correct_per_dollar == pytest.approx(2 / 0.004)
    o = single_model(OUTCOMES, OPUS)
    assert o.accuracy == 1.0 and o.total_cost == pytest.approx(0.04)


def test_score_assignment_adds_jev_overhead_per_call():
    tiers = [Tier.SMALL, Tier.SMALL, Tier.LARGE, Tier.LARGE]
    free = score_assignment(OUTCOMES, tiers)
    paid = score_assignment(OUTCOMES, tiers, jev_cost_per_call=0.002)
    assert free.accuracy == 1.0
    assert free.total_cost == pytest.approx(0.001 * 2 + 0.010 * 2)
    assert paid.total_cost == pytest.approx(free.total_cost + 0.008)
    assert paid.tier_counts == {"SMALL": 2, "LARGE": 2}


def test_oracle_picks_cheapest_correct_model():
    r = oracle(OUTCOMES)
    assert r.accuracy == 1.0
    assert r.total_cost == pytest.approx(0.001 + 0.001 + 0.010 + 0.004)


def test_random_matched_is_exact_expectation_of_random_assignment():
    r = random_matched(OUTCOMES, {"SMALL": 2, "LARGE": 2})
    assert r.accuracy == pytest.approx(0.5 * 0.5 + 0.5 * 1.0)
    assert r.total_cost == pytest.approx(4 * (0.5 * 0.001 + 0.5 * 0.010))


def test_length_matched_gives_longest_prompts_the_biggest_tier():
    r = length_matched(OUTCOMES, {"SMALL": 2, "LARGE": 2})
    assert r.accuracy == 1.0  # the two long (hard) prompts go to Opus
    assert r.total_cost == pytest.approx(0.001 * 2 + 0.010 * 2)


def test_break_even_jev_cost():
    jev = score_assignment(OUTCOMES, [Tier.SMALL, Tier.SMALL, Tier.LARGE, Tier.LARGE])
    other = single_model(OUTCOMES, OPUS)
    per_call = break_even_jev_cost(jev, other)
    assert per_call > 0
    paid = score_assignment(OUTCOMES, [Tier.SMALL] * 2 + [Tier.LARGE] * 2, jev_cost_per_call=per_call)
    assert paid.correct_per_dollar == pytest.approx(other.correct_per_dollar)


def test_break_even_is_negative_when_jev_loses_before_overhead():
    jev = score_assignment(OUTCOMES, [Tier.LARGE] * 4)
    cheaper = single_model(OUTCOMES, HAIKU)
    assert break_even_jev_cost(jev, cheaper) < 0


def feats(d):
    return RoutingFeatures("math", 0.9, d, 0.9, 0.1)


def test_assign_tiers_applies_policy_to_each_item():
    tiers = assign_tiers([feats(0.2), feats(1.5), feats(2.8)], Policy(small_max=0.9, medium_max=2.0))
    assert tiers == [Tier.SMALL, Tier.MEDIUM, Tier.LARGE]


def test_select_policy_takes_cheapest_policy_meeting_quality_floor():
    features = [feats(0.2), feats(0.3), feats(2.5), feats(2.6)]
    chosen, result = select_policy(OUTCOMES, features, quality_floor=1.0)
    tiers = assign_tiers(features, chosen)
    assert result.accuracy == 1.0
    assert tiers[:2] == [Tier.SMALL, Tier.SMALL] and tiers[2:] == [Tier.LARGE, Tier.LARGE]


def test_select_policy_falls_back_to_largest_tier_when_floor_unreachable():
    features = [feats(0.2)] * 4
    chosen, result = select_policy(OUTCOMES, features, quality_floor=1.5)
    assert set(assign_tiers(features, chosen)) == {Tier.LARGE}


def test_split_ids_is_deterministic_and_disjoint():
    ids = [f"id{i}" for i in range(100)]
    dev, test = split_ids(ids, seed=3)
    assert set(dev).isdisjoint(test) and len(dev) + len(test) == 100
    assert split_ids(ids, seed=3) == (dev, test)
    assert 30 < len(dev) < 70


def test_length_tiers_and_correct_vectors():
    from jev_router.routing_eval import correct_vector, length_tiers, random_correct_vector

    counts = {"SMALL": 2, "LARGE": 2}
    tiers = length_tiers(OUTCOMES, counts)
    assert tiers == [Tier.SMALL, Tier.SMALL, Tier.LARGE, Tier.LARGE]
    assert correct_vector(OUTCOMES, tiers) == [1.0, 1.0, 1.0, 1.0]
    expected = random_correct_vector(OUTCOMES, counts)
    assert expected[0] == pytest.approx(1.0)  # both models right
    assert expected[2] == pytest.approx(0.5)  # only Opus right, half the items get Opus
    assert sum(expected) / 4 == pytest.approx(random_matched(OUTCOMES, counts).accuracy)


def test_bootstrap_diff_brackets_the_mean_difference():
    from jev_router.routing_eval import bootstrap_diff

    a = [1.0] * 80 + [0.0] * 20
    b = [1.0] * 60 + [0.0] * 40
    mean, low, high = bootstrap_diff(a, b, seed=1, reps=500)
    assert mean == pytest.approx(0.2)
    assert low < 0.2 < high and low > 0.0
    with pytest.raises(ValueError):
        bootstrap_diff([1.0], [1.0, 0.0], seed=1)
