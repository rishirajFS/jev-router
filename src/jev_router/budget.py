"""Dollar accounting with a hard spending cap for paid model calls."""

from dataclasses import dataclass, field

# USD per 1M tokens: (input, output)
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


# The user has $15 of credit in total and no more to give; stay well clear of it.
HARD_LIMIT_USD = 14.0


# Jev bills input tokens only; output is free.
JEV_INPUT_USD_PER_M = 0.042


def jev_cost_usd(input_tokens: int) -> float:
    return input_tokens * JEV_INPUT_USD_PER_M / 1_000_000


class BudgetExceeded(RuntimeError):
    pass


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = PRICES[model]
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


@dataclass
class BudgetTracker:
    cap_usd: float
    spent: float = 0.0
    by_model: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.cap_usd > HARD_LIMIT_USD:
            raise ValueError(f"cap_usd may not exceed the ${HARD_LIMIT_USD:.2f} hard limit")

    def can_afford(self, model: str, input_tokens: int, output_tokens: int) -> bool:
        return self.spent + cost_usd(model, input_tokens, output_tokens) <= self.cap_usd

    def add(self, model: str, input_tokens: int, output_tokens: int) -> float:
        cost = cost_usd(model, input_tokens, output_tokens)
        self.spent += cost
        self.by_model[model] = self.by_model.get(model, 0.0) + cost
        if self.spent > self.cap_usd:
            raise BudgetExceeded(
                f"spent ${self.spent:.2f}, over the ${self.cap_usd:.2f} cap"
            )
        return cost
