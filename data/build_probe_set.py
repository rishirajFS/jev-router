"""Generates data/probe_prompts.jsonl. Labels are hand-assigned judgements:
0 = a small model answers reliably, 1 = needs a mid-size model, 2 = needs a large model."""

import json
from pathlib import Path

ROWS = [
    # tier 0: small model suffices
    ("chat", 0, "hey, how's it going today?"),
    ("chat", 0, "Thanks, that was helpful!"),
    ("chat", 0, "Can you tell me a fun fact about octopuses?"),
    ("chat", 0, "Say good morning in French, Spanish, and German."),
    ("factual_qa", 0, "What is the capital of Australia?"),
    ("factual_qa", 0, "Who wrote Pride and Prejudice?"),
    ("factual_qa", 0, "How many days are in a leap year?"),
    ("factual_qa", 0, "What does HTTP stand for?"),
    ("classification", 0, "Is this review positive or negative? 'The food was cold and the waiter ignored us.'"),
    ("classification", 0, "Label the language of this text: 'Ich möchte einen Kaffee, bitte.'"),
    ("classification", 0, "Categorize this email subject as spam or not spam: 'You WON a free cruise!!! Click now'"),
    ("extraction", 0, "Extract the email address from: 'Contact Dana at dana.k@example.com before Friday.'"),
    ("extraction", 0, "From 'Order #4471 shipped on 2024-03-02 to Leeds', give me the order number and date."),
    ("extraction", 0, "Convert this list to JSON: apples, pears, plums."),
    ("summarization", 0, "Summarize in one sentence: 'The meeting covered Q3 budget, hiring two engineers, and moving the launch to October.'"),
    ("summarization", 0, "Rewrite this politely: 'Send me the report now.'"),
    ("math", 0, "What is 15% of 80?"),
    ("math", 0, "What is 12 times 9?"),
    ("code", 0, "Write a Python function that returns the square of a number."),
    ("code", 0, "How do I print 'hello' in JavaScript?"),
    ("writing", 0, "Write a two-line birthday message for my coworker."),
    ("writing", 0, "Give me three synonyms for 'happy'."),
    # tier 1: mid-size model needed
    ("math", 1, "A train leaves at 2:15pm going 60 mph and a second leaves the same station at 3:00pm going 80 mph. When does the second catch up?"),
    ("math", 1, "Solve for x: 3x^2 - 12x + 9 = 0, and explain each step."),
    ("math", 1, "A shop marks up items 40% then offers 25% off. What is the net change in price compared to the original cost?"),
    ("code", 1, "Write a Python function that merges overlapping intervals in a list of [start, end] pairs."),
    ("code", 1, "This SQL query is slow: SELECT * FROM orders o JOIN customers c ON c.id = o.customer_id WHERE o.created_at > '2024-01-01'. Suggest indexes and rewrites."),
    ("code", 1, "Write a React hook that debounces a value and cleans up its timer on unmount."),
    ("code", 1, "Explain why this Python code raises 'RuntimeError: dictionary changed size during iteration' and fix it: for k in d: if d[k] < 0: del d[k]"),
    ("code", 1, "Write unit tests with pytest for a function that parses ISO-8601 dates and raises ValueError on bad input."),
    ("summarization", 1, "Summarize the main arguments for and against remote work in three bullet points each, in a neutral tone, for an executive audience."),
    ("summarization", 1, "Rewrite this paragraph for a general audience while keeping every technical claim accurate: 'Gradient descent iteratively updates parameters in the direction opposing the loss gradient, scaled by a learning rate, converging to a local minimum for convex-enough objectives.'"),
    ("extraction", 1, "Given this messy invoice text, extract vendor, line items with quantities and unit prices, tax, and total as JSON, flagging any lines where the arithmetic does not add up: 'ACME Co. 3x widget @ 4.50 = 13.00; 2x gadget @ 10 = 20; tax 8% = 2.64; total 35.64'"),
    ("classification", 1, "Classify each of these 12 support tickets into billing, bug, feature request, or account access, and give a one-line reason for each. [tickets omitted for brevity, assume typical SaaS support messages]"),
    ("factual_qa", 1, "Explain the difference between TCP and UDP and when each is the better choice for a multiplayer game."),
    ("factual_qa", 1, "What caused the 2008 financial crisis, and which policy responses mattered most?"),
    ("factual_qa", 1, "Compare how PostgreSQL and MySQL implement MVCC and what that means for long-running transactions."),
    ("writing", 1, "Write a 300-word cover letter for a backend engineer applying to a fintech startup, emphasizing reliability and testing."),
    ("writing", 1, "Draft a clear incident postmortem for a 40-minute outage caused by an expired TLS certificate."),
    ("reasoning", 1, "I have two job offers: one pays 15% more but has a 90-minute commute; the other is remote. List the factors I should weigh and how."),
    ("reasoning", 1, "Plan a three-day itinerary for Lisbon for a family with two young children, balancing walking and rest."),
    ("reasoning", 1, "Our API returns 500s for about 2% of requests under load. Give me a prioritized debugging plan."),
    ("math", 1, "How many ways can 5 people sit around a round table if two of them refuse to sit next to each other?"),
    ("code", 1, "Convert this recursive Fibonacci function to use memoization and explain the complexity change."),
    ("reasoning", 1, "Two candidate architectures: a monolith with a job queue, or three microservices. Our team is 4 engineers. Which should we pick and why?"),
    # tier 2: large model needed
    ("math", 2, "Prove that there are infinitely many primes congruent to 3 mod 4."),
    ("math", 2, "Let G be a finite group in which every element satisfies x^2 = e. Prove G is abelian, then determine all possible orders of such groups and their structure."),
    ("math", 2, "Compute the expected number of coin flips until the pattern HTH first appears, and derive it using a Markov chain, not by guessing."),
    ("math", 2, "Show that the sum over n of 1/n^2 converges to pi^2/6 using Fourier series of x^2 on [-pi, pi], carefully justifying each step."),
    ("code", 2, "Design and implement a lock-free multi-producer single-consumer queue in C++ with correct memory ordering, and explain why each atomic operation uses the ordering it does."),
    ("code", 2, "Find the race condition in this Go worker-pool that occasionally deadlocks on shutdown, and rewrite it so it is provably correct: [assume 80 lines of channel-based worker code with a shared WaitGroup and a done channel closed twice on error paths]"),
    ("code", 2, "Implement a B-tree with deletion and rebalancing in Rust, including property-based tests, without any unsafe code."),
    ("code", 2, "Write a query planner cost model for a distributed SQL engine that chooses between broadcast and shuffle joins, and justify the cost formulas from first principles."),
    ("reasoning", 2, "A distributed system uses leader election with lease-based leadership. Under what clock-skew and network-partition conditions can two nodes both believe they are leader, and how would fencing tokens fix each case? Give a concrete failure timeline."),
    ("reasoning", 2, "Critique this experimental design for measuring the causal effect of a recommendation algorithm on retention, identifying every threat to validity and proposing a corrected design with a power calculation: [assume a 3-paragraph A/B test description with network effects and novelty bias]"),
    ("reasoning", 2, "Given conflicting evidence from two randomized trials with different populations and endpoints, how should a clinician decide whether to prescribe a new anticoagulant to an elderly patient with renal impairment? Structure the argument formally."),
    ("reasoning", 2, "Analyze the game-theoretic equilibria of a three-firm Cournot competition with asymmetric costs and a regulatory entry threat, and discuss which are subgame perfect."),
    ("factual_qa", 2, "Explain how the Raft consensus protocol guarantees log matching and leader completeness, and construct a scenario showing why the commit rule needs the current-term restriction."),
    ("factual_qa", 2, "Explain why the Transformer's attention cost is quadratic, then compare in depth how FlashAttention, Linformer, and state-space models each address it and what they trade away."),
    ("writing", 2, "Write a rigorous 500-word related-work section positioning a new confidence-calibrated LLM router against RouteLLM, FrugalGPT, and cascade-based methods, with accurate technical distinctions."),
    ("writing", 2, "Draft a persuasive but technically honest architecture decision record for migrating a 10-year-old payments monolith to event sourcing, addressing regulatory audit requirements and rollback risk."),
    ("summarization", 2, "Read this dense legal clause and explain in plain English every way the indemnification obligation could survive termination, citing the interacting subsections: [assume a 600-word clause with cross-references to sections 9.2, 11.4, and 14.1]"),
    ("extraction", 2, "From this 40-page contract excerpt, build a table of every obligation, its owner, its deadline, and conflicting clauses, resolving ambiguity by noting which interpretation is legally safer: [assume long multi-party contract text]"),
    ("classification", 2, "Decide whether each of these 8 borderline social media posts violates a policy that bans 'targeted harassment but not criticism of public figures', explaining the reasoning for edge cases: [assume 8 nuanced posts involving sarcasm and quoted speech]"),
    ("reasoning", 2, "Design a rollout and rollback strategy for a schema migration on a 5TB Postgres table serving 20k QPS with zero downtime, including the exact sequence, locking behavior at each step, and failure modes."),
    ("math", 2, "Derive the closed form of the posterior for a Bayesian linear regression with unknown noise variance under a normal-inverse-gamma prior, and show the marginal likelihood."),
    ("code", 2, "Optimize this matrix multiplication kernel in CUDA for a 4096x4096 float16 problem: explain tiling, shared-memory bank conflicts, and tensor core usage, and give code."),
]


def main() -> None:
    lines = []
    for i, (task, tier, text) in enumerate(ROWS):
        lines.append(json.dumps({"id": f"p{i:03d}", "task_type": task, "expected_tier": tier, "text": text}))
    out = Path(__file__).parent / "probe_prompts.jsonl"
    out.write_text("\n".join(lines) + "\n")
    counts = {t: sum(1 for r in ROWS if r[1] == t) for t in (0, 1, 2)}
    print(f"wrote {len(ROWS)} prompts to {out.name}: {counts}")


if __name__ == "__main__":
    main()
