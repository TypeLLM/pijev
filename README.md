# pijev — Permutation Invariant Jev

**Better Jev probabilities with theoretical guarantees—all with a one-line import change.**

pijev turns Jev's option-order variations into a single averaged prediction, with
Brier score and log loss guaranteed no worse than the average across the included
orderings. All permutations run in one `system_one` request. Keep the same calls
and official response types; pijev handles the rest.

## Problem: Jev's decision is not permutation invariant

The same options can receive different probabilities when reordered. This official
SDK call sends two Choices with identical instructions and descriptions, but
opposite option orders:

```python
from typesafe_sdk import Choice, TypeSafeClient

instructions = "Which team should handle this support ticket? Choose the best primary team."

with TypeSafeClient() as client:
    response = client.system_one(
        state={
            "ticket": "Since upgrading my plan yesterday, I cannot access the "
                      "dashboard. The payment went through, but the page says "
                      "my subscription is inactive. Please fix this."
        },
        questions={
            "forward": Choice(
                instructions=instructions,
                criteria={
                    "billing": "Payments, charges, invoices, and refunds.",
                    "technical": "Product failures, errors, and troubleshooting.",
                    "account": "Login, account access, and account settings.",
                },
            ),
            "reverse": Choice(
                instructions=instructions,
                criteria={
                    "account": "Login, account access, and account settings.",
                    "technical": "Product failures, errors, and troubleshooting.",
                    "billing": "Payments, charges, invoices, and refunds.",
                },
            ),
        },
    )

print(response.choices["forward"].probabilities)
print(response.choices["reverse"].probabilities)
```

One request, two questions, one shared state. A live `jev-1.13.0` run returned:

| Option | billing → technical → account | account → technical → billing |
| --- | ---: | ---: |
| billing | 0.43 | 0.63 |
| technical | 0.21 | 0.10 |
| account | 0.36 | 0.27 |

Both selected `billing`, but its probability changed by **20 percentage points**.
This run also changes question IDs and does not control model randomness; see the
[experiment details](examples/README.md).

## How pijev works

pijev treats option order as a nuisance variable: evaluate different orderings,
align predictions by label, and average their probabilities. The highest mean
probability determines the returned choice.

For three options, there are six permutations. Here is a separate live
`jev-1.13.0` run with **all six in one request**. A = account, B = billing,
T = technical.

| Option | A → B → T | A → T → B | B → A → T | B → T → A | T → A → B | T → B → A | pijev mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| billing | 0.58 | 0.64 | 0.58 | 0.46 | 0.47 | 0.60 | **0.5550** |
| technical | 0.17 | 0.08 | 0.11 | 0.23 | 0.12 | 0.12 | 0.1383 |
| account | 0.25 | 0.28 | 0.31 | 0.31 | 0.41 | 0.28 | 0.3067 |

The `pijev mean` column averages all six predictions: `billing` receives **0.555**,
compared with 0.46–0.64 across individual orderings.
[Reproduce this run](examples/compare_all_orders.py) · [Recorded results](examples/six_orders_result.json)

Ask once using the same SDK interface; pijev handles the six-question batch and
returns one aggregated answer:

```python
from pijev import Choice, TypeSafeClient

with TypeSafeClient() as client:
    response = client.system_one(
        state={
            "ticket": "Since upgrading my plan yesterday, I cannot access the "
                      "dashboard. The payment went through, but the page says "
                      "my subscription is inactive. Please fix this."
        },
        questions={
            "team": Choice(
                instructions="Which team should handle this support ticket? "
                             "Choose the best primary team.",
                criteria={
                    "billing": "Payments, charges, invoices, and refunds.",
                    "technical": "Product failures, errors, and troubleshooting.",
                    "account": "Login, account access, and account settings.",
                },
            ),
        },
    )

answer = response.choices["team"]
print(answer.choice)
print(answer.probabilities)
```

The result is an official `SystemOneResponse` with your original question names.
Live results may differ from the recorded run.

## Usage

Install from this repository (not yet published to PyPI):

```bash
pip install -e .
export TYPESAFE_API_KEY=your-key
```

Change only the import:

```diff
-from typesafe_sdk import TypeSafeClient, Choice, Noul, Score
+from pijev import TypeSafeClient, Choice, Noul, Score
```

Set an optional permutation budget:

```python
client = TypeSafeClient(n_permutations=16, seed=42)
```

The default is **up to 8 distinct permutations per Choice**. Smaller sets use all
permutations; larger sets use uniform sampling without replacement. Set
`n_permutations="all"` for full enumeration. The seed controls local sampling only.
Batches are capped at 720 expanded questions and remain subject to server limits.

Async usage works the same way: import `AsyncTypeSafeClient` from `pijev`, then use
`async with` and `await client.system_one(...)` as usual.

## Guaranteed Better Calibration

Permutation averaging has a **proper-scoring-loss guarantee**, not an unconditional
calibration guarantee. Let $p^{(m)}$ be the label-aligned probability vector for
permutation $m$, and let $y$ be the same correct answer under every ordering. Define

$$
\bar p = \frac{1}{M}\sum_{m=1}^{M}p^{(m)}.
$$

For any loss convex in the predicted probabilities, Jensen's inequality gives

$$
L(\bar p,y) \leq \frac{1}{M}\sum_{m=1}^{M}L(p^{(m)},y).
$$

This holds for the actual set of predictions being averaged, including a sampled
subset of permutations. Two useful cases are:

- **Log loss:** $-\log\bar p_y \leq \frac{1}{M}\sum_m -\log p_y^{(m)}$.
- **Brier score:** writing $e_y$ for the one-hot correct answer, the improvement
  equals the disagreement removed by averaging:

$$
\frac{1}{M}\sum_m\|p^{(m)}-e_y\|_2^2 - \|\bar p-e_y\|_2^2
= \frac{1}{M}\sum_m\|p^{(m)}-\bar p\|_2^2 \geq 0.
$$

The guarantee compares the average prediction with **randomly choosing one of the
included predictions**. Brier improvement is strict when those predictions differ.
For uniform sampling and a fixed predictor, the same inequality holds in expectation
against a uniformly random ordering. Full enumeration gives the group average;
see [Chen, Dobriban, and Lee](https://jmlr.csail.mit.edu/papers/v21/20-163.html)
for related symmetry-averaging theory.

## Cost analysis

For $K$ options and budget $B$, pijev evaluates $M=\min(B,K!)$ permutations in
**one HTTP request**. Multiple Choices contribute their own permutations;
Noul and Score each add one unchanged question.

| Workload | Questions in one request |
| --- | ---: |
| 3 options, default budget 8 | 6 |
| 10 options, default budget 8 | 8 |
| Two Choices with 3 options each | 12 |
| 3 options plus one Noul | 7 |

The state is sent once. For state size $S$ and question size $Q$, request content
is approximately $S+MQ$, versus $M(S+Q)$ for separate calls, excluding protocol
overhead. This is a payload estimate; use `response.usage` for reported tokens.

At TypeSafe's [published rate](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
of **$0.042 per million input tokens, with free output tokens** (checked 2026-09-22):

$$
C = \frac{T}{10^6}\times 0.042
$$

Here, $C$ is the estimated cost in USD and $T$ is `response.usage.input_tokens`.

| Recorded run | Input tokens | Estimated cost |
| --- | ---: | ---: |
| Two permutations | 475 | $0.00001995 |
| All six permutations | 835 | $0.00003507 |

These estimates use reported usage and the published rate, not invoice charges.
One request still contains multiple decisions; latency and billing depend on the
backend, and retries may add requests.

## Notes

- Only Choice answers are averaged. Noul and Score pass through unchanged.
- `confidence` is the selected label's mean probability, **not Jev's native
  confidence score**.
- Sampling approximates full permutation averaging; it does not guarantee exact
  order invariance or better calibration. The live examples demonstrate probability
  differences, not accuracy or calibration gains.
- Official interface: [TypeSafe Python SDK](https://github.com/typesafe-ai/typesafe-sdk-python).
  This is a third-party extension, not an official TypeSafe project.

## License

[Apache License 2.0](LICENSE).
