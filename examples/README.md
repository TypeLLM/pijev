# Jev option-order experiment

[`compare_orders.py`](compare_orders.py) sends two Choices in one `system_one`
request to compare raw probabilities for the same question with different option
orders. It uses the official `typesafe_sdk` directly, without pijev's permutation
expansion or probability aggregation.

## Run

From the pijev repository root:

```bash
python -m pip install -e .
python examples/compare_orders.py
```

Or use the project's existing virtual environment:

```bash
.venv/bin/python examples/compare_orders.py
```

The script reads `TYPESAFE_API_KEY` from the environment first. If unset, it reads
the same variable from `.env` in the repository root. Git ignores `.env`, and the
script does not print the API key. Each run calls the live API and incurs usage;
SDK retries may increase the number of HTTP requests.

## Request design

Both questions share this support ticket:

> Since upgrading my plan yesterday, I cannot access the dashboard. The payment went through, but the page says my subscription is inactive. Please fix this.

Both ask which team should primarily handle the ticket, using identical
instructions, labels, and option descriptions:

| Label | Description |
| --- | --- |
| billing | Payments, charges, invoices, and refunds. |
| technical | Product failures, errors, and troubleshooting. |
| account | Login, account access, and account settings. |

The single request contains these questions:

```python
questions={
    "forward": Choice(instructions=instructions, criteria=criteria),
    "reverse": Choice(
        instructions=instructions,
        criteria=dict(reversed(list(criteria.items()))),
    ),
}
```

The `forward` order is `billing → technical → account`; the `reverse` order is
`account → technical → billing`. The two question IDs also differ.

## Results from one live run

On 2026-09-22, using official Python SDK 0.7.1, the API returned model `jev-1.13.0`:

| Label | forward | reverse | reverse − forward |
| --- | ---: | ---: | ---: |
| billing | 0.43 | 0.63 | +0.20 |
| technical | 0.21 | 0.10 | −0.11 |
| account | 0.36 | 0.27 | −0.09 |

| Field | forward | reverse |
| --- | --- | --- |
| choice | billing | billing |
| confidence | 0.15 | 0.45 |

Reported batch usage: `input_tokens=475`, `output_tokens=73`.
The largest probability difference was **0.20, or 20 percentage points**. This is a
record of one run, not a fixed expected output. Reruns or model changes may produce
different results.

The script prints the model, usage, both orders, complete Choice answers, and
probability differences aligned by label. Here, `confidence` is Jev's original
value; it has not been replaced with the maximum probability.

## Interpretation and limitations

The two Choices in this request returned different probabilities even though they
selected the same label. This provides a concrete example for further investigation
of permutation averaging, but it does not isolate option order as the sole cause.
Question IDs, position within the batch, and model randomness have not been ruled
out through control experiments.

A more rigorous follow-up would repeat identical requests to measure variability,
swap the permutations assigned to the two question IDs, and swap the questions'
positions within the batch. Assessing whether permutation averaging improves
accuracy or calibration also requires a labeled dataset.

Averaging only these two probability vectors gives `billing=0.53`,
`technical=0.155`, and `account=0.315`. This is a calculation from the recorded
results, not a third answer returned by the script or an average over all six
permutations of the three options.

For normal pijev usage, you do not need to create two questions manually. Change
the client import and let pijev expand and aggregate permutations automatically.
See the [project usage guide](../README.md).

## All six permutations

Run the full three-option experiment in one request:

```bash
python examples/compare_all_orders.py
```

The script uses the same question and state, with all six permutations and the
same generated question IDs used by pijev. It reports each raw answer and the
mean after normalizing each probability vector, matching pijev's aggregation.
The [recorded response](six_orders_result.json) from `jev-1.13.0` reports 835 input
tokens and 244 output tokens. Mean probabilities were `billing=0.555`,
`technical=0.138333…`, and `account=0.306667…`. This is a separate live request
from the two-order experiment above.
