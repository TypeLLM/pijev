"""Evaluate all six option orders in one Jev request and report their mean."""
from itertools import permutations
from math import fsum
import json
import os
from pathlib import Path

from typesafe_sdk import Choice, TypeSafeClient


def main():
    env = Path(__file__).resolve().parents[1] / '.env'
    if env.exists() and not os.environ.get('TYPESAFE_API_KEY'):
        for line in env.read_text().splitlines():
            if line.startswith('TYPESAFE_API_KEY='):
                os.environ['TYPESAFE_API_KEY'] = line.split('=', 1)[1].strip()
                break
    criteria = {
        'billing': 'Payments, charges, invoices, and refunds.',
        'technical': 'Product failures, errors, and troubleshooting.',
        'account': 'Login, account access, and account settings.',
    }
    instructions = 'Which team should handle this support ticket? Choose the best primary team.'
    orders = dict(zip((f"__pijev_{i}" for i in range(6)), permutations(sorted(criteria))))
    with TypeSafeClient() as client:
        response = client.system_one(
            state={'ticket': 'Since upgrading my plan yesterday, I cannot access the dashboard. The payment went through, but the page says my subscription is inactive. Please fix this.'},
            questions={
                name: Choice(instructions=instructions,
                             criteria={label: criteria[label] for label in order})
                for name, order in orders.items()
            },
        )
    rows = [response.choices[name].probabilities for name in orders]
    means = {label: fsum(row[label] / fsum(row.values()) for row in rows) / len(rows)
             for label in criteria}
    print(json.dumps({
        'model': response.model,
        'usage': response.usage.model_dump(),
        'orders': orders,
        'answers': {name: response.choices[name].model_dump() for name in orders},
        'pijev_mean': means,
    }, indent=2))



if __name__ == '__main__':
    main()
