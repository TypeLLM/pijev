"""Send two reordered Choices in one real Jev request (no aggregation)."""
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
    with TypeSafeClient() as client:
        response = client.system_one(
            state={'ticket': 'Since upgrading my plan yesterday, I cannot access the dashboard. The payment went through, but the page says my subscription is inactive. Please fix this.'},
            questions={
                'forward': Choice(instructions=instructions, criteria=criteria),
                'reverse': Choice(instructions=instructions, criteria=dict(reversed(list(criteria.items())))),
            },
        )
    a, b = response.choices['forward'], response.choices['reverse']
    differences = {label: b.probabilities[label] - a.probabilities[label] for label in criteria}
    print(json.dumps({
        'model': response.model,
        'usage': response.usage.model_dump(),
        'orders': {'forward': list(criteria), 'reverse': list(reversed(criteria))},
        'answers': {name: answer.model_dump() for name, answer in response.choices.items()},
        'reverse_minus_forward': differences,
        'max_absolute_difference': max(map(abs, differences.values())),
    }, indent=2))


if __name__ == '__main__':
    main()
