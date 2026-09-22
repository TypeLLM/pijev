import asyncio
import json
import unittest

import httpx2
from pydantic import BaseModel
from typesafe_sdk import SystemOneResponse, ChoiceAnswer
from pijev import AsyncTypeSafeClient, Choice, Noul, Score, TypeSafeClient


class Tests(unittest.TestCase):
    def setUp(self):
        self.bodies = []
        self.bad = None
        self.omit = False

    def handle(self, request):
        body = json.loads(request.content)
        self.bodies.append(body)
        answers = {}
        for name, q in reversed(list(body['questions'].items())):
            if q['type'] == 'choice':
                labels = list(q['criteria'])
                probs = dict(zip(labels, [2 / (len(labels) + 1)] + [1 / (len(labels) + 1)] * (len(labels) - 1)))
                answers[name] = dict(type='choice', choice=labels[0], confidence=0.99,
                                     probabilities=self.bad if self.bad is not None else probs)
            elif q['type'] == 'noul':
                answers[name] = dict(type='noul', noul=0.7)
            else:
                answers[name] = dict(type='score', score=0.7, confidence=0.8,
                                     legend={'0': 'low', '1': 'high'}, probabilities={'0': 0.3, '1': 0.7})
        if self.omit:
            answers.pop(next(iter(answers)))
        return httpx2.Response(200, headers={'x-typesafe-request-id': 'test-id'}, json={
            'model': 'mock', 'usage': {'input_tokens': 10, 'output_tokens': len(answers)}, 'answers': answers})

    def client(self, **kwargs):
        client = TypeSafeClient(api_key='test', transport=httpx2.MockTransport(self.handle), **kwargs)
        self.addCleanup(client.close)
        return client

    def test_drop_in_mixed_batch_and_metadata(self):
        q = Choice(criteria={'b': ['B'], 'a': {'hint': 'A'}}, instructions='pick')
        before = q.model_dump()
        result = self.client().system_one('state', {'category': q, '__pijev_0': Noul(), 'rating': Score(criteria=['low', 'high'])})
        self.assertIs(type(result), SystemOneResponse)
        self.assertEqual(result.choices['category'].probabilities, {'a': 0.5, 'b': 0.5})
        self.assertEqual(result.choices['category'].choice, 'a')
        self.assertEqual(result.choices['category'].confidence, 0.5)
        self.assertEqual(result.nouls['__pijev_0'].noul, 0.7)
        self.assertEqual(result.scores['rating'].score, 0.7)
        self.assertEqual(list(result.answers), ['category', '__pijev_0', 'rating'])
        self.assertEqual(set(result.model_dump()), {'answers', 'model', 'usage'})
        self.assertEqual(result.request_id, 'test-id')
        self.assertEqual(result.usage.output_tokens, 4)
        self.assertEqual(result.raw_http_response.json()['answers'], result.model_dump(mode='json')['answers'])
        self.assertEqual(len(self.bodies), 1)
        self.assertEqual(len(self.bodies[0]['questions']), 4)
        self.assertEqual(q.model_dump(), before)

    def test_three_options_one_call_six_questions(self):
        result = self.client().system_one(state='x', questions={'q': Choice(criteria=dict.fromkeys('abc'))})
        self.assertEqual(len(self.bodies), 1)
        self.assertEqual(len(self.bodies[0]['questions']), 6)
        for p in result.choices['q'].probabilities.values():
            self.assertAlmostEqual(p, 1 / 3)

    def test_seed_and_multiple_choices(self):
        client = self.client(seed=42)
        q = {'type': 'choice', 'criteria': dict.fromkeys('abcd')}
        a = client.system_one('x', {'q': q, 'second': Choice(criteria={'only': None})})
        q['criteria'] = dict.fromkeys('dcba')
        b = client.system_one('x', {'q': q, 'second': Choice(criteria={'only': None})})
        self.assertEqual(a.answers, b.answers)
        self.assertEqual(self.bodies[0], self.bodies[1])
        orders = [tuple(q['criteria']) for q in self.bodies[0]['questions'].values()]
        self.assertEqual(len(set(orders)), 9)

    def test_custom_response_models(self):
        class Typed(SystemOneResponse):
            category: ChoiceAnswer
        class Plain(BaseModel):
            answers: dict[str, ChoiceAnswer]
        for target in [Typed, Plain]:
            result = self.client().system_one('x', {'category': Choice(criteria=dict.fromkeys('ab'))}, response_model=target)
            self.assertIsInstance(result, target)
            self.assertEqual(result.answers['category'].probabilities, {'a': 0.5, 'b': 0.5})
            if target is Typed:
                self.assertEqual(result.category, result.answers['category'])

    def test_extra_body_override_and_passthrough(self):
        client = self.client()
        result = client.system_one('old', {'old': Noul()}, extra_body={
            'state': 'new', 'model': 'chosen', 'questions': {'q': Choice(criteria=dict.fromkeys('ab'))}})
        self.assertEqual(list(result.answers), ['q'])
        self.assertEqual(self.bodies[0]['state'], 'new')
        self.assertEqual(self.bodies[0]['model'], 'chosen')
        result = client.system_one('x', {'n': Noul()})
        self.assertEqual(result.nouls['n'].noul, 0.7)
        self.assertEqual(list(self.bodies[1]['questions']), ['n'])

    def test_validation(self):
        for budget in [0, -1, True, 1.5, 'bad']:
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                self.client(n_permutations=budget)
        with self.assertRaises(ValueError):
            self.client(n_permutations='all').system_one('x', {'q': Choice(criteria=dict.fromkeys('abcdefgh'))})
        self.assertEqual(self.bodies, [])
        for bad in [{'a': 1.0}, {'a': -0.1, 'b': 1.1}, {'a': 0.0, 'b': 0.0}]:
            self.bad = bad
            with self.assertRaises(ValueError):
                self.client().system_one('x', {'q': Choice(criteria=dict.fromkeys('ab'))})
        self.bad, self.omit = None, True
        with self.assertRaises(ValueError):
            self.client().system_one('x', {'q': Choice(criteria=dict.fromkeys('ab'))})

    def test_async(self):
        async def run():
            async with AsyncTypeSafeClient(api_key='test', transport=httpx2.MockTransport(self.handle)) as client:
                return await client.system_one('x', {'q': Choice(criteria=dict.fromkeys('ab'))})
        result = asyncio.run(run())
        self.assertIs(type(result), SystemOneResponse)
        self.assertEqual(result.choices['q'].probabilities, {'a': 0.5, 'b': 0.5})
        self.assertEqual(len(self.bodies), 1)


if __name__ == '__main__':
    unittest.main()
