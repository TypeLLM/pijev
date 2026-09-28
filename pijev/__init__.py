"""Drop-in TypeSafe clients with batched permutation averaging for Choice."""
from collections.abc import Mapping
from itertools import permutations
from math import factorial, fsum, isclose, isfinite
from random import Random
from typing import Any, Literal, TypeVar, overload
from pydantic import BaseModel

import httpx2
from typesafe_sdk import *  # Re-export the official public types.
from typesafe_sdk import __all__ as _sdk_exports
from typesafe_sdk import TypeSafeClient as _Client, AsyncTypeSafeClient as _AsyncClient

__all__ = list(_sdk_exports)
ResponseT = TypeVar("ResponseT", bound=BaseModel)


def _validate_budget(budget):
    if budget not in ("all", "auto") and (type(budget) is not int or budget <= 0):
        raise ValueError("n_permutations must be a positive integer, 'auto' or 'all'")


def _balanced_orders(count):
    """A balanced Latin square (Williams design) on positions 0..count-1.

    Every option takes every position equally often and follows every other
    option equally often: count orders when count is even, 2*count when odd.
    """
    first, low, high = [0], 1, count - 1
    while len(first) < count:
        first.append(low)
        low += 1
        if len(first) < count:
            first.append(high)
            high -= 1
    rows = [tuple((item + shift) % count for item in first) for shift in range(count)]
    if count % 2:
        rows += [row[::-1] for row in rows]
    return list(dict.fromkeys(rows))


def _prepare(questions, budget, seed):
    expanded, groups = {}, {}
    # Reserve all original names, including names resembling generated IDs.
    reserved = set(questions)
    for name, question in questions.items():
        if isinstance(question, Choice):
            choice = question
        elif isinstance(question, Mapping) and question.get("type") == "choice":
            choice = Choice.model_validate(question)
        else:
            expanded[name] = question
            continue
        labels = tuple(sorted(choice.criteria))
        if not labels:
            raise ValueError("Choice criteria must not be empty")
        total = factorial(len(labels))
        if budget == "auto":
            # Labels are sorted, so the orders do not depend on how the criteria were written.
            orders = [tuple(labels[i] for i in row) for row in _balanced_orders(len(labels))]
            count = len(orders)
        else:
            count = total if budget == "all" else min(budget, total)
        if len(expanded) + count > 720:
            raise ValueError("At most 720 expanded questions per request")
        if budget != "auto" and count == total:
            orders = permutations(labels)
        elif budget != "auto":
            rng, selected = Random(seed), {}
            while len(selected) < count:
                selected[tuple(rng.sample(labels, len(labels)))] = None
            orders = selected
        names = []
        for order in orders:
            key = f"__pijev_{len(expanded)}"
            while key in reserved:
                key += "_"
            reserved.add(key)
            item = choice.model_copy(deep=True)
            item.criteria = {label: item.criteria[label] for label in order}
            expanded[key] = item
            names.append(key)
        groups[name] = (labels, names)
    if len(expanded) > 720:
        raise ValueError("At most 720 expanded questions per request")
    return expanded, groups


def _aggregate(response, questions, groups, response_model):
    answers = {}
    for name in questions:
        if name not in groups:
            if name not in response.answers:
                raise ValueError(f"Backend did not return answer {name!r}")
            answers[name] = response.answers[name].model_dump(mode="json")
            continue
        labels, names = groups[name]
        rows = []
        for key in names:
            if key not in response.choices:
                raise ValueError(f"Backend did not return Choice answer {key!r}")
            raw = response.choices[key].probabilities
            if set(raw) != set(labels):
                raise ValueError("Backend probabilities must contain exactly the requested labels")
            if any(not isfinite(p) or not 0 <= p <= 1 for p in raw.values()):
                raise ValueError("Backend probabilities must be finite and between 0 and 1")
            mass = fsum(raw.values())
            # The API rounds each probability to two decimals, so the sum may be off by
            # up to 0.005 per option; the renormalisation below absorbs it.
            if mass <= 0 or not isclose(mass, 1.0, rel_tol=0, abs_tol=0.005 * len(labels) + 1e-9):
                raise ValueError("Backend probabilities must sum to approximately 1")
            rows.append({label: raw[label] / mass for label in labels})
        means = {label: fsum(row[label] for row in rows) / len(rows) for label in labels}
        winner = max(labels, key=means.__getitem__)
        answers[name] = ChoiceAnswer(choice=winner, probabilities=means,
                                     confidence=means[winner]).model_dump(mode="json")
    body = response.model_dump(mode="json")
    body["answers"] = answers
    raw = response.raw_http_response
    # Parse through the official response decoder, including custom typed fields.
    # This response represents the transformed body, with actual request metadata.
    headers = {k: v for k, v in raw.headers.items()
               if k.lower() not in {"content-length", "content-encoding"}}
    transformed = httpx2.Response(raw.status_code, headers=headers, json=body,
                                 request=raw.request)
    target = response_model or SystemOneResponse
    if issubclass(target, SystemOneResponse):
        return target.from_http_response(transformed)
    return target.model_validate_json(transformed.content)


def _request_inputs(state, questions, extra_body, budget, seed):
    # Match the SDK's documented extra_body last-write-wins semantics.
    extra = dict(extra_body or {})
    state = extra.pop("state", state)
    questions = extra.pop("questions", questions)
    expanded, groups = _prepare(questions, budget, seed)
    return state, questions, expanded, groups, extra


class TypeSafeClient(_Client):
    """Official sync client interface; Choice permutations share one request."""

    def __init__(self, *args: Any, n_permutations: int | Literal["auto", "all"] = "auto",
                 seed: int | None = None, **kwargs: Any) -> None:
        _validate_budget(n_permutations)
        super().__init__(*args, **kwargs)
        self.n_permutations, self.seed = n_permutations, seed

    @overload
    def system_one(self, state: JSONContent, questions: Questions, *,
                   response_model: type[ResponseT], **kwargs: Any) -> ResponseT: ...

    @overload
    def system_one(self, state: JSONContent, questions: Questions, *,
                   response_model: None = None, **kwargs: Any) -> SystemOneResponse: ...

    def system_one(self, state: JSONContent, questions: Questions, *,
                   model: str | None = None, retry: RetryPolicy | None = None,
                   timeout: float | httpx2.Timeout | None = None,
                   extra_headers: Mapping[str, str] | None = None,
                   extra_body: Mapping[str, JSONValue | None] | None = None,
                   response_model: type[ResponseT] | None = None) -> SystemOneResponse | ResponseT:
        _validate_budget(self.n_permutations)
        state, original, expanded, groups, extra = _request_inputs(
            state, questions, extra_body, self.n_permutations, self.seed)
        result = super().system_one(
            state, expanded, model=model, retry=retry, timeout=timeout,
            extra_headers=extra_headers, extra_body=extra,
            response_model=None if groups else response_model)
        return _aggregate(result, original, groups, response_model) if groups else result


class AsyncTypeSafeClient(_AsyncClient):
    """Official async client interface with the same aggregation as sync."""

    def __init__(self, *args: Any, n_permutations: int | Literal["auto", "all"] = "auto",
                 seed: int | None = None, **kwargs: Any) -> None:
        _validate_budget(n_permutations)
        super().__init__(*args, **kwargs)
        self.n_permutations, self.seed = n_permutations, seed

    @overload
    async def system_one(self, state: JSONContent, questions: Questions, *,
                   response_model: type[ResponseT], **kwargs: Any) -> ResponseT: ...

    @overload
    async def system_one(self, state: JSONContent, questions: Questions, *,
                   response_model: None = None, **kwargs: Any) -> SystemOneResponse: ...

    async def system_one(self, state: JSONContent, questions: Questions, *,
                   model: str | None = None, retry: RetryPolicy | None = None,
                   timeout: float | httpx2.Timeout | None = None,
                   extra_headers: Mapping[str, str] | None = None,
                   extra_body: Mapping[str, JSONValue | None] | None = None,
                   response_model: type[ResponseT] | None = None) -> SystemOneResponse | ResponseT:
        _validate_budget(self.n_permutations)
        state, original, expanded, groups, extra = _request_inputs(
            state, questions, extra_body, self.n_permutations, self.seed)
        result = await super().system_one(
            state, expanded, model=model, retry=retry, timeout=timeout,
            extra_headers=extra_headers, extra_body=extra,
            response_model=None if groups else response_model)
        return _aggregate(result, original, groups, response_model) if groups else result
