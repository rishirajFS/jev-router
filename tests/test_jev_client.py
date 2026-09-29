import json

import httpx
import pytest

from jev_router.jev_client import JevClient, JevError

OK_BODY = {
    "model": "jev-1.13.0",
    "answers": {"q": {"type": "noul", "noul": 0.9}},
    "usage": {"input_tokens": 10, "output_tokens": 2},
}


def make_client(handler, **kwargs):
    return JevClient(
        api_key="secret-key",
        transport=httpx.MockTransport(handler),
        backoff_seconds=0,
        **kwargs,
    )


def test_analyze_sends_bearer_auth_and_payload():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=OK_BODY)

    client = make_client(handler)
    result = client.analyze("hello", {"q": {"type": "noul", "instructions": "?"}})

    assert seen["auth"] == "Bearer secret-key"
    assert seen["body"]["state"] == "hello"
    assert seen["body"]["model"] == "jev-latest"
    assert result.answers["q"]["noul"] == 0.9
    assert result.usage == {"input_tokens": 10, "output_tokens": 2}


def test_retries_on_server_error_then_succeeds():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, json={"detail": "busy"})
        return httpx.Response(200, json=OK_BODY)

    client = make_client(handler, max_retries=3)
    assert client.analyze("x", {}).answers
    assert calls["n"] == 3


def test_retries_on_rate_limit():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, json={"detail": "slow down"})
        return httpx.Response(200, json=OK_BODY)

    assert make_client(handler, max_retries=2).analyze("x", {}).answers
    assert calls["n"] == 2


def test_client_error_is_not_retried_and_hides_key():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(422, json={"detail": "bad schema"})

    client = make_client(handler, max_retries=3)
    with pytest.raises(JevError) as exc:
        client.analyze("x", {})
    assert calls["n"] == 1
    assert "422" in str(exc.value)
    assert "secret-key" not in str(exc.value)


def test_gives_up_after_max_retries():
    def handler(request):
        return httpx.Response(500, json={"detail": "boom"})

    with pytest.raises(JevError):
        make_client(handler, max_retries=2).analyze("x", {})


def test_malformed_body_is_retried_then_raises_jeverror():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(200, content=b"<html>proxy error</html>")

    with pytest.raises(JevError):
        make_client(handler, max_retries=2).analyze("x", {})
    assert calls["n"] == 2


def test_body_without_answers_is_retried_and_recovers():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(200, json={"oops": True})
        return httpx.Response(200, json=OK_BODY)

    assert make_client(handler, max_retries=2).analyze("x", {}).answers


def test_rejects_non_positive_retries():
    with pytest.raises(ValueError):
        JevClient(api_key="k", max_retries=0)


def test_rejects_insecure_base_url_but_allows_localhost():
    with pytest.raises(ValueError):
        JevClient(api_key="k", base_url="http://api.example.com/v1")
    JevClient(api_key="k", base_url="http://localhost:8000/v1").close()


def test_repr_never_contains_key():
    assert "secret-key" not in repr(JevClient(api_key="secret-key"))
