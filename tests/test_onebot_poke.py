from __future__ import annotations

from typing import Any

import pytest

from qichi.transport.onebot_client import OneBotClient, OneBotProtocolError


class FakeResponse:
    def __init__(self, payload: Any, status: int = 200) -> None:
        self._payload, self.status = payload, status

    async def json(self, content_type: Any = None) -> Any:
        return self._payload

    async def __aenter__(self) -> "FakeResponse":
        return self

    async def __aexit__(self, *_: object) -> None:
        return None


class FakeSession:
    def __init__(self, payload: Any, status: int = 200) -> None:
        self.payload, self.status, self.calls = payload, status, []

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append((url, kwargs))
        return FakeResponse(self.payload, self.status)


def client_with(payload: Any, status: int = 200) -> tuple[OneBotClient, FakeSession]:
    client = OneBotClient("http://127.0.0.1:5700", "ws://127.0.0.1:6700", "token", 5.0)
    session = FakeSession(payload, status)

    async def fake_session() -> FakeSession:
        return session

    client._get_session = fake_session  # type: ignore[assignment]
    return client, session


@pytest.mark.asyncio
async def test_a_poke_envelope_without_data_is_a_success():
    # NapCat answers send_poke with {"status":"ok","retcode":0,...} and no data
    # key at all.  Demanding data turned every poke into a protocol error and
    # left fifteen of them stuck as unknown in the outbox.
    client, session = client_with({"status": "ok", "retcode": 0, "message": "", "wording": ""})

    assert await client.send_poke("123456") is None
    assert session.calls[0][0].endswith("/send_poke")
    assert session.calls[0][1]["json"] == {"user_id": "123456"}


@pytest.mark.asyncio
async def test_a_poke_that_napcat_rejects_is_still_an_error():
    client, _ = client_with({"status": "failed", "retcode": 200, "data": None,
                             "message": "unsupported"})
    with pytest.raises(Exception):
        await client.send_poke("123456")


@pytest.mark.asyncio
async def test_actions_that_must_return_data_still_fail_closed():
    # No data key and no usable payload: the per-action rule must still refuse.
    client, _ = client_with({"status": "ok", "retcode": 0})
    with pytest.raises(OneBotProtocolError):
        await client.send_private_msg("123456", [{"type": "text", "data": {"text": "hi"}}])
    with pytest.raises(OneBotProtocolError):
        await client.get_msg("123")


@pytest.mark.asyncio
async def test_a_malformed_envelope_is_still_refused():
    for payload in ({"status": "ok"}, {"retcode": 0}, {"status": "ok", "retcode": "0"}, "nope"):
        client, _ = client_with(payload)
        with pytest.raises(Exception):
            await client.send_poke("123456")
