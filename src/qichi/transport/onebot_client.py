from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any
from uuid import uuid4

import aiohttp


class OneBotError(RuntimeError):
    """Base class for transport and OneBot protocol failures."""


class OneBotTimeoutError(OneBotError):
    pass


class OneBotConnectionError(OneBotError):
    pass


class OneBotHTTPClientError(OneBotError):
    pass


class OneBotHTTPServerError(OneBotError):
    pass


class OneBotHTTPRedirectError(OneBotError):
    pass


class OneBotProtocolError(OneBotError):
    pass


class OneBotActionError(OneBotError):
    pass


class OneBotWebSocketHandshakeError(OneBotError):
    pass


class OneBotWebSocketProtocolError(OneBotError):
    pass


class OneBotWebSocketDisconnectedError(OneBotError):
    pass


class OneBotWebSocketConsumerError(OneBotError):
    pass


class OneBotClient:
    """A single-consumer OneBot 11 HTTP and Forward WebSocket client."""

    def __init__(self, http_url: str, ws_url: str, access_token: str, timeout: float):
        if not isinstance(http_url, str) or not http_url:
            raise ValueError("http_url must be a non-empty string")
        if not isinstance(ws_url, str) or not ws_url:
            raise ValueError("ws_url must be a non-empty string")
        if not isinstance(access_token, str) or not access_token:
            raise ValueError("access_token must be a non-empty string")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("timeout must be positive")
        self._http_url = http_url.rstrip("/")
        self._ws_url = ws_url
        self._access_token = access_token
        self._timeout = float(timeout)
        self._session: aiohttp.ClientSession | None = None
        self._closed = False
        self._stream_lock = asyncio.Lock()
        self._stream_active = False
        self._connection_id: str | None = None
        self._connection_ready = asyncio.Event()

    def __repr__(self) -> str:
        return f"OneBotClient(timeout={self._timeout!r}, closed={self._closed!r})"

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def connection_id(self) -> str | None:
        """The current Forward WS connection identity, if handshaken."""
        return self._connection_id

    async def wait_until_connected(self) -> str:
        """Wait until the single Forward WS has completed its handshake."""
        await self._connection_ready.wait()
        if self._connection_id is None:
            raise OneBotConnectionError("Forward WebSocket is not connected")
        return self._connection_id

    async def __aenter__(self) -> OneBotClient:
        await self._get_session()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def close(self) -> None:
        self._closed = True
        if self._session is not None and not self._session.closed:
            await self._session.close()

    async def send_private_msg(self, user_id: int | str, message: Sequence[Mapping[str, Any]]) -> Any:
        segments = []
        for segment in message:
            if not isinstance(segment, Mapping):
                raise TypeError("message segments must be mappings")
            segments.append(dict(segment))
        return await self._action("send_private_msg", {"user_id": user_id, "message": segments})

    async def get_msg(self, message_id: int | str) -> Any:
        return await self._action("get_msg", {"message_id": message_id})

    async def get_image(self, file: str) -> Mapping[str, Any]:
        """Resolve one inbound image to a cached path and/or a url.

        NapCat answers with the file it already downloaded (and usually a url as
        well).  Nothing is fetched here: the caller decides which source to read
        and owns the size, type and timeout policy.
        """

        if not isinstance(file, str) or not file:
            raise ValueError("file must be a non-empty string")
        data = await self._action("get_image", {"file": file})
        if not isinstance(data, Mapping):
            raise OneBotProtocolError("OneBot get_image response data must be a mapping")
        return data

    async def set_msg_emoji_like(
        self, message_id: int | str, emoji_id: int | str, set: bool = True
    ) -> Any:
        if type(set) is not bool:
            raise TypeError("set must be a bool")
        return await self._action(
            "set_msg_emoji_like",
            {"message_id": message_id, "emoji_id": emoji_id, "set": set},
        )

    async def send_poke(self, user_id: int | str) -> Any:
        return await self._action("send_poke", {"user_id": user_id})

    async def get_login_info(self) -> Mapping[str, Any]:
        data = await self._action("get_login_info", {})
        if not isinstance(data, Mapping):
            raise OneBotProtocolError("get_login_info data must be a mapping")
        return data

    async def event_stream(self) -> AsyncIterator[Mapping[str, Any]]:
        async with self._stream_lock:
            if self._stream_active:
                raise OneBotWebSocketConsumerError("only one Forward WebSocket consumer is allowed")
            self._stream_active = True
        websocket: aiohttp.ClientWebSocketResponse | None = None
        try:
            session = await self._get_session()
            connection_error: OneBotError | None = None
            try:
                websocket = await session.ws_connect(
                    self._ws_url,
                    headers=self._headers(),
                )
            except asyncio.TimeoutError:
                connection_error = OneBotTimeoutError("Forward WebSocket connection timed out")
            except aiohttp.ClientResponseError as error:
                connection_error = OneBotWebSocketHandshakeError(
                    f"Forward WebSocket handshake failed with HTTP {error.status}"
                )
            except aiohttp.ClientConnectionError:
                connection_error = OneBotConnectionError("Forward WebSocket connection failed")
            except aiohttp.ClientError:
                connection_error = OneBotWebSocketHandshakeError("Forward WebSocket handshake failed")
            if connection_error is not None:
                raise connection_error
            self._connection_id = uuid4().hex
            self._connection_ready.set()
            while True:
                receive_error: OneBotError | None = None
                try:
                    message = await websocket.receive()
                except asyncio.TimeoutError:
                    receive_error = OneBotTimeoutError("Forward WebSocket receive timed out")
                except aiohttp.ClientError:
                    receive_error = OneBotWebSocketDisconnectedError("Forward WebSocket disconnected")
                if receive_error is not None:
                    raise receive_error
                if message.type is aiohttp.WSMsgType.TEXT:
                    frame_error: OneBotWebSocketProtocolError | None = None
                    try:
                        payload = json.loads(message.data)
                    except (TypeError, json.JSONDecodeError):
                        frame_error = OneBotWebSocketProtocolError("Forward WebSocket frame is not JSON")
                    if frame_error is not None:
                        raise frame_error
                    if not isinstance(payload, Mapping):
                        raise OneBotWebSocketProtocolError("Forward WebSocket JSON frame must be a mapping")
                    yield payload
                elif message.type is aiohttp.WSMsgType.BINARY:
                    raise OneBotWebSocketProtocolError("Forward WebSocket binary frame is not supported")
                elif message.type is aiohttp.WSMsgType.CLOSE:
                    if self._is_clean_close(message.data):
                        return
                    raise OneBotWebSocketDisconnectedError("Forward WebSocket closed unexpectedly")
                elif message.type in {aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING}:
                    if self._is_clean_close(websocket.close_code):
                        return
                    raise OneBotWebSocketDisconnectedError("Forward WebSocket closed unexpectedly")
                elif message.type is aiohttp.WSMsgType.ERROR:
                    raise OneBotWebSocketDisconnectedError("Forward WebSocket disconnected")
        finally:
            if websocket is not None and not websocket.closed:
                await websocket.close()
            self._connection_ready.clear()
            self._connection_id = None
            async with self._stream_lock:
                self._stream_active = False

    async def _action(self, action: str, payload: Mapping[str, Any]) -> Any:
        session = await self._get_session()
        response_error: OneBotError | None = None
        envelope: Any = None
        try:
            async with session.post(
                f"{self._http_url}/{action}",
                headers=self._headers(),
                json=dict(payload),
                allow_redirects=False,
            ) as response:
                if 300 <= response.status < 400:
                    raise OneBotHTTPRedirectError(f"OneBot HTTP redirect for {action}")
                if 400 <= response.status < 500:
                    raise OneBotHTTPClientError(f"OneBot HTTP {response.status} for {action}")
                if response.status >= 500:
                    raise OneBotHTTPServerError(f"OneBot HTTP {response.status} for {action}")
                try:
                    envelope = await response.json(content_type=None)
                except (aiohttp.ContentTypeError, UnicodeDecodeError, json.JSONDecodeError):
                    response_error = OneBotProtocolError(f"OneBot {action} response is not JSON")
        except asyncio.TimeoutError:
            response_error = OneBotTimeoutError(f"OneBot {action} timed out")
        except aiohttp.ClientConnectionError:
            response_error = OneBotConnectionError(f"OneBot {action} connection failed")
        except aiohttp.ClientError:
            response_error = OneBotConnectionError(f"OneBot {action} network failed")
        if response_error is not None:
            raise response_error
        if not isinstance(envelope, Mapping):
            raise OneBotProtocolError(f"OneBot {action} response must be a mapping")
        status = envelope.get("status")
        retcode = envelope.get("retcode")
        if not isinstance(status, str) or type(retcode) is not int:
            raise OneBotProtocolError(f"OneBot {action} response envelope is invalid")
        if status != "ok" or retcode != 0:
            raise OneBotActionError(f"OneBot {action} returned status={status!r} retcode={retcode}")
        # NapCat answers some actions without any payload: send_poke returns
        # {"status":"ok","retcode":0,...} and no data key at all (verified
        # against the live server on 2026-09-11).  Demanding data there turned
        # every poke into a protocol error and left fifteen of them stuck as
        # unknown in the outbox.  What an action has to return is decided per
        # action below, so an absent payload becomes None and the actions whose
        # result we actually consume still fail closed on it.
        data = envelope.get("data")
        self._validate_action_data(action, data)
        return data

    @staticmethod
    def _validate_action_data(action: str, data: Any) -> None:
        if action == "send_private_msg":
            if not isinstance(data, Mapping) or not OneBotClient._decimal_id(data.get("message_id")):
                raise OneBotProtocolError("OneBot send_private_msg response has invalid message_id")
        elif action == "get_msg":
            if not isinstance(data, Mapping) or not OneBotClient._decimal_id(data.get("message_id")):
                raise OneBotProtocolError("OneBot get_msg response has invalid message_id")
            if "user_id" not in data or "message" not in data:
                raise OneBotProtocolError("OneBot get_msg response is missing required fields")
        elif action == "get_image" and not isinstance(data, Mapping):
            raise OneBotProtocolError("OneBot get_image response data must be a mapping")
        elif action in {"send_poke", "set_msg_emoji_like"} and data is not None and not isinstance(data, Mapping):
            raise OneBotProtocolError(f"OneBot {action} response data must be a mapping")

    @staticmethod
    def _decimal_id(value: Any) -> str | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int) and value >= 0:
            return str(value)
        if isinstance(value, str) and value.isascii() and value.isdecimal():
            return str(int(value, 10))
        return None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._closed:
            raise OneBotConnectionError("OneBot client is closed")
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self._timeout))
        return self._session

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._access_token}"}

    @staticmethod
    def _is_clean_close(close_code: object) -> bool:
        return close_code in {1000, 1001}
