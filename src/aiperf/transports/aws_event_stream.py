# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decoder for AWS event-stream binary framing used by SageMaker streaming endpoints.

SageMaker wraps standard SSE payloads inside AWS event-stream binary messages.
Each message has the format:

    [total_length:4][headers_length:4][prelude_crc:4][headers...][payload...][message_crc:4]

This module provides an async generator that strips the binary framing and yields
the raw SSE payload bytes, making the stream consumable by AsyncSSEStreamReader.
"""

import struct
from collections.abc import AsyncIterator

_PRELUDE_SIZE = 12  # total_length(4) + headers_length(4) + prelude_crc(4)
_MESSAGE_CRC_SIZE = 4


def _is_aws_event_stream(initial_bytes: bytes) -> bool:
    """Detect AWS event-stream binary framing from the first bytes.

    The framing starts with a 4-byte big-endian total message length followed by
    a 4-byte headers length. Valid SSE text never starts with a null byte.
    """
    return len(initial_bytes) >= 4 and initial_bytes[0:1] == b"\x00"


async def decode_aws_event_stream(
    raw_stream: AsyncIterator[bytes],
) -> AsyncIterator[bytes]:
    """Strip AWS event-stream binary framing, yielding SSE payload bytes.

    If the stream is not AWS event-stream framed (i.e. standard SSE), the bytes
    are passed through unchanged.

    Args:
        raw_stream: Async iterator of raw bytes from the HTTP response body.

    Yields:
        Payload bytes suitable for SSE parsing (the `data: {...}\\n\\n` lines).
    """
    buffer = bytearray()
    detected: bool | None = None

    async for chunk in raw_stream:
        if detected is None:
            buffer += chunk
            if len(buffer) < 4:
                continue
            detected = _is_aws_event_stream(bytes(buffer[:4]))
            if not detected:
                yield bytes(buffer)
                buffer.clear()
                async for passthrough_chunk in raw_stream:
                    yield passthrough_chunk
                return
        else:
            buffer += chunk

        while len(buffer) >= _PRELUDE_SIZE:
            total_length = struct.unpack_from("!I", buffer, 0)[0]
            if len(buffer) < total_length:
                break

            headers_length = struct.unpack_from("!I", buffer, 4)[0]
            payload_start = _PRELUDE_SIZE + headers_length
            payload_end = total_length - _MESSAGE_CRC_SIZE
            payload = bytes(buffer[payload_start:payload_end])

            del buffer[:total_length]

            if payload:
                yield payload

    if buffer:
        yield bytes(buffer)
