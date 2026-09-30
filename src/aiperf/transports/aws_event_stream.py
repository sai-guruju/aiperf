# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode AWS event-stream framing used by SageMaker streaming endpoints."""

import struct
from collections.abc import AsyncIterator

_PRELUDE_SIZE = 12
_MESSAGE_CRC_SIZE = 4


def _is_aws_event_stream(initial_bytes: bytes) -> bool:
    """Return whether the first bytes look like AWS event-stream framing."""
    return len(initial_bytes) >= 4 and initial_bytes[0:1] == b"\x00"


async def decode_aws_event_stream(
    raw_stream: AsyncIterator[bytes],
) -> AsyncIterator[bytes]:
    """Strip AWS event-stream frames, passing ordinary SSE through unchanged."""
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
