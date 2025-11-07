from typing import List

import numpy as np
import numpy.typing as npt

from sierraecg.lzw import LzwDecoder


def xli_decode(data: bytes, labels: List[str]) -> List[npt.NDArray[np.int16]]:
    samples: List[npt.NDArray[np.int16]] = []
    offset = 0
    while offset < len(data):
        header = data[offset : offset + 8]
        offset += 8

        size = int.from_bytes(header[0:4], byteorder="little", signed=True)
        start = int.from_bytes(header[6:], byteorder="little", signed=True)
        chunk = data[offset : offset + size]
        offset += size

        decoder = LzwDecoder(chunk, bits=10)

        buffer = []
        while -1 != (b := decoder.read()):
            buffer.append(b & 0xFF)

        if len(buffer) % 2 == 1:
            buffer.append(0)

        deltas = xli_decode_deltas(buffer, start)
        samples.append(deltas)

    return samples


def xli_decode_deltas(buffer: List[int], first: int) -> npt.NDArray[np.int16]:
    """Decode XLI delta compressed samples.

    The algorithm mirrors the reference C implementation in
    ``libsierraecg/sierraecg.c``.  The C version works on 16-bit signed
    integers and silently wraps on overflow.  In NumPy, however, writing a
    value outside the valid range of ``int16`` raises an ``OverflowError``.

    To emulate the C behaviour we perform the calculations in ``int32`` and
    **only** cast back to ``int16`` at the very end – NumPy truncates the
    upper bits during the cast, matching the two-s complement wrap-around
    that happens implicitly in C.
    """

    # Unpack the 16-bit values that make up the delta stream and promote to
    # int32 so intermediate results cannot overflow Python / NumPy checks.
    deltas = xli_unpack(buffer).astype(np.int64, copy=False)

    x = int(deltas[0])
    y = int(deltas[1])
    last = first

    # Start at the third sample and reconstruct the original signal.
    for i in range(2, len(deltas)):
        z = (y + y) - x - last  # equivalent to (2 * y) - x - last
        # The reference C code updates *last* using the *raw* delta value
        # before it is overwritten.
        last = int(deltas[i]) - 64
        deltas[i] = z
        x = y
        y = z

    # Cast back to int16 with wrap-around (matches C behaviour).
    return deltas.astype(np.int16, casting="unsafe", copy=False)


def xli_unpack(buffer: List[int]) -> npt.NDArray[np.int16]:
    unpacked: npt.NDArray[np.int16] = np.array([0 for _ in range(int(len(buffer) / 2))], dtype=np.int16)
    for i in range(len(unpacked)):
        joined_bytes = (((buffer[i] << 8) | buffer[len(unpacked) + i]) << 16) >> 16
        unpacked[i] = np.array(joined_bytes).astype(np.int16)
    return unpacked
