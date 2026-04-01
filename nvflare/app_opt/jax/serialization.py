# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Flax-compatible msgpack helpers for array-tree JAX state dictionaries.

These helpers intentionally avoid importing JAX or Flax so server-side checkpoint
loading can handle JAX state dictionaries without initializing the JAX runtime.
"""

from enum import IntEnum
from typing import Any

import msgpack
import numpy as np

MAX_CHUNK_SIZE = 1024 * 1024 * 1024
_CHUNKED_ARRAY_KEY = "__msgpack_chunked_array__"
_CHUNKS_KEY = "chunks"
_SHAPE_KEY = "shape"


class _MsgpackExtType(IntEnum):
    ndarray = 1
    native_complex = 2
    npscalar = 3


def _tuple_to_dict(values):
    return {str(index): value for index, value in enumerate(values)}


def _dict_to_tuple(values):
    return tuple(values[str(index)] for index in range(len(values)))


def _dtype_from_name(name: str | bytes):
    try:
        return np.dtype(name)
    except TypeError:
        if name in {b"bfloat16", "bfloat16"}:
            try:
                import ml_dtypes
            except ImportError as e:
                raise TypeError("bfloat16 checkpoints require ml_dtypes to deserialize without JAX.") from e
            return ml_dtypes.bfloat16
        raise


def _ndarray_to_bytes(arr) -> bytes:
    arr = np.asarray(arr)
    if arr.dtype.hasobject or arr.dtype.isalignedstruct:
        raise ValueError("Object and structured dtypes are not supported for JAX checkpoint serialization.")
    payload = (arr.shape, arr.dtype.name, arr.tobytes("C"))
    return msgpack.packb(payload, use_bin_type=True)


def _ndarray_from_bytes(data: bytes) -> np.ndarray:
    shape, dtype_name, buffer = msgpack.unpackb(data, raw=True)
    return np.frombuffer(buffer, dtype=_dtype_from_name(dtype_name), count=-1, offset=0).reshape(shape, order="C")


def _is_array_like(value: Any) -> bool:
    return hasattr(value, "__array__") and hasattr(value, "shape") and hasattr(value, "dtype")


def _normalize_tree(tree: Any):
    if isinstance(tree, dict):
        return {key: _normalize_tree(value) for key, value in tree.items()}
    if isinstance(tree, list):
        return [_normalize_tree(value) for value in tree]
    if isinstance(tree, tuple):
        return tuple(_normalize_tree(value) for value in tree)
    if isinstance(tree, (np.ndarray, np.generic)):
        return tree
    if _is_array_like(tree):
        return np.asarray(tree)
    return tree


def _chunk(arr: np.ndarray) -> dict[str, Any]:
    chunk_size = max(1, int(MAX_CHUNK_SIZE / arr.dtype.itemsize))
    flat = arr.reshape(-1)
    chunks = [flat[index : index + chunk_size] for index in range(0, flat.size, chunk_size)]
    return {
        _CHUNKED_ARRAY_KEY: True,
        _SHAPE_KEY: _tuple_to_dict(arr.shape),
        _CHUNKS_KEY: _tuple_to_dict(chunks),
    }


def _unchunk(data: dict[str, Any]) -> np.ndarray:
    shape = _dict_to_tuple(data[_SHAPE_KEY])
    flat = np.concatenate(_dict_to_tuple(data[_CHUNKS_KEY]))
    return flat.reshape(shape)


def _chunk_array_leaves(tree: Any):
    if isinstance(tree, dict):
        return {key: _chunk_array_leaves(value) for key, value in tree.items()}
    if isinstance(tree, list):
        return [_chunk_array_leaves(value) for value in tree]
    if isinstance(tree, tuple):
        return tuple(_chunk_array_leaves(value) for value in tree)
    if isinstance(tree, np.ndarray) and tree.size * tree.dtype.itemsize > MAX_CHUNK_SIZE:
        return _chunk(tree)
    return tree


def _unchunk_array_leaves(tree: Any):
    if isinstance(tree, dict):
        if tree.get(_CHUNKED_ARRAY_KEY):
            return _unchunk(tree)
        return {key: _unchunk_array_leaves(value) for key, value in tree.items()}
    if isinstance(tree, list):
        return [_unchunk_array_leaves(value) for value in tree]
    if isinstance(tree, tuple):
        return tuple(_unchunk_array_leaves(value) for value in tree)
    return tree


def _msgpack_ext_pack(value: Any):
    if isinstance(value, np.ndarray) or _is_array_like(value):
        return msgpack.ExtType(_MsgpackExtType.ndarray, _ndarray_to_bytes(value))
    if isinstance(value, np.generic):
        return msgpack.ExtType(_MsgpackExtType.npscalar, _ndarray_to_bytes(np.asarray(value)))
    if isinstance(value, complex):
        return msgpack.ExtType(
            _MsgpackExtType.native_complex,
            msgpack.packb((value.real, value.imag), use_bin_type=True),
        )
    return value


def _msgpack_ext_unpack(code: int, data: bytes):
    if code == _MsgpackExtType.ndarray:
        return _ndarray_from_bytes(data)
    if code == _MsgpackExtType.native_complex:
        real, imag = msgpack.unpackb(data, raw=False)
        return complex(real, imag)
    if code == _MsgpackExtType.npscalar:
        array = _ndarray_from_bytes(data)
        return array[()]
    return msgpack.ExtType(code, data)


def msgpack_serialize(tree: Any) -> bytes:
    normalized = _normalize_tree(tree)
    chunked = _chunk_array_leaves(normalized)
    return msgpack.packb(chunked, default=_msgpack_ext_pack, strict_types=True, use_bin_type=True)


def msgpack_restore(encoded_tree: bytes):
    state_dict = msgpack.unpackb(encoded_tree, ext_hook=_msgpack_ext_unpack, raw=False)
    return _unchunk_array_leaves(state_dict)
