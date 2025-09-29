from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


@dataclass
class EnviHeader:
    """Container holding the parsed ENVI header metadata."""

    samples: int
    lines: int
    bands: int
    interleave: str
    data_type: int
    byte_order: int
    header_offset: int = 0
    file_type: str | None = None
    data_file: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def shape_bsq(self) -> Tuple[int, int, int]:
        """Return the canonical ``(bands, lines, samples)`` shape."""

        return (self.bands, self.lines, self.samples)


def _normalize_key(key: str) -> str:
    return key.strip().lower().replace(" ", "_")


def _parse_value(value: str) -> Any:
    value = value.strip()
    if value.startswith("{") and value.endswith("}"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        items = [item.strip() for item in inner.split(",")]
        parsed = [_parse_value(item) for item in items]
        return parsed
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if "." in value or "e" in value.lower():
            return float(value)
        return int(value)
    except ValueError:
        return value


def f01_parse_envi_header(path: str | Path) -> EnviHeader:
    """Parse an ENVI ``.hdr`` file.

    The routine supports simple ``key = value`` pairs as well as multiline
    ``{ comma, separated, lists }`` sections. All keys are normalised to
    lowercase snake_case within :attr:`EnviHeader.metadata`.
    """

    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not text:
        raise ValueError(f"Header file '{path}' is empty")
    if not text[0].strip().startswith("ENVI"):
        raise ValueError("ENVI header must start with 'ENVI'")

    metadata: Dict[str, Any] = {}
    key: str | None = None
    multiline: list[str] | None = None

    for raw in text[1:]:
        line = raw.strip()
        if not line:
            continue
        if "=" in line and multiline is None:
            left, right = line.split("=", 1)
            key = _normalize_key(left)
            value = right.strip()
            if value.startswith("{") and not value.endswith("}"):
                multiline = [value]
                continue
            metadata[key] = _parse_value(value)
            key = None
        elif multiline is not None:
            multiline.append(line)
            if line.endswith("}"):
                assert key is not None
                value = "".join(multiline)
                metadata[key] = _parse_value(value)
                multiline = None
                key = None
        elif key is not None:
            # Continuation of previous value.
            metadata[key] = _parse_value(metadata[key] + " " + line)
        else:
            raise ValueError(f"Unable to parse line in header: {raw!r}")

    required = ["samples", "lines", "bands", "interleave", "data_type", "byte_order"]
    missing = [r for r in required if r not in metadata]
    if missing:
        raise ValueError(f"Missing required ENVI keys: {missing}")

    header = EnviHeader(
        samples=int(metadata["samples"]),
        lines=int(metadata["lines"]),
        bands=int(metadata["bands"]),
        interleave=str(metadata["interleave"]).lower(),
        data_type=int(metadata["data_type"]),
        byte_order=int(metadata["byte_order"]),
        header_offset=int(metadata.get("header_offset", 0)),
        file_type=str(metadata.get("file_type")) if "file_type" in metadata else None,
        data_file=str(metadata.get("data_file")) if "data_file" in metadata else None,
        metadata=metadata,
    )
    return header


def _f02_envi_dtype(data_type: int, byte_order: int) -> np.dtype:
    mapping = {
        1: np.uint8,
        2: np.int16,
        3: np.int32,
        4: np.float32,
        5: np.float64,
        6: np.complex64,
        9: np.complex128,
        12: np.uint16,
        13: np.uint32,
        14: np.int64,
        15: np.uint64,
    }
    if data_type not in mapping:
        raise ValueError(f"Unsupported ENVI data type: {data_type}")
    dtype = np.dtype(mapping[data_type])
    if dtype.byteorder not in {"<", ">", "="}:
        dtype = dtype.newbyteorder("=")
    endian = "<" if byte_order == 0 else ">"
    if dtype.byteorder in {"<", ">"}:
        dtype = dtype.newbyteorder(endian)
    else:
        dtype = dtype.newbyteorder(endian)
    return dtype


def f02_open_envi_memmap(path: str | Path, header: EnviHeader) -> np.memmap:
    """Open the binary image described by ``header`` as a memory-map."""

    path = Path(path)
    data_file = header.data_file
    if data_file:
        data_path = Path(data_file)
        if not data_path.is_absolute():
            data_path = path.parent / data_path
    else:
        data_path = path.with_suffix("")
        if not data_path.exists():
            for suffix in (".img", ".bin", ".dat"):
                candidate = path.with_suffix(suffix)
                if candidate.exists():
                    data_path = candidate
                    break
    if not data_path.exists():
        raise FileNotFoundError(f"Could not locate ENVI data file for '{path}'")

    dtype = _f02_envi_dtype(header.data_type, header.byte_order)

    interleave = header.interleave.lower()
    if interleave == "bsq":
        shape = (header.bands, header.lines, header.samples)
    elif interleave == "bil":
        shape = (header.lines, header.bands, header.samples)
    elif interleave == "bip":
        shape = (header.lines, header.samples, header.bands)
    else:
        raise ValueError(f"Unsupported interleave type: {header.interleave}")

    return np.memmap(
        data_path,
        dtype=dtype,
        mode="r",
        offset=header.header_offset,
        shape=shape,
    )


__all__ = [
    "EnviHeader",
    "f01_parse_envi_header",
    "f02_open_envi_memmap",
]
