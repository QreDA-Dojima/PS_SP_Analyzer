from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather

from envi_reader import EnviHeader


def _pa_type_from_numpy(dtype: np.dtype) -> pa.DataType:
    dtype = np.dtype(dtype)
    if dtype.kind == "b":
        return pa.bool_()
    if dtype.kind == "u":
        mapping = {
            1: pa.uint8(),
            2: pa.uint16(),
            4: pa.uint32(),
            8: pa.uint64(),
        }
        if dtype.itemsize not in mapping:
            raise ValueError(f"Unsupported unsigned integer width: {dtype}")
        return mapping[dtype.itemsize]
    if dtype.kind == "i":
        mapping = {
            1: pa.int8(),
            2: pa.int16(),
            4: pa.int32(),
            8: pa.int64(),
        }
        if dtype.itemsize not in mapping:
            raise ValueError(f"Unsupported signed integer width: {dtype}")
        return mapping[dtype.itemsize]
    if dtype.kind == "f":
        mapping = {
            2: pa.float16(),
            4: pa.float32(),
            8: pa.float64(),
        }
        if dtype.itemsize not in mapping:
            raise ValueError(f"Unsupported float width: {dtype}")
        return mapping[dtype.itemsize]
    raise ValueError(f"Unsupported dtype for Arrow conversion: {dtype}")


def _read_chunk_band(
    memmap: np.memmap,
    header: EnviHeader,
    line_slice: Tuple[int, int],
    band_slice: Tuple[int, int],
) -> np.ndarray:
    line_start, line_end = line_slice
    band_start, band_end = band_slice
    interleave = header.interleave.lower()
    if interleave == "bsq":
        data = memmap[band_start:band_end, line_start:line_end, :]
    elif interleave == "bil":
        data = np.transpose(memmap[line_start:line_end, band_start:band_end, :], (1, 0, 2))
    elif interleave == "bip":
        data = np.transpose(memmap[line_start:line_end, :, band_start:band_end], (2, 0, 1))
    else:
        raise ValueError(f"Unsupported interleave: {header.interleave}")
    return np.asarray(data)


def f03_export_long_feather_streaming(
    header: EnviHeader,
    memmap: np.memmap,
    output_path: str | Path,
    *,
    chunk_lines: int = 256,
    chunk_bands: int = 16,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    line_count = header.lines
    sample_count = header.samples
    band_count = header.bands

    value_dtype = np.dtype(memmap.dtype).newbyteorder("=")
    value_type = _pa_type_from_numpy(value_dtype)

    schema = pa.schema(
        [
            pa.field("line", pa.int32()),
            pa.field("sample", pa.int32()),
            pa.field("band", pa.int32()),
            pa.field("value", value_type),
        ]
    )

    with feather.FeatherWriter(str(output_path)) as writer:
        if hasattr(writer, "write_metadata"):
            writer.write_metadata(schema)
        for line_start in range(0, line_count, chunk_lines):
            line_end = min(line_count, line_start + chunk_lines)
            base_lines = np.arange(line_start, line_end, dtype=np.int32)
            repeated_lines = np.repeat(base_lines, sample_count)
            base_samples = np.arange(sample_count, dtype=np.int32)
            repeated_samples = np.tile(base_samples, line_end - line_start)
            pa_lines = pa.array(repeated_lines)
            pa_samples = pa.array(repeated_samples)

            for band_start in range(0, band_count, chunk_bands):
                band_end = min(band_count, band_start + chunk_bands)
                chunk = _read_chunk_band(
                    memmap,
                    header,
                    (line_start, line_end),
                    (band_start, band_end),
                )
                for offset, band_index in enumerate(range(band_start, band_end)):
                    values = np.asarray(chunk[offset], dtype=value_dtype).reshape(-1)
                    pa_band = pa.array(np.full(values.size, band_index, dtype=np.int32))
                    pa_values = pa.array(values, type=value_type)
                    batch = pa.record_batch(
                        [pa_lines, pa_samples, pa_band, pa_values],
                        schema=schema,
                    )
                    writer.write_batches([batch])


def _read_tile_band(
    memmap: np.memmap,
    header: EnviHeader,
    line_slice: Tuple[int, int],
    sample_slice: Tuple[int, int],
) -> np.ndarray:
    line_start, line_end = line_slice
    sample_start, sample_end = sample_slice
    interleave = header.interleave.lower()
    if interleave == "bsq":
        data = memmap[:, line_start:line_end, sample_start:sample_end]
        data = np.moveaxis(data, 0, -1)
    elif interleave == "bil":
        data = memmap[line_start:line_end, :, sample_start:sample_end]
        data = np.transpose(data, (0, 2, 1))
    elif interleave == "bip":
        data = memmap[line_start:line_end, sample_start:sample_end, :]
    else:
        raise ValueError(f"Unsupported interleave: {header.interleave}")
    return np.asarray(data)


def f04_export_excel_spectra_tiled(
    header: EnviHeader,
    memmap: np.memmap,
    output_path: str | Path,
    *,
    tile_lines: int = 64,
    tile_samples: int = 64,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    value_dtype = np.dtype(memmap.dtype).newbyteorder("=")

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        for line_start in range(0, header.lines, tile_lines):
            line_end = min(header.lines, line_start + tile_lines)
            for sample_start in range(0, header.samples, tile_samples):
                sample_end = min(header.samples, sample_start + tile_samples)
                tile = _read_tile_band(
                    memmap,
                    header,
                    (line_start, line_end),
                    (sample_start, sample_end),
                )
                tile = np.asarray(tile, dtype=value_dtype)
                lines = np.arange(line_start, line_end, dtype=np.int32)
                samples = np.arange(sample_start, sample_end, dtype=np.int32)
                line_indices = np.repeat(lines, samples.size)
                sample_indices = np.tile(samples, lines.size)
                spectra = tile.reshape(lines.size * samples.size, header.bands)
                columns = [f"band_{idx}" for idx in range(header.bands)]
                df = pd.DataFrame(spectra, columns=columns)
                df.insert(0, "sample", sample_indices)
                df.insert(0, "line", line_indices)
                sheet_name = f"R{line_start}_C{sample_start}"
                sheet_name = sheet_name[:31]
                df.to_excel(writer, sheet_name=sheet_name, index=False)


__all__ = [
    "f03_export_long_feather_streaming",
    "f04_export_excel_spectra_tiled",
    "_read_chunk_band",
    "_read_tile_band",
]
