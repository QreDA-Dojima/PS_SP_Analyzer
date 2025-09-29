from __future__ import annotations

import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import tkinter as tk
from tkinter import filedialog, messagebox

from envi_reader import f01_parse_envi_header, f02_open_envi_memmap
from exporters import f03_export_long_feather_streaming, f04_export_excel_spectra_tiled


@dataclass
class AppConfig:
    header_path: Path
    output_dir: Path
    export_feather: bool
    export_excel: bool
    chunk_lines: int
    chunk_bands: int
    tile_lines: int
    tile_samples: int


def _update_status(state: Dict[str, object], text: str) -> None:
    root: tk.Tk = state["root"]  # type: ignore[assignment]
    status_var: tk.StringVar = state["status_var"]  # type: ignore[assignment]
    root.after(0, status_var.set, text)


def f10_select_files(state: Dict[str, object], *, kind: str = "header") -> None:
    vars_dict: Dict[str, tk.Variable] = state["vars"]  # type: ignore[assignment]
    if kind == "header":
        filename = filedialog.askopenfilename(
            title="Select ENVI header",
            filetypes=[("ENVI header", "*.hdr"), ("All files", "*.*")],
        )
        if not filename:
            return
        vars_dict["header_path"].set(filename)
        if not vars_dict["output_dir"].get():
            vars_dict["output_dir"].set(str(Path(filename).parent))
        _update_status(state, "Header selected. Ready to export.")
    elif kind == "output":
        folder = filedialog.askdirectory(title="Select output directory")
        if folder:
            vars_dict["output_dir"].set(folder)
            _update_status(state, f"Output directory set to {folder}")
    else:
        raise ValueError(f"Unknown selection kind: {kind}")


def f11_start_export(state: Dict[str, object]) -> None:
    worker: Optional[threading.Thread] = state.get("worker")  # type: ignore[assignment]
    if worker is not None and worker.is_alive():
        messagebox.showinfo("Export running", "Please wait for the current export to finish.")
        return

    vars_dict: Dict[str, tk.Variable] = state["vars"]  # type: ignore[assignment]
    header_path = Path(vars_dict["header_path"].get())  # type: ignore[arg-type]
    output_dir = Path(vars_dict["output_dir"].get())  # type: ignore[arg-type]

    if not header_path.exists():
        messagebox.showerror("Missing file", "Please select a valid ENVI header (.hdr) file.")
        return
    if not output_dir.exists():
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            messagebox.showerror("Output error", f"Unable to create output directory: {exc}")
            return

    export_feather = bool(vars_dict["export_feather"].get())
    export_excel = bool(vars_dict["export_excel"].get())

    if not export_feather and not export_excel:
        messagebox.showwarning("No export selected", "Please choose at least one export format.")
        return

    try:
        chunk_lines = int(vars_dict["chunk_lines"].get())
        chunk_bands = int(vars_dict["chunk_bands"].get())
        tile_lines = int(vars_dict["tile_lines"].get())
        tile_samples = int(vars_dict["tile_samples"].get())
    except (TypeError, ValueError):
        messagebox.showerror("Invalid settings", "Chunk and tile sizes must be integers.")
        return

    config = AppConfig(
        header_path=header_path,
        output_dir=output_dir,
        export_feather=export_feather,
        export_excel=export_excel,
        chunk_lines=max(1, chunk_lines),
        chunk_bands=max(1, chunk_bands),
        tile_lines=max(1, tile_lines),
        tile_samples=max(1, tile_samples),
    )

    thread = threading.Thread(target=_export_one, args=(state, config), daemon=True)
    state["worker"] = thread
    _update_status(state, "Starting export...")
    thread.start()


def _export_one(state: Dict[str, object], config: AppConfig) -> None:
    try:
        _update_status(state, "Parsing header...")
        header = f01_parse_envi_header(config.header_path)
        memmap = f02_open_envi_memmap(config.header_path, header)

        if config.export_feather:
            feather_path = config.output_dir / f"{config.header_path.stem}.feather"
            _update_status(state, f"Exporting Feather to {feather_path}...")
            f03_export_long_feather_streaming(
                header,
                memmap,
                feather_path,
                chunk_lines=config.chunk_lines,
                chunk_bands=config.chunk_bands,
            )

        if config.export_excel:
            excel_path = config.output_dir / f"{config.header_path.stem}.xlsx"
            _update_status(state, f"Exporting Excel to {excel_path}...")
            f04_export_excel_spectra_tiled(
                header,
                memmap,
                excel_path,
                tile_lines=config.tile_lines,
                tile_samples=config.tile_samples,
            )

        _update_status(state, "Export completed successfully.")
    except Exception as exc:  # pylint: disable=broad-except
        tb = traceback.format_exc()
        _update_status(state, f"Export failed: {exc}")
        root: tk.Tk = state["root"]  # type: ignore[assignment]
        root.after(
            0,
            lambda: messagebox.showerror(
                "Export failed",
                f"An error occurred while exporting:\n{exc}\n\nDetails:\n{tb}",
            ),
        )
    finally:
        root: tk.Tk = state["root"]  # type: ignore[assignment]
        root.after(0, lambda: state.__setitem__("worker", None))


def f12_build_gui(root: tk.Tk) -> Dict[str, object]:
    root.title("PS SP Analyzer")

    vars_dict: Dict[str, tk.Variable] = {
        "header_path": tk.StringVar(),
        "output_dir": tk.StringVar(),
        "export_feather": tk.BooleanVar(value=True),
        "export_excel": tk.BooleanVar(value=False),
        "chunk_lines": tk.IntVar(value=256),
        "chunk_bands": tk.IntVar(value=16),
        "tile_lines": tk.IntVar(value=64),
        "tile_samples": tk.IntVar(value=64),
    }
    status_var = tk.StringVar(value="Select an ENVI header to begin.")

    state: Dict[str, object] = {
        "root": root,
        "vars": vars_dict,
        "status_var": status_var,
        "worker": None,
    }

    frame = tk.Frame(root, padx=12, pady=12)
    frame.pack(fill=tk.BOTH, expand=True)
    frame.columnconfigure(1, weight=1)

    tk.Label(frame, text="ENVI header:").grid(row=0, column=0, sticky="w", pady=2)
    tk.Entry(frame, textvariable=vars_dict["header_path"]).grid(row=0, column=1, sticky="ew", pady=2)
    tk.Button(frame, text="Browse...", command=lambda: f10_select_files(state, kind="header")).grid(
        row=0, column=2, padx=(6, 0), pady=2
    )

    tk.Label(frame, text="Output directory:").grid(row=1, column=0, sticky="w", pady=2)
    tk.Entry(frame, textvariable=vars_dict["output_dir"]).grid(row=1, column=1, sticky="ew", pady=2)
    tk.Button(frame, text="Browse...", command=lambda: f10_select_files(state, kind="output")).grid(
        row=1, column=2, padx=(6, 0), pady=2
    )

    tk.Label(frame, text="Feather export:").grid(row=2, column=0, sticky="w", pady=2)
    tk.Checkbutton(frame, variable=vars_dict["export_feather"]).grid(row=2, column=1, sticky="w", pady=2)

    tk.Label(frame, text="Excel export:").grid(row=3, column=0, sticky="w", pady=2)
    tk.Checkbutton(frame, variable=vars_dict["export_excel"]).grid(row=3, column=1, sticky="w", pady=2)

    separator = tk.Frame(frame, height=2, bd=1, relief=tk.SUNKEN)
    separator.grid(row=4, column=0, columnspan=3, sticky="ew", pady=8)

    tk.Label(frame, text="Chunk lines:").grid(row=5, column=0, sticky="w", pady=2)
    tk.Entry(frame, textvariable=vars_dict["chunk_lines"], width=8).grid(row=5, column=1, sticky="w", pady=2)

    tk.Label(frame, text="Chunk bands:").grid(row=6, column=0, sticky="w", pady=2)
    tk.Entry(frame, textvariable=vars_dict["chunk_bands"], width=8).grid(row=6, column=1, sticky="w", pady=2)

    tk.Label(frame, text="Tile lines:").grid(row=7, column=0, sticky="w", pady=2)
    tk.Entry(frame, textvariable=vars_dict["tile_lines"], width=8).grid(row=7, column=1, sticky="w", pady=2)

    tk.Label(frame, text="Tile samples:").grid(row=8, column=0, sticky="w", pady=2)
    tk.Entry(frame, textvariable=vars_dict["tile_samples"], width=8).grid(row=8, column=1, sticky="w", pady=2)

    tk.Button(frame, text="Start export", command=lambda: f11_start_export(state)).grid(
        row=9, column=0, columnspan=3, sticky="ew", pady=(12, 4)
    )

    tk.Label(frame, textvariable=status_var, anchor="w").grid(
        row=10, column=0, columnspan=3, sticky="ew", pady=(8, 0)
    )

    return state


def f00_safe_run() -> None:
    root = tk.Tk()
    state = f12_build_gui(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    f00_safe_run()
