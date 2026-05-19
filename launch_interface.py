"""Lightweight experiment launcher UI.

UX flow:
1) First screen asks whether to use Config presets or a custom preset.
2) Run screen always shows explicit runtime options (environment type,
   fast/slow mode, plotting, GNN, mobility, etc.).
3) Custom preset mode additionally provides full JSON editors for scenario and
   optimization configs so every parameter is editable.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "Config"
HETNET_DIR = CONFIG_DIR / "hetnet"
CELL_FREE_DIR = CONFIG_DIR / "cell_free"
OPT_DIR = CONFIG_DIR / "opt"
HYPERPARAMS_DIR = ROOT / "HyperparameterConfig"
GENERATED_OPT = CONFIG_DIR / "config_runtime_generated_opt.json"
GENERATED_SCE = CONFIG_DIR / "config_runtime_generated_sce.json"

GNN_ARCH_OPTIONS = ["Disabled", "GCN", "GAT", "SAGE", "Transformer"]


def _list_scenario_files(env_type: str):
    """Return sorted list of scenario JSON files for the given environment."""
    subdir = CELL_FREE_DIR if env_type == "cell_free" else HETNET_DIR
    if subdir.is_dir():
        files = sorted([p for p in subdir.glob("*.json") if p.is_file()])
        if files:
            return files
    return []


def _list_net_files():
    """Return sorted list of network hyperparameter JSON files."""
    if HYPERPARAMS_DIR.is_dir():
        files = sorted([p for p in HYPERPARAMS_DIR.glob("network_*.json") if p.is_file()])
        if files:
            return files
    return []


def _net_path_for(size: str) -> Path:
    """Canonical network hyperparameter path for a given size."""
    return HYPERPARAMS_DIR / f"network_{size}.json"


def _default_net_file() -> Path | None:
    """Return the first available network config file, preferring 'small'."""
    for preferred in ("small", "medium", "test", "large"):
        p = _net_path_for(preferred)
        if p.exists():
            return p
    files = _list_net_files()
    return files[0] if files else None


def _list_opt_files():
    """Return sorted list of optimization JSON files from Config/opt/."""
    if OPT_DIR.is_dir():
        files = sorted([p for p in OPT_DIR.glob("*.json") if p.is_file()])
        if files:
            return files
    return []


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, payload):
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


PRESET_SIZES = ["test", "small", "medium", "large"]


def _sce_path_for(size: str, env_type: str) -> Path:
    """Canonical scenario path for a given size and environment."""
    subdir = CELL_FREE_DIR if env_type == "cell_free" else HETNET_DIR
    return subdir / f"config_sce_{size}.json"


def _opt_path_for(size: str) -> Path:
    """Canonical optimization path for a given size."""
    return OPT_DIR / f"config_opt_{size}.json"


def _available_sizes(env_type: str) -> list[str]:
    """Return sizes that have both sce and opt files present."""
    return [
        s for s in PRESET_SIZES
        if _sce_path_for(s, env_type).exists() and _opt_path_for(s).exists()
    ]


def _pick_default_size(env_type: str) -> str:
    available = _available_sizes(env_type)
    for preferred in ("small", "medium", "test", "large"):
        if preferred in available:
            return preferred
    return available[0] if available else "small"


def _preset_label(size: str, env_type: str) -> str:
    """Short info string shown next to the size combo."""
    sce = _sce_path_for(size, env_type)
    opt = _opt_path_for(size)
    sce_name = sce.name if sce.exists() else f"{sce.name} (missing)"
    opt_name = opt.name if opt.exists() else f"{opt.name} (missing)"
    subdir = "cell_free" if env_type == "cell_free" else "hetnet"
    return f"{subdir}/{sce_name}  +  opt/{opt_name}"


def _run_main(
    config_sce: Path,
    config_opt: Path,
    ntrials: int,
    run_name: str | None,
    seed: str | None,
    config_net: Path | None = None,
):
    cmd = [
        sys.executable,
        str(ROOT / "main.py"),
        "-c1",
        str(config_sce),
        "-c2",
        str(config_opt),
        "-n",
        str(ntrials),
    ]
    if config_net:
        cmd.extend(["-c3", str(config_net)])
    if run_name:
        cmd.extend(["--run-name", run_name])
    if seed:
        cmd.extend(["--seed", str(seed)])

    print("Running:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=False)


def _apply_runtime_flags(opt_payload: dict, runtime_vars: dict):
    """Apply explicit runtime options to optimization payload."""
    opt_payload["environment_type"] = runtime_vars["environment_type"].get()
    opt_payload["fast_mode"] = bool(runtime_vars["fast_mode"].get())
    opt_payload["enable_plot"] = bool(runtime_vars["enable_plot"].get())
    opt_payload["network_plot_enabled"] = bool(runtime_vars["network_plot_enabled"].get())
    opt_payload["telecom_plot_enabled"] = bool(runtime_vars["telecom_plot_enabled"].get())
    opt_payload["resource_plot_enabled"] = bool(runtime_vars["resource_plot_enabled"].get())
    opt_payload["deferred_plotting"] = bool(runtime_vars["deferred_plotting"].get())
    opt_payload["mobility_enabled"] = bool(runtime_vars["mobility_enabled"].get())

    # GNN architecture is a single exclusive choice
    arch = runtime_vars["gnn_arch"].get()
    if arch == "Disabled":
        opt_payload["gnn_enabled"] = False
        opt_payload["gnn_transformer_enabled"] = False
        opt_payload.pop("gnn_conv_type", None)
    else:
        opt_payload["gnn_enabled"] = True
        opt_payload["gnn_conv_type"] = arch.lower()
        opt_payload["gnn_transformer_enabled"] = (arch == "Transformer")


def _load_runtime_flags_from_opt(opt_payload: dict, runtime_vars: dict):
    runtime_vars["environment_type"].set(str(opt_payload.get("environment_type", "hetnet")))
    runtime_vars["fast_mode"].set(bool(opt_payload.get("fast_mode", False)))
    runtime_vars["enable_plot"].set(bool(opt_payload.get("enable_plot", True)))
    runtime_vars["network_plot_enabled"].set(bool(opt_payload.get("network_plot_enabled", True)))
    runtime_vars["telecom_plot_enabled"].set(bool(opt_payload.get("telecom_plot_enabled", True)))
    runtime_vars["resource_plot_enabled"].set(bool(opt_payload.get("resource_plot_enabled", True)))
    runtime_vars["deferred_plotting"].set(bool(opt_payload.get("deferred_plotting", False)))
    runtime_vars["mobility_enabled"].set(bool(opt_payload.get("mobility_enabled", False)))

    # Reconstruct single GNN arch choice from flags
    gnn_enabled = bool(opt_payload.get("gnn_enabled", False))
    gnn_transformer = bool(opt_payload.get("gnn_transformer_enabled", False))
    conv_type = str(opt_payload.get("gnn_conv_type", "gcn")).upper()
    if not gnn_enabled:
        runtime_vars["gnn_arch"].set("Disabled")
    elif gnn_transformer or conv_type == "TRANSFORMER":
        runtime_vars["gnn_arch"].set("Transformer")
    else:
        runtime_vars["gnn_arch"].set(conv_type if conv_type in GNN_ARCH_OPTIONS else "GCN")


def _terminal_fallback():
    print("Tkinter unavailable. Using terminal interface.")

    env = input("environment_type [hetnet/cell_free] (hetnet): ").strip() or "hetnet"

    available = _available_sizes(env)
    if not available:
        raise RuntimeError("No preset files found in Config/")

    default_size = _pick_default_size(env)
    print(f"\nAvailable presets: {', '.join(available)}")
    size = input(f"Preset size ({default_size}): ").strip() or default_size
    if size not in available:
        print(f"Unknown size '{size}', using '{default_size}'")
        size = default_size

    sce_path = _sce_path_for(size, env)
    opt_path = _opt_path_for(size)
    opt = _load_json(opt_path)
    sce = _load_json(sce_path)

    opt["environment_type"] = env
    fast_mode = input(f"fast_mode [true/false] ({str(opt.get('fast_mode', False)).lower()}): ").strip().lower()
    gnn_arch = input(f"GNN architecture [{'/'.join(GNN_ARCH_OPTIONS)}] (Disabled): ").strip() or "Disabled"
    plot_enabled = input(f"enable_plot [true/false] ({str(opt.get('enable_plot', True)).lower()}): ").strip().lower()

    if fast_mode in {"true", "false"}:
        opt["fast_mode"] = fast_mode == "true"
    if plot_enabled in {"true", "false"}:
        opt["enable_plot"] = plot_enabled == "true"
    if gnn_arch in GNN_ARCH_OPTIONS:
        if gnn_arch == "Disabled":
            opt["gnn_enabled"] = False
            opt["gnn_transformer_enabled"] = False
        else:
            opt["gnn_enabled"] = True
            opt["gnn_conv_type"] = gnn_arch.lower()
            opt["gnn_transformer_enabled"] = (gnn_arch == "Transformer")

    # Network hyperparameter config
    net_files = _list_net_files()
    net_path = None
    if net_files:
        net_names = [p.name for p in net_files]
        print(f"\nAvailable network configs: {', '.join(net_names)}")
        net_choice = input(f"Network config file (leave blank to skip): ").strip()
        if net_choice:
            matched = [p for p in net_files if p.name == net_choice or p.stem == net_choice]
            net_path = matched[0] if matched else None

    _save_json(GENERATED_SCE, sce)
    _save_json(GENERATED_OPT, opt)

    ntrials = int(input("ntrials (1): ").strip() or "1")
    run_name = input("run_name (optional): ").strip() or None
    seed = input("seed (optional): ").strip() or None

    _run_main(GENERATED_SCE, GENERATED_OPT, ntrials, run_name, seed, config_net=net_path)


def _tk_ui():
    import tkinter as tk
    from tkinter import ttk, messagebox

    default_env = "hetnet"
    default_size = _pick_default_size(default_env)

    if not _available_sizes(default_env):
        raise RuntimeError("No JSON presets found in Config/")

    root = tk.Tk()
    root.title("UARA-DRL Launcher")
    root.geometry("560x270")

    container = ttk.Frame(root, padding=16)
    container.pack(fill="both", expand=True)

    size_var = tk.StringVar(value=default_size)
    net_var = tk.StringVar(value="")
    ntrials_var = tk.StringVar(value="1")
    run_name_var = tk.StringVar(value="")
    seed_var = tk.StringVar(value="")
    mode_var = tk.StringVar(value="preset")

    runtime_vars = {
        "environment_type": tk.StringVar(value="hetnet"),
        "fast_mode": tk.BooleanVar(value=False),
        "enable_plot": tk.BooleanVar(value=True),
        "network_plot_enabled": tk.BooleanVar(value=True),
        "telecom_plot_enabled": tk.BooleanVar(value=True),
        "resource_plot_enabled": tk.BooleanVar(value=True),
        "deferred_plotting": tk.BooleanVar(value=False),
        "gnn_arch": tk.StringVar(value="Disabled"),
        "mobility_enabled": tk.BooleanVar(value=False),
    }

    sce_text = None
    opt_text = None
    env_trace_id: list = [None]  # holds current trace ID to allow removal on re-entry

    def clear_container():
        for child in container.winfo_children():
            child.destroy()

    def _load_to_editor(text_widget, path: Path):
        payload = _load_json(path)
        text_widget.delete("1.0", tk.END)
        text_widget.insert("1.0", json.dumps(payload, indent=2))

    def _parse_editor_json(text_widget, label: str):
        try:
            txt = text_widget.get("1.0", tk.END).strip()
            return json.loads(txt) if txt else {}
        except Exception as e:
            raise ValueError(f"Invalid JSON in {label} editor: {e}")

    def build_runtime_options(parent):
        opts = ttk.LabelFrame(parent, text="Runtime Options (Explicit)", padding=10)
        opts.grid_columnconfigure(1, weight=1)

        ttk.Label(opts, text="Environment").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            opts,
            textvariable=runtime_vars["environment_type"],
            values=["hetnet", "cell_free"],
            state="readonly",
            width=14,
        ).grid(row=0, column=1, sticky="w", padx=8)

        ttk.Checkbutton(opts, text="Fast mode", variable=runtime_vars["fast_mode"]).grid(row=1, column=0, sticky="w")
        ttk.Checkbutton(opts, text="Enable plotting", variable=runtime_vars["enable_plot"]).grid(row=1, column=1, sticky="w")
        ttk.Checkbutton(opts, text="Network plot", variable=runtime_vars["network_plot_enabled"]).grid(row=2, column=0, sticky="w")
        ttk.Checkbutton(opts, text="Telecom plot", variable=runtime_vars["telecom_plot_enabled"]).grid(row=2, column=1, sticky="w")
        ttk.Checkbutton(opts, text="Resource plot", variable=runtime_vars["resource_plot_enabled"]).grid(row=3, column=0, sticky="w")
        ttk.Checkbutton(opts, text="Deferred plotting", variable=runtime_vars["deferred_plotting"]).grid(row=3, column=1, sticky="w")

        ttk.Label(opts, text="GNN architecture").grid(row=4, column=0, sticky="w")
        ttk.Combobox(
            opts,
            textvariable=runtime_vars["gnn_arch"],
            values=GNN_ARCH_OPTIONS,
            state="readonly",
            width=14,
        ).grid(row=4, column=1, sticky="w", padx=8)

        ttk.Checkbutton(opts, text="Mobility enabled", variable=runtime_vars["mobility_enabled"]).grid(row=5, column=0, sticky="w")
        return opts

    def load_opt_flags_from_selected_preset():
        try:
            payload = _load_json(_get_opt_path())
            _load_runtime_flags_from_opt(payload, runtime_vars)
        except Exception:
            pass

    def _get_sce_path() -> Path:
        """Resolve scenario path from current size and environment."""
        env = runtime_vars["environment_type"].get()
        return _sce_path_for(size_var.get(), env)

    def _get_opt_path() -> Path:
        """Resolve optimization path from current size."""
        return _opt_path_for(size_var.get())

    def _get_net_path() -> Path | None:
        """Resolve network hyperparameter path from net_var (full path or None)."""
        val = net_var.get().strip()
        if not val or val == "(none)":
            return None
        p = Path(val)
        return p if p.exists() else None

    def run_clicked():
        try:
            env_type = runtime_vars["environment_type"].get()
            if mode_var.get() == "custom":
                sce_payload = _parse_editor_json(sce_text, "scenario")
                opt_payload = _parse_editor_json(opt_text, "optimization")
            else:
                sce_payload = _load_json(_get_sce_path())
                opt_payload = _load_json(_get_opt_path())

            # Enforce scenario/environment compatibility
            sce_env = sce_payload.get("_env")
            if sce_env and sce_env != env_type:
                messagebox.showerror(
                    "Environment Mismatch",
                    f"Selected scenario is for '{sce_env}' but environment is set to '{env_type}'.\n"
                    "Please select a matching scenario file.",
                )
                return

            _apply_runtime_flags(opt_payload, runtime_vars)
            _save_json(GENERATED_SCE, sce_payload)
            _save_json(GENERATED_OPT, opt_payload)

            ntrials = int(ntrials_var.get().strip() or "1")
            run_name = run_name_var.get().strip() or None
            seed = seed_var.get().strip() or None
            net_path = _get_net_path()
            _run_main(GENERATED_SCE, GENERATED_OPT, ntrials, run_name, seed, config_net=net_path)
        except Exception as e:
            messagebox.showerror("Run Error", str(e))

    def build_run_screen(custom_mode: bool):
        nonlocal sce_text, opt_text
        clear_container()
        mode_var.set("custom" if custom_mode else "preset")
        root.geometry("1160x820" if custom_mode else "960x560")

        header = ttk.Frame(container)
        header.pack(fill="x", pady=(0, 10))
        ttk.Button(header, text="Back", command=build_start_screen).pack(side="left")
        ttk.Label(
            header,
            text="Custom Preset" if custom_mode else "Use Config Presets",
            font=("TkDefaultFont", 12, "bold"),
        ).pack(side="left", padx=10)

        presets = ttk.LabelFrame(container, text="Preset", padding=10)
        presets.pack(fill="x", pady=(0, 10))

        ttk.Label(presets, text="Size").grid(row=0, column=0, sticky="w")
        env_now = runtime_vars["environment_type"].get()
        size_combo = ttk.Combobox(
            presets, textvariable=size_var,
            values=_available_sizes(env_now),
            state="readonly", width=12,
        )
        size_combo.grid(row=0, column=1, sticky="w", padx=8)

        size_label = ttk.Label(
            presets,
            text=_preset_label(size_var.get(), env_now),
            foreground="#555",
        )
        size_label.grid(row=0, column=2, sticky="w", padx=(12, 0))

        # Network hyperparameter file selector
        ttk.Label(presets, text="Network config").grid(row=1, column=0, sticky="w", pady=(6, 0))
        net_files = _list_net_files()
        net_display = ["(none)"] + [str(p) for p in net_files]
        # Pre-select matching size if available
        default_net = _net_path_for(size_var.get())
        net_var.set(str(default_net) if default_net.exists() else "(none)")
        net_combo = ttk.Combobox(
            presets, textvariable=net_var,
            values=net_display,
            state="readonly", width=42,
        )
        net_combo.grid(row=1, column=1, columnspan=2, sticky="w", padx=8, pady=(6, 0))

        def _refresh_size_label(*_):
            env = runtime_vars["environment_type"].get()
            size_label.config(text=_preset_label(size_var.get(), env))

        def _on_env_changed(*_):
            """Refresh size combo and label when environment type changes."""
            env = runtime_vars["environment_type"].get()
            available = _available_sizes(env)
            size_combo["values"] = available
            if size_var.get() not in available:
                size_var.set(_pick_default_size(env))
            _refresh_size_label()
            if custom_mode and sce_text is not None:
                try:
                    _load_to_editor(sce_text, _get_sce_path())
                except Exception:
                    pass

        def _on_size_selected(*_):
            _refresh_size_label()
            load_opt_flags_from_selected_preset()
            # Auto-select matching network config file when size changes
            candidate = _net_path_for(size_var.get())
            if candidate.exists():
                net_var.set(str(candidate))
            if custom_mode:
                if sce_text is not None:
                    _load_to_editor(sce_text, _get_sce_path())
                if opt_text is not None:
                    _load_to_editor(opt_text, _get_opt_path())

        size_combo.bind("<<ComboboxSelected>>", _on_size_selected)

        # Remove stale trace from a previous build_run_screen call (Back → re-enter)
        if env_trace_id[0] is not None:
            try:
                runtime_vars["environment_type"].trace_remove("write", env_trace_id[0])
            except Exception:
                pass
        env_trace_id[0] = runtime_vars["environment_type"].trace_add("write", _on_env_changed)

        runtime_frame = build_runtime_options(container)
        runtime_frame.pack(fill="x", pady=(0, 10))

        if custom_mode:
            editors = ttk.Frame(container)
            editors.pack(fill="both", expand=True, pady=(0, 10))


            left = ttk.LabelFrame(editors, text="Scenario JSON Editor", padding=6)
            right = ttk.LabelFrame(editors, text="Optimization JSON Editor", padding=6)
            left.pack(side="left", fill="both", expand=True, padx=(0, 6))
            right.pack(side="left", fill="both", expand=True, padx=(6, 0))

            sce_text = tk.Text(left, wrap="none", height=20, width=62)
            opt_text = tk.Text(right, wrap="none", height=20, width=62)
            sce_text.pack(fill="both", expand=True)
            opt_text.pack(fill="both", expand=True)

            _load_to_editor(sce_text, _get_sce_path())
            _load_to_editor(opt_text, _get_opt_path())
        else:
            sce_text = None
            opt_text = None

        bottom = ttk.Frame(container)
        bottom.pack(fill="x")

        ttk.Label(bottom, text="Trials").pack(side="left")
        ttk.Entry(bottom, textvariable=ntrials_var, width=8).pack(side="left", padx=(6, 14))

        ttk.Label(bottom, text="Run name").pack(side="left")
        ttk.Entry(bottom, textvariable=run_name_var, width=18).pack(side="left", padx=(6, 14))

        ttk.Label(bottom, text="Seed").pack(side="left")
        ttk.Entry(bottom, textvariable=seed_var, width=12).pack(side="left", padx=(6, 14))

        ttk.Button(bottom, text="Run", command=run_clicked).pack(side="left", padx=(10, 0))

        load_opt_flags_from_selected_preset()

    def build_start_screen():
        clear_container()
        root.geometry("560x270")

        title = ttk.Label(
            container,
            text="UARA-DRL Launcher",
            font=("TkDefaultFont", 14, "bold"),
        )
        title.pack(pady=(10, 8))

        subtitle = ttk.Label(
            container,
            text="Choose how you want to start:",
        )
        subtitle.pack(pady=(0, 18))

        btns = ttk.Frame(container)
        btns.pack()

        ttk.Button(
            btns,
            text="Use Config Presets",
            command=lambda: build_run_screen(custom_mode=False),
            width=24,
        ).pack(pady=6)

        ttk.Button(
            btns,
            text="Custom Preset",
            command=lambda: build_run_screen(custom_mode=True),
            width=24,
        ).pack(pady=6)

    build_start_screen()
    root.mainloop()


def main():
    try:
        _tk_ui()
    except Exception:
        _terminal_fallback()


if __name__ == "__main__":
    main()
