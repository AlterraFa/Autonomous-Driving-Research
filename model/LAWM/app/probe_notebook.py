import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Union


def _load_notebook(notebook_path: Path) -> dict:
    with notebook_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_params_from_notebook(notebook_path: Path) -> dict:
    """
    Extract fully-resolved `PARAMS` from the notebook by executing only the
    config cell that defines:
      - COMMON_SETTINGS
      - LOADER_SETTINGS
      - PARAMS (which uses **COMMON_SETTINGS / **LOADER_SETTINGS)

    Notes:
      - We intentionally do NOT execute the rest of the notebook (it may write
        launcher scripts or perform installation/training).
      - The config cell is expected to be pure-Python dict construction.
    """
    nb = _load_notebook(notebook_path)

    cells = nb.get("cells", [])
    for cell_idx, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)

        # Heuristic: the config cell contains both COMMON_SETTINGS and PARAMS.
        if "COMMON_SETTINGS" not in source or "PARAMS" not in source:
            continue
        if "LOADER_SETTINGS" not in source:
            # Some notebooks might omit LOADER_SETTINGS, but in this project it exists.
            # We'll skip if not present to avoid wrong cells.
            continue

        env: dict = {}
        # Execute the cell in an isolated namespace. This should only define dicts.
        exec(source, env, env)

        if "PARAMS" not in env:
            raise RuntimeError(
                f"Found candidate config cell but `PARAMS` was not defined: {notebook_path} (cell {cell_idx})"
            )

        return {
            "extracted_from_cell_index": cell_idx,
            "COMMON_SETTINGS": env.get("COMMON_SETTINGS"),
            "LOADER_SETTINGS": env.get("LOADER_SETTINGS"),
            "DEVICES": env.get("DEVICES"),
            "CONTINUE_PATH": env.get("CONTINUE_PATH"),
            "PARAMS": env["PARAMS"],
        }

    raise RuntimeError(f"Could not find a config cell defining PARAMS: {notebook_path}")


def extract_nb_params(notebook_path: Union[str, Path]) -> dict[str, Any]:
    """
    Extract fully-resolved `PARAMS` from a single `latent-probe.ipynb`.

    WARNING: This executes the code cell that defines `COMMON_SETTINGS` /
    `LOADER_SETTINGS` / `PARAMS` via `exec()`. Only use this on notebooks you
    trust.
    """
    notebook_path = Path(notebook_path).resolve()
    if not notebook_path.exists():
        raise FileNotFoundError(str(notebook_path))
    extracted = _extract_params_from_notebook(notebook_path)
    # Keep metadata near the extracted params.
    return {
        "notebook_path": str(notebook_path.relative_to(Path.cwd())),
        **extracted,
    }


def extract_params_dict(
    runs_dir: Union[str, Path],
    *,
    run_specs: Optional[list[str]] = None,
    notebook_filename: str = "latent-probe.ipynb",
    expected_run_name_regex: str = r"^run\d+$",
) -> dict[str, dict[str, Any]]:
    """
    Extract fully-resolved `PARAMS` for one or more runs.

    - If `runs_dir` itself contains `latent-probe.ipynb`, it is treated as a single run.
    - Otherwise, it is treated as a directory containing `run*/latent-probe.ipynb`.

    Returns: dict keyed by run name (e.g. `run1`, `run2`, ...).
    """
    runs_dir = Path(runs_dir).resolve()
    if not runs_dir.exists():
        raise FileNotFoundError(str(runs_dir))

    run_name_re = re.compile(expected_run_name_regex)
    notebook_path = runs_dir / notebook_filename

    # Case 1: the caller passed .../runX
    if notebook_path.exists() and notebook_path.is_file():
        run_name = runs_dir.name if run_name_re.match(runs_dir.name) else "run"
        return {run_name: extract_nb_params(notebook_path)}

    # Case 2: the caller passed .../probe (or similar) containing run*/
    if run_specs:
        # Support run specs like `run2` or `probe/run2` (or relative paths).
        def _resolve_run_dir(run_spec: str) -> Path:
            run_spec = run_spec.strip()
            if run_name_re.match(run_spec):
                return runs_dir / run_spec
            # Allow things like "probe/run2" or "./run2"
            candidate = Path(run_spec)
            if not candidate.is_absolute():
                candidate = Path.cwd() / candidate
            return candidate.resolve()

        run_dirs: list[Path] = []
        for spec in run_specs:
            rd = _resolve_run_dir(spec)
            if not rd.exists():
                raise FileNotFoundError(f"Run directory not found: {rd} (from run_specs item {spec!r})")
            run_dirs.append(rd)
    else:
        run_dirs = [
            p
            for p in runs_dir.iterdir()
            if p.is_dir() and run_name_re.match(p.name) and (p / notebook_filename).exists()
        ]

    run_dirs.sort(key=lambda p: int(p.name.replace("run", "")) if run_name_re.match(p.name) else 0)
    if not run_dirs:
        raise RuntimeError(
            f"No run directories found in {runs_dir} "
            f"(expected {notebook_filename} under run*/ with pattern {expected_run_name_regex})"
        )

    index: dict[str, dict[str, Any]] = {}
    for run_dir in run_dirs:
        nb_path = run_dir / notebook_filename
        if not nb_path.exists():
            raise FileNotFoundError(f"Missing notebook: {nb_path}")
        run_name = run_dir.name
        index[run_name] = extract_nb_params(nb_path)
    return index


def find_folders_with_pt_files(root_path: Union[str, Path]) -> dict[str, list[str]]:
    """
    Recursively find folders under `root_path` that contain `.pt` files.

    Returns a dict: {folder_path: [pt_file_name, ...]}.
    """
    root = Path(root_path).resolve()
    if not root.exists():
        raise FileNotFoundError(str(root))
    if root.is_file():
        # If a notebook/file is provided, recurse from its containing directory.
        root = root.parent

    grouped: dict[Path, list[str]] = defaultdict(list)
    for pt_file in root.rglob("*.pt"):
        if pt_file.is_file():
            grouped[pt_file.parent].append(pt_file.name)
            

    return {
        str(folder.relative_to(Path.cwd())): sorted(file_names)
        for folder, file_names in sorted(grouped.items(), key=lambda x: str(x[0]))
    }


def recurse_weight_fd(root_path: Union[str, Path]) -> Optional[str]:
    """
    Return the folder that contains all discovered `.pt` files under `root_path`.

    If `.pt` files are spread across multiple folders, returns None.
    """
    folders = find_folders_with_pt_files(root_path)
    if not folders:
        return None

    # "All weights" means all discovered .pt files are in one folder.
    if len(folders) == 1:
        return next(iter(folders.keys()))

    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "probe",
        help="Directory containing run*/latent-probe.ipynb",
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help=(
            "Run to extract (e.g. run1). Can be passed multiple times. "
            "Also accepts values like `probe/run2`."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "latent_probe_params_index.json",
        help="Output JSON index path",
    )
    parser.add_argument(
        "--write-per-run",
        action="store_true",
        help="Also write individual JSON files (one per run)",
    )
    args = parser.parse_args()

    run_specs: Optional[list[str]] = None
    if args.run:
        # Support comma-separated lists too: --run run1,run2
        run_specs = []
        for item in args.run:
            run_specs.extend([s for s in item.split(",") if s.strip()])

    index = extract_params_dict(args.runs_dir, run_specs=run_specs)

    if args.write_per_run:
        for run_name, entry in index.items():
            per_run_out = args.out.with_name(f"latent_probe_params_{run_name}.json")
            with per_run_out.open("w", encoding="utf-8") as f:
                json.dump(entry, f, indent=2, ensure_ascii=False, default=str)

    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False, default=str)

    print(f"Wrote: {out_path}")
    print(f"Runs extracted: {', '.join(index.keys())}")


if __name__ == "__main__":
    main()

