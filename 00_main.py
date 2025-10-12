"""Orchestrates sequential execution of project notebooks or their Python exports."""

from __future__ import annotations

import argparse
import runpy
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import nbformat
from nbclient import NotebookClient

DEFAULT_DATASET_DIV = "ols"
AUTO_GLOB = "[0-9][0-9]_*.ipynb"
STOP_ON_ERROR = True
KERNEL_NAME = None
TIMEOUT_SEC = 60 * 20
OUTPUT_DIR = Path("_executed")

PIPELINE_RUNS: Dict[str, List[str]] = {
    "ols": [
        #"05_get_data_from.ipynb",
        "17_check_digit.ipynb",
        "20_calc_cost_duration.ipynb",
        "25_assign_region.ipynb",
        "30_assign_cat_dummy.ipynb",
        "35_assign_wgi.ipynb",
        "40_assign_freedomrate.ipynb",
        #"45_assign_gdp_at_pjt_start.ipynb",
        "45_assign_gdp.ipynb",
        "50_assign_population.ipynb",
        "55_assign_ex_eval_flg.ipynb",
        "60_clean_text_variants.ipynb",
        "65_unify_sectors.ipynb",
        "70_assign_rate.ipynb",
        "99_rename_df.ipynb",
    ],
    "ml": [
        #"05_get_data_from.ipynb",
        "17_check_digit.ipynb",
        "20_calc_cost_duration.ipynb",
        "25_assign_region.ipynb",
        "30_assign_cat_dummy.ipynb",
        "35_assign_wgi.ipynb",
        "40_assign_freedomrate.ipynb",
        "45_assign_gdp_at_pjt_start.ipynb",
        #"45_assign_gdp.ipynb",
        "50_assign_population.ipynb",
        "55_assign_ex_eval_flg.ipynb",
        "60_clean_text_variants.ipynb",
        "65_unify_sectors.ipynb",
        "70_assign_rate.ipynb",
        "99_rename_df.ipynb",    
    ],
}


def resolve_dataset(dataset_div: str) -> str:
    if not dataset_div:
        raise ValueError("dataset_div must be provided.")
    key = dataset_div.lower()
    if key not in PIPELINE_RUNS:
        raise ValueError("dataset_div must be 'ols' or 'ml'.")
    return key


def discover_notebooks(root: Path) -> List[str]:
    notebooks = sorted(p.name for p in root.glob(AUTO_GLOB) if p.name != "00_main.ipynb")
    return notebooks


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def detect_execution_env() -> str:
    try:
        from IPython import get_ipython  # type: ignore
    except Exception:
        return "PY"
    shell = get_ipython()
    if shell and shell.__class__.__name__ == "ZMQInteractiveShell":
        return "NOTEBOOK"
    return "PY"


def resolve_exec_env(exec_env_override: Optional[str]) -> str:
    if exec_env_override:
        if exec_env_override.lower() in {"notebook", "nb", "note"}:
            return "NOTEBOOK"
        return "PY"
    return detect_execution_env()


def run_notebook(nb_path: Path) -> dict:
    with nb_path.open("r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    kernel = KERNEL_NAME or nb.metadata.get("kernelspec", {}).get("name", "python3")
    client = NotebookClient(
        nb,
        timeout=TIMEOUT_SEC,
        kernel_name=kernel,
        allow_errors=False,
        record_timing=True,
    )
    start = time.time()
    client.execute()
    duration = time.time() - start

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = OUTPUT_DIR / f"{nb_path.stem}__run-{stamp}.ipynb"
    with out_path.open("w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    return {
        "name": nb_path.name,
        "status": "ok",
        "secs": round(duration, 1),
        "output": str(out_path),
    }


def run_script(py_path: Path, notebook_name: str) -> dict:
    start = time.time()
    runpy.run_path(str(py_path), run_name="__main__")
    duration = time.time() - start
    return {
        "name": notebook_name,
        "status": "ok",
        "secs": round(duration, 1),
        "output": str(py_path),
    }


def execute_entry(path: Path, notebook_name: str, exec_env: str) -> dict:
    if exec_env == "NOTEBOOK":
        return run_notebook(path)
    py_path = path.with_suffix(".py")
    if py_path.exists():
        return run_script(py_path, notebook_name)
    return run_notebook(path)


def summarise_results(results: List[dict], exec_env: str) -> None:
    try:
        import pandas as pd  # type: ignore

        df_res = pd.DataFrame(results)
        if exec_env == "NOTEBOOK":
            try:
                from IPython.display import display  # type: ignore

                display(df_res)
            except Exception:
                print(df_res.to_string(index=False))
        else:
            print(df_res.to_string(index=False))
    except Exception:
        print(results)


def main(dataset_div: str = DEFAULT_DATASET_DIV, exec_env_override: Optional[str] = None) -> List[dict]:
    """
    作成区分に応じて notebook / Python スクリプトを順次実行する。

    Parameters
    ----------
    dataset_div : str
        'ols' または 'ml'
    exec_env_override : Optional[str]
        'notebook' / 'py' / None で実行環境を上書き
    """
    print("running main function...")
    exec_env = resolve_exec_env(exec_env_override)
    print(f"[env] EXEC_ENV = {exec_env}")

    dataset_key = resolve_dataset(dataset_div)
    root = Path(".")
    run_list = list(PIPELINE_RUNS.get(dataset_key, [])) or discover_notebooks(root)

    ensure_output_dir()

    print("Run order:")
    for item in run_list:
        print(f" - {item}")

    results: List[dict] = []
    for name in run_list:
        path = root / name
        print(f"\n=== Running: {name} ===")
        try:
            res = execute_entry(path, name, exec_env)
            secs = res.get("secs")
            output = res.get("output", "")
            if secs is not None:
                print(f"✅ OK  ({secs}s) → {output}")
            else:
                print("✅ OK  (no timing/output info)")
            results.append(res)
        except Exception as exc:
            tb = traceback.format_exc(limit=3)
            print(f"❌ FAIL: {name}\n{tb}")
            results.append(
                {
                    "name": name,
                    "status": "fail",
                    "secs": None,
                    "output": "",
                    "error": str(exc),
                }
            )
            if STOP_ON_ERROR:
                break

    summarise_results(results, exec_env)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pipeline notebooks or exported scripts.")
    parser.add_argument("command", nargs="?", default="main")
    parser.add_argument("dataset_div", nargs="?", default=None, help="ols or ml")
    parser.add_argument("-i", "--input", dest="input_dataset", help="legacy input dataset div")
    parser.add_argument("--env", choices=["auto", "notebook", "py"], default="auto")
    args = parser.parse_args()

    if args.command not in {"main", ""}:
        parser.error(f"Unsupported command '{args.command}'. Only 'main' is available.")

    dataset = args.input_dataset or args.dataset_div or DEFAULT_DATASET_DIV
    env_override = None if args.env == "auto" else args.env
    main(dataset_div=dataset, exec_env_override=env_override)
