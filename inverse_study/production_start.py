"""Run the declared production notebooks in the current Python environment.

There is no smoke mode, user configuration, or expanded sweep in this entry
point. Original solvers and run identities are unchanged. Completed jobs are
reused by each notebook's existing missing-only queue.
"""
from __future__ import annotations

import ast
import gc
import json
import os
from pathlib import Path
from time import perf_counter

from .common import ROOT, RESULTS, write_json


NOTEBOOKS = (
    "00_Start_and_observation_contract.ipynb",
    "01_Forward_verification_and_recoverability.ipynb",
    "02_Classical_baselines_and_QR_audit.ipynb",
    "03_Paired_inverse_PINN_study.ipynb",
    "04_Nonlinear_jump_and_validation.ipynb",
    "05_Brass_experiment.ipynb",
    "07_Audit_tables_figures_and_paper_freeze.ipynb",
)

_EXPECTED = {
    "02": {"RUN_BASELINE_COMPLETION": True},
    "03": {"RUN_PRODUCTION": True, "MAX_NEW_JOBS": None,
           "CONFIG_OVERRIDES": {}, "RUN_SMOKE": False,
           "RUN_MATCHED_CONTROL": False, "RUN_REPLAY_REFINEMENT": True,
           "MAX_NEW_REPLAYS": None},
    "04": {"RUN_NEURAL": True, "MAX_NEW_JOBS": None,
           "RUN_NONLINEAR_BASELINE": True, "MAX_NEW_BASELINE_CASES": None,
           "RUN_REPLAY_REFINEMENT": True, "MAX_NEW_REPLAYS": None,
           "RUN_QR_STUDY": False},
    "05": {"PROFILE": "production", "ALLOW_PRODUCTION": True,
           "RUN_TRAINING": True, "MAX_NEW_TRAINING_JOBS": None,
           "METHODS": ("plain", "fourier_fixed", "fourier_adaptive"),
           "TRAINING_SEEDS": (2026, 2027, 2028),
           "RUN_EVALUATION": True, "EVALUATION_RUN_DIRS": [],
           "RUN_REPLAY": True, "RUN_REFINEMENT": True, "REPLAY_RUN_DIRS": [],
           "RUN_SENSITIVITY": False, "RUN_NONLINEAR_TIKHONOV": False,
           "RUN_QR_AUDIT": False},
    "07": {"FREEZE_FOR_PAPER": False},
}


def prepared_notebooks():
    """Read and compile all selected notebooks before starting computation."""
    from .completion import (compact_linear_plan, compact_nonlinear_plan,
                             verify_protocol_lock)
    verify_protocol_lock(require_reference=True)
    if len(compact_linear_plan()) != 18 or len(compact_nonlinear_plan()) != 12:
        raise RuntimeError("The declared 18 + 12 + 9 production plan changed")
    prepared = []
    for name in NOTEBOOKS:
        file = ROOT / name
        notebook = json.loads(file.read_text(encoding="utf-8"))
        controls, cells = {}, []
        expected = _EXPECTED.get(name[:2], {})
        for index, cell in enumerate(notebook["cells"], start=1):
            if cell["cell_type"] != "code":
                continue
            source = cell.get("source", [])
            source = source if isinstance(source, str) else "".join(source)
            if not source.strip():
                continue
            tree = ast.parse(source, filename=f"{file}:cell{index}")
            for node in tree.body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id in expected:
                            if target.id in controls:
                                raise RuntimeError(f"Duplicate production setting {target.id} in {name}")
                            controls[target.id] = ast.literal_eval(node.value)
            cells.append((index, compile(tree, f"{file}:cell{index}", "exec")))
        for key, value in expected.items():
            if key not in controls or controls[key] != value:
                raise RuntimeError(f"Production setting {key} was changed in {name}. "
                                   "Restore the supplied updated notebook before starting this cohort.")
        prepared.append((name, cells))
    return prepared


def run_all():
    """Execute the complete fixed production workflow with no manual switches."""
    prepared = prepared_notebooks()
    log_path = RESULTS / "reduced_plan/automatic_execution.json"
    log = {"design": "compact_paired_v1", "planned_neural_trainings": 39,
           "notebooks": [], "status": "running"}
    print("PRODUCTION: 18 smooth + 12 nonlinear + 9 brass = 39 total trainings.", flush=True)
    print("Matching completed jobs are skipped; interrupted jobs resume.", flush=True)
    print("Running the selected notebooks sequentially, including validation and the report.", flush=True)
    previous_cwd = Path.cwd()
    os.chdir(ROOT)
    try:
        for position, (name, cells) in enumerate(prepared, start=1):
            entry = {"notebook": name, "status": "running", "last_code_cell": None}
            log["notebooks"].append(entry)
            write_json(log_path, log)
            print(f"\n[{position}/{len(prepared)}] {name}", flush=True)
            namespace = {"__name__": "__main__", "__file__": str(ROOT / name)}
            started = perf_counter()
            for index, code in cells:
                entry["last_code_cell"] = index
                print(f"  Code cell {index}", flush=True)
                exec(code, namespace, namespace)
            entry.update(status="completed", wall_seconds=perf_counter() - started)
            write_json(log_path, log)
            del namespace
            gc.collect()
        summary_file = RESULTS / "reduced_report/completion_summary.json"
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        completed = int(summary["synthetic_completed"]) + int(summary["brass_completed"])
        blocked = int(summary["synthetic_blocked"]) + int(summary["brass_blocked"])
        log["training_completed"] = completed
        log["training_blocked"] = blocked
        log["status"] = "notebooks_executed" if completed == 39 and not blocked else "finished_with_pending_jobs"
        write_json(log_path, log)
        print(f"\nNeural training: {completed}/39 completed; {blocked} blocked.", flush=True)
        print("Results and any pending work or validation failures are in results/reduced_report/.", flush=True)
    except BaseException as exc:
        log["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        log["error_type"] = type(exc).__name__
        log["error"] = str(exc)
        if log["notebooks"]:
            log["notebooks"][-1]["status"] = log["status"]
        write_json(log_path, log)
        print("\nStopped. Completed results are saved. Run this notebook again to continue.", flush=True)
        raise
    finally:
        os.chdir(previous_cwd)
    return log
