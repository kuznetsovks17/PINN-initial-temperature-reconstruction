"""Inspect the reduced queue without importing TensorFlow or starting training.

Usage, from this folder: python verify_resume.py
This can freeze any missing synthetic observation vectors, but it never changes
completed run files, their model weights, or the nonlinear reference.
"""
from __future__ import annotations

import sys
import pandas as pd

from inverse_study.common import DATA, RESULTS
from inverse_study import brass
from inverse_study import brass_completion
from inverse_study import completion
from inverse_study import completion_checks


def main():
    completion.verify_protocol_lock(require_reference=True)
    synthetic = completion.plan_inventory(completion.compact_core_plan())
    problem, observations, _, _ = brass.load_brass(DATA / "brass")
    jobs = brass.retraining_plan(problem, training_seeds=(2026, 2027, 2028),
                                methods=("plain", "fourier_fixed", "fourier_adaptive"),
                                include_sensitivity=False, profile="production")
    experimental = brass_completion.job_inventory(jobs, observations, RESULTS / "brass/production")
    experimental["case"] = "brass"
    all_jobs = pd.concat([synthetic, experimental], ignore_index=True)
    rows = []
    for case, group in all_jobs.groupby("case", sort=False):
        rows.append(dict(case=case, planned=len(group),
                         completed=int(group.status.eq("completed").sum()),
                         pending=int(group.status.isin(["missing", "resumable"]).sum()),
                         blocked=int(group.status.eq("blocked").sum())))
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    print(f"\nTotal: {len(all_jobs)} planned; "
          f"{int(summary.completed.sum())} completed; "
          f"{int(summary.pending.sum())} pending; {int(summary.blocked.sum())} blocked.")
    if summary.blocked.sum():
        print(all_jobs[all_jobs.status.eq("blocked")][["case", "method", "training_seed", "reason"]].to_string(index=False))
        return 2
    for label, plan in [("Smooth", completion.compact_linear_plan()),
                        ("Nonlinear", completion.compact_nonlinear_plan())]:
        following = completion.pending_plan(plan, max_new_jobs=1)
        print(f"\n{label}: next pending training")
        print(following[["method", "noise_level", "noise_seed", "training_seed"]].to_string(index=False)
              if len(following) else "No training remains.")
    baselines = completion_checks.nonlinear_baseline_inventory()
    print("\nNonlinear classical comparisons (separate from neural training counts):")
    print(baselines[["noise_level", "noise_seed", "status", "completed_alphas", "planned_alphas"]].to_string(index=False))
    assert "tensorflow" not in sys.modules, "Audit unexpectedly imported TensorFlow"
    print("\nNo training was started. Completed models and the reference were not changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
