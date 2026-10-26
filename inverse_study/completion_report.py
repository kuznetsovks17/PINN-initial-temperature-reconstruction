"""Descriptive reporting for the prospectively reduced, paired completion study.

This module only reads trained artifacts and writes a separate report directory.
It never trains models, changes the frozen data, selects the best seed, or treats
the three diagonal noise/training pairs as a crossed factorial experiment.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .common import ROOT, DATA, RESULTS, read_json, write_json
from . import workflows as flow
from . import completion


PLAN_KEYS = ["case", "exposure", "obs_per_dim", "noise_level", "noise_seed",
             "training_seed", "method"]
DATA_KEYS = ["case", "exposure", "obs_per_dim", "noise_level", "noise_seed"]
GROUP_KEYS = ["case", "exposure", "obs_per_dim", "noise_level"]
NEURAL_METHODS = ("plain", "fourier_fixed", "fourier_adaptive")
SYNTHETIC_METRICS = (
    "initial_relative_l2", "independent_observation_residual",
    "initial_boundary_linf", "training_wall_time_s", "rmse_over_true_jump",
    "jump_height_relative_error", "jump_location_absolute_error",
    "transition_width_10_90", "overshoot_over_jump", "undershoot_over_jump",
)


def _read_json_if_present(path):
    return read_json(path) if Path(path).is_file() else {}


def _join_candidate_metrics(inventory, candidate):
    """Keep every planned row, including missing or blocked rows."""
    planned = inventory.copy()
    if "repetition" not in planned:
        planned["repetition"] = planned.training_seed.astype(int) - 2025
    if candidate.empty or "run_folder" not in candidate:
        return planned
    allowed = set(planned.loc[planned.status.eq("completed"), "run_folder"])
    chosen = candidate[candidate.run_folder.isin(allowed)].copy()
    if chosen.run_folder.duplicated().any():
        raise ValueError("Multiple candidate rows refer to the same retained run")
    extra_columns = [column for column in chosen.columns
                     if column not in planned.columns or column == "run_folder"]
    return planned.merge(chosen[extra_columns], on="run_folder", how="left", validate="one_to_one")


def descriptive_summary(frame, groups, metrics, *, planned_counts=None):
    """Uncertainty is descriptive; no bootstrap interval or p value is computed."""
    columns = list(groups) + ["metric", "n_available", "n_planned", "mean", "sample_sd",
                             "median", "minimum", "maximum", "summary_status"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for key, group in frame.groupby(list(groups), dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        label = dict(zip(groups, key))
        planned_n = len(group) if planned_counts is None else planned_counts.get(key, len(group))
        for metric in metrics:
            if metric not in group:
                continue
            numeric = pd.to_numeric(group[metric], errors="coerce")
            values = numeric[np.isfinite(numeric)].to_numpy(float)
            n = len(values)
            rows.append(dict(label, metric=metric, n_available=n, n_planned=int(planned_n),
                             mean=float(np.mean(values)) if n else np.nan,
                             sample_sd=float(np.std(values, ddof=1)) if n > 1 else np.nan,
                             median=float(np.median(values)) if n else np.nan,
                             minimum=float(np.min(values)) if n else np.nan,
                             maximum=float(np.max(values)) if n else np.nan,
                             summary_status="complete_descriptive" if n == planned_n else "partial_descriptive"))
    return pd.DataFrame(rows, columns=columns)


def paired_neural_differences(individual, value="initial_relative_l2"):
    """One difference per declared joint repetition; retain unavailable pairs."""
    columns = GROUP_KEYS + ["repetition", "noise_seed", "training_seed", "method_a",
                           "method_b", "metric", "value_a", "value_b", "difference_a_minus_b",
                           "pair_status"]
    if individual.empty:
        return pd.DataFrame(columns=columns)
    result = []
    order = [("fourier_fixed", "plain"), ("fourier_adaptive", "plain"),
             ("fourier_adaptive", "fourier_fixed")]
    for labels, group in individual.groupby(GROUP_KEYS, dropna=False, sort=True):
        label = dict(zip(GROUP_KEYS, labels))
        methods = set(group.method)
        for method_a, method_b in order:
            if not {method_a, method_b}.issubset(methods):
                continue
            for repetition, pair in group.groupby("repetition", sort=True):
                a = pair[pair.method.eq(method_a)]
                b = pair[pair.method.eq(method_b)]
                if len(a) != 1 or len(b) != 1:
                    raise ValueError("The compact plan must have one row per method and repetition")
                av = pd.to_numeric(pd.Series([a.iloc[0].get(value, np.nan)]), errors="coerce").iloc[0]
                bv = pd.to_numeric(pd.Series([b.iloc[0].get(value, np.nan)]), errors="coerce").iloc[0]
                complete = (a.iloc[0]["status"] == b.iloc[0]["status"] == "completed"
                            and np.isfinite(av) and np.isfinite(bv))
                result.append(dict(label, repetition=int(repetition),
                                   noise_seed=int(a.iloc[0].noise_seed), training_seed=int(a.iloc[0].training_seed),
                                   method_a=method_a, method_b=method_b, metric=value,
                                   value_a=float(av), value_b=float(bv),
                                   difference_a_minus_b=float(av-bv) if complete else np.nan,
                                   pair_status="available" if complete else "incomplete"))
    return pd.DataFrame(result, columns=columns)


def classical_comparisons(individual, classical):
    """Join deterministic baselines once by data identity; do not inflate their N."""
    columns = PLAN_KEYS + ["repetition", "dataset_id", "baseline_method",
                           "neural_initial_relative_l2", "baseline_initial_relative_l2",
                           "difference_neural_minus_classical", "comparison_status",
                           "baseline_statistical_unit"]
    if individual.empty or classical.empty or "dataset_id" not in individual:
        return pd.DataFrame(columns=columns)
    rows = []
    for _, model in individual.iterrows():
        matches = classical[classical.dataset_id.eq(model.get("dataset_id"))]
        for _, baseline in matches.iterrows():
            a = model.get("initial_relative_l2", np.nan)
            b = baseline.get("initial_relative_l2", np.nan)
            complete = model["status"] == "completed" and np.isfinite(a) and np.isfinite(b)
            rows.append({**{key: model[key] for key in PLAN_KEYS},
                         "repetition": model["repetition"], "dataset_id": model.get("dataset_id"),
                         "baseline_method": baseline.method,
                         "neural_initial_relative_l2": a, "baseline_initial_relative_l2": b,
                         "difference_neural_minus_classical": a-b if complete else np.nan,
                         "comparison_status": "available" if complete else "incomplete",
                         "baseline_statistical_unit": "one deterministic fit per dataset and method; clean fit reused across training seeds"})
    return pd.DataFrame(rows, columns=columns)


def brass_cohort():
    """Inspect exactly the nine nominal jobs without loading TensorFlow."""
    from . import brass
    from . import brass_completion
    problem, observations, _, _ = brass.load_brass(DATA / "brass")
    jobs = brass.retraining_plan(problem, training_seeds=(2026, 2027, 2028),
                                 methods=NEURAL_METHODS, include_sensitivity=False,
                                 profile="production")
    inventory = brass_completion.job_inventory(jobs, observations, RESULTS / "brass/production")
    inventory["case"] = "brass"
    inventory["repetition"] = inventory.training_seed.astype(int) - 2025
    metric_frames, replay_frames, evidence = [], [], []
    for _, row in inventory.iterrows():
        folder = Path(row.get("run_dir", ROOT / str(row.get("run_folder", ""))))
        if not folder.is_absolute():
            folder = ROOT / folder
        identity = dict(method=row.method, training_seed=int(row.training_seed),
                        repetition=int(row.repetition), scenario="nominal", run_folder=str(folder.relative_to(ROOT)))
        item = dict(identity, status=row.status,
                    withheld_metrics_present=(folder / "brass_metrics_by_time.csv").is_file(),
                    replay_metrics_present=(folder / "brass_independent_replay_metrics_by_time.csv").is_file(),
                    refinement_record_present=(folder / "brass_numerical_floor_study.json").is_file())
        evidence.append(item)
        if row.status != "completed":
            continue
        for filename, frames in [("brass_metrics_by_time.csv", metric_frames),
                                 ("brass_independent_replay_metrics_by_time.csv", replay_frames)]:
            file = folder / filename
            if file.is_file():
                frame = pd.read_csv(file)
                frames.append(frame.assign(**identity))
    metrics = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    replay = pd.concat(replay_frames, ignore_index=True) if replay_frames else pd.DataFrame()
    return inventory, metrics, replay, pd.DataFrame(evidence)


def all_training_statuses():
    """Keep failures and optional/out-of-cohort jobs visible in the audit."""
    rows = []
    parents = [RESULTS / "neural/production", RESULTS / "brass/production"]
    for parent in parents:
        for file in sorted(parent.rglob("status.json")) if parent.exists() else []:
            folder = file.parent
            try:
                status = read_json(file)
                saved = _read_json_if_present(folder / "config.json")
                cfg, problem = saved.get("config", {}), saved.get("problem", {})
                rows.append(dict(run_folder=str(folder.relative_to(ROOT)), case=problem.get("name"),
                                 method=cfg.get("method"), training_seed=cfg.get("training_seed"),
                                 **status))
            except (OSError, ValueError, TypeError) as exc:
                rows.append(dict(run_folder=str(folder.relative_to(ROOT)), status="unreadable_metadata",
                                 error=str(exc)))
    return pd.DataFrame(rows)


def validation_followups(inventory):
    """Keep subsequent replay checks separate from the uploaded evaluation files."""
    allowed = set(inventory.run_folder)
    rows = []
    parent = RESULTS / "reduced_validation/neural_replay"
    for path in sorted(parent.glob("*/*/summary.json")):
        record = read_json(path)
        if record.get("run_folder") in allowed:
            rows.append({**record, "evidence_file": str(path.relative_to(ROOT))})
    return pd.DataFrame(rows)


def nonlinear_baseline_records(inventory):
    """Read the current declared baseline configuration, preserving other variants."""
    from . import completion_checks
    baseline_inventory = completion_checks.nonlinear_baseline_inventory()
    allowed = set(baseline_inventory.loc[baseline_inventory.status.eq("completed"), "run_folder"])
    datasets = set(inventory.loc[inventory.case.eq("nonlinear1d"), "dataset_id"])
    rows, extra = [], []
    for path in sorted((RESULTS / "nonlinear/reduced").glob("*/*/candidate.json")):
        record = read_json(path)
        if record.get("run_folder") in allowed:
            rows.append({**record, "evidence_file": str(path.relative_to(ROOT))})
        elif record.get("dataset_id") in datasets:
            extra.append({**record, "evidence_file": str(path.relative_to(ROOT)),
                          "cohort_note": "preserved baseline variant outside the declared default configuration"})
    frame = pd.DataFrame(rows)
    if len(frame):
        frame["duplicate_variant_requires_review"] = frame.duplicated(["dataset_id", "method"], keep=False)
    return baseline_inventory, frame, pd.DataFrame(extra)


def brass_postprocessing_failures():
    """Retain the cumulative assessment/replay failures, even after a retry."""
    history = RESULTS / "brass/production/brass_postprocessing_failure_history.jsonl"
    latest = RESULTS / "brass/production/brass_postprocessing_failures.json"
    if history.is_file():
        rows = [json.loads(line) for line in history.read_text().splitlines() if line.strip()]
    elif latest.is_file():
        rows = read_json(latest)
    else:
        rows = []
    return pd.DataFrame(rows, columns=["run", "step", "error"])


def build_report(report_dir=None):
    """Export fresh reports only; uploaded run/data/reference files stay unchanged."""
    report_dir = Path(report_dir) if report_dir is not None else RESULTS / "reduced_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    expected = completion.compact_core_plan()
    inventory = completion.plan_inventory(expected)
    candidates, blocked = flow.collect_candidates()
    individual = _join_candidate_metrics(inventory, candidates)
    if candidates.empty:
        classical, extras = pd.DataFrame(), pd.DataFrame()
    else:
        classical = candidates[candidates.method.isin(["tikhonov", "tsvd"])].copy()
        if len(classical):
            classical = classical.merge(expected[DATA_KEYS].drop_duplicates(), on=DATA_KEYS, how="inner", validate="many_to_one")
            classical = classical[classical.kmax.eq(12)].copy()
        retained = set(individual.run_folder) | (set(classical.run_folder) if len(classical) else set())
        extras = candidates[~candidates.run_folder.isin(retained)].copy()
        if len(extras):
            extras["cohort_note"] = "preserved outside the reduced primary cohort; not additional paired repetitions"
    synthetic_summary = descriptive_summary(individual, GROUP_KEYS + ["method"], SYNTHETIC_METRICS)
    classical_summary = descriptive_summary(classical, GROUP_KEYS + ["method"],
                                            ["initial_relative_l2", "independent_observation_residual", "wall_seconds"])
    pair_rows = paired_neural_differences(individual)
    pair_summary = descriptive_summary(pair_rows, GROUP_KEYS + ["method_a", "method_b"], ["difference_a_minus_b"])
    baseline_pairs = classical_comparisons(individual, classical)
    nonlinear_coverage, nonlinear_baselines, nonlinear_variants = nonlinear_baseline_records(inventory)
    replay_followups = validation_followups(inventory)
    brass_inventory, brass_metrics, brass_replay, brass_evidence = brass_cohort()
    brass_initial = (brass_metrics[np.isclose(brass_metrics.time_s, 0)].copy()
                     if len(brass_metrics) and "time_s" in brass_metrics else pd.DataFrame())
    # Build all nine planned rows before summarizing, so missing jobs are not
    # mistaken for a smaller intended sample size.
    brass_individual = brass_inventory.copy()
    if len(brass_initial):
        metrics_columns = [c for c in brass_initial if c not in brass_individual or c == "run_folder"]
        brass_individual = brass_individual.merge(brass_initial[metrics_columns], on="run_folder", how="left", validate="one_to_one")
    brass_summary = descriptive_summary(brass_individual, ["method"],
                                       ["RMSE_K", "MAE_K", "bias_K", "maxabs_K", "shifted_relative_L2"])
    statuses = all_training_statuses()
    if len(statuses):
        retained_folders = set(inventory.run_folder) | set(brass_inventory.run_folder)
        statuses["in_declared_cohort"] = statuses.run_folder.isin(retained_folders)
    files = dict(synthetic_expected_plan=expected, synthetic_coverage=inventory,
                 synthetic_individual_results=individual, synthetic_descriptive_summary=synthetic_summary,
                 paired_neural_differences=pair_rows, paired_difference_summary=pair_summary,
                 classical_unique_fits=classical, classical_descriptive_summary=classical_summary,
                 neural_classical_paired_differences=baseline_pairs,
                 nonlinear_baseline_candidates=nonlinear_baselines,
                 nonlinear_baseline_coverage=nonlinear_coverage,
                 nonlinear_baseline_variants_preserved=nonlinear_variants,
                 independent_replay_refinement=replay_followups,
                 preserved_outside_primary_cohort=extras, blocked_candidate_records=blocked,
                 brass_coverage=brass_inventory, brass_physical_units=brass_metrics,
                 brass_independent_replay=brass_replay, brass_evidence_coverage=brass_evidence,
                 brass_postprocessing_failure_history=brass_postprocessing_failures(),
                 brass_initial_individual_results=brass_individual, brass_initial_descriptive_summary=brass_summary,
                 all_production_training_statuses=statuses)
    for name, frame in files.items():
        frame.to_csv(report_dir / f"{name}.csv", index=False)
    summary = {
        "design": "compact_paired_v1", "synthetic_planned": len(expected),
        "synthetic_completed": int(inventory.status.eq("completed").sum()),
        "synthetic_missing_or_resumable": int(inventory.status.isin(["missing", "resumable"]).sum()),
        "synthetic_blocked": int(inventory.status.eq("blocked").sum()),
        "brass_planned": len(brass_inventory),
        "brass_completed": int(brass_inventory.status.eq("completed").sum()),
        "brass_missing_or_resumable": int(brass_inventory.status.isin(["missing", "resumable"]).sum()),
        "brass_blocked": int(brass_inventory.status.eq("blocked").sum()),
        "retained_classical_unique_fits": len(classical),
        "nonlinear_baseline_candidate_records": len(nonlinear_baselines),
        "independent_replay_refinement_records": len(replay_followups),
        "out_of_cohort_candidates_preserved": len(extras),
        "replications_per_condition": 3,
        "noisy_design": "Three joint noise/training pairs; combined variability, no variance-component separation.",
        "brass_experimental_runs": 1,
        "analysis_scope": "Individual outcomes and descriptive summaries; no crossed bootstrap, significance test, or precision claim.",
        "optional_dimensions": "Four additional feasibility jobs excluded from the 39-job primary budget.",
        "publication_eligible": False,
    }
    write_json(report_dir / "completion_summary.json", summary)
    return dict(files=files, summary=summary, report_dir=report_dir)
