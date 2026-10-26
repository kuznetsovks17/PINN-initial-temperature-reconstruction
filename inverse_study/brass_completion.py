"""Additive, resumable brass-study driver; original training/numerical code is unchanged.

Nominal repeats concern training on ONE experimental dataset. This module never
chooses candidates using withheld errors and never supplies heldout values to
training. Completion and cached assessment do not imply publication approval.
"""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd

from .common import ROOT, array_hash, metadata_hash, read_json, sha256_file, write_json
from . import brass
from . import pinn
from .completion import inspect_training_job


def _relative(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _write_if_changed(path, value):
    path = Path(path)
    if not path.exists() or read_json(path) != value:
        write_json(path, value)


def job_inventory(jobs, observations, output_dir):
    """One row per prospective job, with the exact unchanged engine identity."""
    rows = []
    for i, job in enumerate(jobs):
        checked = inspect_training_job(job['problem'], observations, job['config'], output_dir)
        rows.append(dict(job_index=i, scenario=job['scenario'], method=job['config']['method'],
                         training_seed=int(job['config']['training_seed']),
                         profile=job['config']['profile'], epochs=int(job['config']['epochs']),
                         **checked))
    return pd.DataFrame(rows)


def record_training_identity(run_dir, observations, scenario=None):
    run = Path(run_dir)
    saved = read_json(run / 'config.json')
    cfg = saved['config']
    with np.load(run / 'observations.npz', allow_pickle=False) as z:
        actual_hash = array_hash(z['X_obs'], z['y_obs'])
    expected_hash = array_hash(observations['X_obs'], observations['y_obs'])
    if actual_hash != expected_hash:
        raise ValueError('Saved model did not use the frozen brass observations.')
    path = run / 'study_record.json'
    record = read_json(path) if path.exists() else {}
    scenario = scenario or record.get('scenario', 'nominal' if saved['problem'].get('name') == 'brass' else saved['problem'].get('name', 'unlabelled'))
    record.update(case='brass', profile=cfg['profile'], method=cfg['method'],
                  training_seed=int(cfg['training_seed']), protocol='brass-terminal48-interior-v1',
                  observationhash=actual_hash, observation_sha256=actual_hash,
                  scenario=scenario, dataset_id=observations['dataset_id'],
                  n_terminal_interior=48, n_experimental_runs=1, noise_norm_known=False,
                  public_result=False, run_dir=_relative(run))
    record.setdefault('evidence_status', 'training_complete_pending_assessment')
    _write_if_changed(path, record)
    return record


def train_missing(jobs, observations, output_dir, max_new_jobs=None):
    """Skip completed jobs first; a limit counts only missing/resumable jobs.

    Interruptions propagate so the notebook can be stopped safely. Ordinary
    failures are retained by the trainer and shown in the returned table.
    """
    inventory = job_inventory(jobs, observations, output_dir)
    if max_new_jobs is not None and (isinstance(max_new_jobs, bool) or int(max_new_jobs) != max_new_jobs or max_new_jobs < 0):
        raise ValueError('max_new_jobs must be a nonnegative integer or None')
    blocked = inventory[inventory.status == 'blocked']
    if len(blocked):
        raise RuntimeError('Brass jobs have identity/artifact conflicts; review before training:\n' +
                           blocked[['method', 'training_seed', 'reason']].to_string(index=False))
    pending = inventory[inventory.status.isin(['missing', 'resumable'])]
    if max_new_jobs is not None:
        pending = pending.head(int(max_new_jobs))
    print(f"Brass: {int((inventory.status == 'completed').sum())} reused; "
          f"{len(pending)} pending jobs selected of {int(inventory.status.isin(['missing', 'resumable']).sum())}.")
    if len(pending) and not pinn.tensorflow_available():
        raise RuntimeError('Install requirements-training.txt before running pending brass training.')
    outcomes = []
    for row in pending.itertuples(index=False):
        job = jobs[int(row.job_index)]
        try:
            result = pinn.train_inverse(job['problem'], observations, job['config'], output_dir)
            run = Path(result['run_dir'])
            record_training_identity(run, observations, job['scenario'])
            outcomes.append(dict(job_index=row.job_index, status=result['status'], run_dir=str(run), reason=''))
        except Exception as exc:
            outcomes.append(dict(job_index=row.job_index, status='failed', run_dir=row.run_dir,
                                 reason=f'{type(exc).__name__}: {exc}'))
            print(f"FAILED {job['config']['run_id']}: {exc}")
    return job_inventory(jobs, observations, output_dir), pd.DataFrame(outcomes)


def freeze_candidate_portable(run_dir):
    """Freeze config/weights before assessment, allowing harmless folder relocation.

    An existing historical freeze is checked but never changed. New freezes use
    a run-folder name rather than an absolute machine-specific path.
    """
    run = Path(run_dir)
    status = read_json(run / 'status.json')
    if status.get('status') != 'completed':
        raise ValueError(f'Cannot freeze incomplete brass run: {run.name}')
    files = {name: sha256_file(run / name) for name in ['config.json', 'weights_final.npz']}
    legacy = run / 'brass_candidate_frozen.json'
    if legacy.exists():
        previous = read_json(legacy)
        if previous.get('checkpoint') != 'final' or previous.get('files') != files:
            raise RuntimeError('Historical brass candidate config/weights changed; do not overwrite its freeze.')
    record = dict(schema='portable-brass-candidate-v1', run_id=run.name,
                  checkpoint='final', files=files,
                  declared_selection_rule='final fixed-budget checkpoint; no heldout-error selection',
                  evidence_status='candidate_frozen_for_withheld_evaluation_not_publication_approval')
    path = run / 'brass_candidate_frozen_v2.json'
    if path.exists() and read_json(path) != record:
        raise RuntimeError('Frozen brass candidate differs; preserve it and review the conflict.')
    _write_if_changed(path, record)
    return record


def _assessment_identity(run, observations, heldout, settings, include_forward=False):
    run = Path(run)
    frozen = freeze_candidate_portable(run)
    sources = {'brass.py': sha256_file(Path(brass.__file__)),
               'brass_completion.py': sha256_file(Path(__file__))}
    if include_forward:
        from . import numerics
        sources['numerics.py'] = sha256_file(Path(numerics.__file__))
    else:
        sources['pinn.py'] = sha256_file(Path(pinn.__file__))
    return dict(candidate=frozen, sources=sources,
                observation_sha256=array_hash(observations['X_obs'], observations['y_obs']),
                heldout_sha256=hashlib.sha256(heldout.to_csv(index=False, float_format='%.17g').encode()).hexdigest(),
                settings=settings)


def _cache_valid(folder, cache_name, identity, names):
    folder = Path(folder)
    path = folder / cache_name
    if not path.is_file() or any(not (folder / name).is_file() for name in names):
        return False
    meta = read_json(path)
    return (meta.get('identity_sha256') == metadata_hash(identity) and
            meta.get('outputs_sha256') == {name: sha256_file(folder / name) for name in names})


def _save_cache(folder, cache_name, identity, names):
    folder = Path(folder)
    write_json(folder / cache_name, dict(identity_sha256=metadata_hash(identity), identity=identity,
               outputs_sha256={name: sha256_file(folder / name) for name in names},
               status='completed'))


def assessment_cached(run_dir, observations, heldout):
    """Return (metrics, points, cache_status), preserving all candidate results."""
    run = Path(run_dir)
    identity = _assessment_identity(run, observations, heldout, {'checkpoint': 'final'})
    names = ['brass_metrics_by_time.csv', 'brass_withheld_predictions_K.csv']
    if _cache_valid(run, 'brass_assessment_cache.json', identity, names):
        return pd.read_csv(run / names[0]), pd.read_csv(run / names[1]), 'cached'
    metrics, points = brass.evaluate_saved_pinn(run, heldout, checkpoint='final')
    metrics.to_csv(run / names[0], index=False)
    points.to_csv(run / names[1], index=False)
    record = record_training_identity(run, observations)
    initial = metrics.loc[np.isclose(metrics.time_s, 0)].iloc[0]
    record.update(evidence_status='withheld_assessment_complete_pending_forward_and_provenance_checks',
                  initial_RMSE_K=float(initial['RMSE_K']), initial_MAE_K=float(initial['MAE_K']),
                  initial_bias_K=float(initial['bias_K']), initial_maxabs_K=float(initial['maxabs_K']),
                  initial_relative_l2=float(initial['shifted_relative_L2']))
    _write_if_changed(run / 'study_record.json', record)
    _save_cache(run, 'brass_assessment_cache.json', identity, names)
    return metrics, points, 'computed'


def replay_cached(run_dir, observations, heldout, nx=401, rtol=1e-8, atol=1e-10,
                  max_step=.02, output_dir=None):
    """Independent BDF replay with cached numeric fields and physical metrics."""
    run = Path(run_dir)
    out = run if output_dir is None else Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    settings = dict(checkpoint='final', nx=int(nx), rtol=float(rtol), atol=float(atol), max_step=float(max_step))
    identity = _assessment_identity(run, observations, heldout, settings, include_forward=True)
    names = ['brass_independent_replay_metrics_by_time.csv', 'brass_independent_replay_predictions_K.csv',
             'brass_independent_terminal_fit.json', 'brass_independent_forward_fields.npz',
             'brass_replay_details.json']
    if _cache_valid(out, 'brass_replay_cache.json', identity, names):
        with np.load(out / names[3], allow_pickle=False) as z:
            fields = {key: z[key].copy() for key in z.files}
        details = read_json(out / names[4])
        return dict(nx=int(nx), heldout_metrics=pd.read_csv(out / names[0]),
                    heldout_points=pd.read_csv(out / names[1]), terminal_metrics=read_json(out / names[2]),
                    terminal_prediction=fields['terminal_prediction'],
                    initial_boundary_correction_K=np.asarray(details['initial_boundary_correction_K']),
                    forward=dict(x=fields['x'], t=fields['t'], u=fields['theta'],
                                 wall_seconds=details['wall_seconds']), cache_status='cached')
    saved = read_json(run / 'config.json')
    if saved['problem'].get('name') != 'brass':
        raise ValueError('The nominal replay study requires a nominal brass candidate.')
    predictor = lambda X: pinn.predict_saved(run, X, times=0.0, checkpoint='final')
    result = brass.replay_initial(saved['problem'], predictor, observations, heldout,
                                  nx=int(nx), rtol=rtol, atol=atol, max_step=max_step)
    result['heldout_metrics'].to_csv(out / names[0], index=False)
    result['heldout_points'].to_csv(out / names[1], index=False)
    write_json(out / names[2], result['terminal_metrics'])
    np.savez_compressed(out / names[3], x=result['forward']['x'], t=result['forward']['t'],
                        theta=result['forward']['u'], terminal_prediction=result['terminal_prediction'])
    write_json(out / names[4], dict(initial_boundary_correction_K=result['initial_boundary_correction_K'],
                                   wall_seconds=result['forward']['wall_seconds']))
    _save_cache(out, 'brass_replay_cache.json', identity, names)
    result['cache_status'] = 'computed'
    return result


def refinement_cached(run_dir, observations, heldout, meshes=(101, 201, 401),
                      rtol=1e-8, atol=1e-10, max_step=.02, guard_factor=2.):
    """Refine every candidate; reuse the finest primary replay and cached checks."""
    run = Path(run_dir)
    meshes = tuple(int(n) for n in meshes)
    if len(meshes) < 2 or any(n < 3 for n in meshes) or any(b <= a for a, b in zip(meshes, meshes[1:])):
        raise ValueError('Provide at least two strictly increasing meshes')
    if not np.isfinite(guard_factor) or guard_factor <= 0:
        raise ValueError('guard_factor must be positive and finite')
    settings = dict(meshes=meshes, rtol=float(rtol), atol=float(atol), max_step=float(max_step),
                    guard_factor=float(guard_factor))
    identity = _assessment_identity(run, observations, heldout, settings, include_forward=True)
    names = ['brass_replay_refinement.csv', 'brass_numerical_floor_study.json']
    if _cache_valid(run, 'brass_refinement_cache.json', identity, names):
        return pd.read_csv(run / names[0]), read_json(run / names[1]), 'cached'
    replays = []
    for nx in meshes:
        settings_key = metadata_hash(dict(nx=nx, rtol=rtol, atol=atol, max_step=max_step))[:16]
        out = None if nx == meshes[-1] else run / 'replay_checks' / settings_key
        replays.append(replay_cached(run, observations, heldout, nx, rtol, atol, max_step, output_dir=out))
    finest = replays[-1]
    rows = []
    for replay in replays:
        diff = replay['terminal_prediction'] - finest['terminal_prediction']
        points_diff = brass.SCALE_K * (replay['heldout_points'].prediction_theta.to_numpy() -
                                       finest['heldout_points'].prediction_theta.to_numpy())
        rows.append(dict(nx=replay['nx'], rtol=rtol, atol=atol, max_step=max_step,
                         terminal_fit_RMSE_K=replay['terminal_metrics']['RMSE_K'],
                         terminal_l2_difference_to_finest_theta=float(np.linalg.norm(diff)),
                         terminal_RMSE_difference_to_finest_K=float(brass.SCALE_K*np.sqrt(np.mean(diff**2))),
                         heldout_max_difference_to_finest_K=float(np.max(np.abs(points_diff))),
                         boundary_correction_maxabs_K=float(np.max(np.abs(replay['initial_boundary_correction_K']))),
                         wall_seconds=replay['forward']['wall_seconds'],
                         comparison='difference_to_finest_numerical_grid_not_exact_error'))
    table = pd.DataFrame(rows)
    tighter_settings = dict(nx=meshes[-1], rtol=rtol/10, atol=atol/10, max_step=max_step/2)
    tighter = replay_cached(run, observations, heldout, **tighter_settings,
                           output_dir=run / 'replay_checks' / metadata_hash(tighter_settings)[:16])
    spatial_change = float(table.iloc[-2].terminal_l2_difference_to_finest_theta)
    time_change = float(np.linalg.norm(tighter['terminal_prediction'] - finest['terminal_prediction']))
    proposed = float(guard_factor * (spatial_change + time_change))
    record = dict(spatial_terminal_l2_increment_theta=spatial_change,
                  time_tolerance_terminal_l2_increment_theta=time_change, guard_factor=float(guard_factor),
                  proposed_numerical_noise_floor=proposed if np.isfinite(proposed) and proposed > 0 else None,
                  meaning='observed numerical sensitivity, not a certified bound or measurement uncertainty',
                  candidate=_relative(run), case='brass', profile=read_json(run / 'config.json')['config']['profile'],
                  observation_sha256=array_hash(observations['X_obs'], observations['y_obs']),
                  accepted_by_notebook_switch=False)
    table.to_csv(run / names[0], index=False)
    write_json(run / names[1], record)
    _save_cache(run, 'brass_refinement_cache.json', identity, names)
    return table, record, 'computed'
