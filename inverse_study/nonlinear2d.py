"""Independent conservative reference, observation freezing, and audits for a
strongly temperature-dependent two-dimensional inverse heat experiment.

This module does not alter the previously completed smooth-2D, nonlinear-1D,
or brass runs. It deliberately avoids importing the neural engine.

PDE: theta_t = Fo * div(a(theta) grad theta) on (0,1)^2,
     theta = 0 on all four boundaries, no source,
     a(theta) = a_lo + (a_hi - a_lo)/(1 + exp[-k*(theta-theta_mid)]).

A symmetric face-flux finite difference/finite volume operator and a converged
backward Euler Picard solve are used for references and independent replays.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve
from scipy.special import expit

SPEC = dict(
    name="strong_nonlinear2d", dim=2, law="sigmoid20_2d", bounds=[[0.0, 1.0], [0.0, 1.0]],
    Fo=0.002, T=1.0, boundary_type="dirichlet0", dirichlet_value=0.0,
    a_lo=1.0, a_hi=20.0, beta=12.0, theta_mid=0.5,
    initial_profile="two_interior_hotspots_v1", initial_normalization=1.0,
    source=0.0,
)


def _initial_raw(x, y):
    s = np.sin(np.pi*x)*np.sin(np.pi*y)
    p1 = np.exp(-((x-0.32)/0.115)**2 - ((y-0.43)/0.145)**2)
    p2 = np.exp(-((x-0.71)/0.105)**2 - ((y-0.68)/0.115)**2)
    return s * (0.20 + 1.05*p1 + 0.78*p2)

# Grid-independent constant chosen once from a fixed, dense grid; the profile
# is NOT recomputed or rescaled separately on different numerical meshes.
_MAX_GRID = np.linspace(0.0, 1.0, 1001)
_XX, _YY = np.meshgrid(_MAX_GRID, _MAX_GRID, indexing="ij")
INITIAL_NORMALIZER = float(np.max(_initial_raw(_XX, _YY)))
del _XX, _YY
SPEC["initial_normalization"] = INITIAL_NORMALIZER


def problem_spec():
    return dict(SPEC)


def initial(x, y):
    return _initial_raw(np.asarray(x), np.asarray(y)) / INITIAL_NORMALIZER


def coefficient(u, spec=SPEC):
    """Bounded, strictly positive, strongly temperature-sensitive coefficient."""
    return float(spec["a_lo"]) + (float(spec["a_hi"]) - float(spec["a_lo"])) * expit(
        float(spec["beta"]) * (np.asarray(u) - float(spec["theta_mid"])))


def grid(n):
    n=int(n)
    if n < 7: raise ValueError("Grid has at least 7 nodes per axis")
    v=np.linspace(0., 1., n)
    xx,yy=np.meshgrid(v,v,indexing="ij")
    return v,xx,yy


def _matrix(current, dt, spec):
    """Positive definite frozen-face implicit matrix I+dt*Fo*(-div a grad).

    Neighbor faces share exactly one coefficient; Dirichlet boundary faces have
    their own half-state coefficients. The matrix is symmetric M-matrix for dt>0.
    """
    u=np.asarray(current, dtype=float)
    n=u.shape[0]
    if u.shape != (n,n): raise ValueError("square interior field required")
    h=1.0/(n+1)
    scale= float(dt)*float(spec["Fo"])/h**2
    d=np.zeros((n,n), dtype=float)
    # x-neighbor internal edges
    ax=coefficient(0.5*(u[:-1,:]+u[1:,:]),spec)*scale
    d[:-1,:]+=ax
    d[1:,:]+=ax
    # y-neighbor internal edges
    ay=coefficient(0.5*(u[:,:-1]+u[:,1:]),spec)*scale
    d[:,:-1]+=ay
    d[:,1:]+=ay
    # exact zero Dirichlet boundaries
    d[0,:] += scale*coefficient(0.5*u[0,:], spec)
    d[-1,:] += scale*coefficient(0.5*u[-1,:], spec)
    d[:,0] += scale*coefficient(0.5*u[:,0], spec)
    d[:,-1] += scale*coefficient(0.5*u[:,-1], spec)
    # boundary diagonals occur once per incident boundary face, including corners
    sz=n*n
    off_x=-ax.ravel()
    off_y=np.zeros(sz-1,dtype=float)
    index=(np.arange(n)[:,None]*n+np.arange(n-1)[None,:]).ravel()
    off_y[index]=-ay.ravel()
    return diags([1.0+d.ravel(),off_y,off_y,off_x,off_x], [0,1,-1,n,-n],shape=(sz,sz),format="csc")


def _single_step(previous, dt, spec, rtol=2e-9, atol=1e-11, max_picard=90):
    u = np.asarray(previous, dtype=float)
    trial=u.copy()
    for k in range(1,max_picard+1):
        candidate=spsolve(_matrix(trial,dt,spec),u.ravel()).reshape(u.shape)
        if not np.all(np.isfinite(candidate)):
            raise FloatingPointError("Nonfinite nonlinear finite-volume field")
        if np.max(np.abs(candidate-trial)) <= atol+rtol*np.max(np.abs(candidate)):
            return candidate,k
        trial=candidate
    raise RuntimeError(f"Picard iteration failed after {max_picard} steps at dt={dt}, max-change={np.max(np.abs(candidate-trial)):.3e}; reduce dt")


def forward(n=65,dt=0.01, spec=None, initial_field=None, times=(0.0,0.5,1.0),
            save_path=None, verbose=False):
    """Converged Picard backward-Euler finite-volume solution on full n*n grid.

    Requested times must be multiples of dt (up to roundoff), to avoid
    interpolating an inaccurate intermediate timestep. Prefer dt=1/100 etc.
    """
    spec = dict(SPEC if spec is None else spec)
    n = int(n); v,xx,yy=grid(n)
    T=float(spec["T"])
    nsteps=round(T/float(dt))
    if nsteps < 1 or abs(nsteps*dt-T)>1e-12: raise ValueError("dt must divide T exactly")
    times = tuple(float(t) for t in times)
    indices = [round(t/float(dt)) for t in times]
    if any(abs(ii*dt-t)>1e-11 or not 0<=ii<=nsteps for ii,t in zip(indices,times)):
        raise ValueError("snapshot times must lie on timesteps; choose compatible dt")
    if initial_field is None:
        u=initial(xx,yy)
    else:
        u=np.asarray(initial_field,dtype=float).copy()
        if u.shape!=(n,n): raise ValueError("initial_field shape must match grid")
    if not np.allclose(u[[0,-1],:],0,atol=1e-12) or not np.allclose(u[:,[0,-1]],0,atol=1e-12):
        raise ValueError("Initial field must satisfy zero Dirichlet boundary values")
    records={0:u.copy()}
    maxiter=0
    tic=perf_counter()
    for step in range(1,nsteps+1):
        uinner,its=_single_step(u[1:-1,1:-1],dt,spec)
        u[1:-1,1:-1]=uinner
        maxiter=max(its,maxiter)
        if step in indices: records[step]=u.copy()
        if verbose and (step%max(1,nsteps//5)==0): print(f"{step}/{nsteps}: max(theta)={u.max():.5f}, Picard={its}")
    snapshots=np.stack([records[ii] for ii in indices])
    out=dict(n=n,dt=float(dt),times=np.asarray(times),axis=v,initial=snapshots[0],
             snapshots=snapshots,terminal=snapshots[times.index(T)] if T in times else u.copy(),
             max_picard_iterations=maxiter,wall_seconds=perf_counter()-tic,
             diffusivity_range_initial=np.array([coefficient(records[0],spec).min(),coefficient(records[0],spec).max()]),
             diffusivity_range_terminal=np.array([coefficient(u,spec).min(),coefficient(u,spec).max()]),
             spec=spec)
    if save_path is not None:
        save_path=Path(save_path); save_path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(save_path,**{k:v for k,v in out.items() if k!="spec"})
        (save_path.with_suffix(".json")).write_text(json.dumps({"spec":spec,"n":n,"dt":dt,"times":times,"max_picard_iterations":maxiter,"wall_seconds":out["wall_seconds"]},indent=2))
    return out


def interpolate(result, X, t=None):
    X=np.asarray(X,dtype=float)
    if X.ndim!=2 or X.shape[1]!=2: raise ValueError("X must be (N,2)")
    arr=np.asarray(result["terminal"] if t is None else result["snapshots"][np.argmin(np.abs(result["times"]-t))])
    return RegularGridInterpolator((result["axis"],result["axis"]), arr, bounds_error=True)(X)


def create_observations(reference, noise=0.05, seed=3100, n_obs_grid=33, root="results/nonlinear2d"):
    """Fixed paired interior sensor observations; true initial is stored only in reference."""
    spec=reference["spec"]
    v,xx,yy=grid(n_obs_grid)
    X=np.column_stack([xx[1:-1,1:-1].ravel(),yy[1:-1,1:-1].ravel()])
    clean=interpolate(reference,X)
    sigma=float(noise)*float(np.sqrt(np.mean(clean**2)))
    eps=np.random.default_rng(int(seed)).normal(0,sigma,size=clean.shape)
    y=clean+eps
    tag=f"noise{noise:g}_seed{seed}"
    folder=Path(root)/"observations"/tag
    folder.mkdir(parents=True, exist_ok=True)
    obs=dict(X_obs=X,y_obs=y,noise_norm=float(np.linalg.norm(eps)),noise_seed=int(seed))
    chk=hashlib.sha256(np.column_stack([X,y]).tobytes()).hexdigest()
    file=folder/"observations.npz"
    if file.exists():
        with np.load(file,allow_pickle=False) as old:
            old_hash=hashlib.sha256(np.column_stack([old["X_obs"],old["y_obs"]]).tobytes()).hexdigest()
        if chk!=old_hash: raise RuntimeError(f"Existing frozen observation changed: {file}")
    else:
        np.savez_compressed(file,**obs)
    meta={"experiment":spec,"sigma":sigma,"noise_level":float(noise),"noise_seed":int(seed),
          "reference_sha256":hashlib.sha256(np.ascontiguousarray(reference["terminal"]).tobytes()).hexdigest(),
          "noise_norm":obs["noise_norm"],"observation_sha256":chk,
          "reference_n":int(reference["n"]),"reference_dt":float(reference["dt"]),
          "sensor_count":len(X),"boundary_policy":"exact homogeneous Dirichlet", "truth_in_training":False}
    metafile=folder/"metadata.json"
    if metafile.exists() and json.loads(metafile.read_text()) != meta:
        raise RuntimeError(f"Frozen metadata differs from current reference: {metafile}")
    if not metafile.exists(): metafile.write_text(json.dumps(meta,indent=2))
    return folder,obs


def relative_l2(u,v):
    return float(np.linalg.norm(np.asarray(u)-np.asarray(v))/np.linalg.norm(v))


def compare_reference(ref_a,ref_b,n_evaluation=65):
    """Compare snapshots of two distinct-grid solutions at the same coordinates."""
    v,xx,yy=grid(n_evaluation)
    X=np.column_stack([xx.ravel(),yy.ravel()])
    a=interpolate(ref_a,X).reshape(n_evaluation,n_evaluation)
    b=interpolate(ref_b,X).reshape(n_evaluation,n_evaluation)
    return dict(terminal_relative_l2=relative_l2(a,b),terminal_absolute_max=float(np.max(np.abs(a-b))))


def load_reference(path):
    """Load a prior reference and verify its recorded PDE/initial condition spec."""
    path=Path(path)
    if not path.is_file() or not path.with_suffix('.json').exists():
        raise FileNotFoundError(f'Reference and metadata expected at {path}')
    metadata=json.loads(path.with_suffix('.json').read_text())
    if metadata['spec'] != SPEC:
        raise RuntimeError('Cached reference uses a different physical law or initial profile; do not mix datasets')
    with np.load(path,allow_pickle=False) as z:
        out={k:z[k].copy() for k in z.files}
    out.update(spec=metadata['spec'],n=int(out['n']),dt=float(out['dt']),
               max_picard_iterations=int(out['max_picard_iterations']),
               wall_seconds=float(out['wall_seconds']))
    if out['snapshots'].shape != (len(out['times']),out['n'],out['n']):
        raise RuntimeError('Corrupted cached reference shape')
    return out


def ensure_reference(path,n=97,dt=0.005):
    path=Path(path)
    if path.is_file():
        ref=load_reference(path)
        if int(ref['n'])!=int(n) or abs(ref['dt']-float(dt))>1e-14:
            raise RuntimeError('Reference resolution changed; refuse unsafe cache reuse')
        return ref, 'cached'
    ref=forward(n=n,dt=dt,save_path=path)
    return ref, 'computed'
