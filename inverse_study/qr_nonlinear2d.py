"""Nonlinear 2D mixed finite-element quasi-reversibility (QR).

Matches the manuscript mixed equations, for homogeneous Dirichlet boundaries:
    w_s + Fo div(a(w) grad w) + gamma Delta^2 w = 0,
    q = sqrt(gamma) Delta w.
Weak backward Euler with a(w_old) frozen, for phi in H_0^1, psi in H^1:
    (wnew,phi)-dt Fo(a(wold) grad(wnew),grad(phi))
        -dt sqrt(gamma)(grad(qnew),grad(phi)) = (wold,phi)
    (qnew,psi)+sqrt(gamma)(grad(wnew),grad(psi)) = 0.

Temperature w is essential Dirichlet (0) on the boundary; q is unconstrained.
That faithfully reproduces the paper's natural derivative closure on w. This is
NOT a Tikhonov initial-profile optimization or the alternative q=0 QR closure.

Structured conforming P1 triangles, consistent mass matrix, and element-average
lagged nonlinear coefficient. Synthetic terminal measurements are only loaded
from the frozen observation files, never from the known initial truth.

Post-processing initial errors is done only after gamma is selected using
independent forward replay discrepancies at the actual terminal sensors.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import bmat, coo_matrix, csc_matrix
from scipy.sparse.linalg import spsolve

from . import nonlinear2d as n2


class QRNumericalFailure(RuntimeError):
    """Candidate inversion is unstable or numerical linear algebra fails."""


@dataclass
class MixedFEM2D:
    nodes_per_dim: int = 33

    def __post_init__(self):
        n = int(self.nodes_per_dim)
        if n < 5:
            raise ValueError('At least five grid nodes required')
        self.n = n
        self.axis = np.linspace(0., 1., n)
        X,Y = np.meshgrid(self.axis,self.axis,indexing='ij')
        self.points = np.column_stack((X.ravel(),Y.ravel()))
        self.size = n*n
        ii, jj = np.meshgrid(np.arange(n-1),np.arange(n-1),indexing='ij')
        start = (ii*n+jj).ravel()
        triangles = []
        for k in start:
            triangles.append((k,k+n,k+1))
            triangles.append((k+n+1,k+1,k+n))
        self.tri = np.asarray(triangles,dtype=np.int64)
        p=self.points[self.tri]
        twice=(p[:,1,0]-p[:,0,0])*(p[:,2,1]-p[:,0,1])-(p[:,2,0]-p[:,0,0])*(p[:,1,1]-p[:,0,1])
        if np.any(twice<=0.):
            raise ValueError('Expected counterclockwise triangles')
        self.area=0.5*twice
        grads=np.empty((len(self.tri),3,2))
        for j in range(3):
            a,b=(j+1)%3,(j+2)%3
            grads[:,j,0]=(p[:,a,1]-p[:,b,1])/twice
            grads[:,j,1]=(p[:,b,0]-p[:,a,0])/twice
        self.k_local=np.einsum('tik,tjk,t->tij',grads,grads,self.area)
        self.irow=np.repeat(self.tri,3,axis=1).reshape(-1)
        self.icol=np.tile(self.tri,(1,3)).reshape(-1)
        ml=(np.ones((3,3))+np.eye(3))[None,:,:]*self.area[:,None,None]/12.
        self.mass=coo_matrix((ml.ravel(),(self.irow,self.icol)),shape=(self.size,self.size)).tocsc()
        self.k_lap=self.stiffness(np.ones(len(self.tri)))
        ids=np.arange(self.size).reshape((n,n))
        self.interior=ids[1:-1,1:-1].ravel()
        self.boundary=np.setdiff1d(np.arange(self.size),self.interior)
        self.mass_ii=self.mass[self.interior,:][:,self.interior].tocsc()
        self.k_i=self.k_lap[self.interior,:].tocsc()
        self.k_col=self.k_lap[:,self.interior].tocsc()
        self.mass_norm=float(np.linalg.norm(self.mass.diagonal()))

    def stiffness(self,coeffs):
        coeffs=np.asarray(coeffs,dtype=float)
        if coeffs.shape != (len(self.tri),):raise ValueError('One coefficient per triangle')
        return coo_matrix(((self.k_local*coeffs[:,None,None]).ravel(),
                           (self.irow,self.icol)),shape=(self.size,self.size)).tocsc()

    def nonlinear_stiffness(self,field,spec):
        field=np.asarray(field).reshape(self.size)
        triangle_avgs=np.mean(field[self.tri],axis=1)
        return self.stiffness(n2.coefficient(triangle_avgs,spec))

    def sample_sensor_values(self,obs):
        # Interpolate the *observed* 33x33 terminal grid, with known zero boundary;
        # do not interpolate the noiseless reference or the initial truth.
        X=np.asarray(obs['X_obs']);y=np.asarray(obs['y_obs'])
        s=int(round(np.sqrt(len(y))))
        if s*s!=len(y) or X.shape!=(s*s,2):raise ValueError('Expected square tensor interior observations')
        reference_axis=np.linspace(0.,1.,s+2)
        XX,YY=np.meshgrid(reference_axis[1:-1],reference_axis[1:-1],indexing='ij')
        if not np.allclose(X,np.column_stack((XX.ravel(),YY.ravel())),rtol=0.,atol=1e-13):
            raise ValueError('Sensor ordering differs from frozen grid')
        grid=np.zeros((s+2,s+2))
        grid[1:-1,1:-1]=y.reshape(s,s)
        if self.n==s+2:
            return grid.copy(),{'terminal_initialization':'direct nodal sensor values; zero Dirichlet boundary'}
        tmp=RegularGridInterpolator((reference_axis,reference_axis),grid,method='linear',bounds_error=True)(self.points).reshape(self.n,self.n)
        tmp[0,:]=tmp[-1,:]=0.
        tmp[:,0]=tmp[:,-1]=0.
        return tmp,{'terminal_initialization':'bilinear interpolation of noisy observations plus zero Dirichlet boundary'}

    def discrete_residual(self,new,old,q,spec,gamma,dt):
        old=np.asarray(old).ravel(); new=np.asarray(new).ravel();q=np.asarray(q).ravel()
        ka=self.nonlinear_stiffness(old,spec)
        ii=self.interior
        lhs1=self.mass_ii@(new[ii]-old[ii])-dt*spec['Fo']*(ka[ii,:]@new)-dt*np.sqrt(gamma)*(self.k_i@q)
        lhs2=self.mass@q+np.sqrt(gamma)*(self.k_lap@new)
        return float(np.linalg.norm(lhs1)),float(np.linalg.norm(lhs2))

    def run(self,terminal, spec, gamma,dt=0.02,field_cap=20.,report_interval=0):
        gamma=float(gamma);dt=float(dt)
        if gamma<=0 or dt<=0:raise ValueError('gamma/dt must be positive')
        steps=int(round(float(spec['T'])/dt))
        if abs(steps*dt-float(spec['T']))>1e-12:raise ValueError('dt must divide T')
        previous=np.asarray(terminal,dtype=float)
        if previous.shape!=(self.n,self.n):raise ValueError('terminal shape mismatch')
        if np.max(np.abs(previous[self.boundary//self.n,self.boundary%self.n]))>1e-12:
            raise ValueError('terminal boundary must be exactly zero')
        previous=previous.ravel().copy()
        iq=self.interior
        sq=np.sqrt(gamma)
        top_right=(-dt*sq*self.k_i).tocsc()
        bottom_left=(sq*self.k_col).tocsc()
        tic=perf_counter()
        snapshots={0:previous.reshape(self.n,self.n).copy()}
        max_first=max_second=0.;
        for step in range(1,steps+1):
            stiffness=self.nonlinear_stiffness(previous,spec)
            top_left=(self.mass_ii-dt*float(spec['Fo'])*stiffness[iq,:][:,iq]).tocsc()
            block=bmat([[top_left,top_right],[bottom_left,self.mass]],format='csc')
            rhs=np.r_[self.mass_ii@previous[iq],np.zeros(self.size)]
            try:
                solution=spsolve(block,rhs)
            except Exception as ex:
                raise QRNumericalFailure(f'mixed block solve failed at step {step}: {ex}') from ex
            if not np.isfinite(solution).all():
                raise QRNumericalFailure(f'nonfinite block solution at step {step}')
            curr=np.zeros(self.size,dtype=float)
            curr[iq]=solution[:len(iq)]
            q=solution[len(iq):]
            if np.max(np.abs(curr))>field_cap:
                raise QRNumericalFailure(f'field exceeded stated diagnostic cap {field_cap} at step {step}, amplitude={np.max(np.abs(curr)):.4g}; do not clip')
            if step in {1,steps}:
                res1,res2=self.discrete_residual(curr,previous,q,spec,gamma,dt)
                max_first=max(max_first,res1)
                max_second=max(max_second,res2)
            previous=curr
            if step==steps//2:snapshots[step]=curr.reshape(self.n,self.n).copy()
            if report_interval and step%report_interval==0:
                print(f'QR step={step}/{steps}, min/max={curr.min():.4g}/{curr.max():.4g}')
        snapshots[steps]=previous.reshape(self.n,self.n).copy()
        return dict(initial=previous.reshape(self.n,self.n),mid_reverse=snapshots.get(steps//2),
                    initial_terminal=snapshots[0],sqrt_gamma=sq,gamma=gamma,dt=dt,steps=steps,
                    nodes=self.n,max_temperature=float(previous.max()),min_temperature=float(previous.min()),
                    max_weak1_residual=max_first,max_weak2_residual=max_second,
                    inverse_wall_seconds=perf_counter()-tic)


def sensor_fit_from_replay(qr_field,model,obs,spec,replay_n=65,replay_dt=0.01):
    """Independent forward consistency; uses observation vectors, not true initial field."""
    qr_axis=model.axis
    replay_axis,xx,yy=n2.grid(replay_n)
    X=np.column_stack((xx.ravel(),yy.ravel()))
    projected=RegularGridInterpolator((qr_axis,qr_axis),np.asarray(qr_field),bounds_error=True)(X).reshape(replay_n,replay_n)
    projected[0,:]=0.;projected[-1,:]=0.;projected[:,0]=0.;projected[:,-1]=0.
    replay=n2.forward(n=replay_n,dt=replay_dt,spec=spec,initial_field=projected,times=(0.,float(spec['T'])))
    values=RegularGridInterpolator((replay['axis'],replay['axis']),replay['terminal'])(obs['X_obs'])
    y=obs['y_obs']
    return dict(replay=replay,sensor_discrepancy_absolute=float(np.linalg.norm(values-y)),
                sensor_discrepancy_relative=float(np.linalg.norm(values-y)/np.linalg.norm(y)),
                sensor_pred=values,projected_initial=projected)


def evaluate_with_truth_only_after_selection(model,candidate,reference):
    """Paper metrics, deliberately separate from regularization parameter selection."""
    axis=model.axis
    xx,yy=np.meshgrid(axis,axis,indexing='ij')
    truth=n2.initial(xx,yy)
    y=np.asarray(candidate['initial'])
    norm=float(np.linalg.norm(y-truth)/np.linalg.norm(truth))
    return {'initial_relative_l2':norm,'initial_max_abs_error':float(np.max(np.abs(y-truth)))}


def load_observation(folder):
    folder=Path(folder)
    with np.load(folder/'observations.npz',allow_pickle=False) as z:
        obs={k:z[k].copy() for k in z.files}
    meta=json.loads((folder/'metadata.json').read_text())
    obs_hash=hashlib.sha256(np.column_stack([obs['X_obs'],obs['y_obs']]).tobytes()).hexdigest()
    if obs_hash!=meta['observation_sha256']:
        raise ValueError('Frozen observations changed; refusing to run QR')
    return obs,meta


def _job_id(gamma,n,dt,obs_sha,replay_n,replay_dt):
    payload=json.dumps({'gamma':float(gamma),'n':int(n),'dt':float(dt),'observations':obs_sha,'replay_n':int(replay_n),'replay_dt':float(replay_dt),'impl':'mixed_P1_consistent_mass_lagged_a_v1'},sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:14]


def run_or_load_candidate(folder,observation,meta,spec,gamma,n=33,dt=0.02,
                          replay_n=33,replay_dt=0.02,redo_failed=False):
    """Cached QR candidate; saves status and actual arrays, never invented metrics."""
    folder=Path(folder)
    ident=_job_id(gamma,n,dt,meta['observation_sha256'],replay_n,replay_dt)
    out=folder/f'qr_mixed_n{n}_dt{dt:g}_gamma{gamma:.2g}_{ident}'
    out.mkdir(parents=True,exist_ok=True)
    status_path=out/'result.json';array_path=out/'reconstruction.npz'
    if status_path.exists():
        record=json.loads(status_path.read_text())
        if record.get('status')=='completed' and array_path.exists():
            return record,out,'cached'
        if record.get('status')=='failed' and not redo_failed:
            return record,out,'cached_failure'
    fm=MixedFEM2D(n)
    terminal,initialization=fm.sample_sensor_values(observation)
    record=dict(status='started',gamma=float(gamma),n=n,dt=dt,identifier=ident,
                observation_sha256=meta['observation_sha256'],
                boundary_closure='essential w=0, unconstrained q (natural normal derivative closure)',
                scheme='lagged-coefficient backward Euler mixed P1, consistent mass',
                **initialization)
    try:
        qr=fm.run(terminal,spec,gamma,dt=dt)
        replay=sensor_fit_from_replay(qr['initial'],fm,observation,spec,replay_n,replay_dt)
        record.update(status='completed',max_temperature=qr['max_temperature'],
                      min_temperature=qr['min_temperature'],inverse_wall_seconds=qr['inverse_wall_seconds'],
                      max_weak1_residual=qr['max_weak1_residual'],max_weak2_residual=qr['max_weak2_residual'],
                      sensor_discrepancy_absolute=replay['sensor_discrepancy_absolute'],
                      sensor_discrepancy_relative=replay['sensor_discrepancy_relative'],
                      replay_n=replay_n,replay_dt=replay_dt)
        np.savez_compressed(array_path,axis=fm.axis,
                            initial=qr['initial'],terminal_mesh_observation=terminal,
                            reconstructed_mid_reverse=qr['mid_reverse'],
                            replay_initial=replay['projected_initial'],
                            replay_terminal=replay['replay']['terminal'],
                            replay_pred_at_sensors=replay['sensor_pred'])
    except Exception as exc:
        record.update(status='failed',reason=str(exc),exception_type=type(exc).__name__)
    status_path.write_text(json.dumps(record,indent=2))
    return record,out,'computed'


def choose_by_discrepancy(table, meta):
    """Select from *successful* candidate QR solutions without evaluating truth.

    The noise standard deviation follows the published nominal observation law.
    Prefer nearest positive discrepancy to sqrt(N)*sigma (Morozov style). For
    zero-noise choose the smallest forward discrepancy. Boundary/FE bias may
    prevent attaining the target; this is recorded rather than hidden.
    """
    successful=[r for r in table if r.get('status')=='completed']
    if not successful:raise QRNumericalFailure('No successful QR candidate; do not select')
    target=np.sqrt(meta['sensor_count'])*float(meta['sigma'])
    selected=min(successful,key=lambda r:(abs(r['sensor_discrepancy_absolute']-target),r['gamma']))
    return dict(selected_run=selected['identifier'],selected_gamma=selected['gamma'],
                target_noise_norm_nominal=float(target),
                observed_discrepancy_absolute=selected['sensor_discrepancy_absolute'],
                distance_to_discrepancy_target=float(abs(selected['sensor_discrepancy_absolute']-target)),
                target_attained_within_20pct=bool(abs(selected['sensor_discrepancy_absolute']-target)<=0.2*max(target,1e-10)),
                selection_rule='nearest nominal Morozov-type target using independent forward replay; clean = minimum forward discrepancy; never initial truth')
