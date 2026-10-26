#!/usr/bin/env python3
"""Regenerate numeric tables for the paper's historical noise/grid sweeps.

Reads actual saved single-run PINN metrics and historical QR gamma sweeps.
Boundary-inclusive legacy data and preprocessing are NOT identical to the
later interior-only paired comparisons. No training or solver is executed.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
LEG=ROOT/'data/legacy_runs'
QR=ROOT/'data/legacy_qr'
SHARED='noise_and_observation_shared__smooth2d__d2__obs33__noise005p0'


def read_initial(run):
    return float(json.loads((LEG/run/'metrics.json').read_text())['initial_relative_l2'])


def qr_table(filename):
    data=pd.read_csv(QR/filename)
    filt=np.isclose(data['gamma'],1e-6,rtol=1e-8,atol=0)
    data=data.loc[filt].set_index('run_id')
    return data


def noise():
    q=qr_table('qr_exp01_noise_summary_v2.csv')
    rows=[]
    for pct in [0,1,5,10,15,20]:
        run=(SHARED if pct==5 else f'noise_sweep__smooth2d__d2__obs33__noise{pct:03d}p0')
        rows.append({'noise_percent':pct,'total_tensor_nodes':1089,
                     'PINN_relative_L2':read_initial(run),
                     'QR_relative_L2':float(q.loc[run,'initial_rel_l2_validation'])})
    return pd.DataFrame(rows)


def density():
    q=qr_table('qr_exp02_observation_count_summary_v2.csv')
    rows=[]
    for n in [5,9,17,33]:
        run=SHARED if n==33 else f'observation_sweep__smooth2d__d2__obs{n}__noise005p0'
        rows.append({'tensor_grid':f'{n}x{n}','total_nodes':n*n,'interior_nodes':(n-2)**2,
                     'PINN_relative_L2':read_initial(run),
                     'QR_relative_L2':float(q.loc[run,'initial_rel_l2_validation'])})
    return pd.DataFrame(rows)


def main():
    a,b=noise(),density()
    print('Full terminal tensor grid: noise sensitivity (single-run legacy data)')
    print(a.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('\nTerminal tensor-grid density (single-run legacy data, nominal 5% noise)')
    print(b.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    assert len(a)==6 and len(b)==4
    assert abs(float(a.loc[a.noise_percent==5,'PINN_relative_L2'].iloc[0]) -
               float(b.loc[b.tensor_grid=='33x33','PINN_relative_L2'].iloc[0]))<1e-12
    assert np.allclose(a.PINN_relative_L2,[.12208664291695,.13505600640021,.18035870137514,.20853773029576,.21997803667466,.22102665270995],atol=1e-12)
    assert np.allclose(b.PINN_relative_L2,[.2192,.2238,.2124,.1804],atol=0.00005)
    print('\nPASS: same 5%-noise / 33x33 run reused exactly once in each table view.')

if __name__=='__main__':main()
