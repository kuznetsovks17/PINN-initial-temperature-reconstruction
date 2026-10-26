#!/usr/bin/env python3
"""Re-create nonlinear 2D QR vs fixed-Fourier figure from archived NUMERICAL FIELDS.

Uses the same plotting function as Notebook 10 and never modifies the author's
saved figure files. This regenerates the field comparison; it does not re-run QR
or the PINN training, and Matplotlib raster output may vary between versions.
"""
from pathlib import Path
from argparse import ArgumentParser
import json
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from inverse_study import nonlinear2d as n2, nonlinear2d_figures as plots


def reproduce(cond, out_dir):
    tag={'clean':'noise0_seed0','noisy5':'noise0.05_seed3100'}[cond]
    sel=json.loads((ROOT/'results/nonlinear2d/qr_mixed'/tag/'selected_gamma.json').read_text())
    # Prefer the exact file path reported in the article results table, avoiding assumptions about JSON keys.
    import pandas as pd
    df=pd.read_csv(ROOT/'published_results/nonlinear2d/report/nonlinear2d_qr_selected_results.csv')
    match=df[df['condition'].eq(cond)]
    assert len(match)==1
    qr_file=ROOT/str(match.iloc[0]['run_folder'])/'reconstruction.npz'
    with np.load(qr_file) as z:
        qa=z['axis'];q0=z['initial']
    opts=[]
    for d in (ROOT/'published_results/nonlinear2d/neural').glob('*'):
        if not d.is_dir() or not (d/'config.json').exists():continue
        cfg=json.loads((d/'config.json').read_text())
        if cfg['config']['method']=='fourier_fixed' and f'strong2d_{cond}_' in cfg['config'].get('run_id',''):
            opts.append(d)
    assert len(opts)==1, (cond,len(opts))
    with np.load(opts[0]/'predictions.npz') as z:
        xev=z['X_eval'];theta0=z['theta_initial']
    axis,xx,yy=n2.grid(65)
    grid=np.column_stack((xx.ravel(),yy.ravel()))
    assert np.allclose(grid,xev)
    q65=RegularGridInterpolator((qa,qa),q0,bounds_error=True)(grid).reshape(65,65)
    out_dir.mkdir(parents=True,exist_ok=True)
    dst=out_dir/f'nonlinear2d_qr_vs_fourier_{cond}_original_style'
    plots.plot_qr_vs_fourier({'initial':n2.initial(xx,yy)},q65,theta0,dst)
    print(f'{cond}: {dst}.pdf')

if __name__=='__main__':
    parser=ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'generated_figures')
    args=parser.parse_args()
    for c in ['clean','noisy5']:reproduce(c,args.output)
