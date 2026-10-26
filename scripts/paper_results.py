#!/usr/bin/env python3
"""Print source-backed result summaries used in the inverse-conduction manuscript.

Uses saved CSV tables from the author's completed heat(3).zip archive.
Does not import TensorFlow, perform training, select parameter values, or rewrite results.
"""
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'published_results'


def summary(file, metric, group='case'):
    d=pd.read_csv(P/file)
    sub=d[d['metric'].eq(metric)].copy()
    cols=list(dict.fromkeys(c for c in [group,'noise_level','method','n_available','mean','sample_sd'] if c in sub))
    sort_keys=list(dict.fromkeys(c for c in [group,'noise_level','method'] if c in cols))
    return sub[cols].sort_values(sort_keys).reset_index(drop=True)


def report():
    for title,path,metric in [
        ('Linear and nonlinear synthetic PINNs: initial relative L2',
         'reduced_report/synthetic_descriptive_summary.csv','initial_relative_l2'),
        ('Linear 2D spectral Tikhonov and TSVD',
         'reduced_report/classical_descriptive_summary.csv','initial_relative_l2'),
        ('Brass initial RMSE in kelvin',
         'reduced_report/brass_initial_descriptive_summary.csv','RMSE_K'),
        ('Brass initial signed bias in kelvin',
         'reduced_report/brass_initial_descriptive_summary.csv','bias_K'),
    ]:
        print('\n'+title)
        group='case' if 'reduced_report/brass' not in path else 'method'
        print(summary(path,metric,group).to_string(index=False, float_format=lambda x:f'{x:.6g}'))
    print('\nNonlinear 2D — QR and PINNs: measured initial errors')
    d=pd.read_csv(P/'nonlinear2d/report/nonlinear2d_qr_pinn_comparison.csv')
    print(d[['condition','method','initial_relative_l2','independent_sensor_relative_discrepancy']]
          .to_string(index=False,float_format=lambda x:f'{x:.6g}'))
    assert len(d)==8, f'Expected 2 conditions x 4 methods, found {len(d)}'
    s=pd.read_csv(P/'reduced_report/synthetic_descriptive_summary.csv')
    for case,methods in [('smooth2d',3),('nonlinear1d',2)]:
        x=s[(s.case==case)&(s.metric=='initial_relative_l2')]
        assert len(x)==methods*2 and (x.n_available==3).all(),(case,len(x),x.n_available.tolist())
    b=pd.read_csv(P/'reduced_report/brass_initial_descriptive_summary.csv')
    assert len(b[(b.metric=='RMSE_K') & (b.n_available==3)])==3
    print('\nPASS: saved published summary cohorts have the expected sizes (18+12 synthetic, 9 brass, 8 nonlinear 2D methods/conditions).')

if __name__=='__main__':report()
