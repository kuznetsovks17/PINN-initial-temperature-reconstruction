"""Figures for the 2D nonlinear study, following the author's ORIGINAL plot conventions.

Conventions taken from epinn_publication_figures_selected_only(2).ipynb:
jet field colormap, pcolormesh with a 16-level translucent contour overlay,
matched limits and axis aspect, shared colorbars, mathematical labels and
lettered panels. Does not modify ANY of the author's original figures.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .nonlinear2d import coefficient, interpolate, grid, initial

TITLE_FS=14
PANEL_FS=12
TICK_FS=12
AXIS_FS=13
COLORBAR_LABEL_FS=13
COLORBAR_TICK_FS=12


def _field(ax, values, vmin, vmax, ticks=(0,.25,.5,.75,1)):
    values=np.asarray(values,dtype=float)
    n=values.shape[0]
    assert values.shape == (n,n)
    z=values.T  # x-first numerical tensor -> (y,x) Matplotlib grid
    xx=np.linspace(0.0,1.0,n)
    mesh=ax.pcolormesh(xx,xx,z,shading='auto',cmap='jet',vmin=vmin,vmax=vmax,rasterized=True)
    if vmax>vmin:
        levels=np.linspace(vmin,vmax,16)
        ax.contourf(xx,xx,z,levels=levels,cmap='jet',vmin=vmin,vmax=vmax,alpha=.30)
    ax.set_aspect('equal',adjustable='box')
    ax.set_xlim(0,1);ax.set_ylim(0,1)
    ax.set_xticks(ticks);ax.set_yticks(ticks)
    ax.tick_params(labelsize=TICK_FS,length=3)
    return mesh


def _label(ax,title,letter,show_y):
    ax.set_title(title,fontsize=TITLE_FS,pad=6)
    ax.set_xlabel(r'$x$',fontsize=AXIS_FS,labelpad=3)
    if show_y: ax.set_ylabel(r'$y$',fontsize=AXIS_FS,labelpad=4)
    else: ax.tick_params(labelleft=False)
    ax.text(.03,.97,f'({letter})',transform=ax.transAxes,ha='left',va='top',
            fontsize=PANEL_FS,fontweight='bold',color='black',
            bbox=dict(facecolor='white',edgecolor='none',alpha=.7,pad=.5),zorder=30)


def _save(fig,base):
    base=Path(base);base.parent.mkdir(exist_ok=True,parents=True)
    fig.savefig(base.with_suffix('.png'),dpi=250,bbox_inches='tight')
    fig.savefig(base.with_suffix('.pdf'),bbox_inches='tight')
    plt.close(fig)


def plot_reference(ref,base):
    """Actual computed forward fields only; never fabricates inverse output."""
    data=[np.asarray(ref['initial']),np.asarray(ref['terminal']),coefficient(ref['initial'],ref['spec'])]
    fig=plt.figure(figsize=(11.8,4.6))
    gs=fig.add_gridspec(2,3,height_ratios=[1,.055],left=.065,right=.96,
                         bottom=.20,top=.90,wspace=.17,hspace=.34)
    axs=[fig.add_subplot(gs[0,i]) for i in range(3)]
    cm=ca=None
    for i,(ax,v,title) in enumerate(zip(axs,data, [r'Initial field, $\theta_0$',r'Terminal field, $\theta_T$',r'Initial diffusivity, $a(\theta_0)$'])):
        lo,hi=(0,1) if i<=1 else (1,20)
        im=_field(ax,v,lo,hi)
        _label(ax,title,'abc'[i],show_y=i==0)
        if i==1: cm=im
        if i==2: ca=im
    cax1=fig.add_subplot(gs[1,:2]); cax2=fig.add_subplot(gs[1,2])
    c1=fig.colorbar(cm,cax=cax1,orientation='horizontal')
    c1.set_label(r'Temperature, $\theta$',fontsize=COLORBAR_LABEL_FS,labelpad=3)
    c2=fig.colorbar(ca,cax=cax2,orientation='horizontal')
    c2.set_label(r'Diffusivity ratio, $a$',fontsize=COLORBAR_LABEL_FS,labelpad=3)
    for c in (c1,c2): c.ax.tick_params(labelsize=COLORBAR_TICK_FS,length=3)
    _save(fig,base)

def plot_qr_vs_fourier(ref,qr_initial,neural_initial,base,neural_label='Fourier PINN, fixed'):
    """Publication comparison: true initial, mixed nonlinear QR, and ONE predeclared PINN.

    A full 3-PINN-only field montage is intentionally not produced.
    Numeric rankings for all PINN variants belong in the comparison table.
    This routine does not fabricate or infer results if either solver is missing.
    """
    truth=np.asarray(ref['initial'],dtype=float)
    n=truth.shape[0]
    qr=np.asarray(qr_initial,dtype=float).reshape(n,n)
    nn=np.asarray(neural_initial,dtype=float).reshape(n,n)
    fields=[truth,qr,nn]
    low=min(float(np.min(z)) for z in fields)
    high=max(float(np.max(z)) for z in fields)
    fig=plt.figure(figsize=(11.3,4.3))
    gs=fig.add_gridspec(1,4,width_ratios=[1,1,1,.065],left=.067,right=.97,
                        bottom=.15,top=.86,wspace=.13)
    axes=[fig.add_subplot(gs[0,i]) for i in range(3)]
    titles=[r'Reference, $\theta_0$',r'Nonlinear QR',neural_label]
    for i,(ax,field,title) in enumerate(zip(axes,fields,titles)):
        im=_field(ax,field,low,high)
        _label(ax,title,'abc'[i],show_y=i==0)
    cb=fig.colorbar(im,cax=fig.add_subplot(gs[0,3]))
    cb.set_label(r'Initial temperature, $\theta_0$',fontsize=COLORBAR_LABEL_FS)
    cb.ax.tick_params(labelsize=COLORBAR_TICK_FS)
    _save(fig,base)

    # Error panels use common absolute-error scale, not a separate color limit.
    errq=np.abs(qr-truth);errn=np.abs(nn-truth)
    mx=max(float(errq.max()),float(errn.max()),1e-10)
    fig=plt.figure(figsize=(8.1,4.0))
    gs=fig.add_gridspec(1,3,width_ratios=[1,1,.06],left=.08,right=.94,
                        bottom=.16,top=.85,wspace=.15)
    for i,(field,title) in enumerate(((errq,'Nonlinear QR'),(errn,neural_label))):
        ax=fig.add_subplot(gs[0,i]);im=_field(ax,field,0,mx)
        _label(ax,title,'ab'[i],show_y=i==0)
    cb=fig.colorbar(im,cax=fig.add_subplot(gs[0,2]))
    cb.set_label(r'Absolute initial error',fontsize=COLORBAR_LABEL_FS)
    cb.ax.tick_params(labelsize=COLORBAR_TICK_FS)
    _save(fig,str(base)+'_absolute_error')
