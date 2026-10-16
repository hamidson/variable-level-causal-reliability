#!/usr/bin/env python3
"""Reproducibility script for the ACIJ paper

Variable-Level Causal Reliability from Heterogeneous Causal Discovery
Methods in Multivariate Time Series.

Generated from the final experimental notebook. Development-only smoke,
recovery, and display-test cells were omitted. The scientific computation
order is preserved.

For interactive use in VS Code/Jupyter, each `# %%` block can be run as a cell.
"""


# %%
# Environment and package setup
import os, sys, subprocess, importlib.util
from pathlib import Path

IN_COLAB = 'google.colab' in sys.modules
USE_DRIVE_PACKAGES = True
ALLOW_INSTALL_IF_MISSING = False
PKG_DIR = None

if IN_COLAB and USE_DRIVE_PACKAGES:
    from google.colab import drive
    if not Path('/content/drive/MyDrive').is_dir():
        drive.mount('/content/drive', force_remount=False)

    PKG_DIR = Path('/content/drive/MyDrive/PhD_Boyxorozova/colab_packages')
    PKG_DIR.mkdir(parents=True, exist_ok=True)

    if str(PKG_DIR) not in sys.path:
        sys.path.insert(0, str(PKG_DIR))

packages = {
    'tigramite': 'tigramite==5.2.10.1',
    'lingam': 'lingam==1.13.0',
}

missing = []

for module_name, package_name in packages.items():
    importlib.invalidate_caches()
    spec = importlib.util.find_spec(module_name)

    if spec is not None:
        print(f'{module_name}: tayyor — qayta o‘rnatilmadi.')
    else:
        missing.append((module_name, package_name))

if missing and not ALLOW_INSTALL_IF_MISSING:
    names = ', '.join(m for m, _ in missing)
    raise ModuleNotFoundError(
        f'Quyidagi paket(lar) Drive paketlar papkasidan topilmadi: {names}. '
        f'Paketlar papkasi: {PKG_DIR}. '
        'Avval oldingi ishlagan Colab muhitidagi paketlar shu papkada ekanini tekshiring. '
        'Faqat zarurat bo‘lsa ALLOW_INSTALL_IF_MISSING=True qiling.'
    )

if missing and ALLOW_INSTALL_IF_MISSING:
    for module_name, package_name in missing:
        print(f'{module_name}: bir marta Drive papkasiga o‘rnatilmoqda...')
        cmd = [
            sys.executable, '-m', 'pip', 'install',
            '--prefer-binary', '--disable-pip-version-check'
        ]
        if PKG_DIR is not None:
            cmd += ['--target', str(PKG_DIR)]
        cmd += [package_name]
        subprocess.check_call(cmd)
        importlib.invalidate_caches()

        if importlib.util.find_spec(module_name) is None:
            raise ImportError(f'{module_name} o‘rnatildi, ammo import qilinmadi.')

print('Muhit tayyor. Mavjud paketlar qayta o‘rnatilmadi.')

# %%
# CausalTime pre-run validation
# CausalTime — final pre-run validation (no download)
from pathlib import Path
import numpy as np

_ct_root = Path(
    "/content/drive/MyDrive/PhD_Boyxorozova/data/causaltime_gen_ver1.0"
)

_expected = {
    "medical": ((480, 40, 40), (20, 20)),
    "pm25":    ((480, 40, 72), (36, 36)),
    "traffic": ((480, 40, 40), (20, 20)),
}

print("CausalTime pre-run validation:")

for _domain, (_data_shape, _graph_shape) in _expected.items():
    _data_file = _ct_root / _domain / "gen_data.npy"
    _graph_file = _ct_root / _domain / "graph.npy"

    if not _data_file.exists() or not _graph_file.exists():
        raise FileNotFoundError(
            f"{_domain}: gen_data.npy yoki graph.npy topilmadi: {_ct_root}"
        )

    _data = np.load(_data_file, mmap_mode="r")
    _graph = np.load(_graph_file, mmap_mode="r")

    print(
        f"{_domain}: data={_data.shape}, graph={_graph.shape}"
    )

    assert tuple(_data.shape) == _data_shape, (
        _domain, _data.shape, _data_shape
    )
    assert tuple(_graph.shape) == _graph_shape, (
        _domain, _graph.shape, _graph_shape
    )

print("CausalTime: OK")

# %%
# Imports, configuration, and paths
import json, math, time, hashlib, warnings, shutil, re
from dataclasses import dataclass, asdict
from pathlib import Path
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.optimize import minimize
from scipy.stats import wilcoxon, spearmanr
from sklearn.metrics import precision_recall_fscore_support
from IPython.display import display

from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.lpcmci import LPCMCI
from tigramite.independence_tests.parcorr import ParCorr
from lingam import VARLiNGAM

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 120)
pd.set_option('display.width', 200)

RUN_MODE = 'paper'  # FINAL: 9 scenario × 10 seed + external benchmarks
FAST_MODE = RUN_MODE in {'smoke','fast'}
SHOW_PLOTS = True
FORCE_RECOMPUTE = False
NOTEBOOK_VERSION = '0.8-ACIJ-FINAL-RUN'
CACHE_SCHEMA = 8

@dataclass
class Config:
    target: str = 'Y'
    max_lag: int = 5
    pcmci_pc_alpha: float = 0.05
    varlingam_coef_threshold: float = 0.08
    varlingam_prune: bool = True
    dynotears_lambda_w: float = 0.03
    dynotears_lambda_a: float = 0.03
    dynotears_w_threshold: float = 0.08
    dynotears_max_iter: int = 10 if FAST_MODE else 40
    dynotears_optimizer_maxiter: int = 100 if FAST_MODE else 400
    dynotears_h_tol: float = 1e-8
    dynotears_rho_max: float = 1e16
    kernel: str = 'exponential'
    kernel_bandwidth: float = 1.0
    theta_v: float = 2/3
    theta_exact: float = 0.50
    theta_proximity: float = 0.55
    theta_margin: float = 0.05
    tie_tol: float = 1e-10
    rv_strategy: str = 'family_normalized'

CFG = Config()
if IN_COLAB and Path('/content/drive/MyDrive').is_dir():
    PROJECT = Path('/content/drive/MyDrive/PhD_Boyxorozova/v08_ACIJ_FINAL')
else:
    PROJECT = Path('/content/PhD_Boyxorozova_v08_ACIJ_FINAL')
DIRS={
    'root':PROJECT,
    'cache':PROJECT/'cache',
    'results':PROJECT/'results',
    'figures':PROJECT/'figures',
    'data':PROJECT/'data',
    'benchmark_cache':PROJECT/'benchmark_cache',
}
for p in DIRS.values():
    p.mkdir(parents=True,exist_ok=True)

def _ct_root_has_files(p):
    p = Path(p)
    return all(
        (p/d/'gen_data.npy').exists() and
        (p/d/'graph.npy').exists()
        for d in ['medical','pm25','traffic']
    )

if IN_COLAB and Path('/content/drive/MyDrive').is_dir():
    _ct_candidates = [
        Path('/content/drive/MyDrive/PhD_Boyxorozova/data/causaltime_gen_ver1.0'),
        Path('/content/drive/MyDrive/PhD_Boyxorozova/data/CausalTime'),
        Path('/content/drive/MyDrive/PhD_Boyxorozova/CausalTime'),
        Path('/content/drive/MyDrive/CausalTime'),
        DIRS['data'] / 'causaltime_gen_ver1.0',
    ]
else:
    _ct_candidates = [
        DIRS['data'] / 'causaltime_gen_ver1.0',
        DIRS['data'] / 'CausalTime',
    ]

CAUSALTIME_ROOT = next(
    (p for p in _ct_candidates if _ct_root_has_files(p)),
    _ct_candidates[0]
)

CAUSALTIME_DOMAINS = ['medical','pm25','traffic']
CAUSALTIME_MAX_LAG = 1
DOWNLOAD_CAUSALTIME_IF_MISSING = False
REQUIRE_CAUSALTIME_FOR_PAPER = True

print('CausalTime root candidate =', CAUSALTIME_ROOT)

if RUN_MODE == 'paper' and REQUIRE_CAUSALTIME_FOR_PAPER and not _ct_root_has_files(CAUSALTIME_ROOT):
    raise FileNotFoundError(
        'FINAL paper run boshlanmadi, chunki CausalTime uch domeni topilmadi.\\n'
        'Quyidagi tuzilmani Drive ga bir marta joylashtiring:\\n'
        '/content/drive/MyDrive/PhD_Boyxorozova/data/causaltime_gen_ver1.0/medical/gen_data.npy\\n'
        '/content/drive/MyDrive/PhD_Boyxorozova/data/causaltime_gen_ver1.0/medical/graph.npy\\n'
        '/content/drive/MyDrive/PhD_Boyxorozova/data/causaltime_gen_ver1.0/pm25/gen_data.npy\\n'
        '/content/drive/MyDrive/PhD_Boyxorozova/data/causaltime_gen_ver1.0/pm25/graph.npy\\n'
        '/content/drive/MyDrive/PhD_Boyxorozova/data/causaltime_gen_ver1.0/traffic/gen_data.npy\\n'
        '/content/drive/MyDrive/PhD_Boyxorozova/data/causaltime_gen_ver1.0/traffic/graph.npy\\n'
        'Fayllar joylashtirilgach Runtime > Run all ni bosing.'
    )
print('RUN_MODE =',RUN_MODE)
print('FAST_MODE =',FAST_MODE)
print('PROJECT =',PROJECT)
print('CAUSALTIME_ROOT =',CAUSALTIME_ROOT)
print(json.dumps(asdict(CFG),ensure_ascii=False,indent=2))

# %%
# Notebook code cell 8
import sys
checks = {
    'python': sys.version.split()[0],
    'pandas': pd.__version__,
    'numpy': np.__version__,
    'PCMCI': PCMCI.__name__,
    'LPCMCI': LPCMCI.__name__,
    'VARLiNGAM': VARLiNGAM.__name__,
}
print('PRE-FLIGHT: OK')
for k, v in checks.items():
    print(f'{k}: {v}')

# %%
# Edge normalization
# PRE-FLIGHT GUARD: bu yacheykadan oldin Importlar yacheykasi bajarilgan bo‘lishi shart.
_required_names = ['pd', 'np', 'CFG', 'display']
_missing_names = [name for name in _required_names if name not in globals()]
if _missing_names:
    raise RuntimeError(
        "Oldingi 'Importlar va umumiy sozlamalar' yacheykasi joriy runtime-da bajarilmagan. "
        "Colab menyusidan Runtime > Run all (yoki Ctrl+F9) ni bosing. "
        f"Yetishmayotgan nomlar: {_missing_names}"
    )

METHOD_FAMILY = {
    'PCMCI+': 'conditional_independence',
    'PCMCI': 'conditional_independence',
    'LPCMCI': 'conditional_independence',
    'VAR-LiNGAM': 'non_gaussian_svar',
    'DYNOTEARS': 'continuous_score_optimization',
}

FAMILY_DESCRIPTION = {
    'conditional_independence': 'constraint/conditional-independence time-series causal discovery',
    'non_gaussian_svar': 'non-Gaussian structural VAR identification',
    'continuous_score_optimization': 'continuous score-based structure optimization',
}

EXPECTED_METHODS = list(METHOD_FAMILY)

EDGE_COLUMNS = ['method','family','cause','target','lag','score','p_value','selected']

def empty_edges():
    return pd.DataFrame(columns=EDGE_COLUMNS)

def normalize_edges(df):
    if df is None or len(df) == 0:
        return empty_edges()
    out=df.copy()
    for c in EDGE_COLUMNS:
        if c not in out.columns:
            out[c]=np.nan
    out['lag']=pd.to_numeric(out['lag'],errors='raise').astype(int)
    if out['selected'].dtype==object:
        out['selected']=(
            out['selected'].astype(str).str.lower()
            .map({'true':True,'false':False}).fillna(False)
        )
    out['selected']=out['selected'].astype(bool)
    return out[EDGE_COLUMNS].sort_values(['method','cause','lag']).reset_index(drop=True)

print('v0.6 algorithmic-family tarkibi:')
display(pd.DataFrame([
    {'Method':m,'Family':f,'Family description':FAMILY_DESCRIPTION[f]}
    for m,f in METHOD_FAMILY.items()
]))

# %%
# Causal-discovery methods
def _tigramite_frame(df):
    cols=list(df.columns)
    return cols, pp.DataFrame(
        df.to_numpy(dtype=float),
        datatime=np.arange(len(df)),
        var_names=cols
    )

def _extract_tigramite_target_edges(result, cols, target, max_lag, method_name, alpha):
    target_idx=cols.index(target)
    graph=np.asarray(result['graph'])
    pmat=np.asarray(result['p_matrix'])
    vmat=np.asarray(result['val_matrix'])
    rows=[]

    for i,cause in enumerate(cols):
        if cause==target:
            continue
        for lag in range(1,max_lag+1):
            mark=str(graph[i,target_idx,lag])
            p=float(pmat[i,target_idx,lag])
            score=float(vmat[i,target_idx,lag])

            # Only lagged edges oriented toward the target are used as positive presence evidence.
            is_directed_to_target = mark.endswith('>')
            if is_directed_to_target and np.isfinite(p) and p<=alpha:
                rows.append({
                    'method':method_name,
                    'family':METHOD_FAMILY[method_name],
                    'cause':cause,
                    'target':target,
                    'lag':lag,
                    'score':score,
                    'p_value':p,
                    'selected':True
                })
    return normalize_edges(pd.DataFrame(rows))

def run_pcmciplus(df,target,max_lag,pc_alpha=0.05):
    cols,tig_df=_tigramite_frame(df)
    model=PCMCI(
        dataframe=tig_df,
        cond_ind_test=ParCorr(significance='analytic'),
        verbosity=0
    )
    result=model.run_pcmciplus(
        tau_min=1,
        tau_max=max_lag,
        pc_alpha=pc_alpha,
        fdr_method='none'
    )
    return _extract_tigramite_target_edges(
        result,cols,target,max_lag,'PCMCI+',pc_alpha
    )

def run_pcmci(df,target,max_lag,pc_alpha=0.05):
    cols,tig_df=_tigramite_frame(df)
    model=PCMCI(
        dataframe=tig_df,
        cond_ind_test=ParCorr(significance='analytic'),
        verbosity=0
    )
    result=model.run_pcmci(
        tau_min=1,
        tau_max=max_lag,
        pc_alpha=pc_alpha,
        alpha_level=pc_alpha,
        fdr_method='none'
    )
    return _extract_tigramite_target_edges(
        result,cols,target,max_lag,'PCMCI',pc_alpha
    )

def run_lpcmci(df,target,max_lag,pc_alpha=0.05):
    cols,tig_df=_tigramite_frame(df)
    model=LPCMCI(
        dataframe=tig_df,
        cond_ind_test=ParCorr(significance='analytic'),
        verbosity=0
    )
    result=model.run_lpcmci(
        tau_min=1,
        tau_max=max_lag,
        pc_alpha=pc_alpha
    )
    return _extract_tigramite_target_edges(
        result,cols,target,max_lag,'LPCMCI',pc_alpha
    )

def run_varlingam(df,target,max_lag,coef_threshold=0.08,prune=True,seed=42):
    cols=list(df.columns); target_idx=cols.index(target)
    model=VARLiNGAM(lags=max_lag,criterion=None,prune=prune,random_state=seed)
    model.fit(df.to_numpy(dtype=float)); mats=np.asarray(model.adjacency_matrices_)
    rows=[]
    for lag in range(1,min(max_lag,mats.shape[0]-1)+1):
        B=np.asarray(mats[lag],dtype=float)
        for i,cause in enumerate(cols):
            if cause==target: continue
            coef=float(B[target_idx,i])
            if np.isfinite(coef) and abs(coef)>=coef_threshold:
                rows.append({'method':'VAR-LiNGAM','family':METHOD_FAMILY['VAR-LiNGAM'],'cause':cause,'target':target,'lag':lag,
                             'score':coef,'p_value':np.nan,'selected':True})
    return normalize_edges(pd.DataFrame(rows)),model

def make_lag_design(X,p):
    X=np.asarray(X,dtype=float)
    if X.ndim!=2: raise ValueError('X must be 2D [time, variables].')
    if p<1 or p>=len(X): raise ValueError('p must satisfy 1 <= p < n_samples.')
    return X[p:],np.concatenate([X[p-k:len(X)-k] for k in range(1,p+1)],axis=1)

def _acyclicity(W):
    d=W.shape[0]; E=expm(W*W); h=float(np.trace(E)-d); grad=E.T*(2.0*W)
    return h,grad

def fit_dynotears_linear(X,p,lambda_w=0.03,lambda_a=0.03,max_iter=50,h_tol=1e-8,rho_max=1e16,w_threshold=0.08,optimizer_maxiter=500):
    X_now,X_lag=make_lag_design(X,p); n,d=X_now.shape; q=p*d
    sd_now=X_now.std(axis=0,ddof=0); sd_now[sd_now<1e-12]=1.0; Xn=(X_now-X_now.mean(axis=0))/sd_now
    sd_lag=X_lag.std(axis=0,ddof=0); sd_lag[sd_lag<1e-12]=1.0; Z=(X_lag-X_lag.mean(axis=0))/sd_lag
    nw=d*d; na=q*d; size=2*nw+2*na; z=np.zeros(size,dtype=float); bounds=[]
    for _ in range(2):
        for i in range(d):
            for j in range(d): bounds.append((0.0,0.0) if i==j else (0.0,None))
    bounds.extend([(0.0,None)]*(2*na))
    def unpack(v):
        o=0; Wp=v[o:o+nw].reshape(d,d); o+=nw; Wm=v[o:o+nw].reshape(d,d); o+=nw
        Ap=v[o:o+na].reshape(q,d); o+=na; Am=v[o:o+na].reshape(q,d)
        return Wp,Wm,Ap,Am,Wp-Wm,Ap-Am
    rho,alpha,h_prev=1.0,0.0,np.inf; last_result=None
    for _outer in range(max_iter):
        while rho<rho_max:
            def objective(v):
                Wp,Wm,Ap,Am,W,A=unpack(v); R=Xn-Xn@W-Z@A; loss=0.5/n*np.sum(R*R); h,gh=_acyclicity(W)
                reg=lambda_w*(Wp.sum()+Wm.sum())+lambda_a*(Ap.sum()+Am.sum())
                f=loss+reg+0.5*rho*h*h+alpha*h
                gW=-(Xn.T@R)/n+(rho*h+alpha)*gh; gA=-(Z.T@R)/n
                g=np.concatenate([(gW+lambda_w).ravel(),(-gW+lambda_w).ravel(),(gA+lambda_a).ravel(),(-gA+lambda_a).ravel()])
                return f,g
            last_result=minimize(objective,z,method='L-BFGS-B',jac=True,bounds=bounds,
                                 options={'maxiter':int(optimizer_maxiter),'ftol':1e-12,'gtol':1e-8,'maxls':30})
            z_new=last_result.x; *_,W_new,A_new=unpack(z_new); h_new,_=_acyclicity(W_new)
            if h_new>0.25*h_prev: rho*=10.0
            else: z=z_new; break
        *_,W,A=unpack(z); h,_=_acyclicity(W); alpha+=rho*h; h_prev=h
        if h<=h_tol or rho>=rho_max: break
    W[np.abs(W)<w_threshold]=0.0; A[np.abs(A)<w_threshold]=0.0
    lag_matrices=[]
    for lag in range(1,p+1): lag_matrices.append(A[(lag-1)*d:lag*d,:].T.copy())
    return {'W0':W.T.copy(),'lag_matrices':np.asarray(lag_matrices),'acyclicity':float(_acyclicity(W)[0]),
            'optimizer_success':bool(last_result.success) if last_result is not None else False}

def run_dynotears(df,target,max_lag,cfg):
    cols=list(df.columns); target_idx=cols.index(target)
    fit=fit_dynotears_linear(df.to_numpy(dtype=float),p=max_lag,lambda_w=cfg.dynotears_lambda_w,lambda_a=cfg.dynotears_lambda_a,
        max_iter=cfg.dynotears_max_iter,h_tol=cfg.dynotears_h_tol,rho_max=cfg.dynotears_rho_max,
        w_threshold=cfg.dynotears_w_threshold,optimizer_maxiter=cfg.dynotears_optimizer_maxiter)
    rows=[]
    for lag,B in enumerate(fit['lag_matrices'],start=1):
        for i,cause in enumerate(cols):
            if cause==target: continue
            coef=float(B[target_idx,i])
            if np.isfinite(coef) and abs(coef)>=cfg.dynotears_w_threshold:
                rows.append({'method':'DYNOTEARS','family':METHOD_FAMILY['DYNOTEARS'],'cause':cause,'target':target,'lag':lag,
                             'score':coef,'p_value':np.nan,'selected':True})
    return normalize_edges(pd.DataFrame(rows)),fit

# %%
# Reliability aggregation
def build_presence_matrix(edges,variables,methods):
    P=pd.DataFrame(0.0,index=variables,columns=methods)
    for _,r in edges[edges['selected']].iterrows():
        if r['cause'] in P.index and r['method'] in P.columns:
            P.loc[r['cause'],r['method']]=1.0
    return P

def effective_method_weights(methods,method_family,strategy='family_normalized',method_quality=None):
    methods=list(methods)
    if not methods: return {}
    if strategy=='equal_method': return {m:1/len(methods) for m in methods}
    if strategy!='family_normalized': raise ValueError('strategy must be equal_method or family_normalized')
    families={}
    for m in methods: families.setdefault(method_family[m],[]).append(m)
    fam_w={f:1/len(families) for f in families}; out={}
    for fam,ms in families.items():
        inner={m:1/len(ms) for m in ms}
        if method_quality is not None:
            vals=np.array([max(float(method_quality.get(m,0.0)),0.0) for m in ms])
            if vals.sum()>0: inner={m:float(v/vals.sum()) for m,v in zip(ms,vals)}
        for m in ms: out[m]=fam_w[fam]*inner[m]
    return out

def compute_rv(edges,variables,method_family,strategy='family_normalized',method_quality=None):
    methods=list(method_family); P=build_presence_matrix(edges,variables,methods)
    w=effective_method_weights(methods,method_family,strategy,method_quality)
    return P.mul(pd.Series(w),axis=1).sum(axis=1).rename('R_V'),P,{'method_weights':w}

def lag_kernel(a,b,h=1.0,kind='exponential'):
    if h<=0: raise ValueError('kernel bandwidth h must be > 0')
    d=abs(float(a)-float(b))
    if kind=='exponential': return math.exp(-d/h)
    if kind=='gaussian': return math.exp(-(d*d)/(2*h*h))
    if kind=='triangular': return max(0.0,1.0-d/h)
    raise ValueError('Unknown kernel')

def method_lag_sets(edges,variables,methods):
    out={v:{m:[] for m in methods} for v in variables}; selected=edges[edges['selected']]
    for v in variables:
        for m in methods:
            vals=selected.loc[(selected['cause']==v)&(selected['method']==m),'lag'].astype(int).tolist()
            out[v][m]=sorted(set(vals))
    return out

def compute_lag_consensus(edges,variables,methods,method_family,h,kernel_kind,strategy='family_normalized',method_quality=None):
    lagsets=method_lag_sets(edges,variables,methods)
    w=effective_method_weights(methods,method_family,strategy,method_quality); rows=[]
    for v in variables:
        selected_methods=[m for m in methods if lagsets[v][m]]
        denom=sum(w[m] for m in selected_methods)
        candidates=sorted({tau for m in selected_methods for tau in lagsets[v][m]})
        for tau in candidates:
            if denom<=0: exact=prox=0.0
            else:
                exact=sum(w[m] for m in selected_methods if tau in lagsets[v][m])/denom
                prox=sum(w[m]*max(lag_kernel(tau,t0,h,kernel_kind) for t0 in lagsets[v][m]) for m in selected_methods)/denom
            rows.append({'cause':v,'lag':int(tau),'E_L':float(exact),'R_L':float(prox),'D_L':float(1.0-prox),
                         'n_exact_methods':int(sum(tau in lagsets[v][m] for m in selected_methods)),
                         'n_variable_methods':int(len(selected_methods))})
    cols=['cause','lag','E_L','R_L','D_L','n_exact_methods','n_variable_methods']
    return pd.DataFrame(rows,columns=cols),lagsets,w

def select_rv_exact_only(rv,lag_table,theta_v,theta_exact,eps=1e-12):
    out=lag_table.copy()
    if len(out)==0:
        out['selected']=[]; return out
    out['R_V']=out['cause'].map(rv.to_dict()).fillna(0.0)
    out['selected']=(out['R_V']+eps>=theta_v)&(out['E_L']>theta_exact+eps)
    return out

def select_temporal_features_v05(rv,lag_table,theta_v,theta_exact,theta_proximity,theta_margin,tie_tol=1e-10,eps=1e-12):
    # 1) R_V gate; 2) exact majority; 3) otherwise unique R_L medoid; 4) else abstain.
    out=lag_table.copy()
    cols=['R_V','variable_pass','exact_majority','proximity_candidate','proximity_gap','selected','decision_mode','reason']
    if len(out)==0:
        for c in cols: out[c]=[]
        return out
    out['R_V']=out['cause'].map(rv.to_dict()).fillna(0.0)
    out['variable_pass']=out['R_V']+eps>=theta_v
    out['exact_majority']=out['E_L']>theta_exact+eps
    out['proximity_candidate']=False; out['proximity_gap']=np.nan; out['selected']=False
    out['decision_mode']='none'; out['reason']='tanlanmadi'
    for cause,idx0 in out.groupby('cause').groups.items():
        idx=list(idx0); g=out.loc[idx].copy()
        if not bool(g['variable_pass'].iloc[0]):
            out.loc[idx,'decision_mode']='variable_rejected'; out.loc[idx,'reason']='o‘zgaruvchi ishonchliligi past'; continue
        exact_idx=g.index[g['exact_majority']].tolist()
        if exact_idx:
            out.loc[idx,'decision_mode']='exact_consensus_context'; out.loc[idx,'reason']='aniq ko‘pchilikka kirmadi'
            out.loc[exact_idx,'selected']=True; out.loc[exact_idx,'decision_mode']='exact_consensus'
            out.loc[exact_idx,'reason']='aniq vaqt kechikishi bo‘yicha ko‘pchilik'; continue
        ranked=g.sort_values(['R_L','E_L','lag'],ascending=[False,False,True])
        best_idx=ranked.index[0]; best_rl=float(ranked.iloc[0]['R_L'])
        second_rl=float(ranked.iloc[1]['R_L']) if len(ranked)>=2 else 0.0
        gap=best_rl-second_rl
        n_tied=int(np.sum(np.abs(ranked['R_L'].to_numpy()-best_rl)<=tie_tol))
        out.loc[idx,'proximity_gap']=gap; out.loc[best_idx,'proximity_candidate']=True
        if best_rl+eps<theta_proximity:
            out.loc[idx,'decision_mode']='abstain_low_proximity'; out.loc[idx,'reason']='yaqinlik bo‘yicha kelishuv yetarli emas'; continue
        if n_tied>1 or gap+eps<theta_margin:
            out.loc[idx,'decision_mode']='abstain_ambiguous'; out.loc[idx,'reason']='yaqinlik bo‘yicha yagona ustun vaqt kechikishi yo‘q'; continue
        out.loc[idx,'decision_mode']='proximity_context'; out.loc[idx,'reason']='yaqinlik bo‘yicha markaziy nomzod emas'
        out.loc[best_idx,'selected']=True; out.loc[best_idx,'decision_mode']='proximity_consensus'
        out.loc[best_idx,'reason']='yaqin vaqt kechikishlari bo‘yicha yagona kelishuv markazi'
    return out

# %%
# Disagreement diagnostics
def binary_entropy01(p,eps=1e-12):
    p=float(np.clip(p,0,1))
    if p<=eps or p>=1-eps: return 0.0
    return float(-(p*math.log2(p)+(1-p)*math.log2(1-p)))

def lagset_similarity(A,B,h=1.0,kind='exponential'):
    A=sorted(set(A)); B=sorted(set(B))
    if not A or not B: return np.nan
    ab=np.mean([max(lag_kernel(a,b,h,kind) for b in B) for a in A])
    ba=np.mean([max(lag_kernel(b,a,h,kind) for a in A) for b in B])
    return float((ab+ba)/2)

def variable_disagreement(rv,lagsets,methods,method_weights,h,kind):
    rows=[]
    for v in rv.index:
        p=float(rv.loc[v]); d_v=binary_entropy01(p); selected=[m for m in methods if lagsets[v][m]]
        num=den=0.0; pair_sims=[]
        for a,b in combinations(selected,2):
            sim=lagset_similarity(lagsets[v][a],lagsets[v][b],h,kind)
            if np.isfinite(sim):
                pw=method_weights[a]*method_weights[b]; num+=pw*sim; den+=pw; pair_sims.append(sim)
        d_l=np.nan if den<=0 else float(1-num/den)
        rows.append({'cause':v,'R_V':p,'D_V':d_v,'D_L_variable':d_l,'n_support_methods':len(selected),
                     'mean_pair_similarity':np.nan if not pair_sims else float(np.mean(pair_sims))})
    return pd.DataFrame(rows)

# %%
# Synthetic stress scenarios
@dataclass(frozen=True)
class Scenario:
    name: str
    n_samples: int = 500
    burn_in: int = 350
    rho_x: float = 0.30
    rho_y: float = 0.30
    rho_noise: float = 0.25
    noise_y: float = 0.80
    beta1: float = 0.80
    beta2: float = 0.60
    lag1: int = 2
    lag2: int = 4
    x3_corr: float = 0.0
    n_extra_noise: int = 0
    nonlinear: bool = False
    noise_dist: str = 'laplace'   # v0.7 default: non-Gaussian
    note: str = ''

def _unit_variance_noise(rng, size, kind):
    if kind == 'laplace':
        # Laplace variance = 2*b^2; b=1/sqrt(2) -> variance 1
        return rng.laplace(0.0, 1.0/np.sqrt(2.0), size=size)
    if kind == 'gaussian':
        return rng.normal(0.0, 1.0, size=size)
    raise ValueError("noise_dist must be 'laplace' or 'gaussian'")

def generate_stress_series(s,seed):
    rng=np.random.default_rng(seed)
    max_needed=max(CFG.max_lag,s.lag1,s.lag2)
    n=s.n_samples+s.burn_in+max_needed+10
    n_eps=5+s.n_extra_noise
    e=_unit_variance_noise(rng,(n,n_eps),s.noise_dist)

    x1=np.zeros(n); x2=np.zeros(n); x3=np.zeros(n); x4=np.zeros(n); y=np.zeros(n)
    extras=[np.zeros(n) for _ in range(s.n_extra_noise)]

    for t in range(1,n):
        x1[t]=s.rho_x*x1[t-1]+e[t,0]
        x2[t]=s.rho_x*x2[t-1]+e[t,1]
        x3[t]=s.rho_x*x3[t-1]+(s.x3_corr*x1[t-1] if s.x3_corr>0 else 0.0)+e[t,2]
        x4[t]=s.rho_noise*x4[t-1]+e[t,3]

        for j,arr in enumerate(extras):
            arr[t]=s.rho_noise*arr[t-1]+e[t,4+j]

        if t>=max_needed:
            a=x1[t-s.lag1]
            b=x2[t-s.lag2]
            if s.nonlinear:
                term1=s.beta1*np.tanh(a)
                term2=s.beta2*np.sign(b)*(b*b)/(1.0+abs(b))
            else:
                term1=s.beta1*a
                term2=s.beta2*b

            y_noise_col=4+s.n_extra_noise
            y[t]=s.rho_y*y[t-1]+term1+term2+s.noise_y*e[t,y_noise_col]

    start=s.burn_in+max_needed+5
    data={
        'X1':x1[start:start+s.n_samples],
        'X2':x2[start:start+s.n_samples],
        'X3':x3[start:start+s.n_samples],
        'X4':x4[start:start+s.n_samples]
    }
    for j,arr in enumerate(extras,start=5):
        data[f'X{j}']=arr[start:start+s.n_samples]
    data['Y']=y[start:start+s.n_samples]

    df=pd.DataFrame(data)
    df=(df-df.mean())/df.std(ddof=0).replace(0,1)

    gt={('X1',s.lag1),('X2',s.lag2)}
    return df,gt

SCENARIOS_ALL=[
    Scenario('S1_clean',note='nazorat uchun nisbatan oson chiziqli Laplace-noise holat'),
    Scenario('S2_tiny_weak',n_samples=110,noise_y=2.0,beta1=0.45,beta2=0.35,
             note='juda kichik tanlanma + kuchsiz ta’sir'),
    Scenario('S3_noisy_weak',n_samples=180,noise_y=3.0,beta1=0.40,beta2=0.30,
             note='kuchli Laplace shovqin + kuchsiz ta’sir'),
    Scenario('S4_extreme_autocorr',n_samples=180,rho_x=0.95,rho_y=0.90,noise_y=1.8,
             beta1=0.50,beta2=0.40,note='juda kuchli avtokorrelyatsiya'),
    Scenario('S5_correlated_distractor',n_samples=180,x3_corr=0.95,noise_y=1.8,
             beta1=0.45,beta2=0.35,note='X3 X1 bilan kuchli bog‘langan, lekin Y ning bevosita sababi emas'),
    Scenario('S6_high_dimensional',n_samples=160,n_extra_noise=6,rho_x=0.65,rho_noise=0.55,
             noise_y=2.0,beta1=0.45,beta2=0.35,note='kichikroq N + ko‘p ahamiyatsiz o‘zgaruvchilar'),
    Scenario('S7_mixed_extreme',n_samples=130,rho_x=0.90,rho_y=0.80,x3_corr=0.90,
             n_extra_noise=4,noise_y=2.8,beta1=0.40,beta2=0.30,
             note='kichik N + autocorr + distractor + shovqin + kuchsiz ta’sir'),
    Scenario('S8_nonlinear',n_samples=220,nonlinear=True,n_extra_noise=2,noise_y=1.5,
             beta1=0.80,beta2=0.60,note='nochiziqli sababli ta’sirlar'),
    Scenario('S9_gaussian_assumption_violation',n_samples=220,noise_y=1.5,
             beta1=0.60,beta2=0.45,noise_dist='gaussian',
             note='VAR-LiNGAM non-Gaussian farazi ataylab buzilgan Gaussian ssenariy'),
]

if RUN_MODE == 'smoke':
    wanted={'S1_clean','S5_correlated_distractor'}
    SCENARIOS=[s for s in SCENARIOS_ALL if s.name in wanted]
    SEEDS=[11]
elif RUN_MODE == 'fast':
    wanted={'S1_clean','S2_tiny_weak','S4_extreme_autocorr','S5_correlated_distractor',
            'S9_gaussian_assumption_violation'}
    SCENARIOS=[s for s in SCENARIOS_ALL if s.name in wanted]
    SEEDS=[11,22]
elif RUN_MODE == 'paper':
    SCENARIOS=SCENARIOS_ALL
    SEEDS=list(range(101,111))
elif RUN_MODE == 'full':
    SCENARIOS=SCENARIOS_ALL
    SEEDS=list(range(101,121))
else:
    raise ValueError("RUN_MODE must be 'smoke', 'fast', 'paper', or 'full'")

print('Ssenariylar:',[s.name for s in SCENARIOS])
print('Seedlar:',SEEDS)
print('Dataset runlar:',len(SCENARIOS)*len(SEEDS))
print('Har datasetdagi algoritmlar:',EXPECTED_METHODS)
print('Noise:', {s.name:s.noise_dist for s in SCENARIOS})

# %%
# Metrics and baseline aggregations
def edge_set_from_method(edges,method):
    x=edges[(edges['method']==method)&(edges['selected'])]
    return set(zip(x['cause'],x['lag'].astype(int)))

def binary_metrics(pred_set,true_set,universe):
    y_true=[1 if x in true_set else 0 for x in sorted(universe)]; y_pred=[1 if x in pred_set else 0 for x in sorted(universe)]
    p,r,f,_=precision_recall_fscore_support(y_true,y_pred,average='binary',zero_division=0)
    return {'Precision':float(p),'Recall':float(r),'F1':float(f),
            'TP':sum(x in pred_set and x in true_set for x in universe),
            'FP':sum(x in pred_set and x not in true_set for x in universe),
            'FN':sum(x not in pred_set and x in true_set for x in universe)}

def variable_metrics(pred_set,true_set,variables): return binary_metrics({v for v,_ in pred_set},{v for v,_ in true_set},set(variables))

def nearest_lag_mae(pred_set,true_set):
    errors=[]
    for v,tau_true in true_set:
        cands=[tau for vv,tau in pred_set if vv==v]
        if cands: errors.append(min(abs(tau-tau_true) for tau in cands))
    return float(np.mean(errors)) if errors else np.nan

def exact_majority_pred(edges,methods,threshold=0.5):
    if not methods: return set()
    g=edges[edges['selected']].groupby(['cause','lag'])['method'].nunique()/len(methods)
    return {(v,int(lag)) for (v,lag),support in g.items() if support>threshold}

def union_pred(edges):
    x=edges[edges['selected']]; return set(zip(x['cause'],x['lag'].astype(int)))

def pairwise_jaccard_disagreement(edge_sets):
    vals=[]; names=list(edge_sets)
    for a,b in combinations(names,2):
        A=edge_sets[a]; B=edge_sets[b]; vals.append(0.0 if not A and not B else 1-len(A&B)/len(A|B))
    return float(np.mean(vals)) if vals else np.nan

# %%
# Caching and single-run evaluation
def stress_cache_key(s,seed,method):
    payload={
        'scenario':asdict(s),
        'seed':seed,
        'method':method,
        'cfg':asdict(CFG),
        'schema':CACHE_SCHEMA,
        'version':NOTEBOOK_VERSION
    }
    return hashlib.sha256(
        json.dumps(payload,sort_keys=True,default=str).encode()
    ).hexdigest()[:24]

def cache_file(s,seed,method):
    safe=method.lower().replace('+','plus').replace('-','_').replace('/','_')
    return DIRS['cache']/f'{s.name}_{seed}_{safe}_{stress_cache_key(s,seed,method)}.csv'

def run_stack_for_dataset(df,s,seed):
    all_edges=[]
    status={}
    times={}

    for m in EXPECTED_METHODS:
        path=cache_file(s,seed,m)

        if path.exists() and not FORCE_RECOMPUTE:
            try:
                e=normalize_edges(pd.read_csv(path))
                status[m]='OK'
                times[m]=0.0
                all_edges.append(e)
                continue
            except Exception:
                pass

        t0=time.perf_counter()
        try:
            if m=='PCMCI+':
                e=run_pcmciplus(df,CFG.target,CFG.max_lag,CFG.pcmci_pc_alpha)
            elif m=='PCMCI':
                e=run_pcmci(df,CFG.target,CFG.max_lag,CFG.pcmci_pc_alpha)
            elif m=='LPCMCI':
                e=run_lpcmci(df,CFG.target,CFG.max_lag,CFG.pcmci_pc_alpha)
            elif m=='VAR-LiNGAM':
                e,_=run_varlingam(
                    df,CFG.target,CFG.max_lag,
                    CFG.varlingam_coef_threshold,
                    CFG.varlingam_prune,
                    seed
                )
            elif m=='DYNOTEARS':
                e,_=run_dynotears(df,CFG.target,CFG.max_lag,CFG)
            else:
                raise ValueError(f'Unknown method: {m}')

            e=normalize_edges(e)
            status[m]='OK'
            e.to_csv(path,index=False)
            all_edges.append(e)
        except Exception as ex:
            status[m]=f'ERROR: {type(ex).__name__}: {ex}'

        times[m]=time.perf_counter()-t0

    edges=normalize_edges(pd.concat(all_edges,ignore_index=True)) if all_edges else empty_edges()
    return edges,status,times

def variable_set_metrics(pred_vars,true_vars,variables):
    return binary_metrics(set(pred_vars),set(true_vars),set(variables))

def variable_reliability_comparison(edges,variables,active_family,gt,scenario,seed):
    rv_family,P,meta_family=compute_rv(
        edges,variables,active_family,'family_normalized'
    )
    rv_equal,_,meta_equal=compute_rv(
        edges,variables,active_family,'equal_method'
    )

    true_vars={v for v,_ in gt}
    eps=1e-12

    strategies={
        'FamilyNormalized_RV':{
            v for v in variables
            if rv_family.get(v,0.0)+eps>=CFG.theta_v
        },
        'EqualMethod_sameTheta':{
            v for v in variables
            if rv_equal.get(v,0.0)+eps>=CFG.theta_v
        },
        'SimpleMajority_0.5':{
            v for v in variables
            if rv_equal.get(v,0.0)>0.5+eps
        },
        'Union_variable':{
            v for v in variables
            if rv_equal.get(v,0.0)>eps
        },
        'Intersection_variable':{
            v for v in variables
            if rv_equal.get(v,0.0)>=1.0-eps
        }
    }

    rows=[]
    for name,pred in strategies.items():
        met=variable_set_metrics(pred,true_vars,variables)
        rows.append({
            'Scenario':scenario,
            'Seed':seed,
            'Strategy':name,
            **met,
            'N_selected_variables':len(pred)
        })

    scores=[]
    for v in variables:
        support_methods=[
            m for m in active_family
            if P.loc[v,m]>0
        ]
        support_families=sorted({
            active_family[m] for m in support_methods
        })
        scores.append({
            'Scenario':scenario,
            'Seed':seed,
            'Variable':v,
            'TrueVariable':v in true_vars,
            'R_V_family':float(rv_family.get(v,0.0)),
            'R_V_equal_method':float(rv_equal.get(v,0.0)),
            'Delta_family_minus_equal':float(
                rv_family.get(v,0.0)-rv_equal.get(v,0.0)
            ),
            'FamilySelected':bool(
                rv_family.get(v,0.0)+eps>=CFG.theta_v
            ),
            'EqualSelectedSameTheta':bool(
                rv_equal.get(v,0.0)+eps>=CFG.theta_v
            ),
            'SimpleMajoritySelected':bool(
                rv_equal.get(v,0.0)>0.5+eps
            ),
            'MethodSupportCount':len(support_methods),
            'FamilySupportCount':len(support_families),
            'SupportMethods':'|'.join(support_methods),
            'SupportFamilies':'|'.join(support_families),
        })

    return (
        pd.DataFrame(rows),
        pd.DataFrame(scores),
        meta_family,
        meta_equal
    )

def evaluate_one_run(s,seed,verbose=False):
    df,gt=generate_stress_series(s,seed)
    variables=[c for c in df.columns if c!=CFG.target]
    universe={(v,lag) for v in variables for lag in range(1,CFG.max_lag+1)}

    edges,status,times=run_stack_for_dataset(df,s,seed)
    active=[m for m in EXPECTED_METHODS if status.get(m)=='OK']
    active_family={m:METHOD_FAMILY[m] for m in active}

    # Reviewer-facing comparisons require the same 3:1:1 stack in every run.
    if active != EXPECTED_METHODS:
        raise RuntimeError(
            f'{s.name}/{seed}: complete 3:1:1 stack required. Status={status}'
        )

    rv,_,_=compute_rv(
        edges,variables,active_family,CFG.rv_strategy
    )
    lt,lagsets,lw=compute_lag_consensus(
        edges,variables,active,active_family,
        CFG.kernel_bandwidth,CFG.kernel,CFG.rv_strategy
    )

    sel_prop=select_temporal_features_v05(
        rv,lt,CFG.theta_v,CFG.theta_exact,
        CFG.theta_proximity,CFG.theta_margin,CFG.tie_tol
    )
    sel_exact=select_rv_exact_only(
        rv,lt,CFG.theta_v,CFG.theta_exact
    )

    pred_prop=set(zip(
        sel_prop.loc[sel_prop['selected'],'cause'],
        sel_prop.loc[sel_prop['selected'],'lag'].astype(int)
    ))
    pred_exact=set(zip(
        sel_exact.loc[sel_exact['selected'],'cause'],
        sel_exact.loc[sel_exact['selected'],'lag'].astype(int)
    ))

    preds={m:edge_set_from_method(edges,m) for m in active}
    preds['Union']=union_pred(edges)
    preds['ExactMajority']=exact_majority_pred(edges,active,0.5)
    preds['RV_ExactOnly']=pred_exact
    preds['Proposed_v06']=pred_prop

    rows=[]
    for method,pred in preds.items():
        met=binary_metrics(pred,gt,universe)
        vmet=variable_metrics(pred,gt,variables)
        rows.append({
            'Scenario':s.name,
            'Seed':seed,
            'Method':method,
            **met,
            'Var_Precision':vmet['Precision'],
            'Var_Recall':vmet['Recall'],
            'Var_F1':vmet['F1'],
            'Lag_MAE':nearest_lag_mae(pred,gt),
            'N_selected':len(pred),
            'Runtime_s':sum(times.values())
                        if method=='Proposed_v06'
                        else times.get(method,np.nan)
        })

    single_sets={m:preds[m] for m in active}

    if len(sel_prop):
        modes=sel_prop.loc[
            sel_prop['selected'],
            ['cause','decision_mode']
        ].drop_duplicates()
        n_prox=int((modes['decision_mode']=='proximity_consensus').sum())
        n_exact=int((modes['decision_mode']=='exact_consensus').sum())
        abstain_vars=int(sel_prop.loc[
            sel_prop['decision_mode'].isin(
                ['abstain_low_proximity','abstain_ambiguous']
            ),
            'cause'
        ].nunique())
    else:
        n_prox=n_exact=abstain_vars=0

    varcmp,rvscores,fam_meta,eq_meta=variable_reliability_comparison(
        edges,variables,active_family,gt,s.name,seed
    )

    run_meta={
        'Scenario':s.name,
        'Seed':seed,
        'N':len(df),
        'P':len(variables),
        'GroundTruth':sorted(gt),
        'ActiveMethods':active,
        'ActiveFamilies':active_family,
        'CompleteStack':active==EXPECTED_METHODS,
        'FamilyNormalizedWeights':fam_meta['method_weights'],
        'EqualMethodWeights':eq_meta['method_weights'],
        'PairwiseJaccardDisagreement':pairwise_jaccard_disagreement(single_sets),
        'RL_changed_decisions':len(pred_prop.symmetric_difference(pred_exact)),
        'ProximitySelections':n_prox,
        'ExactSelections':n_exact,
        'AbstainedVariables':abstain_vars,
        'MethodStatus':status,
        'MethodTimes':times
    }

    dis=variable_disagreement(
        rv,lagsets,active,lw,
        CFG.kernel_bandwidth,CFG.kernel
    )
    dis['Scenario']=s.name
    dis['Seed']=seed

    if verbose:
        print()
        print(s.name,'seed=',seed,'GT=',sorted(gt))
        print('status=',status)
        display(pd.DataFrame(rows).set_index('Method').round(3))
        print('Variable-level aggregation:')
        display(varcmp.round(3))
        print('R_V scores:')
        display(rvscores.round(3))
        if len(sel_prop):
            display(sel_prop[
                ['cause','lag','R_V','E_L','R_L','D_L',
                 'selected','decision_mode','reason']
            ].round(3))
        display(dis.round(3))

    return (
        pd.DataFrame(rows),
        run_meta,
        dis,
        edges,
        sel_prop,
        varcmp,
        rvscores
    )

# %%
# Full synthetic stress evaluation
all_result_frames=[]
all_meta=[]
all_dis=[]
all_varcmp=[]
all_rvscores=[]
all_edge_frames=[]
start_all=time.perf_counter()

for si,s in enumerate(SCENARIOS,1):
    print(f'[{si}/{len(SCENARIOS)}] {s.name}: {s.note}')
    for seed in SEEDS:
        try:
            r,meta,dis,edges,_,vc,vs=evaluate_one_run(
                s,seed,verbose=False
            )
            all_result_frames.append(r)
            all_meta.append(meta)
            all_dis.append(dis)
            all_varcmp.append(vc)
            all_rvscores.append(vs)

            e2=edges.copy()
            e2['Scenario']=s.name
            e2['Seed']=seed
            all_edge_frames.append(e2)

            prop=r[r['Method']=='Proposed_v06'].iloc[0]
            fam=vc[vc['Strategy']=='FamilyNormalized_RV'].iloc[0]
            maj=vc[vc['Strategy']=='SimpleMajority_0.5'].iloc[0]

            print(
                f'  seed={seed}: temporal F1={prop.F1:.3f}; '
                f'VarF1 family={fam.F1:.3f}, majority={maj.F1:.3f}; '
                f'JaccardD={meta["PairwiseJaccardDisagreement"]:.3f}; '
                f'RL-diff={meta["RL_changed_decisions"]}'
            )
        except Exception as ex:
            print(
                f'  seed={seed}: ERROR -> '
                f'{type(ex).__name__}: {ex}'
            )

RESULTS=pd.concat(
    all_result_frames,ignore_index=True
) if all_result_frames else pd.DataFrame()

DISAGREEMENT=pd.concat(
    all_dis,ignore_index=True
) if all_dis else pd.DataFrame()

VARIABLE_ABLATION=pd.concat(
    all_varcmp,ignore_index=True
) if all_varcmp else pd.DataFrame()

RV_SCORES=pd.concat(
    all_rvscores,ignore_index=True
) if all_rvscores else pd.DataFrame()

EDGE_RUNS=pd.concat(
    all_edge_frames,ignore_index=True
) if all_edge_frames else pd.DataFrame()

META=pd.DataFrame(all_meta)

print()
print(f'Jami vaqt: {time.perf_counter()-start_all:.1f} s')
print('Temporal result rows:',len(RESULTS))
print('Variable ablation rows:',len(VARIABLE_ABLATION))
print('R_V score rows:',len(RV_SCORES))
print('Cached edge rows for v0.7 ablations:',len(EDGE_RUNS))

# %%
# Scenario summaries
if len(RESULTS):
    SUMMARY=(RESULTS.groupby(['Scenario','Method']).agg(
        F1_mean=('F1','mean'),
        F1_std=('F1','std'),
        Precision_mean=('Precision','mean'),
        Recall_mean=('Recall','mean'),
        Var_F1_mean=('Var_F1','mean'),
        Lag_MAE_mean=('Lag_MAE','mean'),
        N_selected_mean=('N_selected','mean')
    ).reset_index())
    display(SUMMARY.round(3))
else:
    SUMMARY=pd.DataFrame()

print('\n=== REVIEWER UCHUN VARIABLE-LEVEL COMPARISON ===')
if len(VARIABLE_ABLATION):
    VAR_SUMMARY=(VARIABLE_ABLATION.groupby(
        ['Scenario','Strategy']
    ).agg(
        F1_mean=('F1','mean'),
        F1_std=('F1','std'),
        Precision_mean=('Precision','mean'),
        Recall_mean=('Recall','mean'),
        N_selected_mean=('N_selected_variables','mean')
    ).reset_index())
    display(VAR_SUMMARY.round(3))
else:
    VAR_SUMMARY=pd.DataFrame()

# %%
# Overall summaries
if len(RESULTS):
    OVERALL=(RESULTS.groupby('Method').agg(
        F1_mean=('F1','mean'),
        F1_std=('F1','std'),
        Precision_mean=('Precision','mean'),
        Recall_mean=('Recall','mean'),
        Var_F1_mean=('Var_F1','mean'),
        Lag_MAE_mean=('Lag_MAE','mean'),
        N_selected_mean=('N_selected','mean')
    ).sort_values('F1_mean',ascending=False))

    display(OVERALL.round(3))

    if SHOW_PLOTS:
        fig, ax = plt.subplots(figsize=(12,5))

        order = OVERALL.index.tolist()
        vals = OVERALL.loc[order,'F1_mean']
        errs = OVERALL.loc[order,'F1_std'].fillna(0)

        bars = ax.bar(
            order,
            vals,
            yerr=errs,
            capsize=3
        )

        ax.bar_label(
            bars,
            labels=[f'{v:.3f}' for v in vals],
            padding=3,
            fontsize=9
        )

        ax.set_ylim(0, 1.12)
        ax.set_ylabel('Temporal F1')
        ax.set_title('v0.6 stress-testlar bo‘yicha o‘rtacha temporal F1')
        ax.tick_params(axis='x', rotation=35)

        plt.tight_layout()
        plt.show()

else:
    OVERALL = pd.DataFrame()


if len(VARIABLE_ABLATION):
    VAR_OVERALL=(VARIABLE_ABLATION.groupby('Strategy').agg(
        F1_mean=('F1','mean'),
        F1_std=('F1','std'),
        Precision_mean=('Precision','mean'),
        Recall_mean=('Recall','mean'),
        N_selected_mean=('N_selected_variables','mean')
    ).sort_values('F1_mean',ascending=False))

    print('\nVariable-level overall:')
    display(VAR_OVERALL.round(3))

    if SHOW_PLOTS:
        fig, ax = plt.subplots(figsize=(10,5))

        order = VAR_OVERALL.index.tolist()
        vals = VAR_OVERALL.loc[order,'F1_mean']
        errs = VAR_OVERALL.loc[order,'F1_std'].fillna(0)

        bars = ax.bar(
            order,
            vals,
            yerr=errs,
            capsize=3
        )

        ax.bar_label(
            bars,
            labels=[f'{v:.3f}' for v in vals],
            padding=3,
            fontsize=9
        )

        ax.set_ylim(0, 1.12)
        ax.set_ylabel('Variable-level F1')
        ax.set_title('Family-normalized R_V vs baselines')
        ax.tick_params(axis='x', rotation=30)

        plt.tight_layout()
        plt.show()

else:
    VAR_OVERALL = pd.DataFrame()

# %%
# Legacy lag-ablation diagnostic
if len(RESULTS):
    a=RESULTS[RESULTS['Method']=='Proposed_v06'][['Scenario','Seed','F1','Precision','Recall']].rename(columns={'F1':'F1_prop','Precision':'P_prop','Recall':'R_prop'})
    b=RESULTS[RESULTS['Method']=='RV_ExactOnly'][['Scenario','Seed','F1','Precision','Recall']].rename(columns={'F1':'F1_noRL','Precision':'P_noRL','Recall':'R_noRL'})
    ABL=a.merge(b,on=['Scenario','Seed'],how='inner'); ABL['Delta_F1']=ABL['F1_prop']-ABL['F1_noRL']
    if len(META): ABL=ABL.merge(META[['Scenario','Seed','RL_changed_decisions','ProximitySelections','AbstainedVariables','PairwiseJaccardDisagreement']],on=['Scenario','Seed'],how='left')
    display(ABL.round(3))
    changed=int((ABL['RL_changed_decisions'].fillna(0)>0).sum()); rescued=int((ABL['ProximitySelections'].fillna(0)>0).sum())
    print('R_L yakuniy qarorni o‘zgartirgan runlar:',changed,'/',len(ABL))
    print('Yaqinlik bo‘yicha tanlash ishga tushgan runlar:',rescued,'/',len(ABL))
    print('O‘rtacha ΔF1:',round(float(ABL['Delta_F1'].mean()),4))
    if changed==0: print('DIQQAT: real stress ssenariylarida ham R_L qarorni o‘zgartirmadi; 2-gipotezani qayta ko‘rish kerak.')
else: ABL=pd.DataFrame()

# %%
# Controlled lag-agreement tests
def fake_edges_for_lags(lags_by_method,cause='X1',target='Y'):
    rows=[]
    for m,lags in lags_by_method.items():
        for tau in lags:
            rows.append({
                'method':m,
                'family':METHOD_FAMILY[m],
                'cause':cause,
                'target':target,
                'lag':tau,
                'score':1.0,
                'p_value':np.nan,
                'selected':True
            })
    return normalize_edges(pd.DataFrame(rows))

def controlled_case(name,lags_by_method):
    methods=list(lags_by_method)
    mf={m:METHOD_FAMILY[m] for m in methods}
    e=fake_edges_for_lags(lags_by_method)

    rv,_,_=compute_rv(e,['X1'],mf,CFG.rv_strategy)
    lt,_,_=compute_lag_consensus(
        e,['X1'],methods,mf,
        CFG.kernel_bandwidth,CFG.kernel,CFG.rv_strategy
    )
    sel=select_temporal_features_v05(
        rv,lt,CFG.theta_v,CFG.theta_exact,
        CFG.theta_proximity,CFG.theta_margin,CFG.tie_tol
    )
    chosen=set(zip(
        sel.loc[sel['selected'],'cause'],
        sel.loc[sel['selected'],'lag'].astype(int)
    ))

    print()
    print(name,'lags=',lags_by_method)
    print('Tanlandi:',sorted(chosen))
    display(sel[
        ['cause','lag','R_V','E_L','R_L','D_L',
         'proximity_gap','selected','decision_mode','reason']
    ].round(3))
    return chosen,sel

A='PCMCI+'
B='VAR-LiNGAM'
C='DYNOTEARS'

C1,_=controlled_case('C1_UNANIMOUS',{A:[2],B:[2],C:[2]})
C2,_=controlled_case('C2_EXACT_MAJORITY_NEAR',{A:[2],B:[3],C:[2]})
C3,_=controlled_case('C3_EXACT_MAJORITY_FAR',{A:[2],B:[5],C:[2]})
C4,_=controlled_case('C4_PROXIMITY_CENTER',{A:[2],B:[3],C:[4]})
C5,_=controlled_case('C5_DISPERSED',{A:[1],B:[3],C:[5]})
C6,_=controlled_case('C6_TWO_METHOD_TIE',{A:[2],B:[3],C:[]})
C7,_=controlled_case('C7_TWO_TRUE_LAGS_ALL_METHODS',{A:[2,3],B:[2,3],C:[2,3]})

assert C1=={('X1',2)}
assert C2=={('X1',2)}
assert C3=={('X1',2)}
assert C4=={('X1',3)}
assert C5==set()
assert C6==set()
assert C7=={('X1',2),('X1',3)}

print('Lag-consensus nazorat testlari: OK')

# %%
# Disagreement summaries
if len(DISAGREEMENT):
    DIS_SUM=DISAGREEMENT.groupby('Scenario').agg(D_V_mean=('D_V','mean'),D_L_mean=('D_L_variable','mean'),support_mean=('n_support_methods','mean')).reset_index()
    display(DIS_SUM.round(3))
if len(META):
    display(META[['Scenario','Seed','P','PairwiseJaccardDisagreement','RL_changed_decisions','ProximitySelections','AbstainedVariables']].round(3))
    if len(DISAGREEMENT):
        d_run=DISAGREEMENT.groupby(['Scenario','Seed']).agg(D_V_run=('D_V','mean'),D_L_run=('D_L_variable','mean')).reset_index()
        diag=META.merge(d_run,on=['Scenario','Seed'],how='left')
        for col in ['D_V_run','D_L_run']:
            tmp=diag[[col,'PairwiseJaccardDisagreement']].dropna()
            if len(tmp)>=5 and tmp[col].nunique()>1:
                rho,p=spearmanr(tmp[col],tmp['PairwiseJaccardDisagreement'])
                print(f'Spearman({col}, JaccardDisagreement): rho={rho:.3f}, p={p:.4g}, n={len(tmp)}')

# %%
# Paired synthetic statistical tests
def paired_wilcoxon_table(results,proposed='Proposed_v06'):
    if len(results)==0:
        return pd.DataFrame()
    piv=results.pivot_table(
        index=['Scenario','Seed'],
        columns='Method',
        values='F1',
        aggfunc='first'
    )
    rows=[]
    if proposed not in piv.columns:
        return pd.DataFrame()

    for m in piv.columns:
        if m==proposed:
            continue
        pair=piv[[proposed,m]].dropna()
        diff=pair[proposed]-pair[m]
        if len(pair)<5 or np.allclose(diff,0):
            stat=p=np.nan
        else:
            try:
                stat,p=wilcoxon(
                    pair[proposed],
                    pair[m],
                    alternative='two-sided',
                    zero_method='wilcox'
                )
            except Exception:
                stat=p=np.nan

        rows.append({
            'Baseline':m,
            'N_pairs':len(pair),
            'Mean_Delta_F1':float(diff.mean()) if len(diff) else np.nan,
            'Median_Delta_F1':float(diff.median()) if len(diff) else np.nan,
            'Wins':int((diff>0).sum()),
            'Ties':int(np.isclose(diff,0).sum()),
            'Losses':int((diff<0).sum()),
            'Wilcoxon_stat':stat,
            'p_value':p
        })
    return pd.DataFrame(rows).sort_values(
        'Mean_Delta_F1',ascending=False
    )

def paired_variable_table(var_results,proposed='FamilyNormalized_RV'):
    if len(var_results)==0:
        return pd.DataFrame()

    piv=var_results.pivot_table(
        index=['Scenario','Seed'],
        columns='Strategy',
        values='F1',
        aggfunc='first'
    )
    rows=[]
    if proposed not in piv.columns:
        return pd.DataFrame()

    for m in piv.columns:
        if m==proposed:
            continue
        pair=piv[[proposed,m]].dropna()
        diff=pair[proposed]-pair[m]

        if len(pair)<5 or np.allclose(diff,0):
            stat=p=np.nan
        else:
            try:
                stat,p=wilcoxon(
                    pair[proposed],
                    pair[m],
                    alternative='two-sided',
                    zero_method='wilcox'
                )
            except Exception:
                stat=p=np.nan

        rows.append({
            'Baseline':m,
            'N_pairs':len(pair),
            'Mean_Delta_F1':float(diff.mean()) if len(diff) else np.nan,
            'Median_Delta_F1':float(diff.median()) if len(diff) else np.nan,
            'Wins':int((diff>0).sum()),
            'Ties':int(np.isclose(diff,0).sum()),
            'Losses':int((diff<0).sum()),
            'Wilcoxon_stat':stat,
            'p_value':p
        })

    return pd.DataFrame(rows).sort_values(
        'Mean_Delta_F1',ascending=False
    )

STAT_TESTS=paired_wilcoxon_table(RESULTS)
VAR_STAT_TESTS=paired_variable_table(VARIABLE_ABLATION)

print('Temporal paired tests:')
display(STAT_TESTS.round(4))
print('Variable-level paired tests:')
display(VAR_STAT_TESTS.round(4))

if FAST_MODE:
    print('Eslatma: smoke/fast rejim statistik xulosa uchun emas.')

# %%
# Controlled 3:1:1 family-imbalance experiment
# Controlled 3:1:1 family-imbalance test

def make_controlled_edges(support_methods, variable='X_test', lag=1):
    rows = []
    for method in EXPECTED_METHODS:
        if method in support_methods:
            rows.append({
                'method': method,
                'family': METHOD_FAMILY[method],
                'cause': variable,
                'target': CFG.target,
                'lag': lag,
                'score': 1.0,
                'p_value': np.nan,
                'selected': True
            })
    return normalize_edges(pd.DataFrame(rows))

controlled_cases = [
    {
        'Case': 'Redundant-family spurious',
        'Variable': 'X_spurious',
        'SupportMethods': ['PCMCI+', 'PCMCI', 'LPCMCI'],
        'ExpectedMeaning':
            'Three redundant CI-family methods support a spurious variable'
    },
    {
        'Case': 'Cross-family true',
        'Variable': 'X_true',
        'SupportMethods': ['PCMCI+', 'VAR-LiNGAM', 'DYNOTEARS'],
        'ExpectedMeaning':
            'Support is distributed across three distinct families'
    }
]

_controlled_rows = []

for case in controlled_cases:
    variable = case['Variable']
    edges_case = make_controlled_edges(
        support_methods=case['SupportMethods'],
        variable=variable,
        lag=1
    )

    rv_family, _, _ = compute_rv(
        edges_case,
        variables=[variable],
        method_family=METHOD_FAMILY,
        strategy='family_normalized'
    )

    rv_equal, _, _ = compute_rv(
        edges_case,
        variables=[variable],
        method_family=METHOD_FAMILY,
        strategy='equal_method'
    )

    r_family = float(rv_family.loc[variable])
    r_equal = float(rv_equal.loc[variable])

    _controlled_rows.append({
        'Case': case['Case'],
        'Variable': variable,
        'SupportMethods': '|'.join(case['SupportMethods']),
        'N_support_methods': len(case['SupportMethods']),
        'R_V_family': r_family,
        'R_V_equal': r_equal,
        'FamilySelected_theta_2_3':
            r_family + 1e-12 >= CFG.theta_v,
        'EqualSelected_theta_2_3':
            r_equal + 1e-12 >= CFG.theta_v,
        'SimpleMajoritySelected_0_5':
            r_equal > 0.5 + 1e-12,
        'ExpectedMeaning': case['ExpectedMeaning']
    })

FAMILY_IMBALANCE_CONTROLLED = pd.DataFrame(_controlled_rows)

print('Controlled 3:1:1 family-imbalance test:')
display(FAMILY_IMBALANCE_CONTROLLED.round(3))

_spurious = FAMILY_IMBALANCE_CONTROLLED[
    FAMILY_IMBALANCE_CONTROLLED['Case'] == 'Redundant-family spurious'
].iloc[0]

_crossfam = FAMILY_IMBALANCE_CONTROLLED[
    FAMILY_IMBALANCE_CONTROLLED['Case'] == 'Cross-family true'
].iloc[0]

# Exact mathematical checks
assert np.isclose(_spurious['R_V_family'], 1/3)
assert np.isclose(_spurious['R_V_equal'], 3/5)
assert np.isclose(_crossfam['R_V_family'], 7/9)
assert np.isclose(_crossfam['R_V_equal'], 3/5)

print(
    'Redundant-family spurious:',
    f"family={_spurious['R_V_family']:.3f},",
    f"equal={_spurious['R_V_equal']:.3f}"
)
print(
    'Cross-family true:',
    f"family={_crossfam['R_V_family']:.3f},",
    f"equal={_crossfam['R_V_equal']:.3f}"
)

if SHOW_PLOTS:
    plot_df = FAMILY_IMBALANCE_CONTROLLED.set_index('Case')[
        ['R_V_family', 'R_V_equal']
    ]

    ax = plot_df.plot(kind='bar', figsize=(10, 5))

    for bar in ax.patches:
        height = bar.get_height()
        if np.isfinite(height):
            ax.annotate(
                f'{height:.3f}',
                xy=(bar.get_x()+bar.get_width()/2, height),
                xytext=(0, 5),
                textcoords='offset points',
                ha='center',
                va='bottom',
                fontsize=9,
                fontweight='bold'
            )

    ax.axhline(
        CFG.theta_v,
        linestyle='--',
        linewidth=1.5,
        label=r'$\theta_V=2/3$'
    )
    ax.axhline(
        0.5,
        linestyle=':',
        linewidth=1.5,
        label='simple majority threshold'
    )

    ax.set_ylim(0, 1.12)
    ax.set_ylabel('Variable-level support score')
    ax.set_xlabel('Case')
    ax.set_title('Controlled 3:1:1 family-imbalance test')
    ax.legend()

    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()

    fig_path = DIRS['figures'] / 'family_imbalance_3_1_1.png'
    plt.savefig(fig_path, dpi=220, bbox_inches='tight')
    plt.show()

    print('Figure saved:', fig_path)

# %%
# Consistency checks
# v0.8 FINAL consistency checks

for k in ['exponential','gaussian','triangular']:
    assert abs(lag_kernel(3,3,1.2,k)-1.0) < 1e-12
    assert 0 <= lag_kernel(2,5,1.2,k) <= 1
    assert abs(
        lag_kernel(2,5,1.2,k) - lag_kernel(5,2,1.2,k)
    ) < 1e-12

if 'RESULTS' in globals() and isinstance(RESULTS, pd.DataFrame) and len(RESULTS):
    for c in [
        'Precision','Recall','F1',
        'Var_Precision','Var_Recall','Var_F1'
    ]:
        assert RESULTS[c].dropna().between(0,1).all(), c
    assert (RESULTS['N_selected'] >= 0).all()

if (
    'VARIABLE_ABLATION' in globals()
    and isinstance(VARIABLE_ABLATION, pd.DataFrame)
    and len(VARIABLE_ABLATION)
):
    for c in ['Precision','Recall','F1']:
        assert VARIABLE_ABLATION[c].dropna().between(0,1).all(), c

if 'RV_SCORES' in globals() and isinstance(RV_SCORES, pd.DataFrame) and len(RV_SCORES):
    assert RV_SCORES['R_V_family'].between(-1e-12,1+1e-12).all()
    assert RV_SCORES['R_V_equal_method'].between(-1e-12,1+1e-12).all()

if (
    'DISAGREEMENT' in globals()
    and isinstance(DISAGREEMENT, pd.DataFrame)
    and len(DISAGREEMENT)
):
    assert DISAGREEMENT['D_V'].dropna().between(0,1).all()
    assert DISAGREEMENT['D_L_variable'].dropna().between(0,1).all()

# Equal family prior must sum to one.
_families = sorted(set(METHOD_FAMILY.values()))
family_mass = {f: 1.0/len(_families) for f in _families}

assert abs(sum(family_mass.values()) - 1.0) < 1e-12
assert all(0 <= v <= 1 for v in family_mass.values())

# Reviewer-critical controlled experiment must exist and match theory.
assert 'FAMILY_IMBALANCE_CONTROLLED' in globals()
assert len(FAMILY_IMBALANCE_CONTROLLED) == 2

_sp = FAMILY_IMBALANCE_CONTROLLED[
    FAMILY_IMBALANCE_CONTROLLED['Case'] == 'Redundant-family spurious'
].iloc[0]
_cf = FAMILY_IMBALANCE_CONTROLLED[
    FAMILY_IMBALANCE_CONTROLLED['Case'] == 'Cross-family true'
].iloc[0]

assert np.isclose(_sp['R_V_family'], 1/3)
assert np.isclose(_sp['R_V_equal'], 3/5)
assert np.isclose(_cf['R_V_family'], 7/9)
assert np.isclose(_cf['R_V_equal'], 3/5)

# Old lag-consensus diagnostics are checked only if they were executed.
if all(
    name in globals()
    for name in ['C1','C2','C3','C4','C5','C6','C7']
):
    assert C1 == {('X1',2)}
    assert C2 == {('X1',2)}
    assert C3 == {('X1',2)}
    assert C4 == {('X1',3)}
    assert C5 == set()
    assert C6 == set()
    assert C7 == {('X1',2),('X1',3)}

print('Barcha v0.8 FINAL consistency checks: OK')

# %%
# Family-composition ablation
COMPOSITIONS = {
    '1:1:1':['PCMCI+','VAR-LiNGAM','DYNOTEARS'],
    '2:1:1':['PCMCI+','PCMCI','VAR-LiNGAM','DYNOTEARS'],
    '3:1:1':['PCMCI+','PCMCI','LPCMCI','VAR-LiNGAM','DYNOTEARS'],
}

scenario_map={s.name:s for s in SCENARIOS_ALL}
comp_rows=[]

if len(EDGE_RUNS):
    for (scenario,seed),g in EDGE_RUNS.groupby(['Scenario','Seed']):
        s=scenario_map[scenario]
        df,gt=generate_stress_series(s,int(seed))
        variables=[c for c in df.columns if c!=CFG.target]
        true_vars={v for v,_ in gt}

        for comp_name,methods in COMPOSITIONS.items():
            mf={m:METHOD_FAMILY[m] for m in methods}
            eg=g[g['method'].isin(methods)].copy()

            rv_family,_,_=compute_rv(eg,variables,mf,'family_normalized')
            rv_equal,_,_=compute_rv(eg,variables,mf,'equal_method')

            strategies={
                'FamilyNormalized_RV':{v for v in variables if rv_family.get(v,0)+1e-12>=CFG.theta_v},
                'EqualMethod_sameTheta':{v for v in variables if rv_equal.get(v,0)+1e-12>=CFG.theta_v},
                'SimpleMajority_0.5':{v for v in variables if rv_equal.get(v,0)>0.5+1e-12},
                'Union_variable':{v for v in variables if rv_equal.get(v,0)>1e-12},
                'Intersection_variable':{v for v in variables if rv_equal.get(v,0)>=1-1e-12},
            }

            for strategy,pred in strategies.items():
                met=variable_set_metrics(pred,true_vars,variables)
                comp_rows.append({
                    'Scenario':scenario,'Seed':int(seed),'Composition':comp_name,
                    'Strategy':strategy,**met,'N_selected_variables':len(pred)
                })

FAMILY_COMPOSITION_ABLATION=pd.DataFrame(comp_rows)

if len(FAMILY_COMPOSITION_ABLATION):
    FAMILY_COMPOSITION_SUMMARY=(
        FAMILY_COMPOSITION_ABLATION
        .groupby(['Composition','Strategy'])
        .agg(F1_mean=('F1','mean'),F1_std=('F1','std'),
             Precision_mean=('Precision','mean'),
             Recall_mean=('Recall','mean'),
             N_selected_mean=('N_selected_variables','mean'))
        .reset_index()
    )
    display(FAMILY_COMPOSITION_SUMMARY.round(3))
else:
    FAMILY_COMPOSITION_SUMMARY=pd.DataFrame()

# %%
# Threshold and family-prior sensitivity
THETA_GRID=[0.34,0.50,0.60,2/3,0.75,0.85,1.00]

def compute_rv_with_family_priors(edges,variables,method_family,family_priors):
    methods=list(method_family)
    P=build_presence_matrix(edges,variables,methods)
    families={}
    for m in methods:
        families.setdefault(method_family[m],[]).append(m)

    raw={f:max(float(family_priors.get(f,0.0)),0.0) for f in families}
    z=sum(raw.values())
    if z<=0:
        raise ValueError('family_priors must contain positive mass')
    pi={f:raw[f]/z for f in families}

    weights={}
    for f,ms in families.items():
        for m in ms:
            weights[m]=pi[f]/len(ms)

    rv=P.mul(pd.Series(weights),axis=1).sum(axis=1).rename('R_V')
    return rv,weights,pi

theta_rows=[]
pi_rows=[]

PI_SETTINGS={
    'equal':{
        'conditional_independence':1/3,
        'non_gaussian_svar':1/3,
        'continuous_score_optimization':1/3,
    },
    'CI_heavy':{
        'conditional_independence':0.50,
        'non_gaussian_svar':0.25,
        'continuous_score_optimization':0.25,
    },
    'CI_light':{
        'conditional_independence':0.20,
        'non_gaussian_svar':0.40,
        'continuous_score_optimization':0.40,
    },
    'SVAR_heavy':{
        'conditional_independence':0.25,
        'non_gaussian_svar':0.50,
        'continuous_score_optimization':0.25,
    },
    'Score_heavy':{
        'conditional_independence':0.25,
        'non_gaussian_svar':0.25,
        'continuous_score_optimization':0.50,
    },
}

if len(EDGE_RUNS):
    for (scenario,seed),g in EDGE_RUNS.groupby(['Scenario','Seed']):
        s=scenario_map[scenario]
        df,gt=generate_stress_series(s,int(seed))
        variables=[c for c in df.columns if c!=CFG.target]
        true_vars={v for v,_ in gt}

        mf=METHOD_FAMILY.copy()
        rv_family,_,_=compute_rv(g,variables,mf,'family_normalized')

        for theta in THETA_GRID:
            pred={v for v in variables if rv_family.get(v,0)+1e-12>=theta}
            met=variable_set_metrics(pred,true_vars,variables)
            theta_rows.append({
                'Scenario':scenario,'Seed':int(seed),'theta_v':float(theta),
                **met,'N_selected_variables':len(pred)
            })

        for label,priors in PI_SETTINGS.items():
            rv,weights,pi=compute_rv_with_family_priors(g,variables,mf,priors)
            pred={v for v in variables if rv.get(v,0)+1e-12>=CFG.theta_v}
            met=variable_set_metrics(pred,true_vars,variables)
            pi_rows.append({
                'Scenario':scenario,'Seed':int(seed),'PriorSetting':label,
                **met,'N_selected_variables':len(pred),
                'Pi_CI':pi.get('conditional_independence',np.nan),
                'Pi_SVAR':pi.get('non_gaussian_svar',np.nan),
                'Pi_SCORE':pi.get('continuous_score_optimization',np.nan),
            })

THETA_SENSITIVITY=pd.DataFrame(theta_rows)
PI_SENSITIVITY=pd.DataFrame(pi_rows)

if len(THETA_SENSITIVITY):
    THETA_SUMMARY=(THETA_SENSITIVITY.groupby('theta_v')
                   .agg(F1_mean=('F1','mean'),F1_std=('F1','std'),
                        Precision_mean=('Precision','mean'),
                        Recall_mean=('Recall','mean'))
                   .reset_index())
    print('theta_V sensitivity:')
    display(THETA_SUMMARY.round(3))
else:
    THETA_SUMMARY=pd.DataFrame()

if len(PI_SENSITIVITY):
    PI_SUMMARY=(PI_SENSITIVITY.groupby('PriorSetting')
                .agg(F1_mean=('F1','mean'),F1_std=('F1','std'),
                     Precision_mean=('Precision','mean'),
                     Recall_mean=('Recall','mean'))
                .reset_index())
    print('pi_f sensitivity:')
    display(PI_SUMMARY.round(3))
else:
    PI_SUMMARY=pd.DataFrame()

# %%
# External-benchmark utilities
def _extract_tigramite_all_edges(result,cols,max_lag,method_name,alpha):
    graph=np.asarray(result['graph'])
    pmat=np.asarray(result['p_matrix'])
    vmat=np.asarray(result['val_matrix'])
    rows=[]

    for i,cause in enumerate(cols):
        for j,target in enumerate(cols):
            if i==j:
                continue
            for lag in range(1,max_lag+1):
                mark=str(graph[i,j,lag])
                p=float(pmat[i,j,lag])
                score=float(vmat[i,j,lag])
                if mark.endswith('>') and np.isfinite(p) and p<=alpha:
                    rows.append({
                        'method':method_name,'family':METHOD_FAMILY[method_name],
                        'cause':cause,'target':target,'lag':lag,
                        'score':score,'p_value':p,'selected':True
                    })
    return normalize_edges(pd.DataFrame(rows))

def run_tigramite_all(df,max_lag,method_name,pc_alpha=0.05):
    cols,tig_df=_tigramite_frame(df)

    if method_name=='LPCMCI':
        model=LPCMCI(
            dataframe=tig_df,
            cond_ind_test=ParCorr(significance='analytic'),
            verbosity=0
        )
        result=model.run_lpcmci(
            tau_min=1,tau_max=max_lag,pc_alpha=pc_alpha
        )
    else:
        model=PCMCI(
            dataframe=tig_df,
            cond_ind_test=ParCorr(significance='analytic'),
            verbosity=0
        )
        if method_name=='PCMCI+':
            result=model.run_pcmciplus(
                tau_min=1,tau_max=max_lag,
                pc_alpha=pc_alpha,fdr_method='none'
            )
        elif method_name=='PCMCI':
            result=model.run_pcmci(
                tau_min=1,tau_max=max_lag,
                pc_alpha=pc_alpha,alpha_level=pc_alpha,
                fdr_method='none'
            )
        else:
            raise ValueError(method_name)

    return _extract_tigramite_all_edges(
        result,cols,max_lag,method_name,pc_alpha
    )

def run_varlingam_all(df,max_lag,coef_threshold=0.08,prune=True,seed=42):
    cols=list(df.columns)
    model=VARLiNGAM(
        lags=max_lag,criterion=None,prune=prune,random_state=seed
    )
    model.fit(df.to_numpy(dtype=float))
    mats=np.asarray(model.adjacency_matrices_)
    rows=[]

    for lag in range(1,min(max_lag,mats.shape[0]-1)+1):
        B=np.asarray(mats[lag],dtype=float)
        for j,target in enumerate(cols):
            for i,cause in enumerate(cols):
                if i==j:
                    continue
                coef=float(B[j,i])
                if np.isfinite(coef) and abs(coef)>=coef_threshold:
                    rows.append({
                        'method':'VAR-LiNGAM','family':METHOD_FAMILY['VAR-LiNGAM'],
                        'cause':cause,'target':target,'lag':lag,
                        'score':coef,'p_value':np.nan,'selected':True
                    })
    return normalize_edges(pd.DataFrame(rows))

def run_dynotears_all(df,max_lag,cfg):
    cols=list(df.columns)
    fit=fit_dynotears_linear(
        df.to_numpy(dtype=float),p=max_lag,
        lambda_w=cfg.dynotears_lambda_w,
        lambda_a=cfg.dynotears_lambda_a,
        max_iter=cfg.dynotears_max_iter,
        h_tol=cfg.dynotears_h_tol,
        rho_max=cfg.dynotears_rho_max,
        w_threshold=cfg.dynotears_w_threshold,
        optimizer_maxiter=cfg.dynotears_optimizer_maxiter
    )
    rows=[]
    for lag,B in enumerate(fit['lag_matrices'],start=1):
        for j,target in enumerate(cols):
            for i,cause in enumerate(cols):
                if i==j:
                    continue
                coef=float(B[j,i])
                if np.isfinite(coef) and abs(coef)>=cfg.dynotears_w_threshold:
                    rows.append({
                        'method':'DYNOTEARS','family':METHOD_FAMILY['DYNOTEARS'],
                        'cause':cause,'target':target,'lag':lag,
                        'score':coef,'p_value':np.nan,'selected':True
                    })
    return normalize_edges(pd.DataFrame(rows))

def benchmark_cache_file(dataset_key,method,max_lag):
    safe=re.sub(r'[^A-Za-z0-9_.-]+','_',dataset_key)
    msafe=method.replace('+','plus').replace('-','_')
    return DIRS['benchmark_cache']/f'{safe}_L{max_lag}_{msafe}_schema{CACHE_SCHEMA}.csv'

def run_all_target_stack(df,dataset_key,max_lag,seed=42):
    all_edges=[]
    status={}
    times={}

    for method in EXPECTED_METHODS:
        path=benchmark_cache_file(dataset_key,method,max_lag)

        if path.exists() and not FORCE_RECOMPUTE:
            try:
                e=normalize_edges(pd.read_csv(path))
                status[method]='OK(cache)'
                times[method]=0.0
                all_edges.append(e)
                continue
            except Exception:
                pass

        t0=time.perf_counter()
        try:
            if method in {'PCMCI+','PCMCI','LPCMCI'}:
                e=run_tigramite_all(
                    df,max_lag,method,CFG.pcmci_pc_alpha
                )
            elif method=='VAR-LiNGAM':
                e=run_varlingam_all(
                    df,max_lag,
                    CFG.varlingam_coef_threshold,
                    CFG.varlingam_prune,seed
                )
            elif method=='DYNOTEARS':
                e=run_dynotears_all(df,max_lag,CFG)
            else:
                raise ValueError(method)

            e=normalize_edges(e)
            e.to_csv(path,index=False)
            status[method]='OK'
            all_edges.append(e)
        except Exception as ex:
            status[method]=f'ERROR: {type(ex).__name__}: {ex}'

        times[method]=time.perf_counter()-t0

    edges=normalize_edges(pd.concat(all_edges,ignore_index=True)) if all_edges else empty_edges()
    return edges,status,times

def evaluate_variable_graph(edges,variables,gt_matrix,dataset_name,theta_v=None):
    if theta_v is None:
        theta_v=CFG.theta_v

    gt=(np.asarray(gt_matrix)!=0)
    if gt.shape!=(len(variables),len(variables)):
        raise ValueError(f'GT shape {gt.shape} != {(len(variables),len(variables))}')

    np.fill_diagonal(gt,False)
    rows=[]

    for j,target in enumerate(variables):
        candidates=[v for v in variables if v!=target]
        true_vars={variables[i] for i in range(len(variables)) if gt[i,j] and i!=j}

        et=edges[edges['target']==target].copy()
        rv_family,_,_=compute_rv(et,candidates,METHOD_FAMILY,'family_normalized')
        rv_equal,_,_=compute_rv(et,candidates,METHOD_FAMILY,'equal_method')

        pred_sets={
            'FamilyNormalized_RV':{v for v in candidates if rv_family.get(v,0)+1e-12>=theta_v},
            'EqualMethod_sameTheta':{v for v in candidates if rv_equal.get(v,0)+1e-12>=theta_v},
            'SimpleMajority_0.5':{v for v in candidates if rv_equal.get(v,0)>0.5+1e-12},
            'Union_variable':{v for v in candidates if rv_equal.get(v,0)>1e-12},
            'Intersection_variable':{v for v in candidates if rv_equal.get(v,0)>=1-1e-12},
        }

        # individual methods
        for m in EXPECTED_METHODS:
            pred_sets[m]=set(
                et.loc[(et['method']==m)&(et['selected']),'cause']
            )

        for method,pred in pred_sets.items():
            met=variable_set_metrics(pred,true_vars,candidates)
            rows.append({
                'Dataset':dataset_name,'Target':target,'Strategy':method,
                **met,'N_true':len(true_vars),'N_selected':len(pred)
            })

    return pd.DataFrame(rows)

# %%
# Lorenz-96 benchmark
def lorenz96_rhs(x,F=8.0):
    x=np.asarray(x,dtype=float)
    d=len(x)
    dx=np.zeros(d,dtype=float)
    for i in range(d):
        dx[i]=(x[(i+1)%d]-x[(i-2)%d])*x[(i-1)%d]-x[i]+F
    return dx

def simulate_lorenz96(d=10,F=8.0,dt=0.01,n_samples=2500,burn_in=1500,sample_every=5,seed=42):
    rng=np.random.default_rng(seed)
    x=F+0.01*rng.normal(size=d)
    out=[]
    total=burn_in+n_samples*sample_every

    def rk4(z):
        k1=lorenz96_rhs(z,F)
        k2=lorenz96_rhs(z+0.5*dt*k1,F)
        k3=lorenz96_rhs(z+0.5*dt*k2,F)
        k4=lorenz96_rhs(z+dt*k3,F)
        return z+(dt/6.0)*(k1+2*k2+2*k3+k4)

    for t in range(total):
        x=rk4(x)
        if t>=burn_in and (t-burn_in)%sample_every==0:
            out.append(x.copy())
            if len(out)>=n_samples:
                break

    arr=np.asarray(out)
    cols=[f'L{i}' for i in range(d)]
    df=pd.DataFrame(arr,columns=cols)
    df=(df-df.mean())/df.std(ddof=0).replace(0,1)

    gt=np.zeros((d,d),dtype=int)
    for j in range(d):
        for i in ((j-2)%d,(j-1)%d,(j+1)%d):
            gt[i,j]=1
    return df,gt

LORENZ_RESULTS=pd.DataFrame()
LORENZ_STATUS={}

try:
    lor_n=1200 if RUN_MODE=='smoke' else (2000 if RUN_MODE=='fast' else 3500)
    LORENZ_DF,LORENZ_GT=simulate_lorenz96(n_samples=lor_n,seed=123)
    lor_edges,LORENZ_STATUS,lor_times=run_all_target_stack(
        LORENZ_DF,
        dataset_key=f'Lorenz96_d10_n{lor_n}_seed123',
        max_lag=1,
        seed=123
    )
    print('Lorenz method status:',LORENZ_STATUS)

    ok_methods=[m for m,s in LORENZ_STATUS.items() if str(s).startswith('OK')]
    if set(ok_methods)==set(EXPECTED_METHODS):
        LORENZ_RESULTS=evaluate_variable_graph(
            lor_edges,list(LORENZ_DF.columns),LORENZ_GT,'Lorenz96'
        )
        display(
            LORENZ_RESULTS.groupby('Strategy')
            .agg(F1_mean=('F1','mean'),Precision_mean=('Precision','mean'),
                 Recall_mean=('Recall','mean'))
            .sort_values('F1_mean',ascending=False).round(3)
        )
    else:
        print('Lorenz benchmark skipped: complete 5-method stack talab qilinadi.')
except Exception as ex:
    print('Lorenz benchmark ERROR:',type(ex).__name__,ex)

# %%
# CausalTime loader and preprocessing
CAUSALTIME_DROPBOX = (
    'https://www.dropbox.com/scl/fo/f68sxf49u8bceisd9iai3/h'
    '?rlkey=6nw2fjvuc2qzwvz11woi6hnhn&dl=1'
)

def _looks_like_causaltime_root(root):
    root=Path(root)
    return all(
        (root/d/'gen_data.npy').exists() and (root/d/'graph.npy').exists()
        for d in ['medical','pm25','traffic']
    )

def try_download_causaltime(root):
    """
    Best-effort one-time download. Agar Dropbox shared-folder ZIP bloklansa,
    datasetni rasmiy CausalTime sahifasidan bir marta yuklab,
    CAUSALTIME_ROOT ichiga ochish kifoya.
    """
    root=Path(root)
    if _looks_like_causaltime_root(root):
        print('CausalTime allaqachon mavjud:',root)
        return True

    if not DOWNLOAD_CAUSALTIME_IF_MISSING:
        return False

    import urllib.request, zipfile
    root.parent.mkdir(parents=True,exist_ok=True)
    zpath=root.parent/'causaltime_v1.zip'

    print('CausalTime yuklanmoqda...')
    try:
        urllib.request.urlretrieve(CAUSALTIME_DROPBOX,zpath)
        extract_dir=root.parent/'_causaltime_extract'
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True,exist_ok=True)

        with zipfile.ZipFile(zpath,'r') as zf:
            zf.extractall(extract_dir)

        # ichki papkani avtomatik topish
        candidates=[p for p in extract_dir.rglob('*') if p.is_dir() and _looks_like_causaltime_root(p)]
        if not candidates:
            raise FileNotFoundError('ZIP ichidan medical/pm25/traffic tuzilmasi topilmadi.')

        source=candidates[0]
        if root.exists():
            shutil.rmtree(root)
        shutil.copytree(source,root)
        print('CausalTime saqlandi:',root)
        return True
    except Exception as ex:
        print('Avtomatik download muvaffaqiyatsiz:',type(ex).__name__,ex)
        print('Rasmiy sahifa: https://www.causaltime.cc/dataset/')
        print('Datasetni bir marta quyidagi papkaga oching:',root)
        return False

def inspect_causaltime_domain(root,domain):
    root=Path(root)
    droot=root/domain
    data_path=droot/'gen_data.npy'
    graph_path=droot/'graph.npy'

    if not data_path.exists() or not graph_path.exists():
        raise FileNotFoundError(f'{domain}: gen_data.npy yoki graph.npy topilmadi: {droot}')

    data=np.load(data_path,mmap_mode='r')
    graph=np.load(graph_path)

    if data.ndim!=3:
        raise ValueError(f'{domain}: gen_data ndim={data.ndim}, 3 kutilgan')
    if graph.ndim!=2 or graph.shape[0]!=graph.shape[1]:
        raise ValueError(f'{domain}: graph shape noto‘g‘ri: {graph.shape}')

    N=graph.shape[0]
    if data.shape[-1] != 2*N:
        raise ValueError(
            f'{domain}: last dim={data.shape[-1]}, 2*N={2*N} bilan mos emas'
        )

    return {
        'Domain':domain,
        'DataFileMB':data_path.stat().st_size/(1024**2),
        'GraphFileKB':graph_path.stat().st_size/1024,
        'DataShape':str(tuple(data.shape)),
        'GraphShape':str(tuple(graph.shape)),
        'Dtype':str(data.dtype),
        'NaN_count':int(np.isnan(np.asarray(data[:min(2,len(data))])).sum()),
        'GraphEdges_offdiag':int((np.abs(graph)>1e-12).sum()-np.count_nonzero(np.diag(graph))),
        'FirstValues':np.asarray(data[0,:min(2,data.shape[1]),:min(5,data.shape[2])]).round(4).tolist(),
    }

def load_causaltime_domain(root,domain,max_lag=1,max_sequences=None,transpose_graph=False):
    root=Path(root)
    data=np.load(root/domain/'gen_data.npy')
    graph=np.load(root/domain/'graph.npy')

    N=graph.shape[0]
    X=np.asarray(data[...,:N],dtype=float)

    if max_sequences is not None:
        X=X[:min(int(max_sequences),len(X))]

    # Har variable bo‘yicha global standardization
    flat=X.reshape(-1,N)
    mu=np.nanmean(flat,axis=0)
    sd=np.nanstd(flat,axis=0)
    sd[sd<1e-12]=1.0
    X=(X-mu)/sd
    X=np.nan_to_num(X,nan=0.0,posinf=0.0,neginf=0.0)

    # sequence boundary lag leakage ni bloklash uchun zero separator
    parts=[]
    sep=np.zeros((max_lag,N),dtype=float)
    for k,seq in enumerate(X):
        if k:
            parts.append(sep.copy())
        parts.append(seq)
    linked=np.vstack(parts)

    cols=[f'V{i}' for i in range(N)]
    df=pd.DataFrame(linked,columns=cols)

    gt=np.asarray(graph).T if transpose_graph else np.asarray(graph)
    gt=(np.abs(gt)>1e-12).astype(int)
    np.fill_diagonal(gt,0)

    return df,gt,{
        'domain':domain,
        'n_sequences':len(X),
        'time_step':X.shape[1],
        'n_nodes':N,
        'linked_length':len(df),
        'transpose_graph':bool(transpose_graph),
    }

if not _looks_like_causaltime_root(CAUSALTIME_ROOT):
    try_download_causaltime(CAUSALTIME_ROOT)

inspect_rows=[]
for domain in ['medical','pm25','traffic']:
    try:
        inspect_rows.append(inspect_causaltime_domain(CAUSALTIME_ROOT,domain))
    except Exception as ex:
        inspect_rows.append({
            'Domain':domain,
            'ERROR':f'{type(ex).__name__}: {ex}'
        })

CAUSALTIME_INSPECTION=pd.DataFrame(inspect_rows)
display(CAUSALTIME_INSPECTION)

# %%
# CausalTime benchmark
CAUSALTIME_RESULTS=pd.DataFrame()
CAUSALTIME_META=[]

if _looks_like_causaltime_root(CAUSALTIME_ROOT):
    ct_frames=[]

    if RUN_MODE=='smoke':
        max_seq=20
    elif RUN_MODE=='fast':
        max_seq=100
    else:
        max_seq=None

    for domain in CAUSALTIME_DOMAINS:
        try:
            df_ct,gt_ct,meta_ct=load_causaltime_domain(
                CAUSALTIME_ROOT,
                domain,
                max_lag=CAUSALTIME_MAX_LAG,
                max_sequences=max_seq,
                transpose_graph=False
            )
            print('\nCausalTime',domain,meta_ct)

            key=(
                f'CausalTime_{domain}_seq{meta_ct["n_sequences"]}'
                f'_T{meta_ct["time_step"]}_N{meta_ct["n_nodes"]}'
            )
            edges_ct,status_ct,times_ct=run_all_target_stack(
                df_ct,key,CAUSALTIME_MAX_LAG,seed=321
            )

            meta_ct['status']=status_ct
            meta_ct['times']=times_ct
            CAUSALTIME_META.append(meta_ct)
            print('status:',status_ct)

            ok_methods=[m for m,s in status_ct.items() if str(s).startswith('OK')]
            if set(ok_methods)!=set(EXPECTED_METHODS):
                print(domain,': complete 5-method stack yo‘q, metric chiqarilmadi.')
                continue

            r=evaluate_variable_graph(
                edges_ct,
                list(df_ct.columns),
                gt_ct,
                f'CausalTime-{domain}'
            )
            ct_frames.append(r)

        except Exception as ex:
            print(domain,'ERROR:',type(ex).__name__,ex)

    if ct_frames:
        CAUSALTIME_RESULTS=pd.concat(ct_frames,ignore_index=True)
        display(
            CAUSALTIME_RESULTS.groupby(['Dataset','Strategy'])
            .agg(F1_mean=('F1','mean'),
                 Precision_mean=('Precision','mean'),
                 Recall_mean=('Recall','mean'))
            .reset_index()
            .round(3)
        )
else:
    print('CausalTime dataset topilmadi. Synthetic + Lorenz natijalari ishlaydi.')
    print('Datasetni rasmiy manbadan bir marta CAUSALTIME_ROOT ga oching:')
    print(CAUSALTIME_ROOT)

EXTERNAL_BENCHMARK_RESULTS=pd.concat(
    [x for x in [LORENZ_RESULTS,CAUSALTIME_RESULTS] if len(x)],
    ignore_index=True
) if (len(LORENZ_RESULTS) or len(CAUSALTIME_RESULTS)) else pd.DataFrame()

if len(EXTERNAL_BENCHMARK_RESULTS):
    EXTERNAL_SUMMARY=(
        EXTERNAL_BENCHMARK_RESULTS.groupby(['Dataset','Strategy'])
        .agg(F1_mean=('F1','mean'),F1_std=('F1','std'),
             Precision_mean=('Precision','mean'),
             Recall_mean=('Recall','mean'))
        .reset_index()
    )
    print('\nExternal benchmark summary:')
    display(EXTERNAL_SUMMARY.round(3))
else:
    EXTERNAL_SUMMARY=pd.DataFrame()

# %%
# Practical CausalTime 2:1:1 evaluation
# ===== CausalTime external benchmark: practical 2:1:1 stack =====

CT_METHOD_FAMILY = {
    'PCMCI+': 'conditional_independence',
    'PCMCI': 'conditional_independence',
    'VAR-LiNGAM': 'non_gaussian_svar',
    'DYNOTEARS': 'continuous_score_optimization',
}

CT_METHODS = list(CT_METHOD_FAMILY.keys())


def run_ct_selected_stack(df, dataset_key, max_lag=1, seed=321):
    all_edges = []
    status = {}
    times = {}

    for method in CT_METHODS:
        path = benchmark_cache_file(
            dataset_key, method, max_lag
        )

        if path.exists() and not FORCE_RECOMPUTE:
            try:
                e = normalize_edges(pd.read_csv(path))
                status[method] = 'OK(cache)'
                times[method] = 0.0
                all_edges.append(e)

                print(
                    f'  {method}: OK(cache), '
                    f'edges={len(e)}'
                )
                continue

            except Exception:
                pass

        print(f'  {method}: START...')

        t0 = time.perf_counter()

        try:
            if method in {'PCMCI+', 'PCMCI'}:
                e = run_tigramite_all(
                    df,
                    max_lag,
                    method,
                    CFG.pcmci_pc_alpha
                )

            elif method == 'VAR-LiNGAM':
                e = run_varlingam_all(
                    df,
                    max_lag,
                    CFG.varlingam_coef_threshold,
                    CFG.varlingam_prune,
                    seed
                )

            elif method == 'DYNOTEARS':
                e = run_dynotears_all(
                    df,
                    max_lag,
                    CFG
                )

            else:
                raise ValueError(method)

            e = normalize_edges(e)

            e.to_csv(path, index=False)

            status[method] = 'OK'
            all_edges.append(e)

        except Exception as ex:
            status[method] = (
                f'ERROR: {type(ex).__name__}: {ex}'
            )

        times[method] = time.perf_counter() - t0

        print(
            f'  {method}: {status[method]}, '
            f'time={times[method]:.1f}s'
        )

    if all_edges:
        edges = normalize_edges(
            pd.concat(all_edges, ignore_index=True)
        )
    else:
        edges = empty_edges()

    return edges, status, times


def evaluate_ct_211(
    edges,
    variables,
    gt_matrix,
    dataset_name,
    theta_v=None
):
    if theta_v is None:
        theta_v = CFG.theta_v

    gt = (np.asarray(gt_matrix) != 0)

    np.fill_diagonal(gt, False)

    rows = []

    for j, target in enumerate(variables):
        candidates = [
            v for v in variables
            if v != target
        ]

        true_vars = {
            variables[i]
            for i in range(len(variables))
            if gt[i, j] and i != j
        }

        et = edges[
            edges['target'] == target
        ].copy()

        rv_family, _, _ = compute_rv(
            et,
            candidates,
            CT_METHOD_FAMILY,
            'family_normalized'
        )

        rv_equal, _, _ = compute_rv(
            et,
            candidates,
            CT_METHOD_FAMILY,
            'equal_method'
        )

        pred_sets = {
            'FamilyNormalized_RV': {
                v for v in candidates
                if rv_family.get(v, 0)
                + 1e-12 >= theta_v
            },

            'EqualMethod_sameTheta': {
                v for v in candidates
                if rv_equal.get(v, 0)
                + 1e-12 >= theta_v
            },

            'SimpleMajority_0.5': {
                v for v in candidates
                if rv_equal.get(v, 0)
                > 0.5 + 1e-12
            },

            'Union_variable': {
                v for v in candidates
                if rv_equal.get(v, 0)
                > 1e-12
            },

            'Intersection_variable': {
                v for v in candidates
                if rv_equal.get(v, 0)
                >= 1 - 1e-12
            },
        }

        for m in CT_METHODS:
            pred_sets[m] = set(
                et.loc[
                    (et['method'] == m)
                    & (et['selected']),
                    'cause'
                ]
            )

        for method, pred in pred_sets.items():
            met = variable_set_metrics(
                pred,
                true_vars,
                candidates
            )

            rows.append({
                'Dataset': dataset_name,
                'Target': target,
                'Strategy': method,
                **met,
                'N_true': len(true_vars),
                'N_selected': len(pred),
            })

    return pd.DataFrame(rows)


CAUSALTIME_RESULTS = pd.DataFrame()
CAUSALTIME_META = []

ct_frames = []

for domain in ['medical', 'pm25', 'traffic']:

    df_ct, gt_ct, meta_ct = load_causaltime_domain(
        CAUSALTIME_ROOT,
        domain,
        max_lag=CAUSALTIME_MAX_LAG,
        max_sequences=None,
        transpose_graph=False
    )

    print(
        '\n=== CausalTime',
        domain,
        meta_ct,
        '==='
    )

    key = (
        f'CausalTime_{domain}'
        f'_seq{meta_ct["n_sequences"]}'
        f'_T{meta_ct["time_step"]}'
        f'_N{meta_ct["n_nodes"]}'
    )

    edges_ct, status_ct, times_ct = (
        run_ct_selected_stack(
            df_ct,
            key,
            CAUSALTIME_MAX_LAG,
            seed=321
        )
    )

    meta_ct['status'] = status_ct
    meta_ct['times'] = times_ct
    meta_ct['method_composition'] = '2:1:1'

    CAUSALTIME_META.append(meta_ct)

    print('status:', status_ct)

    ok_methods = [
        m for m, s in status_ct.items()
        if str(s).startswith('OK')
    ]

    if set(ok_methods) != set(CT_METHODS):
        print(
            domain,
            ': complete 2:1:1 stack yo‘q.'
        )
        continue

    r = evaluate_ct_211(
        edges_ct,
        list(df_ct.columns),
        gt_ct,
        f'CausalTime-{domain}'
    )

    ct_frames.append(r)


if ct_frames:
    CAUSALTIME_RESULTS = pd.concat(
        ct_frames,
        ignore_index=True
    )

    CT_SUMMARY = (
        CAUSALTIME_RESULTS
        .groupby(['Dataset', 'Strategy'])
        .agg(
            F1_mean=('F1', 'mean'),
            Precision_mean=('Precision', 'mean'),
            Recall_mean=('Recall', 'mean')
        )
        .reset_index()
    )

    display(
        CT_SUMMARY.round(3)
    )

else:
    CT_SUMMARY = pd.DataFrame()

print(
    '\nCausalTime 2:1:1 benchmark tugadi.'
)

# %%
# Save final publication outputs
run_id=pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
out_dir=DIRS['results']/f'run_{run_id}'
out_dir.mkdir(parents=True,exist_ok=True)

def _save_csv_if(name,obj,index=False):
    if isinstance(obj,pd.DataFrame) and len(obj):
        obj.to_csv(out_dir/name,index=index)

_save_csv_if('stress_results.csv',RESULTS)
_save_csv_if('scenario_summary.csv',SUMMARY)
if isinstance(OVERALL,pd.DataFrame) and len(OVERALL):
    OVERALL.to_csv(out_dir/'overall_summary.csv')

_save_csv_if('variable_ablation.csv',VARIABLE_ABLATION)
_save_csv_if('variable_ablation_summary.csv',VAR_SUMMARY)
if isinstance(VAR_OVERALL,pd.DataFrame) and len(VAR_OVERALL):
    VAR_OVERALL.to_csv(out_dir/'variable_ablation_overall.csv')

_save_csv_if('rv_scores.csv',RV_SCORES)
_save_csv_if('family_imbalance_controlled.csv',globals().get('FAMILY_IMBALANCE_CONTROLLED',pd.DataFrame()))
_save_csv_if('family_composition_ablation.csv',globals().get('FAMILY_COMPOSITION_ABLATION',pd.DataFrame()))
_save_csv_if('family_composition_summary.csv',globals().get('FAMILY_COMPOSITION_SUMMARY',pd.DataFrame()))
_save_csv_if('theta_sensitivity.csv',globals().get('THETA_SENSITIVITY',pd.DataFrame()))
_save_csv_if('theta_sensitivity_summary.csv',globals().get('THETA_SUMMARY',pd.DataFrame()))
_save_csv_if('pi_sensitivity.csv',globals().get('PI_SENSITIVITY',pd.DataFrame()))
_save_csv_if('pi_sensitivity_summary.csv',globals().get('PI_SUMMARY',pd.DataFrame()))

_save_csv_if('lorenz96_results.csv',globals().get('LORENZ_RESULTS',pd.DataFrame()))
_save_csv_if('causaltime_inspection.csv',globals().get('CAUSALTIME_INSPECTION',pd.DataFrame()))
_save_csv_if('causaltime_results.csv',globals().get('CAUSALTIME_RESULTS',pd.DataFrame()))
_save_csv_if('causaltime_summary.csv',globals().get('CT_SUMMARY',pd.DataFrame()))
_save_csv_if('external_benchmark_results.csv',globals().get('EXTERNAL_BENCHMARK_RESULTS',pd.DataFrame()))
_save_csv_if('external_benchmark_summary.csv',globals().get('EXTERNAL_SUMMARY',pd.DataFrame()))

_save_csv_if('disagreement.csv',globals().get('DISAGREEMENT',pd.DataFrame()))
_save_csv_if('ablation_lag_diagnostic_only.csv',globals().get('ABL',pd.DataFrame()))
_save_csv_if('paired_tests_temporal_diagnostic.csv',globals().get('STAT_TESTS',pd.DataFrame()))
_save_csv_if('paired_tests_variable.csv',globals().get('VAR_STAT_TESTS',pd.DataFrame()))

if len(META):
    META.to_json(
        out_dir/'run_meta.json',
        orient='records',
        force_ascii=False,
        indent=2
    )

with open(out_dir/'external_meta.json','w',encoding='utf-8') as f:
    json.dump({
        'lorenz_status':globals().get('LORENZ_STATUS',{}),
        'causaltime_meta':globals().get('CAUSALTIME_META',[]),
        'causaltime_root':str(CAUSALTIME_ROOT),
        'causaltime_domains':CAUSALTIME_DOMAINS,
        'causaltime_max_lag':CAUSALTIME_MAX_LAG,
    },f,ensure_ascii=False,indent=2,default=str)

with open(out_dir/'config.json','w',encoding='utf-8') as f:
    json.dump({
        'notebook':NOTEBOOK_VERSION,
        'RUN_MODE':RUN_MODE,
        'FAST_MODE':FAST_MODE,
        'config':asdict(CFG),
        'method_family':METHOD_FAMILY,
        'family_description':FAMILY_DESCRIPTION,
        'scenarios':[asdict(s) for s in SCENARIOS],
        'seeds':SEEDS,
        'compositions':COMPOSITIONS,
        'theta_grid':THETA_GRID,
        'pi_settings':PI_SETTINGS,
    },f,ensure_ascii=False,indent=2,default=str)


# Publication gate
_gate = {
    'synthetic_runs_expected': len(SCENARIOS) * len(SEEDS),
    'synthetic_runs_completed': int(META.shape[0]) if isinstance(META,pd.DataFrame) else 0,
    'family_imbalance_controlled_ok': bool(
        'FAMILY_IMBALANCE_CONTROLLED' in globals()
        and len(FAMILY_IMBALANCE_CONTROLLED) == 2
    ),
    'family_composition_rows': int(len(globals().get('FAMILY_COMPOSITION_ABLATION',pd.DataFrame()))),
    'theta_rows': int(len(globals().get('THETA_SENSITIVITY',pd.DataFrame()))),
    'pi_rows': int(len(globals().get('PI_SENSITIVITY',pd.DataFrame()))),
    'lorenz_rows': int(len(globals().get('LORENZ_RESULTS',pd.DataFrame()))),
    'causaltime_rows': int(len(globals().get('CAUSALTIME_RESULTS',pd.DataFrame()))),
}

with open(out_dir/'publication_gate.json','w',encoding='utf-8') as f:
    json.dump(_gate,f,ensure_ascii=False,indent=2)

print('Publication gate:')
print(json.dumps(_gate,ensure_ascii=False,indent=2))

if RUN_MODE == 'paper':
    if _gate['synthetic_runs_completed'] != _gate['synthetic_runs_expected']:
        print('WARNING: barcha synthetic paper-runlar tugamagan.')
    if _gate['lorenz_rows'] == 0:
        print('WARNING: Lorenz-96 natijasi yo‘q.')
    if _gate['causaltime_rows'] == 0:
        print('WARNING: CausalTime natijasi yo‘q.')

# figures papkasidagi mavjud publication figures ni ham nusxalaymiz
fig_out=out_dir/'figures'
fig_out.mkdir(exist_ok=True)
for fp in DIRS['figures'].glob('*'):
    if fp.is_file():
        try:
            shutil.copy2(fp,fig_out/fp.name)
        except Exception:
            pass

zip_base=DIRS['results']/f'paper_results_{run_id}'
zip_path=shutil.make_archive(
    str(zip_base),'zip',root_dir=out_dir
)

print('Saqlandi:',out_dir)
print('Paper ZIP:',zip_path)

if IN_COLAB:
    try:
        from google.colab import files
        print('ZIP Drive da saqlandi.')
        print(f"files.download(r'{zip_path}')")
    except Exception:
        pass

# %%
# External binary selection at theta=0.50
# ============================================================
# FINAL EXTERNAL VALIDATION at theta_V = 0.50
# NO causal-discovery recomputation — CACHE ONLY
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd

THETA_FINAL = 0.50

# ------------------------------------------------------------
# 1. Cache'dan edge natijalarini o'qish
# ------------------------------------------------------------
def load_cached_edges(dataset_key, methods, max_lag=1):
    frames = []
    status = {}

    for method in methods:
        p = benchmark_cache_file(
            dataset_key,
            method,
            max_lag
        )

        if not p.exists():
            status[method] = f'MISSING: {p.name}'
            continue

        try:
            e = normalize_edges(pd.read_csv(p))
            frames.append(e)
            status[method] = f'OK(cache), edges={len(e)}'
        except Exception as ex:
            status[method] = (
                f'ERROR: {type(ex).__name__}: {ex}'
            )

    if frames:
        edges = normalize_edges(
            pd.concat(frames, ignore_index=True)
        )
    else:
        edges = empty_edges()

    return edges, status


# ------------------------------------------------------------
# 2. Custom family composition bilan variable-level baholash
# ------------------------------------------------------------
def evaluate_variable_graph_custom(
    edges,
    variables,
    gt_matrix,
    dataset_name,
    method_family,
    theta_v=0.50
):
    gt = (np.asarray(gt_matrix) != 0).copy()

    if gt.shape != (len(variables), len(variables)):
        raise ValueError(
            f'GT shape {gt.shape} != '
            f'{(len(variables), len(variables))}'
        )

    np.fill_diagonal(gt, False)

    methods = list(method_family.keys())
    rows = []

    for j, target in enumerate(variables):

        candidates = [
            v for v in variables
            if v != target
        ]

        true_vars = {
            variables[i]
            for i in range(len(variables))
            if gt[i, j] and i != j
        }

        et = edges[
            edges['target'] == target
        ].copy()

        rv_family, _, _ = compute_rv(
            et,
            candidates,
            method_family,
            'family_normalized'
        )

        rv_equal, _, _ = compute_rv(
            et,
            candidates,
            method_family,
            'equal_method'
        )

        pred_sets = {
            'FamilyNormalized_RV_theta0.50': {
                v for v in candidates
                if rv_family.get(v, 0.0) + 1e-12
                >= theta_v
            },

            'EqualMethod_theta0.50': {
                v for v in candidates
                if rv_equal.get(v, 0.0) + 1e-12
                >= theta_v
            },

            # Reviewer comparison:
            # strict simple majority = > 50%
            'SimpleMajority_gt0.50': {
                v for v in candidates
                if rv_equal.get(v, 0.0)
                > 0.5 + 1e-12
            },

            'Union_variable': {
                v for v in candidates
                if rv_equal.get(v, 0.0)
                > 1e-12
            },
        }

        # Individual methods — faqat nazorat uchun
        for m in methods:
            pred_sets[m] = set(
                et.loc[
                    (et['method'] == m)
                    & (et['selected']),
                    'cause'
                ]
            )

        for strategy, pred in pred_sets.items():

            met = variable_set_metrics(
                pred,
                true_vars,
                candidates
            )

            rows.append({
                'Dataset': dataset_name,
                'Target': target,
                'theta_v': theta_v,
                'Strategy': strategy,
                **met,
                'N_true': len(true_vars),
                'N_selected': len(pred),
            })

    return pd.DataFrame(rows)


# ============================================================
# 3. LORENZ-96 — 3:1:1
# ============================================================

LORENZ_METHOD_FAMILY_FINAL = {
    'PCMCI+': 'conditional_independence',
    'PCMCI': 'conditional_independence',
    'LPCMCI': 'conditional_independence',
    'VAR-LiNGAM': 'non_gaussian_svar',
    'DYNOTEARS': 'continuous_score_optimization',
}

lor_n = 3500

lor_key = (
    f'Lorenz96_d10_n{lor_n}_seed123'
)

lor_edges_50, lor_status_50 = load_cached_edges(
    lor_key,
    list(LORENZ_METHOD_FAMILY_FINAL.keys()),
    max_lag=1
)

print('\n=== LORENZ CACHE ===')
print(lor_status_50)

# Ground truthni qayta qurish — causal methods ishlamaydi
_, LORENZ_GT_50 = simulate_lorenz96(
    n_samples=lor_n,
    seed=123
)

lor_vars = [
    f'L{i}' for i in range(LORENZ_GT_50.shape[0])
]

LORENZ_RESULTS_T05 = evaluate_variable_graph_custom(
    lor_edges_50,
    lor_vars,
    LORENZ_GT_50,
    'Lorenz96',
    LORENZ_METHOD_FAMILY_FINAL,
    theta_v=THETA_FINAL
)


# ============================================================
# 4. CAUSALTIME — practical 2:1:1
# ============================================================

CT_METHOD_FAMILY_FINAL = {
    'PCMCI+': 'conditional_independence',
    'PCMCI': 'conditional_independence',
    'VAR-LiNGAM': 'non_gaussian_svar',
    'DYNOTEARS': 'continuous_score_optimization',
}

ct_frames_50 = []

for domain in ['medical', 'pm25', 'traffic']:

    graph_path = (
        Path(CAUSALTIME_ROOT)
        / domain
        / 'graph.npy'
    )

    gt_ct = np.load(graph_path)

    gt_ct = (
        np.abs(np.asarray(gt_ct)) > 1e-12
    ).astype(int)

    np.fill_diagonal(gt_ct, 0)

    N = gt_ct.shape[0]

    variables = [
        f'V{i}' for i in range(N)
    ]

    key = (
        f'CausalTime_{domain}'
        f'_seq480_T40_N{N}'
    )

    edges_ct_50, status_ct_50 = load_cached_edges(
        key,
        list(CT_METHOD_FAMILY_FINAL.keys()),
        max_lag=1
    )

    print(
        f'\n=== CAUSALTIME {domain} CACHE ==='
    )
    print(status_ct_50)

    expected = set(
        CT_METHOD_FAMILY_FINAL.keys()
    )

    ok = {
        m for m, s in status_ct_50.items()
        if str(s).startswith('OK')
    }

    if ok != expected:
        print(
            domain,
            ': cache to‘liq emas — SKIP'
        )
        continue

    r = evaluate_variable_graph_custom(
        edges_ct_50,
        variables,
        gt_ct,
        f'CausalTime-{domain}',
        CT_METHOD_FAMILY_FINAL,
        theta_v=THETA_FINAL
    )

    ct_frames_50.append(r)


CAUSALTIME_RESULTS_T05 = (
    pd.concat(
        ct_frames_50,
        ignore_index=True
    )
    if ct_frames_50
    else pd.DataFrame()
)


# ============================================================
# 5. FINAL SUMMARY
# ============================================================

EXTERNAL_T05 = pd.concat(
    [
        LORENZ_RESULTS_T05,
        CAUSALTIME_RESULTS_T05
    ],
    ignore_index=True
)

EXTERNAL_SUMMARY_T05 = (
    EXTERNAL_T05
    .groupby(
        ['Dataset', 'Strategy']
    )
    .agg(
        F1_mean=('F1', 'mean'),
        F1_std=('F1', 'std'),
        Precision_mean=('Precision', 'mean'),
        Recall_mean=('Recall', 'mean'),
        N_selected_mean=('N_selected', 'mean')
    )
    .reset_index()
)

focus = EXTERNAL_SUMMARY_T05[
    EXTERNAL_SUMMARY_T05['Strategy'].isin([
        'FamilyNormalized_RV_theta0.50',
        'EqualMethod_theta0.50',
        'SimpleMajority_gt0.50',
        'Union_variable',
        'PCMCI+',
        'PCMCI',
        'LPCMCI',
        'VAR-LiNGAM',
        'DYNOTEARS',
    ])
].copy()

print(
    '\n======================================'
)
print(
    ' FINAL EXTERNAL VALIDATION: theta=0.50'
)
print(
    '======================================'
)

display(
    focus.sort_values(
        ['Dataset', 'F1_mean'],
        ascending=[True, False]
    ).round(3)
)


# ------------------------------------------------------------
# 6. Faqat asosiy 3 aggregatorni ixcham taqqoslash
# ------------------------------------------------------------
main_compare = (
    EXTERNAL_SUMMARY_T05[
        EXTERNAL_SUMMARY_T05[
            'Strategy'
        ].isin([
            'FamilyNormalized_RV_theta0.50',
            'EqualMethod_theta0.50',
            'SimpleMajority_gt0.50',
        ])
    ][
        [
            'Dataset',
            'Strategy',
            'F1_mean',
            'Precision_mean',
            'Recall_mean',
            'N_selected_mean'
        ]
    ]
    .sort_values(
        ['Dataset', 'Strategy']
    )
)

print(
    '\n=== ASOSIY AGGREGATORLAR ==='
)

display(
    main_compare.round(3)
)


# ------------------------------------------------------------
# 7. Natijani Drive'ga saqlash
# ------------------------------------------------------------
post_dir = (
    DIRS['results']
    / 'postprocess_theta050'
)

post_dir.mkdir(
    parents=True,
    exist_ok=True
)

LORENZ_RESULTS_T05.to_csv(
    post_dir / 'lorenz96_theta050.csv',
    index=False
)

CAUSALTIME_RESULTS_T05.to_csv(
    post_dir / 'causaltime_theta050.csv',
    index=False
)

EXTERNAL_SUMMARY_T05.to_csv(
    post_dir / 'external_summary_theta050.csv',
    index=False
)

print(
    '\nSaqlandi:',
    post_dir
)

print(
    '\nTAYYOR — causal discovery qayta hisoblanmadi.'
)

# %%
# Threshold-free AP/AUC diagnostics
# ============================================================
# Threshold-free RV diagnostic
# Uses cached causal-discovery outputs; no recomputation
# ============================================================

import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score
)

# ------------------------------------------------------------
# Cache reader
# ------------------------------------------------------------
def load_cached_edges_diag(
    dataset_key,
    methods,
    max_lag=1
):
    frames = []
    status = {}

    for method in methods:

        p = benchmark_cache_file(
            dataset_key,
            method,
            max_lag
        )

        if not p.exists():
            status[method] = 'MISSING'
            continue

        try:
            e = normalize_edges(
                pd.read_csv(p)
            )

            frames.append(e)

            status[method] = (
                f'OK({len(e)})'
            )

        except Exception as ex:
            status[method] = (
                f'ERROR: {type(ex).__name__}'
            )

    if frames:
        edges = normalize_edges(
            pd.concat(
                frames,
                ignore_index=True
            )
        )
    else:
        edges = empty_edges()

    return edges, status


# ------------------------------------------------------------
# Threshold-free scoring
# ------------------------------------------------------------
def evaluate_score_quality(
    edges,
    variables,
    gt_matrix,
    method_family,
    dataset_name,
    orientation='as_loaded'
):

    G = (
        np.asarray(gt_matrix).T
        if orientation == 'transposed'
        else np.asarray(gt_matrix)
    )

    G = (np.abs(G) > 1e-12).astype(int)

    np.fill_diagonal(G, 0)

    detail_rows = []

    pooled = {
        'FamilyNormalized_RV': {
            'y': [],
            'score': []
        },
        'EqualMethodScore': {
            'y': [],
            'score': []
        },
    }

    for j, target in enumerate(variables):

        candidates = [
            v for v in variables
            if v != target
        ]

        candidate_idx = [
            i for i in range(len(variables))
            if i != j
        ]

        y_true = np.array([
            G[i, j]
            for i in candidate_idx
        ], dtype=int)

        et = edges[
            edges['target'] == target
        ].copy()

        rv_family, _, _ = compute_rv(
            et,
            candidates,
            method_family,
            'family_normalized'
        )

        rv_equal, _, _ = compute_rv(
            et,
            candidates,
            method_family,
            'equal_method'
        )

        score_sets = {
            'FamilyNormalized_RV':
                np.array([
                    rv_family.get(v, 0.0)
                    for v in candidates
                ], dtype=float),

            'EqualMethodScore':
                np.array([
                    rv_equal.get(v, 0.0)
                    for v in candidates
                ], dtype=float),
        }

        for strategy, scores in score_sets.items():

            pooled[strategy]['y'].extend(
                y_true.tolist()
            )

            pooled[strategy]['score'].extend(
                scores.tolist()
            )

            # Targetda hech bo'lmasa bitta true parent bo'lsa AP
            ap = np.nan

            if y_true.sum() > 0:
                ap = average_precision_score(
                    y_true,
                    scores
                )

            # ROC-AUC uchun ikki klass kerak
            auc = np.nan

            if len(np.unique(y_true)) == 2:
                auc = roc_auc_score(
                    y_true,
                    scores
                )

            detail_rows.append({
                'Dataset': dataset_name,
                'Orientation': orientation,
                'Target': target,
                'Strategy': strategy,
                'AP': ap,
                'ROC_AUC': auc,
                'N_true': int(y_true.sum()),
            })

    DETAIL = pd.DataFrame(detail_rows)

    summary_rows = []

    for strategy in pooled:

        y = np.asarray(
            pooled[strategy]['y'],
            dtype=int
        )

        s = np.asarray(
            pooled[strategy]['score'],
            dtype=float
        )

        pooled_ap = (
            average_precision_score(y, s)
            if y.sum() > 0
            else np.nan
        )

        pooled_auc = (
            roc_auc_score(y, s)
            if len(np.unique(y)) == 2
            else np.nan
        )

        d = DETAIL[
            DETAIL['Strategy'] == strategy
        ]

        summary_rows.append({
            'Dataset': dataset_name,
            'Orientation': orientation,
            'Strategy': strategy,

            'AP_macro':
                d['AP'].mean(),

            'ROC_AUC_macro':
                d['ROC_AUC'].mean(),

            'AP_pooled':
                pooled_ap,

            'ROC_AUC_pooled':
                pooled_auc,

            'Positive_rate':
                float(y.mean()),
        })

    return DETAIL, pd.DataFrame(
        summary_rows
    )


# ============================================================
# 1. Lorenz-96
# ============================================================

LOR_FAMILY = {
    'PCMCI+':
        'conditional_independence',

    'PCMCI':
        'conditional_independence',

    'LPCMCI':
        'conditional_independence',

    'VAR-LiNGAM':
        'non_gaussian_svar',

    'DYNOTEARS':
        'continuous_score_optimization',
}

lor_n = 3500

lor_key = (
    f'Lorenz96_d10_n{lor_n}_seed123'
)

lor_edges, lor_status = (
    load_cached_edges_diag(
        lor_key,
        list(LOR_FAMILY.keys()),
        1
    )
)

_, lor_gt = simulate_lorenz96(
    n_samples=lor_n,
    seed=123
)

lor_vars = [
    f'L{i}'
    for i in range(
        lor_gt.shape[0]
    )
]

lor_detail, lor_summary = (
    evaluate_score_quality(
        lor_edges,
        lor_vars,
        lor_gt,
        LOR_FAMILY,
        'Lorenz96',
        'as_loaded'
    )
)


# ============================================================
# 2. CausalTime
#    ikkala orientation faqat diagnostika uchun
# ============================================================

CT_FAMILY = {
    'PCMCI+':
        'conditional_independence',

    'PCMCI':
        'conditional_independence',

    'VAR-LiNGAM':
        'non_gaussian_svar',

    'DYNOTEARS':
        'continuous_score_optimization',
}

ct_details = []
ct_summaries = []

for domain in [
    'medical',
    'pm25',
    'traffic'
]:

    gt = np.load(
        Path(CAUSALTIME_ROOT)
        / domain
        / 'graph.npy'
    )

    N = gt.shape[0]

    variables = [
        f'V{i}'
        for i in range(N)
    ]

    key = (
        f'CausalTime_{domain}'
        f'_seq480_T40_N{N}'
    )

    edges, status = (
        load_cached_edges_diag(
            key,
            list(CT_FAMILY.keys()),
            1
        )
    )

    print(
        '\n',
        domain,
        status
    )

    for orientation in [
        'as_loaded',
        'transposed'
    ]:

        d, s = evaluate_score_quality(
            edges,
            variables,
            gt,
            CT_FAMILY,
            f'CausalTime-{domain}',
            orientation
        )

        ct_details.append(d)
        ct_summaries.append(s)


CT_SCORE_DETAIL = pd.concat(
    ct_details,
    ignore_index=True
)

CT_SCORE_SUMMARY = pd.concat(
    ct_summaries,
    ignore_index=True
)

SCORE_SUMMARY_FINAL = pd.concat(
    [
        lor_summary,
        CT_SCORE_SUMMARY
    ],
    ignore_index=True
)


# ------------------------------------------------------------
# Main table
# ------------------------------------------------------------

print(
    '\n======================================'
)
print(
    'THRESHOLD-FREE SCORE DIAGNOSTIC'
)
print(
    '======================================'
)

display(
    SCORE_SUMMARY_FINAL
    .sort_values([
        'Dataset',
        'Orientation',
        'Strategy'
    ])
    .round(3)
)


# ------------------------------------------------------------
# Family vs equal delta
# ------------------------------------------------------------

pivot = (
    SCORE_SUMMARY_FINAL
    .pivot_table(
        index=[
            'Dataset',
            'Orientation'
        ],
        columns='Strategy',
        values=[
            'AP_macro',
            'AP_pooled',
            'ROC_AUC_macro',
            'ROC_AUC_pooled'
        ]
    )
)

delta_rows = []

for idx in pivot.index:

    dataset, orientation = idx

    row = {
        'Dataset': dataset,
        'Orientation': orientation
    }

    for metric in [
        'AP_macro',
        'AP_pooled',
        'ROC_AUC_macro',
        'ROC_AUC_pooled'
    ]:

        try:
            row[
                f'Delta_{metric}'
            ] = (
                pivot.loc[
                    idx,
                    (
                        metric,
                        'FamilyNormalized_RV'
                    )
                ]
                -
                pivot.loc[
                    idx,
                    (
                        metric,
                        'EqualMethodScore'
                    )
                ]
            )

        except Exception:
            row[
                f'Delta_{metric}'
            ] = np.nan

    delta_rows.append(row)


SCORE_DELTA = pd.DataFrame(
    delta_rows
)

print(
    '\n=== FAMILY RV minus EQUAL METHOD ==='
)

display(
    SCORE_DELTA.round(3)
)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

diag_dir = (
    DIRS['results']
    / 'postprocess_score_diagnostic'
)

diag_dir.mkdir(
    parents=True,
    exist_ok=True
)

SCORE_SUMMARY_FINAL.to_csv(
    diag_dir
    / 'score_summary.csv',
    index=False
)

SCORE_DELTA.to_csv(
    diag_dir
    / 'score_delta.csv',
    index=False
)

CT_SCORE_DETAIL.to_csv(
    diag_dir
    / 'causaltime_score_detail.csv',
    index=False
)

lor_detail.to_csv(
    diag_dir
    / 'lorenz_score_detail.csv',
    index=False
)

print(
    '\nSaqlandi:',
    diag_dir
)

print(
    '\nTAYYOR — causal discovery qayta ishlamadi.'
)

# %%
# Paired AP/AUC tests
# ============================================================
# Paired statistical test:
# FamilyNormalized_RV vs EqualMethodScore
# no causal-discovery recomputation
# ============================================================

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# Lorenz + CausalTime target-level natijalarini birlashtiramiz
score_detail_all = pd.concat(
    [
        lor_detail,
        CT_SCORE_DETAIL
    ],
    ignore_index=True
)

stat_rows = []

for (dataset, orientation), g in score_detail_all.groupby(
    ['Dataset', 'Orientation']
):

    for metric in ['AP', 'ROC_AUC']:

        pvt = (
            g.pivot_table(
                index='Target',
                columns='Strategy',
                values=metric,
                aggfunc='first'
            )
            .dropna()
        )

        if not {
            'FamilyNormalized_RV',
            'EqualMethodScore'
        }.issubset(pvt.columns):
            continue

        fam = pvt['FamilyNormalized_RV'].to_numpy()
        eq = pvt['EqualMethodScore'].to_numpy()

        diff = fam - eq

        # Agar barcha farqlar 0 bo'lsa Wilcoxon ishlamaydi
        if np.allclose(diff, 0):
            W = 0.0
            p = 1.0
        else:
            try:
                W, p = wilcoxon(
                    fam,
                    eq,
                    alternative='two-sided',
                    zero_method='wilcox'
                )
            except Exception:
                W, p = np.nan, np.nan

        stat_rows.append({
            'Dataset': dataset,
            'Orientation': orientation,
            'Metric': metric,
            'N_targets': len(pvt),
            'Family_mean': np.mean(fam),
            'Equal_mean': np.mean(eq),
            'Mean_delta': np.mean(diff),
            'Median_delta': np.median(diff),
            'Family_better_targets': int(np.sum(diff > 0)),
            'Equal_better_targets': int(np.sum(diff < 0)),
            'Ties': int(np.sum(np.isclose(diff, 0))),
            'Wilcoxon_W': W,
            'p_value': p
        })

PAIRED_SCORE_TESTS = pd.DataFrame(stat_rows)

print(
    '\n======================================'
)
print(
    'PAIRED SCORE TESTS'
)
print(
    '======================================'
)

display(
    PAIRED_SCORE_TESTS.round(4)
)

# Pooled external targets
pooled_rows = []

for metric in ['AP', 'ROC_AUC']:

    g = score_detail_all.copy()

    pvt = (
        g.pivot_table(
            index=[
                'Dataset',
                'Orientation',
                'Target'
            ],
            columns='Strategy',
            values=metric,
            aggfunc='first'
        )
        .dropna()
    )

    if {
        'FamilyNormalized_RV',
        'EqualMethodScore'
    }.issubset(pvt.columns):

        fam = pvt[
            'FamilyNormalized_RV'
        ].to_numpy()

        eq = pvt[
            'EqualMethodScore'
        ].to_numpy()

        diff = fam - eq

        if np.allclose(diff, 0):
            W = 0.0
            p = 1.0
        else:
            W, p = wilcoxon(
                fam,
                eq,
                alternative='two-sided',
                zero_method='wilcox'
            )

        pooled_rows.append({
            'Metric': metric,
            'N_targets': len(pvt),
            'Family_mean': np.mean(fam),
            'Equal_mean': np.mean(eq),
            'Mean_delta': np.mean(diff),
            'Median_delta': np.median(diff),
            'Family_better_targets':
                int(np.sum(diff > 0)),
            'Equal_better_targets':
                int(np.sum(diff < 0)),
            'Ties':
                int(np.sum(np.isclose(diff, 0))),
            'Wilcoxon_W': W,
            'p_value': p
        })

POOLED_SCORE_TESTS = pd.DataFrame(
    pooled_rows
)

print(
    '\n=== POOLED EXTERNAL TARGETS ==='
)

display(
    POOLED_SCORE_TESTS.round(4)
)

# Save
stat_dir = (
    DIRS['results']
    / 'postprocess_score_diagnostic'
)

PAIRED_SCORE_TESTS.to_csv(
    stat_dir / 'paired_score_tests.csv',
    index=False
)

POOLED_SCORE_TESTS.to_csv(
    stat_dir / 'pooled_score_tests.csv',
    index=False
)

print(
    '\nSaqlandi:',
    stat_dir
)

# %%
# Final unique-target statistics with Holm correction
# ============================================================
# Final unique-target statistics
# One CausalTime orientation only + Lorenz
# No recomputation
# ============================================================

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


# ------------------------------------------------------------
# 1. Faqat bitta orientation
# ------------------------------------------------------------

clean_detail = pd.concat(
    [
        lor_detail,
        CT_SCORE_DETAIL[
            CT_SCORE_DETAIL['Orientation'] == 'as_loaded'
        ]
    ],
    ignore_index=True
)


# ------------------------------------------------------------
# 2. Pooled test — unique targets only
# ------------------------------------------------------------

pooled_rows = []

for metric in ['AP', 'ROC_AUC']:

    pvt = (
        clean_detail
        .pivot_table(
            index=[
                'Dataset',
                'Target'
            ],
            columns='Strategy',
            values=metric,
            aggfunc='first'
        )
        .dropna()
    )

    fam = pvt[
        'FamilyNormalized_RV'
    ].to_numpy()

    eq = pvt[
        'EqualMethodScore'
    ].to_numpy()

    diff = fam - eq

    if np.allclose(diff, 0):
        W = 0.0
        p = 1.0
    else:
        W, p = wilcoxon(
            fam,
            eq,
            alternative='two-sided',
            zero_method='wilcox'
        )

    pooled_rows.append({
        'Metric': metric,
        'N_unique_targets': len(pvt),
        'Family_mean': fam.mean(),
        'Equal_mean': eq.mean(),
        'Mean_delta': diff.mean(),
        'Median_delta': np.median(diff),
        'Family_better': int(
            np.sum(diff > 0)
        ),
        'Equal_better': int(
            np.sum(diff < 0)
        ),
        'Ties': int(
            np.sum(np.isclose(diff, 0))
        ),
        'Wilcoxon_W': W,
        'p_value': p
    })


CLEAN_POOLED_TEST = pd.DataFrame(
    pooled_rows
)


print(
    '\n=== CLEAN POOLED EXTERNAL TEST ==='
)

display(
    CLEAN_POOLED_TEST.round(5)
)


# ------------------------------------------------------------
# 3. Domain-level testlar — faqat as_loaded + Lorenz
# ------------------------------------------------------------

domain_rows = []

for (dataset, orientation), g in clean_detail.groupby(
    ['Dataset', 'Orientation']
):

    for metric in ['AP', 'ROC_AUC']:

        pvt = (
            g.pivot_table(
                index='Target',
                columns='Strategy',
                values=metric,
                aggfunc='first'
            )
            .dropna()
        )

        if not {
            'FamilyNormalized_RV',
            'EqualMethodScore'
        }.issubset(pvt.columns):
            continue

        fam = pvt[
            'FamilyNormalized_RV'
        ].to_numpy()

        eq = pvt[
            'EqualMethodScore'
        ].to_numpy()

        diff = fam - eq

        if np.allclose(diff, 0):
            W = 0.0
            p = 1.0
        else:
            W, p = wilcoxon(
                fam,
                eq,
                alternative='two-sided',
                zero_method='wilcox'
            )

        domain_rows.append({
            'Dataset': dataset,
            'Orientation': orientation,
            'Metric': metric,
            'N_targets': len(pvt),
            'Family_mean': fam.mean(),
            'Equal_mean': eq.mean(),
            'Mean_delta': diff.mean(),
            'Family_better': int(
                np.sum(diff > 0)
            ),
            'Equal_better': int(
                np.sum(diff < 0)
            ),
            'Ties': int(
                np.sum(np.isclose(diff, 0))
            ),
            'Wilcoxon_W': W,
            'p_raw': p
        })


DOMAIN_TEST_CLEAN = pd.DataFrame(
    domain_rows
)


# ------------------------------------------------------------
# 4. Holm correction — 8 domain/metric test
# ------------------------------------------------------------

reject, p_holm, _, _ = multipletests(
    DOMAIN_TEST_CLEAN['p_raw'],
    alpha=0.05,
    method='holm'
)

DOMAIN_TEST_CLEAN[
    'p_holm'
] = p_holm

DOMAIN_TEST_CLEAN[
    'Significant_Holm_005'
] = reject


print(
    '\n=== DOMAIN TESTS + HOLM CORRECTION ==='
)

display(
    DOMAIN_TEST_CLEAN.round(5)
)


# ------------------------------------------------------------
# 5. Save
# ------------------------------------------------------------

stat_dir = (
    DIRS['results']
    / 'postprocess_score_diagnostic'
)

CLEAN_POOLED_TEST.to_csv(
    stat_dir
    / 'clean_pooled_external_test.csv',
    index=False
)

DOMAIN_TEST_CLEAN.to_csv(
    stat_dir
    / 'domain_tests_holm.csv',
    index=False
)

print(
    '\nSaqlandi:',
    stat_dir
)
