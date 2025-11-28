# Creating a ready-to-submit Python script for Exercise 8
# The script will be saved to /mnt/data/submission_ex8_ready.py
# It runs the same pipeline as the notebook but saves key figures and prints summaries.
# After this runs, you'll get a download link to the .py file.

script = r'''
#!/usr/bin/env python3
# submission_ex8_ready.py
# Auto-generated script for Exercise 8 (Python for ML)
# Usage: python submission_ex8_ready.py
# Make sure the CSV/XLSX files are in the current working directory or adjust paths below.

import os
import sys
import warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_absolute_error
from sklearn.manifold import TSNE
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

plt.rcParams['figure.figsize'] = (8,5)

# ---------- USER SETTINGS ----------
PATIENT_CSV = "patient-data.csv"
MNIST_CSV = "mnist_test_nolabels.csv"   # set to None if not available
CURSE_XLSX = "curse-of-dimensionality.xlsx"
TARGET_COLUMN = "Ailment"  # detected automatically earlier; change if incorrect
FORCE_TASK = None  # 'classification' or 'regression' or None

OUT_DIR = "submission_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

def compute_vif(df):
    X = df.select_dtypes(include=[np.number]).copy()
    X = X.dropna(axis=1, how='all')
    if X.shape[1] < 2:
        return pd.DataFrame(columns=['feature','VIF'])
    Xc = sm.add_constant(X)
    vif = []
    for i, col in enumerate(X.columns):
        try:
            v = variance_inflation_factor(Xc.values, i+1)
        except Exception:
            v = float('nan')
        vif.append((col, float(v)))
    return pd.DataFrame(vif, columns=['feature','VIF']).sort_values('VIF', ascending=False)

def basic_preprocess(df, target_col=None, drop_cols=None):
    df = df.copy()
    if drop_cols:
        for c in drop_cols:
            if c in df.columns:
                df = df.drop(columns=c)
    for col in df.select_dtypes(include=['object']).columns:
        if col == target_col:
            continue
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            df[col] = df[col].astype('category').cat.codes
    if target_col is not None:
        df = df.dropna(subset=[target_col])
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    return df

def save_fig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    print("Saved figure:", path)

def main():
    # Load patient data
    if not os.path.exists(PATIENT_CSV):
        # try alternative names in cwd
        candidates = [f for f in os.listdir('.') if f.lower().startswith('patient') and f.lower().endswith('.csv')]
        if candidates:
            print("Using alternative patient CSV:", candidates[0])
            patient_file = candidates[0]
        else:
            raise FileNotFoundError(f"Patient CSV '{PATIENT_CSV}' not found in working dir.")
    else:
        patient_file = PATIENT_CSV
    print("Loading patient file:", patient_file)
    df = pd.read_csv(patient_file)
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    # Determine target
    TARGET = TARGET_COLUMN if TARGET_COLUMN in df.columns else None
    if TARGET is None:
        # fallback heuristics
        prefs = ['Outcome','outcome','target','Target','label','Label','Y','y','Diagnosis','diagnosis','Class','class','response','Response']
        for c in prefs:
            if c in df.columns:
                TARGET = c
                break
    if TARGET is None:
        TARGET = df.columns[-1]
        print("Falling back to last column as target:", TARGET)
    print("Using target column:", TARGET)

    # Preprocess
    df_p = basic_preprocess(df, target_col=TARGET)
    print("After preprocessing shape:", df_p.shape)
    print("Target unique values:", df_p[TARGET].nunique())

    # Detect task
    n_unique = df_p[TARGET].nunique(dropna=True)
    if FORCE_TASK in ('classification','regression'):
        task = FORCE_TASK
    else:
        if (df_p[TARGET].dtype.kind in 'biufc' and n_unique > 20):
            task = 'regression'
        else:
            task = 'classification'
    print("Detected task:", task)

    # Correlation heatmap
    num_df = df_p.select_dtypes(include=[np.number]).copy()
    if TARGET in num_df.columns:
        num_df = num_df.drop(columns=[TARGET])
    corr = num_df.corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title('Pairwise correlation heatmap (numeric features)')
    heatmap_path = os.path.join(OUT_DIR, "correlation_heatmap.png")
    plt.savefig(heatmap_path, bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved correlation heatmap to", heatmap_path)

    # VIF and progressive dropping
    X = df_p.drop(columns=[TARGET]).select_dtypes(include=[np.number])
    y = df_p[TARGET]
    print("Numeric feature count:", X.shape[1])
    vif_df = compute_vif(X)
    print("\\nTop VIFs:\\n", vif_df.head(20).to_string(index=False))

    # Progressive drop tracking
    def eval_metric_for_model(est, Xmat, yvec, task):
        try:
            strat = yvec if (task=='classification' and len(np.unique(yvec))>2) else None
            Xtr, Xte, ytr, yte = train_test_split(Xmat, yvec, test_size=0.25, random_state=42, stratify=strat)
        except Exception:
            Xtr, Xte, ytr, yte = train_test_split(Xmat, yvec, test_size=0.25, random_state=42)
        sc = StandardScaler(); Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)
        est.fit(Xtr_s, ytr)
        yp = est.predict(Xte_s)
        if task=='classification':
            return float(f1_score(yte, np.round(yp), average='weighted', zero_division=0))
        else:
            return float(r2_score(yte, yp))

    history = []
    X_work = X.copy()
    step = 0
    est = RandomForestClassifier(n_estimators=100, random_state=0) if task=='classification' else RandomForestRegressor(n_estimators=100, random_state=0)
    while True:
        vif_now = compute_vif(X_work)
        metric_now = eval_metric_for_model(est, X_work, y, task)
        history.append({'step': step, 'n_features': X_work.shape[1], 'metric': metric_now, 'top_vif': vif_now.head(3)})
        print(f"Step {step}: features={X_work.shape[1]}, metric={metric_now:.4f}")
        if vif_now.empty:
            break
        maxv = vif_now['VIF'].iloc[0]
        if pd.isna(maxv) or maxv <= 10 or X_work.shape[1] <= 2:
            break
        dropf = vif_now['feature'].iloc[0]
        print(" Dropping feature due to high VIF:", dropf, " (VIF=", maxv, ")")
        X_work = X_work.drop(columns=[dropf])
        step += 1

    # Save history
    hist_df = pd.DataFrame([{'step':h['step'],'n_features':h['n_features'],'metric':h['metric']} for h in history])
    hist_df.to_csv(os.path.join(OUT_DIR, "vif_progression.csv"), index=False)
    print("Saved VIF progression to", os.path.join(OUT_DIR, "vif_progression.csv"))

    # PCA explained variance
    Xnum = df_p.drop(columns=[TARGET]).select_dtypes(include=[np.number])
    sc = StandardScaler(); Xs = sc.fit_transform(Xnum)
    pca = PCA(); pca.fit(Xs)
    var_ratio = pca.explained_variance_ratio_
    # plot variance and cumulative
    plt.figure(figsize=(10,4))
    plt.bar(np.arange(1, len(var_ratio)+1), var_ratio)
    plt.xlabel('PC'); plt.ylabel('Explained variance ratio'); plt.title('PCA: variance explained')
    plt.savefig(os.path.join(OUT_DIR, "pca_variance.png"), bbox_inches='tight', dpi=150); plt.close()
    plt.figure(figsize=(10,4))
    plt.plot(np.arange(1, len(var_ratio)+1), np.cumsum(var_ratio), marker='o')
    plt.xlabel('PC'); plt.ylabel('Cumulative explained variance'); plt.title('PCA: cumulative explained')
    plt.savefig(os.path.join(OUT_DIR, "pca_cumulative.png"), bbox_inches='tight', dpi=150); plt.close()
    print("Saved PCA variance plots to output folder.")

    # Models on top-k PCs vs original
    Xp = pca.transform(Xs)
    if task == 'classification':
        base_est = RandomForestClassifier(n_estimators=200, random_state=0)
        pcs = [2,3,5,10,20]
    else:
        base_est = RandomForestRegressor(n_estimators=200, random_state=0)
        pcs = [2,3,5,10,20]

    results = []
    for k in [p for p in pcs if p <= Xp.shape[1]]:
        Xk = Xp[:, :k]
        try:
            strat = y if (task=='classification' and len(np.unique(y))>2) else None
            Xtr, Xte, ytr, yte = train_test_split(Xk, y, test_size=0.25, random_state=42, stratify=strat)
        except Exception:
            Xtr, Xte, ytr, yte = train_test_split(Xk, y, test_size=0.25, random_state=42)
        base_est.fit(Xtr, ytr)
        yp = base_est.predict(Xte)
        if task=='classification':
            res = {'k':k, 'f1': float(f1_score(yte, np.round(yp), average='weighted', zero_division=0)), 'accuracy': float(accuracy_score(yte, np.round(yp)))}
        else:
            res = {'k':k, 'r2': float(r2_score(yte, yp)), 'mae': float(mean_absolute_error(yte, yp))}
        results.append(res)
    # baseline original
    try:
        strat = y if (task=='classification' and len(np.unique(y))>2) else None
        Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.25, random_state=42, stratify=strat)
    except Exception:
        Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.25, random_state=42)
    base_est.fit(Xtr, ytr)
    yp = base_est.predict(Xte)
    if task=='classification':
        base_res = {'k':'original', 'f1': float(f1_score(yte, np.round(yp), average='weighted', zero_division=0)), 'accuracy': float(accuracy_score(yte, np.round(yp)))}
    else:
        base_res = {'k':'original', 'r2': float(r2_score(yte, yp)), 'mae': float(mean_absolute_error(yte, yp))}
    results.append(base_res)
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUT_DIR, "pca_model_results.csv"), index=False)
    print("Saved PCA model comparison to", os.path.join(OUT_DIR, "pca_model_results.csv"))
    print(results_df)

    # MNIST part (if present)
    if MNIST_CSV and os.path.exists(MNIST_CSV):
        mn = pd.read_csv(MNIST_CSV, header=None)
        print("MNIST shape:", mn.shape)
        sc = StandardScaler(); Xm = sc.fit_transform(mn)
        pca_m = PCA(n_components=50); Xm_p = pca_m.fit_transform(Xm)
        plt.figure(figsize=(10,4))
        plt.bar(np.arange(1,51), pca_m.explained_variance_ratio_); plt.title('MNIST: first 50 PC variance')
        plt.savefig(os.path.join(OUT_DIR, "mnist_pca_variance.png"), bbox_inches='tight', dpi=150); plt.close()
        plt.figure(figsize=(6,6)); plt.scatter(Xm_p[:,0], Xm_p[:,1], s=2, alpha=0.6); plt.xlabel('PC1'); plt.ylabel('PC2'); plt.title('MNIST PC1 vs PC2'); plt.savefig(os.path.join(OUT_DIR, "mnist_pc1_pc2.png"), bbox_inches='tight', dpi=150); plt.close()
        subs = min(3000, Xm.shape[0])
        idx = np.random.RandomState(42).choice(Xm.shape[0], subs, replace=False)
        tsne = TSNE(n_components=2, init='pca', random_state=42, perplexity=30)
        Xm_emb = tsne.fit_transform(Xm[idx])
        plt.figure(figsize=(7,6)); plt.scatter(Xm_emb[:,0], Xm_emb[:,1], s=3, alpha=0.7); plt.title('t-SNE (mnist subsample)'); plt.savefig(os.path.join(OUT_DIR, "mnist_tsne.png"), bbox_inches='tight', dpi=150); plt.close()
        print("Saved MNIST visualizations to output folder.")

    print("\\nAll done. Outputs saved in folder:", OUT_DIR)
    print("Files:", os.listdir(OUT_DIR))

if __name__ == '__main__':
    main()
'''

out_path = 'submission_ex8_ready.py'
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(script)

out_path

