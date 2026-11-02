"""Week 5: predictive modelling on the real Sephora features made by real_eda.py (real_features.csv).
Run: python w5_models.py   (needs real_features.csv in the same folder; pandas numpy scipy scikit-learn matplotlib)
Outputs: w5_results.json, fig_w1..fig_w6.png
"""
import json, warnings, time
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, precision_recall_curve,
                             mean_squared_error, mean_absolute_error, r2_score, silhouette_score, brier_score_loss)
from sklearn.inspection import permutation_importance
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')
BASE = Path(__file__).resolve().parent
SEED = 42; RES = {}
BLUE, ORANGE, AQUA, GREY = '#2a78d6', '#eb6834', '#1baf7a', '#8a8a8a'
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 130})
def save(fig, n): fig.savefig(BASE / n, bbox_inches='tight', facecolor='#fcfcfb'); plt.close(fig)

d = pd.read_csv(BASE / 'real_features.csv')
d['product_id'] = d.review_id.str.split('_').str[-1]
TEXT = ['pos_count', 'neg_count', 'sentiment_score', 'negation_count', 'patch_x_deep', 'sens_x_smell'] + [c for c in d.columns if c.startswith('flag_')]
META = ['review_id', 'rating', 'low_rating', 'split', 'product_id']
ALL = [c for c in d.columns if c not in META]
LEAKY = ['log_votes_given']          # helpful votes arrive AFTER a review is posted -> not available at prediction time
ALL_WITH_LEAK = list(ALL)
ALL = [c for c in ALL if c not in LEAKY]
STRUCT = [c for c in ALL if c not in TEXT]
RES['data'] = {'rows': len(d), 'products': int(d.product_id.nunique()), 'n_all_features': len(ALL), 'n_structured': len(STRUCT), 'n_text': len(ALL) - len(STRUCT),
               'low_rating_share': float(d.low_rating.mean())}
tr = d[d.split == 'train'].reset_index(drop=True); te = d[d.split == 'test'].reset_index(drop=True)
RES['data'].update({'train_rows': len(tr), 'test_rows': len(te), 'train_products': int(tr.product_id.nunique()), 'test_products': int(te.product_id.nunique()),
                    'test_low_share': float(te.low_rating.mean())})
gkf = GroupKFold(n_splits=5)

# ===================================================== 1. CLASSIFICATION: will a review be 1-2 stars?
def clf_models():
    return {
        'Logistic regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight='balanced')),
        'Decision tree (depth 6)': DecisionTreeClassifier(max_depth=6, min_samples_leaf=50, class_weight='balanced', random_state=SEED),
        'Random forest': RandomForestClassifier(n_estimators=120, max_depth=14, min_samples_leaf=20, max_samples=0.3, class_weight='balanced_subsample', n_jobs=-1, random_state=SEED),
        'Gradient boosting': HistGradientBoostingClassifier(max_depth=5, learning_rate=0.06, max_iter=200, l2_regularization=1.0, random_state=SEED),
    }
def evaluate(model, Xtr, ytr, Xte, yte):
    m = model.fit(Xtr, ytr); p = m.predict_proba(Xte)[:, 1]
    return p
rng = np.random.default_rng(SEED)
cv_out = {}; oof = {}
for fs_name, cols in [('structured only', STRUCT), ('structured + text', ALL)]:
    X = tr[cols].values; y = tr.low_rating.values; g = tr.product_id.values
    for mname in clf_models():
        t0 = time.time(); aucs, aps, fold_p = [], [], np.zeros(len(tr))
        for tri, vai in gkf.split(X, y, g):
            m = clf_models()[mname].fit(X[tri], y[tri]); p = m.predict_proba(X[vai])[:, 1]; fold_p[vai] = p
            aucs.append(roc_auc_score(y[vai], p)); aps.append(average_precision_score(y[vai], p))
        cv_out[f'{fs_name} | {mname}'] = {'auc_mean': float(np.mean(aucs)), 'auc_sd': float(np.std(aucs)), 'ap_mean': float(np.mean(aps)), 'ap_sd': float(np.std(aps)), 'secs': round(time.time() - t0, 1)}
        oof[f'{fs_name} | {mname}'] = fold_p
        print('cv', fs_name, mname, round(np.mean(aucs), 3), round(time.time() - t0, 1), flush=True)
RES['clf_cv'] = cv_out
# baselines
RES['baseline'] = {'prevalence_train': float(tr.low_rating.mean()), 'prevalence_test': float(te.low_rating.mean()), 'always_negative_accuracy_test': float(1 - te.low_rating.mean()), 'random_ap': float(te.low_rating.mean())}

# held-out test (touched once, after model choice on CV)
test_out = {}; test_p = {}
for fs_name, cols in [('structured only', STRUCT), ('structured + text', ALL)]:
    for mname in clf_models():
        m = clf_models()[mname].fit(tr[cols].values, tr.low_rating.values); p = m.predict_proba(te[cols].values)[:, 1]
        test_p[f'{fs_name} | {mname}'] = p
        test_out[f'{fs_name} | {mname}'] = {'auc': float(roc_auc_score(te.low_rating, p)), 'ap': float(average_precision_score(te.low_rating, p)),
                                            'brier': float(brier_score_loss(te.low_rating, np.clip(p, 0, 1)))}
RES['clf_test'] = test_out

# threshold choice from out-of-fold predictions of the best structured+text model
best = max([k for k in cv_out if k.startswith('structured + text')], key=lambda k: cv_out[k]['ap_mean']); RES['best_model'] = best
pr, rc, th = precision_recall_curve(tr.low_rating.values, oof[best])
rows = []
for target_recall in [0.5, 0.7, 0.8, 0.9]:
    i = np.where(rc[:-1] >= target_recall)[0][-1]; t_ = th[i]
    pt = test_p[best]; yhat = (pt >= t_).astype(int)
    rows.append({'target_recall_cv': target_recall, 'threshold': float(t_), 'cv_precision': float(pr[i]), 'test_precision': float(precision_score(te.low_rating, yhat)),
                 'test_recall': float(recall_score(te.low_rating, yhat)), 'test_f1': float(f1_score(te.low_rating, yhat)), 'flagged_share_test': float(yhat.mean())})
RES['thresholds'] = rows
# F1-optimal threshold
f1s = 2 * pr * rc / (pr + rc + 1e-12); ib = int(np.nanargmax(f1s[:-1])); tb = th[ib]; yb = (test_p[best] >= tb).astype(int)
RES['f1_opt'] = {'threshold': float(tb), 'cv_f1': float(f1s[ib]), 'test_precision': float(precision_score(te.low_rating, yb)), 'test_recall': float(recall_score(te.low_rating, yb)),
                 'test_f1': float(f1_score(te.low_rating, yb)), 'tp': int(((yb == 1) & (te.low_rating == 1)).sum()), 'fp': int(((yb == 1) & (te.low_rating == 0)).sum()),
                 'fn': int(((yb == 0) & (te.low_rating == 1)).sum()), 'tn': int(((yb == 0) & (te.low_rating == 0)).sum())}

# ===================================================== 2. OVER / UNDER-FITTING: decision-tree depth, boosting iterations
depths = list(range(1, 21)); dt_tr, dt_va = [], []
Xs, ys, gs = tr[ALL].values, tr.low_rating.values, tr.product_id.values
folds = list(gkf.split(Xs, ys, gs))
for dep in depths:
    a_tr, a_va = [], []
    for tri, vai in folds:
        m = DecisionTreeClassifier(max_depth=dep, min_samples_leaf=5, random_state=SEED).fit(Xs[tri], ys[tri])
        a_tr.append(roc_auc_score(ys[tri], m.predict_proba(Xs[tri])[:, 1])); a_va.append(roc_auc_score(ys[vai], m.predict_proba(Xs[vai])[:, 1]))
    dt_tr.append(float(np.mean(a_tr))); dt_va.append(float(np.mean(a_va)))
RES['tree_depth_curve'] = {'depths': depths, 'train_auc': dt_tr, 'cv_auc': dt_va, 'best_depth': depths[int(np.argmax(dt_va))], 'best_cv_auc': float(max(dt_va)),
                           'depth20_gap': float(dt_tr[-1] - dt_va[-1]), 'depth1_cv': float(dt_va[0])}
# boosting: train vs validation AUC as iterations grow (one product-grouped split of the training data)
tri, vai = folds[0]
hg = HistGradientBoostingClassifier(max_depth=5, learning_rate=0.1, max_iter=400, l2_regularization=0.0, early_stopping=False, random_state=SEED).fit(Xs[tri], ys[tri])
its = list(range(10, 401, 10)); b_tr, b_va = [], []
sub = np.random.default_rng(SEED).choice(len(tri), 15000, replace=False)
for k, (pt_, pv_) in enumerate(zip(hg.staged_predict_proba(Xs[tri][sub]), hg.staged_predict_proba(Xs[vai]))):
    if (k + 1) in its: b_tr.append(float(roc_auc_score(ys[tri][sub], pt_[:, 1]))); b_va.append(float(roc_auc_score(ys[vai], pv_[:, 1])))
RES['boost_curve'] = {'iters': its, 'train_auc': b_tr, 'val_auc': b_va, 'best_iter': its[int(np.argmax(b_va))], 'best_val': float(max(b_va)), 'final_gap': float(b_tr[-1] - b_va[-1])}

# ===================================================== 3. REGRESSION: predict the star rating
def reg_models():
    return {'Mean baseline': None, 'Linear regression': make_pipeline(StandardScaler(), LinearRegression()),
            'Ridge (alpha 10)': make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
            'Random forest': RandomForestRegressor(n_estimators=120, max_depth=14, min_samples_leaf=20, max_samples=0.3, n_jobs=-1, random_state=SEED),
            'Gradient boosting': HistGradientBoostingRegressor(max_depth=5, learning_rate=0.06, max_iter=200, l2_regularization=1.0, random_state=SEED)}
yr = tr.rating.values.astype(float); reg_cv = {}
for mname in reg_models():
    r, ma, r2 = [], [], []
    for tri, vai in folds:
        if mname == 'Mean baseline': p = np.full(len(vai), yr[tri].mean())
        else: p = reg_models()[mname].fit(Xs[tri], yr[tri]).predict(Xs[vai])
        p = np.clip(p, 1, 5); r.append(np.sqrt(mean_squared_error(yr[vai], p))); ma.append(mean_absolute_error(yr[vai], p)); r2.append(r2_score(yr[vai], p))
    reg_cv[mname] = {'rmse': float(np.mean(r)), 'rmse_sd': float(np.std(r)), 'mae': float(np.mean(ma)), 'r2': float(np.mean(r2))}
    print('reg', mname, round(np.mean(r), 3), flush=True)
RES['reg_cv'] = reg_cv
reg_te = {}; yte = te.rating.values.astype(float)
for mname in reg_models():
    p = np.full(len(te), yr.mean()) if mname == 'Mean baseline' else reg_models()[mname].fit(Xs, yr).predict(te[ALL].values)
    p = np.clip(p, 1, 5); reg_te[mname] = {'rmse': float(np.sqrt(mean_squared_error(yte, p))), 'mae': float(mean_absolute_error(yte, p)), 'r2': float(r2_score(yte, p))}
    if mname == 'Gradient boosting':
        err = np.abs(yte - p); RES['reg_error_by_rating'] = {int(k): float(err[yte == k].mean()) for k in range(1, 6)}
RES['reg_test'] = reg_te

# ===================================================== 4. INTERPRETATION: odds ratios (LR) + permutation importance (GB)
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(tr[ALL], tr.low_rating)
co = pd.Series(lr[-1].coef_[0], index=ALL); orr = np.exp(co).sort_values()
RES['lr_odds_per_sd'] = {'highest': orr.tail(8).round(2).to_dict(), 'lowest': orr.head(8).round(2).to_dict()}
gbm = HistGradientBoostingClassifier(max_depth=5, learning_rate=0.06, max_iter=200, l2_regularization=1.0, random_state=SEED).fit(tr[ALL], tr.low_rating)
samp = te.sample(8000, random_state=SEED)
pi = permutation_importance(gbm, samp[ALL], samp.low_rating, scoring='roc_auc', n_repeats=5, random_state=SEED, n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=ALL).sort_values(ascending=False); impsd = pd.Series(pi.importances_std, index=ALL)
RES['perm_importance'] = {k: {'auc_drop': float(v), 'sd': float(impsd[k])} for k, v in imp.head(12).items()}
# group-level calibration check on test set
p_ = test_p[best]; q = pd.qcut(p_, 10, duplicates='drop'); cal = pd.DataFrame({'p': p_, 'y': te.low_rating.values, 'q': q}).groupby('q', observed=True).agg(p=('p', 'mean'), y=('y', 'mean'))
RES['calibration_gb'] = {'decile_pred': cal.p.round(3).tolist(), 'decile_obs': cal.y.round(3).tolist()}
# product-level check: does the model rank products by low-rating share?
pr_df = te.assign(p=test_p[best]).groupby('product_id').agg(pred=('p', 'mean'), obs=('low_rating', 'mean'), n=('low_rating', 'size'))
from scipy import stats as st
RES['product_level'] = {'n_products': int(len(pr_df)), 'spearman': float(st.spearmanr(pr_df.pred, pr_df.obs)[0]), 'p': float(st.spearmanr(pr_df.pred, pr_df.obs)[1])}

# ===================================================== LEAK CHECK: what if the post-hoc vote feature is allowed?
leak = {}
for nm_, cols_ in [('without votes (used)', ALL), ('with votes (leaky)', ALL_WITH_LEAK)]:
    X_ = tr[cols_].values; aucs_ = []
    for tri, vai in gkf.split(X_, tr.low_rating.values, tr.product_id.values):
        m_ = HistGradientBoostingClassifier(max_depth=5, learning_rate=0.06, max_iter=200, l2_regularization=1.0, random_state=SEED).fit(X_[tri], tr.low_rating.values[tri])
        aucs_.append(roc_auc_score(tr.low_rating.values[vai], m_.predict_proba(X_[vai])[:, 1]))
    m_ = HistGradientBoostingClassifier(max_depth=5, learning_rate=0.06, max_iter=200, l2_regularization=1.0, random_state=SEED).fit(tr[cols_].values, tr.low_rating.values)
    leak[nm_] = {'cv_auc': float(np.mean(aucs_)), 'cv_sd': float(np.std(aucs_)), 'test_auc': float(roc_auc_score(te.low_rating, m_.predict_proba(te[cols_].values)[:, 1]))}
vt = d.groupby('low_rating').log_votes_given.mean(); leak['mean_log_votes_low'] = float(vt.loc[1]); leak['mean_log_votes_not_low'] = float(vt.loc[0])
RES['leak_check'] = leak

# ===================================================== 5. CLUSTERING: group products by complaint profile
prof = d.groupby('product_id').agg(n=('rating', 'size'), mean_rating=('rating', 'mean'), low_rate=('low_rating', 'mean'), irrit=('flag_irrit', 'mean'), smell=('flag_smell', 'mean'),
                                   breakout=('flag_breakout', 'mean'), not_eff=('flag_not_eff', 'mean'), praise=('flag_praise', 'mean'), log_price=('log_price', 'mean'),
                                   fragranced=('fragranced', 'first'), retinoid=('act_retinoid', 'first'), patch=('type_patch', 'first'), facewash=('type_facewash', 'first'))
prof = prof[prof.n >= 100]; cols_c = ['mean_rating', 'low_rate', 'irrit', 'smell', 'breakout', 'not_eff', 'praise', 'log_price', 'fragranced', 'retinoid', 'facewash']
Z = StandardScaler().fit_transform(prof[cols_c]); sil = {}
for k in range(2, 8):
    km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(Z); sil[k] = float(silhouette_score(Z, km.labels_))
kbest = max(sil, key=sil.get); km = KMeans(n_clusters=kbest, n_init=20, random_state=SEED).fit(Z); prof['cluster'] = km.labels_
# stability: re-run with different seeds, adjusted rand index
from sklearn.metrics import adjusted_rand_score
aris = [adjusted_rand_score(km.labels_, KMeans(n_clusters=kbest, n_init=20, random_state=s).fit(Z).labels_) for s in range(1, 6)]
RES['cluster'] = {'n_products': int(len(prof)), 'silhouette_by_k': sil, 'k': int(kbest), 'ari_mean': float(np.mean(aris)),
                  'profiles': prof.groupby('cluster')[['n', 'mean_rating', 'low_rate', 'irrit', 'smell', 'breakout', 'not_eff', 'praise', 'log_price', 'fragranced', 'retinoid', 'facewash', 'patch']].mean().round(3).to_dict('index'),
                  'sizes': prof.cluster.value_counts().sort_index().to_dict(), 'patch_cluster': int(prof[prof.patch == 1].cluster.iloc[0]) if (prof.patch == 1).any() else None}
pca = PCA(n_components=2, random_state=SEED).fit(Z); P2 = pca.transform(Z); RES['cluster']['pca_var'] = pca.explained_variance_ratio_.round(3).tolist()

# ===================================================== 6. FORECASTING a trend: monthly share of 1-2 star reviews
ds0 = pd.Timestamp('2008-08-28'); dd = d.assign(date=ds0 + pd.to_timedelta(d.days_since_start, unit='D')); dd['ym'] = dd.date.dt.to_period('M')
ms = dd.groupby('ym').agg(n=('low_rating', 'size'), low=('low_rating', 'mean')); ms = ms[(ms.n >= 300)]
ms = ms.loc['2019-01':'2022-12'] if len(ms.loc['2019-01':'2022-12']) >= 24 else ms
series = ms.low.values; months = ms.index.astype(str).tolist(); h = 1; start = 24; res_f = {'naive': [], 'moving_avg_3': [], 'linear_trend': [], 'seasonal_naive': []}; act = []
for t in range(start, len(series)):
    hist = series[:t]; act.append(series[t])
    res_f['naive'].append(hist[-1]); res_f['moving_avg_3'].append(hist[-3:].mean())
    xs = np.arange(len(hist)); b = np.polyfit(xs, hist, 1); res_f['linear_trend'].append(np.polyval(b, len(hist)))
    res_f['seasonal_naive'].append(hist[-12])
act = np.array(act); RES['forecast'] = {'months': len(series), 'first': months[0], 'last': months[-1], 'test_months': int(len(act)), 'mean_low_share': float(series.mean()),
    'sd_low_share': float(series.std()), 'models': {k: {'rmse': float(np.sqrt(np.mean((np.array(v) - act) ** 2))), 'mae': float(np.mean(np.abs(np.array(v) - act)))} for k, v in res_f.items()}}

# ===================================================== figures
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6)); names = list(clf_models())
for a_, key, ttl in [(ax[0], 'auc', 'ROC AUC (higher is better)'), (ax[1], 'ap', 'PR AUC / average precision')]:
    w = .38
    for j, (fs_name, col) in enumerate([('structured only', BLUE), ('structured + text', ORANGE)]):
        v = [cv_out[f'{fs_name} | {n}'][f'{key}_mean'] for n in names]; e = [cv_out[f'{fs_name} | {n}'][f'{key}_sd'] for n in names]
        a_.bar(np.arange(4) + (j - .5) * w, v, w, yerr=e, color=col, capsize=2, label=fs_name)
        for i_, (v_, e_) in enumerate(zip(v, e)): a_.text(i_ + (j - .5) * w, v_ + e_ + .01, f'{v_:.2f}', ha='center', fontsize=7)
    a_.set_xticks(range(4)); a_.set_xticklabels(['Logistic', 'Tree d=6', 'Random\nforest', 'Gradient\nboosting'], fontsize=8); a_.set_title(ttl, fontsize=10)
ax[0].axhline(.5, color=GREY, lw=.8, ls='--'); ax[0].set_ylim(.4, 1.0); ax[0].legend(frameon=False, fontsize=8, loc='upper left')
ax[1].axhline(RES['baseline']['random_ap'], color=GREY, lw=.8, ls='--'); ax[1].set_ylim(0, 1.0)
save(fig, 'fig_w1_classifiers.png')

fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.4))
ax[0].plot(depths, dt_tr, color=ORANGE, label='train'); ax[0].plot(depths, dt_va, color=BLUE, label='validation (grouped CV)')
ax[0].axvline(RES['tree_depth_curve']['best_depth'], color=GREY, ls=':', lw=1); ax[0].set_xlabel('Tree depth'); ax[0].set_ylabel('ROC AUC'); ax[0].set_title('Decision tree: underfit to overfit', fontsize=10); ax[0].legend(frameon=False, fontsize=8)
ax[1].plot(its, b_tr, color=ORANGE, label='train'); ax[1].plot(its, b_va, color=BLUE, label='validation'); ax[1].set_xlabel('Boosting iterations'); ax[1].set_title('Gradient boosting (no regularisation)', fontsize=10); ax[1].legend(frameon=False, fontsize=8)
save(fig, 'fig_w2_fit.png')

fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6))
pt_all = test_p[best]; prt, rct, tht = precision_recall_curve(te.low_rating, pt_all); ax[0].plot(rct, prt, color=BLUE); ax[0].axhline(te.low_rating.mean(), color=GREY, ls='--', lw=.8)
for r_ in rows: ax[0].scatter(r_['test_recall'], r_['test_precision'], color=ORANGE, zorder=3, s=25); ax[0].annotate(f"recall {r_['target_recall_cv']:.0%} target", (r_['test_recall'], r_['test_precision']), textcoords='offset points', xytext=(5, 5), fontsize=7)
ax[0].set_xlabel('Recall'); ax[0].set_ylabel('Precision'); ax[0].set_title('Precision-recall curve (held-out products)', fontsize=10)
ax[1].plot([0, .6], [0, .6], color=GREY, ls='--', lw=.8); ax[1].plot(cal.p.values, cal.y.values, color=BLUE, marker='o'); ax[1].set_xlabel('Mean predicted probability (decile)'); ax[1].set_ylabel('Observed share 1-2 stars'); ax[1].set_title('Calibration on held-out products', fontsize=10)
save(fig, 'fig_w3_pr_calibration.png')

fig, ax = plt.subplots(figsize=(6.2, 3.6)); top = imp.head(10)[::-1]; ax.barh(top.index, top.values, xerr=impsd[top.index].values, color=BLUE); ax.set_xlabel('Drop in ROC AUC when the feature is shuffled'); ax.set_title('Permutation importance (held-out sample)', fontsize=10)
save(fig, 'fig_w4_importance.png')

fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.4)); nm = list(reg_cv); v = [reg_cv[n]['rmse'] for n in nm]; e = [reg_cv[n]['rmse_sd'] for n in nm]
ax[0].bar(range(len(nm)), v, yerr=e, color=[GREY] + [BLUE] * 4, capsize=3)
for i_, (v_, e_) in enumerate(zip(v, e)): ax[0].text(i_, v_ + e_ + .01, f'{v_:.2f}', ha='center', fontsize=8)
ax[0].set_xticks(range(len(nm))); ax[0].set_xticklabels(['Mean\nbaseline', 'Linear', 'Ridge', 'Random\nforest', 'Gradient\nboosting'], fontsize=8); ax[0].set_ylabel('RMSE (stars)'); ax[0].set_title('Predicting the star rating (CV)', fontsize=10)
eb = RES['reg_error_by_rating']; ax[1].bar(list(eb.keys()), list(eb.values()), color=ORANGE); ax[1].set_xlabel('True rating'); ax[1].set_ylabel('Mean absolute error'); ax[1].set_title('Error by true rating (boosting, test)', fontsize=10)
save(fig, 'fig_w5_regression.png')

fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6)); pal = [BLUE, ORANGE, AQUA, '#c9a227', '#8a5fd6', '#d65f8a', GREY]
for k_ in range(kbest): m_ = prof.cluster.values == k_; ax[0].scatter(P2[m_, 0], P2[m_, 1], s=prof.n.values[m_] / 40, color=pal[k_], alpha=.75, label=f'cluster {k_} (n={m_.sum()})')
if (prof.patch == 1).any(): pi_ = int(np.where(prof.patch.values == 1)[0][0]); ax[0].annotate('patch', P2[pi_], xytext=(6, 6), textcoords='offset points', fontsize=8)
ax[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.0%})'); ax[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.0%})'); ax[0].legend(frameon=False, fontsize=7); ax[0].set_title('Products by complaint profile', fontsize=10)
ax[1].plot(list(sil.keys()), list(sil.values()), color=BLUE, marker='o'); ax[1].set_xlabel('Number of clusters k'); ax[1].set_ylabel('Silhouette'); ax[1].set_title('Choosing k', fontsize=10)
save(fig, 'fig_w6_clusters.png')

json.dump(RES, open(BASE / 'w5_results.json', 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
print('done')
