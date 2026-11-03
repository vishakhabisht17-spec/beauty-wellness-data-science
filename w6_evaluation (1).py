"""Week 6: evaluation framework for the review-risk model (real Sephora data).
Uses real_features.csv from real_eda.py (Week 4). Run: python w6_evaluation.py   (needs pandas numpy scipy scikit-learn matplotlib)
Outputs: w6_results.json, fig_x1..fig_x6.png
"""
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, precision_score, recall_score, f1_score
warnings.filterwarnings('ignore')
BASE = Path(__file__).resolve().parent
SEED = 42; RES = {}
BLUE, ORANGE, AQUA, GREY, GOLD = '#2a78d6', '#eb6834', '#1baf7a', '#8a8a8a', '#c9a227'
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 130})
def save(fig, n): fig.savefig(BASE / n, bbox_inches='tight', facecolor='#fcfcfb'); plt.close(fig)
def gb(): return HistGradientBoostingClassifier(max_depth=5, learning_rate=0.06, max_iter=200, l2_regularization=1.0, random_state=SEED)

d = pd.read_csv(BASE / 'real_features.csv'); d['product_id'] = d.review_id.str.split('_').str[-1]
META = ['review_id', 'rating', 'low_rating', 'split', 'product_id']
FEATS = [c for c in d.columns if c not in META and c != 'log_votes_given']      # votes arrive after posting -> excluded
d['n_words'] = np.expm1(d.log_words).round()
d['type'] = np.select([d.type_patch == 1, d.type_serum == 1, d.type_facewash == 1], ['patch', 'serum', 'facewash'], 'other')
d['skin'] = np.select([d.skin_combination == 1, d.skin_dry == 1, d.skin_normal == 1, d.skin_oily == 1], ['combination', 'dry', 'normal', 'oily'], 'unknown')
tr = d[d.split == 'train'].reset_index(drop=True); te = d[d.split == 'test'].reset_index(drop=True)
RES['data'] = {'rows': len(d), 'train': len(tr), 'test': len(te), 'test_products': int(te.product_id.nunique()), 'test_types': te.groupby('type').product_id.nunique().to_dict(),
               'prevalence_test': float(te.low_rating.mean())}

# ---------- Stage 1-2: model, out-of-fold predictions on training products, threshold rules
gkf = GroupKFold(n_splits=5); X, y, g = tr[FEATS].values, tr.low_rating.values, tr.product_id.values; oof = np.zeros(len(tr))
for a, b in gkf.split(X, y, g): oof[b] = gb().fit(X[a], y[a]).predict_proba(X[b])[:, 1]
RES['cv'] = {'auc': float(roc_auc_score(y, oof)), 'ap': float(average_precision_score(y, oof))}
pr, rc, th = precision_recall_curve(y, oof)
def thr_for_recall(t): return float(th[np.where(rc[:-1] >= t)[0][-1]])
THR = {'recall70': thr_for_recall(0.70), 'recall50': thr_for_recall(0.50), 'recall80': thr_for_recall(0.80)}
f1s = 2 * pr * rc / (pr + rc + 1e-12); THR['f1opt'] = float(th[int(np.nanargmax(f1s[:-1]))]); RES['thresholds'] = THR

# ---------- Stage 3: final test on unseen products
model = gb().fit(X, y); pt = model.predict_proba(te[FEATS].values)[:, 1]; yt = te.low_rating.values
RES['test'] = {'auc': float(roc_auc_score(yt, pt)), 'ap': float(average_precision_score(yt, pt))}
def at(thr, p=pt, yy=yt):
    yh = (p >= thr).astype(int); tp = int(((yh == 1) & (yy == 1)).sum()); fp = int(((yh == 1) & (yy == 0)).sum()); fn = int(((yh == 0) & (yy == 1)).sum()); tn = int(((yh == 0) & (yy == 0)).sum())
    return {'thr': float(thr), 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'precision': tp / max(tp + fp, 1), 'recall': tp / max(tp + fn, 1), 'f1': 2 * tp / max(2 * tp + fp + fn, 1),
            'flagged_share': float(yh.mean()), 'accuracy': (tp + tn) / len(yy), 'specificity': tn / max(tn + fp, 1)}
RES['ops'] = {k: at(v) for k, v in THR.items()}
RES['always_negative_accuracy'] = float(1 - yt.mean())

# bootstrap over PRODUCTS (reviews of one product are not independent)
rng = np.random.default_rng(SEED); pids = te.product_id.unique(); idx = {p: np.where(te.product_id.values == p)[0] for p in pids}; B = {'auc': [], 'ap': [], 'prec70': [], 'rec70': []}
for _ in range(1000):
    pick = rng.choice(pids, len(pids)); ii = np.concatenate([idx[p] for p in pick]); pp, yy = pt[ii], yt[ii]
    if yy.sum() == 0 or yy.sum() == len(yy): continue
    B['auc'].append(roc_auc_score(yy, pp)); B['ap'].append(average_precision_score(yy, pp)); o = at(THR['recall70'], pp, yy); B['prec70'].append(o['precision']); B['rec70'].append(o['recall'])
RES['bootstrap_ci'] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in B.items()}

# calibration (expected calibration error, deciles) and lift
q = pd.qcut(pt, 10, duplicates='drop'); cal = pd.DataFrame({'p': pt, 'y': yt, 'q': q}).groupby('q', observed=True).agg(p=('p', 'mean'), y=('y', 'mean'), n=('y', 'size'))
RES['calibration'] = {'pred': cal.p.round(4).tolist(), 'obs': cal.y.round(4).tolist(), 'ece': float((cal.n * (cal.p - cal.y).abs()).sum() / cal.n.sum())}
order = np.argsort(-pt); ys = yt[order]; tot = ys.sum(); lift = {}
for frac in [0.05, 0.10, 0.20, 0.30, 0.50]:
    k = int(frac * len(ys)); lift[str(frac)] = {'captured': float(ys[:k].sum() / tot), 'lift': float(ys[:k].mean() / yt.mean())}
RES['lift'] = lift

# ---------- Subgroups on unseen products
def sub(mask):
    m = mask.values if hasattr(mask, 'values') else mask
    if m.sum() < 200 or yt[m].sum() < 20: return None
    o = at(THR['recall70'], pt[m], yt[m]); return {'n': int(m.sum()), 'unhappy_share': float(yt[m].mean()), 'auc': float(roc_auc_score(yt[m], pt[m])), 'ap': float(average_precision_score(yt[m], pt[m])), 'recall': o['recall'], 'precision': o['precision']}
RES['by_type'] = {k: sub(te.type == k) for k in ['facewash', 'serum', 'patch'] if (te.type == k).any()}
RES['by_skin'] = {k: sub(te.skin == k) for k in ['combination', 'dry', 'normal', 'oily', 'unknown']}
bins = [(0, 19, '0-19 words'), (20, 59, '20-59'), (60, 119, '60-119'), (120, 10000, '120+')]
RES['by_length'] = {lab: sub((te.n_words >= lo) & (te.n_words <= hi)) for lo, hi, lab in bins}
RES['by_product'] = {}
pp_ = te.assign(p=pt).groupby('product_id').agg(n=('p', 'size'), obs=('low_rating', 'mean'), pred=('p', 'mean'))
from scipy import stats as st
RES['product_rank'] = {'spearman': float(st.spearmanr(pp_.pred, pp_.obs)[0]), 'n': int(len(pp_)), 'top5_overlap': int(len(set(pp_.sort_values('pred').tail(5).index) & set(pp_.sort_values('obs').tail(5).index)))}

# ---------- Stage 4: time-based test (train on older reviews, test on newer)
cut = d.days_since_start.quantile(0.75); old = d[d.days_since_start <= cut].reset_index(drop=True); new = d[d.days_since_start > cut].reset_index(drop=True)
Xo, yo, go = old[FEATS].values, old.low_rating.values, old.product_id.values; oof_o = np.zeros(len(old))
for a, b in gkf.split(Xo, yo, go): oof_o[b] = gb().fit(Xo[a], yo[a]).predict_proba(Xo[b])[:, 1]
pro, rco, tho = precision_recall_curve(yo, oof_o); thr_o = float(tho[np.where(rco[:-1] >= 0.70)[0][-1]])
mo = gb().fit(Xo, yo); pn = mo.predict_proba(new[FEATS].values)[:, 1]; yn = new.low_rating.values
d0 = pd.Timestamp('2008-08-28'); RES['time_split'] = {'cutoff_date': str((d0 + pd.Timedelta(days=float(cut))).date()), 'train_rows': len(old), 'test_rows': len(new), 'train_unhappy': float(yo.mean()), 'test_unhappy': float(yn.mean()),
    'auc': float(roc_auc_score(yn, pn)), 'ap': float(average_precision_score(yn, pn)), 'op70': at(thr_o, pn, yn), 'thr': thr_o}
# monthly stability of the test-period AUC
nm = new.assign(p=pn, m=(d0 + pd.to_timedelta(new.days_since_start, unit='D')).dt.to_period('Q').astype(str)); qa = {}
for k, gq in nm.groupby('m'):
    if len(gq) > 1000 and 0 < gq.low_rating.sum(): qa[k] = {'n': int(len(gq)), 'auc': float(roc_auc_score(gq.low_rating, gq.p))}
RES['time_split']['by_quarter'] = qa

# ---------- Business illustration per 10,000 new reviews (counts only; no money figures invented)
ill = {}
for k in ['recall50', 'recall70', 'recall80']:
    o = RES['ops'][k]; unhappy = 10000 * yt.mean(); ill[k] = {'unhappy': float(unhappy), 'flagged': float(10000 * o['flagged_share']), 'caught': float(unhappy * o['recall']), 'false_alarms': float(10000 * o['flagged_share'] - unhappy * o['recall']), 'missed': float(unhappy * (1 - o['recall']))}
RES['per_10000'] = ill

# ---------- Figures
fig, ax = plt.subplots(figsize=(10.5, 3.1)); ax.axis('off'); ax.set_xlim(0, 100); ax.set_ylim(0, 30)
stages = [('1  Split by product', 'Keep 24 products\nfully unseen'), ('2  Cross-validation', '5 folds, products\nnever shared'), ('3  Final test', 'Unseen products,\nopened once'), ('4  Time test', 'Train on older,\ntest on newer'), ('5  Live monitoring', 'Weekly metrics,\nretrain on drift')]
for i, (t, s) in enumerate(stages):
    x = 1 + i * 19.6; ax.add_patch(FancyBboxPatch((x, 6), 17, 18, boxstyle='round,pad=0.4', fc=[BLUE, BLUE, BLUE, ORANGE, GREY][i], ec='none', alpha=.92)); ax.text(x + 8.5, 19, t, ha='center', va='center', color='white', fontsize=8.5, fontweight='bold'); ax.text(x + 8.5, 11.5, s, ha='center', va='center', color='white', fontsize=8)
    if i < 4: ax.add_patch(FancyArrowPatch((x + 17.6, 15), (x + 19.2, 15), arrowstyle='-|>', mutation_scale=10, color='#444'))
ax.text(1, 1.2, 'Blue = run in this report;  orange = run as an extra check;  grey = plan for deployment', fontsize=8, color='#555')
save(fig, 'fig_x1_stages.png')

fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8)); fpr, tpr, _ = roc_curve(yt, pt); ax[0].plot(fpr, tpr, color=BLUE, lw=2); ax[0].plot([0, 1], [0, 1], color=GREY, ls='--', lw=.8)
for k, col in [('recall50', GOLD), ('recall70', ORANGE), ('recall80', AQUA)]:
    o = RES['ops'][k]; ax[0].scatter(1 - o['specificity'], o['recall'], color=col, zorder=3, s=30); ax[0].annotate(f"{int(round(o['recall'] * 100))}% recall", (1 - o['specificity'], o['recall']), xytext=(6, -10), textcoords='offset points', fontsize=7.5)
ax[0].set_xlabel('False alarm rate (happy reviews flagged)'); ax[0].set_ylabel('Share of unhappy reviews found'); ax[0].set_title(f"ROC curve, unseen products (AUC {RES['test']['auc']:.2f})", fontsize=10)
p2, r2, _ = precision_recall_curve(yt, pt); ax[1].plot(r2, p2, color=BLUE, lw=2); ax[1].axhline(yt.mean(), color=GREY, ls='--', lw=.8); ax[1].text(0.02, yt.mean() + 0.02, f'random guess = {yt.mean():.0%}', fontsize=8, color='#555')
for k, col in [('recall50', GOLD), ('recall70', ORANGE), ('recall80', AQUA)]:
    o = RES['ops'][k]; ax[1].scatter(o['recall'], o['precision'], color=col, zorder=3, s=30); ax[1].annotate(f"{o['precision']:.0%} precise", (o['recall'], o['precision']), xytext=(6, 4), textcoords='offset points', fontsize=7.5)
ax[1].set_xlabel('Share of unhappy reviews found (recall)'); ax[1].set_ylabel('Share of flags that are right (precision)'); ax[1].set_title(f"Precision-recall curve (PR AUC {RES['test']['ap']:.2f})", fontsize=10)
save(fig, 'fig_x2_roc_pr.png')

fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.5)); o = RES['ops']['recall70']; cm = np.array([[o['tn'], o['fp']], [o['fn'], o['tp']]])
ax[0].imshow(np.array([[0, 1], [1, 0]]), cmap=matplotlib.colors.ListedColormap(['#d9ecff', '#ffe3d6']), vmin=0, vmax=1)
for i in range(2):
    for j in range(2): ax[0].text(j, i, f'{cm[i, j]:,}\n' + ['happy, not flagged', 'false alarm', 'missed', 'unhappy, caught'][i * 2 + j], ha='center', va='center', fontsize=9)
ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(['Not flagged', 'Flagged']); ax[0].set_yticks([0, 1]); ax[0].set_yticklabels(['Happy', 'Unhappy']); ax[0].set_title('Outcomes at the 70% recall target (29,226 test reviews)', fontsize=10)
for s_ in ax[0].spines.values(): s_.set_visible(False)
fr = [0.05, 0.1, 0.2, 0.3, 0.5]; cap = [lift[str(f)]['captured'] for f in fr]
ax[1].bar([f'top {int(f * 100)}%' for f in fr], cap, color=BLUE, width=.6); ax[1].plot(range(5), fr, color=GREY, marker='o', ms=3, ls='--', lw=1)
for i, v in enumerate(cap): ax[1].text(i, v + .015, f'{v:.0%}', ha='center', fontsize=8)
ax[1].set_ylim(0, 1); ax[1].set_ylabel('Share of unhappy reviews captured'); ax[1].set_title('Reading only the highest-risk reviews (dashed = random)', fontsize=10)
save(fig, 'fig_x3_confusion_lift.png')

fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.4), gridspec_kw={'width_ratios': [2, 4, 4]})
def bars(a, dct, ttl):
    keys = [k for k, v in dct.items() if v]; v = [dct[k]['auc'] for k in keys]; a.bar(range(len(keys)), v, color=BLUE, width=.6); a.axhline(RES['test']['auc'], color=ORANGE, ls='--', lw=1)
    for i, x in enumerate(v): a.text(i, x + .01, f'{x:.2f}', ha='center', fontsize=8)
    a.set_xticks(range(len(keys))); a.set_xticklabels(keys, fontsize=8, rotation=20 if len(keys) > 3 else 0); a.set_ylim(.5, 1); a.set_title(ttl, fontsize=9.5)
bars(ax[0], RES['by_type'], 'By product type'); bars(ax[1], RES['by_skin'], 'By skin type'); bars(ax[2], RES['by_length'], 'By review length'); ax[0].set_ylabel('ROC AUC (dashed = overall)')
save(fig, 'fig_x4_subgroups.png')

fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.4)); lab = ['Unseen products\n(final test)', 'Newer reviews\n(time test)']; v = [RES['test']['auc'], RES['time_split']['auc']]; v2 = [RES['test']['ap'], RES['time_split']['ap']]
ax[0].bar(np.arange(2) - .2, v, .38, color=BLUE, label='ROC AUC'); ax[0].bar(np.arange(2) + .2, v2, .38, color=ORANGE, label='PR AUC')
for i in range(2): ax[0].text(i - .2, v[i] + .01, f'{v[i]:.2f}', ha='center', fontsize=8); ax[0].text(i + .2, v2[i] + .01, f'{v2[i]:.2f}', ha='center', fontsize=8)
ax[0].set_xticks(range(2)); ax[0].set_xticklabels(lab, fontsize=8.5); ax[0].set_ylim(0, 1); ax[0].legend(frameon=False, fontsize=8); ax[0].set_title('Does it hold up on new products and on newer reviews?', fontsize=10)
ks = list(RES['time_split']['by_quarter']); ax[1].plot(range(len(ks)), [RES['time_split']['by_quarter'][k]['auc'] for k in ks], color=BLUE, marker='o'); ax[1].set_xticks(range(len(ks))); ax[1].set_xticklabels(ks, rotation=60, fontsize=7); ax[1].set_ylim(.7, 1); ax[1].set_title('Test-period AUC by quarter', fontsize=10)
save(fig, 'fig_x5_time.png')

json.dump(RES, open(BASE / 'w6_results.json', 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)); print('done')
