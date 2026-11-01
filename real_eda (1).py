"""Week 4 (REAL DATA): EDA + feature engineering on the Sephora skincare reviews (Kaggle, file reviews_500-750.csv).
Run:  python real_eda.py   (put reviews_500-750.csv in the same folder; needs pandas numpy scipy scikit-learn matplotlib seaborn)
Outputs: real_results.json, real_features.csv, fig_r1..fig_r6.png
"""
import json, re, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
warnings.filterwarnings('ignore')
BASE = Path(__file__).resolve().parent
rng = np.random.default_rng(42)
BLUE, ORANGE, AQUA, GREY = '#2a78d6', '#eb6834', '#1baf7a', '#8a8a8a'
COL = {'facewash': BLUE, 'serum': ORANGE, 'patch': AQUA}
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 130})
RES = {}

# ------------------------------------------------------------ load + product type from product name
files = sorted(BASE.glob('reviews_*.csv'))
raw = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
RES['files'] = [f.name for f in files]
RES['raw_rows'] = len(raw); RES['raw_products'] = int(raw.product_id.nunique())

def ptype(name):
    s = name.lower()
    if 'eye' in s or 'mask' in s or 'wipe' in s or 'makeup' in s or 'oil' in s or 'toner' in s or 'moisturizer' in s: return None
    if 'patch' in s: return 'patch'
    if re.search(r'serum|essence|ampoule', s): return 'serum'
    if re.search(r'cleanser|wash|cleansing|foam|bar\b', s): return 'facewash'
    return None
prod = raw.drop_duplicates('product_id').set_index('product_id')
prod['type'] = prod.product_name.map(ptype)
d = raw.copy(); d['type'] = d.product_id.map(prod['type'])
d = d[d['type'].notna()].copy()
d = d.drop_duplicates(['author_id', 'product_id', 'review_text'])
d['review_text'] = d['review_text'].fillna(''); d['review_title'] = d['review_title'].fillna('')
# product_info.csv (ingredients, categories)
info = pd.read_csv(BASE / 'product_info.csv', usecols=['product_id', 'ingredients', 'primary_category', 'secondary_category', 'tertiary_category', 'loves_count'])
d = d.merge(info, on='product_id', how='left')
FRAG_RE = r'fragrance|parfum|limonene|linalool|citronellol|geraniol|citral|eugenol|coumarin|benzyl salicylate|hexyl cinnamal|isomethyl'
d['ing_known'] = d.ingredients.notna()
d['fragranced'] = d.ingredients.fillna('').str.lower().str.contains(FRAG_RE, regex=True).astype(int)
d['n_ingredients'] = d.ingredients.fillna('').str.count(',') + 1
d.loc[~d.ing_known, 'n_ingredients'] = 0
for k_, pat_ in {'salicylic': 'salicylic', 'niacinamide': 'niacinamide', 'retinoid': 'retinol|retinal|retinyl', 'vitc': 'ascorb', 'bpo': 'benzoyl peroxide', 'hyaluronic': 'hyaluron'}.items():
    d['act_' + k_] = d.ingredients.fillna('').str.lower().str.contains(pat_, regex=True).astype(int)
d['text'] = (d.review_title + '. ' + d.review_text).str.lower()
d['date'] = pd.to_datetime(d.submission_time, errors='coerce')
d['low'] = (d.rating <= 2).astype(int)
RES['n_rows'] = len(d); RES['n_products'] = int(d.product_id.nunique())
RES['type_counts'] = d['type'].value_counts().to_dict()
RES['type_products'] = d.groupby('type').product_id.nunique().to_dict()
RES['date_min'] = str(d.date.min().date()); RES['date_max'] = str(d.date.max().date())
RES['missing_pct'] = (d[['skin_type', 'is_recommended', 'helpfulness', 'review_text', 'price_usd', 'date']].isna().mean() * 100).round(1).to_dict()
RES['rating_share'] = (d.rating.value_counts(normalize=True).sort_index() * 100).round(1).to_dict()
RES['describe'] = {c: {'mean': float(d[c].mean()), 'median': float(d[c].median()), 'skew': float(stats.skew(d[c].dropna())),
                       'min': float(d[c].min()), 'max': float(d[c].max())}
                   for c in ['rating', 'price_usd', 'total_feedback_count']}
d['n_words'] = d.review_text.str.split().str.len()
RES['describe']['n_words'] = {'mean': float(d.n_words.mean()), 'median': float(d.n_words.median()), 'skew': float(stats.skew(d.n_words)),
                              'min': float(d.n_words.min()), 'max': float(d.n_words.max())}
q1, q3 = d.n_words.quantile([.25, .75]); RES['words_iqr_outliers'] = int((d.n_words > q3 + 3 * (q3 - q1)).sum())

# ------------------------------------------------------------ keyword flags (negation-aware where needed)
NEG = r"(?<!no )(?<!not )(?<!without )(?<!never )(?<!n't )"
def has(p): return d.text.str.contains(p, regex=True)
d['deep'] = has(r'cyst|deep|under the skin|underground|painful bump|nodul|hormonal')
d['irrit'] = has(NEG + r'(burn|sting|irritat|redness|red and|rash|peel|itch|allerg|reaction|inflam)')
d['smell'] = has(NEG + r'(smell|fragrance|scent|perfum)')
d['breakout'] = has(NEG + r'(break ?out|broke me out|zits|clog|purg|cystic)')
d['not_eff'] = has(r"(did not work|didn't work|doesn't work|does not work|no difference|waste|not effective|nothing)")
d['praise'] = has(r'(love|great|works|clear|smooth|favorite|best|amazing|holy grail)')
d['sensitive_mention'] = has(r'sensitive')
d['reason'] = np.select([d.irrit | d.smell, d.breakout, d.not_eff, d.deep], ['irritation_or_smell', 'breakout', 'not_effective', 'deep_lesion'], 'other')

# ------------------------------------------------------------ PART A: EDA statistics
# A1 ratings by type
tp = d.groupby('type').agg(n=('rating', 'size'), mean=('rating', 'mean'), median=('rating', 'median'), low=('low', 'mean'),
                           price=('price_usd', 'mean'), rec=('is_recommended', 'mean'))
RES['by_type'] = tp.round(3).to_dict('index')
groups = [d.loc[d.type == t, 'rating'] for t in ['facewash', 'serum', 'patch']]
H, p = stats.kruskal(*groups); RES['kruskal'] = {'H': float(H), 'p': float(p), 'pairs': {}}
for a, b in [('patch', 'serum'), ('patch', 'facewash'), ('serum', 'facewash')]:
    x, y = d.loc[d.type == a, 'rating'], d.loc[d.type == b, 'rating']
    u, pp = stats.mannwhitneyu(x, y); rb = 1 - 2 * u / (len(x) * len(y))
    RES['kruskal']['pairs'][f'{a} vs {b}'] = {'mean_diff': float(x.mean() - y.mean()), 'p_bonf': float(min(1, pp * 3)), 'rank_biserial': float(-rb)}
# single patch product vs rest: which patch product(s)
RES['patch_products'] = d[d.type == 'patch'].groupby('product_name').rating.agg(['size', 'mean']).round(2).reset_index().values.tolist()

# A2 reasons among 1-2 star reviews by type (chi-square, Cramer's V)
low = d[d.low == 1]
ct = pd.crosstab(low.type, low.reason); chi2, p, dof, exp = stats.chi2_contingency(ct)
RES['reasons'] = {'table_pct': (ct.div(ct.sum(1), axis=0) * 100).round(1).to_dict('index'), 'n_low': ct.sum(1).to_dict(),
                  'chi2': float(chi2), 'p': float(p), 'dof': int(dof), 'cramers_v': float(np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1))))}
res = (ct - exp) / np.sqrt(exp); RES['reasons']['residuals'] = res.round(1).to_dict('index')

# A3 patch lesion-depth mentions among low-rated patch reviews (bootstrap CI)
lp = low[low.type == 'patch']['deep'].values
bs = [rng.choice(lp, len(lp)).mean() for _ in range(2000)]
RES['patch_deep'] = {'n': int(len(lp)), 'share': float(lp.mean()), 'ci': [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                     'share_nonpatch_low': float(low[low.type != 'patch']['deep'].mean())}
# deep-mention vs rating on patches
pp_ = d[d.type == 'patch']
RES['patch_deep']['mean_rating_deep'] = float(pp_[pp_.deep].rating.mean()); RES['patch_deep']['mean_rating_notdeep'] = float(pp_[~pp_.deep].rating.mean())
RES['patch_deep']['n_deep_all'] = int(pp_.deep.sum())
u, pu = stats.mannwhitneyu(pp_[pp_.deep].rating, pp_[~pp_.deep].rating); RES['patch_deep']['p'] = float(pu)

# A4 skin type and sensitive-mention (no 'sensitive' skin-type option exists in the data)
RES['skin_counts'] = d.skin_type.fillna('unknown').value_counts().to_dict()
RES['irrit_by_skin'] = d.groupby(d.skin_type.fillna('unknown')).irrit.mean().round(4).to_dict()
sens = d[d.sensitive_mention]
RES['sensitive'] = {'n': int(len(sens)), 'irrit_rate': float(sens.irrit.mean()), 'irrit_rate_other': float(d[~d.sensitive_mention].irrit.mean()),
                    'by_type': sens.groupby('type').irrit.agg(['size', 'mean']).round(3).to_dict('index'),
                    'rating_sens': float(sens.rating.mean()), 'rating_other': float(d[~d.sensitive_mention].rating.mean())}
a = sens[sens.type == 'patch']; b = sens[sens.type == 'serum']
tab = [[a.irrit.sum(), len(a) - a.irrit.sum()], [b.irrit.sum(), len(b) - b.irrit.sum()]]
RES['sensitive']['fisher_patch_vs_serum'] = {'p': float(stats.fisher_exact(tab)[1]), 'odds_ratio': float(stats.fisher_exact(tab)[0])}

# A5 fragrance / smell mentions
sm = d.groupby('smell').rating.agg(['size', 'mean']); RES['smell'] = {'n': int(d.smell.sum()), 'share': float(d.smell.mean()),
    'mean_rating_smell': float(sm.loc[True, 'mean']), 'mean_rating_nosmell': float(sm.loc[False, 'mean']),
    'p': float(stats.mannwhitneyu(d[d.smell].rating, d[~d.smell].rating)[1]), 'by_type': d.groupby('type').smell.mean().round(3).to_dict()}

# A6 rating vs recommendation
RES['rating_vs_recommend'] = {'corr_spearman': float(stats.spearmanr(d.rating, d.is_recommended, nan_policy='omit')[0]),
    'rec_among_rating1_2': float(d[d.low == 1].is_recommended.mean()), 'rec_among_rating5': float(d[d.rating == 5].is_recommended.mean()),
    'rating5_not_rec': int(((d.rating == 5) & (d.is_recommended == 0)).sum()), 'rating1_rec': int(((d.rating == 1) & (d.is_recommended == 1)).sum())}

# A7 time + helpfulness
d['year'] = d.date.dt.year; yr = d.groupby('year').agg(n=('rating', 'size'), mean=('rating', 'mean'))
RES['time'] = {'by_year': yr.round(3).to_dict('index'), 'rho': float(stats.spearmanr(d.date.map(pd.Timestamp.toordinal), d.rating, nan_policy='omit')[0])}
RES['time']['p'] = float(stats.spearmanr(d.date.map(pd.Timestamp.toordinal), d.rating, nan_policy='omit')[1])
RES['votes_by_rating'] = d.groupby('rating').total_feedback_count.mean().round(2).to_dict()

# A8 correlations
cc = d.assign(log_words=np.log1p(d.n_words), price=d.price_usd, votes=d.total_feedback_count, recommended=d.is_recommended)[
    ['rating', 'recommended', 'price', 'log_words', 'votes', 'irrit', 'smell', 'breakout', 'not_eff', 'deep', 'praise']].astype(float)
corr = cc.corr(method='spearman'); RES['top_corr'] = [(i, j, float(corr.loc[i, j])) for i in corr for j in corr if i < j and abs(corr.loc[i, j]) > .25]

# A9 top terms in 1-2 star reviews
terms = {}
for t in ['facewash', 'serum', 'patch']:
    sub = low[low.type == t].text
    if len(sub) < 20: continue
    cv = CountVectorizer(ngram_range=(1, 2), stop_words=[w for w in 'a an the and or of to is it in on for with this that i my me was were be as at but have has had so very'.split()],
                         min_df=3, max_features=2000)
    X = cv.fit_transform(sub); s = np.asarray(X.sum(0)).ravel(); idx = s.argsort()[::-1][:10]
    terms[t] = [(cv.get_feature_names_out()[i], int(s[i])) for i in idx]
RES['top_terms'] = terms

# A10 fragrance from the ingredient list (H4)
k_ = d[d.ing_known]
RES['frag'] = {'products_known': int(k_.product_id.nunique()), 'products_fragranced': int(k_[k_.fragranced == 1].product_id.nunique()),
               'reviews_known': int(len(k_)), 'ingredients_missing_products': int(d[~d.ing_known].product_id.nunique())}
ft = k_.groupby('type').apply(lambda g: g.drop_duplicates('product_id').fragranced.mean()).round(3); RES['frag']['share_products_by_type'] = ft.to_dict()
for nm_, g_ in [('fragranced', k_[k_.fragranced == 1]), ('free', k_[k_.fragranced == 0])]:
    RES['frag'][nm_] = {'n': int(len(g_)), 'mean_rating': float(g_.rating.mean()), 'low_rate': float(g_.low.mean()), 'irrit_rate': float(g_.irrit.mean()), 'products': int(g_.product_id.nunique())}
tab_ = [[k_[k_.fragranced == 1].irrit.sum(), (k_.fragranced == 1).sum() - k_[k_.fragranced == 1].irrit.sum()], [k_[k_.fragranced == 0].irrit.sum(), (k_.fragranced == 0).sum() - k_[k_.fragranced == 0].irrit.sum()]]
chi_, p_, _, _ = stats.chi2_contingency(tab_); RES['frag']['irrit_chi2'] = float(chi_); RES['frag']['irrit_p'] = float(p_); RES['frag']['irrit_or'] = float((tab_[0][0] / tab_[0][1]) / (tab_[1][0] / tab_[1][1]))
RES['frag']['rating_mwu_p'] = float(stats.mannwhitneyu(k_[k_.fragranced == 1].rating, k_[k_.fragranced == 0].rating)[1])
# adjusted: irritation ~ fragranced + type + sensitive mention (bootstrap over PRODUCTS because reviews of one product are not independent)
Xl = pd.DataFrame({'fragranced': k_.fragranced, 'sensitive': k_.sensitive_mention.astype(int), 'serum': (k_.type == 'serum').astype(int), 'facewash': (k_.type == 'facewash').astype(int)})
lr = LogisticRegression(C=1e6, max_iter=2000).fit(Xl, k_.irrit.astype(int)); RES['frag']['adj_or'] = {c_: float(np.exp(v_)) for c_, v_ in zip(Xl.columns, lr.coef_[0])}
pids = k_.product_id.unique(); grp = {p_: np.where(k_.product_id.values == p_)[0] for p_ in pids}; bs_ = []
for _ in range(200):
    pick = rng.choice(pids, len(pids)); ix = np.concatenate([grp[p_] for p_ in pick])
    try: bs_.append(float(np.exp(LogisticRegression(C=1e6, max_iter=500).fit(Xl.iloc[ix], k_.irrit.values[ix].astype(int)).coef_[0][0])))
    except Exception: pass
RES['frag']['adj_or_fragranced_ci_product_bootstrap'] = [float(np.percentile(bs_, 2.5)), float(np.percentile(bs_, 97.5))]
# product level: irritation rate per product vs fragranced
pl = k_.groupby('product_id').agg(frag=('fragranced', 'first'), irr=('irrit', 'mean'), low=('low', 'mean'), n=('rating', 'size'), type=('type', 'first'))
pl = pl[pl.n >= 50]; RES['frag']['product_level'] = {'n_products': int(len(pl)), 'irr_fragranced': float(pl[pl.frag == 1].irr.mean()), 'irr_free': float(pl[pl.frag == 0].irr.mean()),
    'n_fragranced': int((pl.frag == 1).sum()), 'n_free': int((pl.frag == 0).sum()), 'mwu_p': float(stats.mannwhitneyu(pl[pl.frag == 1].irr, pl[pl.frag == 0].irr)[1])}
# sensitive mention x fragrance
RES['frag']['sens_irrit'] = {'fragranced': float(k_[(k_.sensitive_mention) & (k_.fragranced == 1)].irrit.mean()), 'free': float(k_[(k_.sensitive_mention) & (k_.fragranced == 0)].irrit.mean()),
    'n_f': int(((k_.sensitive_mention) & (k_.fragranced == 1)).sum()), 'n_free': int(((k_.sensitive_mention) & (k_.fragranced == 0)).sum())}
# category cross-check
RES['category_check'] = d.groupby('type').tertiary_category.agg(lambda s: s.value_counts().head(3).to_dict()).to_dict()
RES['patch_info'] = d[d.type == 'patch'][['tertiary_category', 'fragranced', 'n_ingredients']].drop_duplicates().astype(str).values.tolist()
RES['acne_cat'] = d[d.tertiary_category == 'Blemish & Acne Treatments'].groupby('product_name').rating.agg(['size', 'mean']).round(2).reset_index().values.tolist()

# A11 irritation by active ingredient (fragrance-free products often carry strong actives)
RES['actives'] = {}
for c_ in [c_ for c_ in d.columns if c_.startswith('act_')]:
    g1, g0 = d[d[c_] == 1], d[d[c_] == 0]
    if len(g1):
        RES['actives'][c_[4:]] = {'products': int(g1.product_id.nunique()), 'n': int(len(g1)), 'irrit_with': float(g1.irrit.mean()), 'irrit_without': float(g0.irrit.mean()), 'share_fragranced_products': float(g1.drop_duplicates('product_id').fragranced.mean())}
RES['frag']['frag_free_with_strong_active'] = float(d[(d.fragranced == 0)].drop_duplicates('product_id')[['act_salicylic', 'act_retinoid', 'act_bpo']].max(axis=1).mean())
RES['frag']['frag_with_strong_active'] = float(d[(d.fragranced == 1)].drop_duplicates('product_id')[['act_salicylic', 'act_retinoid', 'act_bpo']].max(axis=1).mean())
# ------------------------------------------------------------ PART B: features
split_prod = pd.Series(d.product_id.unique()); rng2 = np.random.default_rng(42)
test_prods = set(rng2.choice(split_prod, size=int(.25 * len(split_prod)), replace=False))
d['split'] = np.where(d.product_id.isin(test_prods), 'test', 'train'); tr = (d.split == 'train')
RES['split'] = d.split.value_counts().to_dict()
POS = set('love great best amazing perfect favorite smooth gentle works clear soft glowing happy recommend holy grail obsessed'.split())
NGW = set('burn burning stings stinging irritated irritation rash broke breakout breakouts terrible awful worst waste greasy clogged painful redness dry itchy disappointed'.split())
NEGATORS = set("not no never don't didn't doesn't won't wasn't isn't without".split())
tok = d.text.str.findall(r"[a-z']+")
def sent(ws):
    p = n = 0
    for i, w in enumerate(ws):
        flip = i > 0 and ws[i - 1] in NEGATORS
        if w in POS: p += (0 if flip else 1); n += (1 if flip else 0)
        elif w in NGW: n += (0 if flip else 1); p += (1 if flip else 0)
    return p, n
ps = tok.map(sent)
f = pd.DataFrame(index=d.index)
for t in ['patch', 'serum', 'facewash']: f['type_' + t] = (d.type == t).astype(int)
for s in ['combination', 'dry', 'normal', 'oily']: f['skin_' + s] = (d.skin_type == s).astype(int)
f['skin_unknown'] = d.skin_type.isna().astype(int)
f['log_price'] = np.log1p(d.price_usd); f['log_words'] = np.log1p(d.n_words); f['log_title_len'] = np.log1p(d.review_title.str.len())
f['log_product_reviews'] = np.log1p(d.groupby('product_id').rating.transform('size'))
stat = f[tr].assign(t=d.type[tr]).groupby('t')['log_price'].agg(['mean', 'std'])
f['price_z_in_type'] = ((f.log_price - d.type.map(stat['mean'])) / d.type.map(stat['std'])).fillna(0)
f['price_tier'] = pd.cut(f.price_z_in_type, [-np.inf, -.5, .5, np.inf], labels=[0, 1, 2]).astype(int)
f['skin_missing'] = d.skin_type.isna().astype(int); f['text_missing'] = (d.review_text.str.len() == 0).astype(int)
f['has_words_ge_100'] = (d.n_words >= 100).astype(int)
f['sens_x_nonpatch'] = d.sensitive_mention.astype(int) * (d.type != 'patch').astype(int)
f['oily_x_serum'] = f.skin_oily * f.type_serum
f['patch_x_deep'] = f.type_patch * d.deep.astype(int)
f['sens_x_smell'] = d.sensitive_mention.astype(int) * d.smell.astype(int)
f['pos_count'] = ps.map(lambda x: x[0]); f['neg_count'] = ps.map(lambda x: x[1])
f['sentiment_score'] = (f.pos_count - f.neg_count) / (f.pos_count + f.neg_count + 1)
f['negation_count'] = tok.map(lambda ws: sum(w in NEGATORS for w in ws))
for c in ['irrit', 'smell', 'breakout', 'not_eff', 'deep', 'praise', 'sensitive_mention']: f['flag_' + c] = d[c].astype(int)
f['month_sin'] = np.sin(2 * np.pi * d.date.dt.month / 12).fillna(0); f['month_cos'] = np.cos(2 * np.pi * d.date.dt.month / 12).fillna(0)
f['days_since_start'] = (d.date - d.date.min()).dt.days.fillna(0)
f['log_votes_given'] = np.log1p(d.total_feedback_count)
f['fragranced'] = d.fragranced; f['fragrance_unknown'] = (~d.ing_known).astype(int); f['n_ingredients'] = d.n_ingredients
f['sens_x_fragranced'] = d.sensitive_mention.astype(int) * d.fragranced; f['log_loves'] = np.log1p(d.loves_count.fillna(0))
for c_ in [c_ for c_ in d.columns if c_.startswith('act_')]: f[c_] = d[c_]
f['n_exclaim'] = d.review_text.str.count('!'); f['caps_ratio'] = d.review_text.map(lambda s: sum(ch.isupper() for ch in s) / (len(s) + 1))
RES['n_features'] = f.shape[1]
RES['skew'] = {'price': [float(stats.skew(d.price_usd.dropna())), float(stats.skew(f.log_price.dropna()))],
               'words': [float(stats.skew(d.n_words)), float(stats.skew(f.log_words))],
               'votes': [float(stats.skew(d.total_feedback_count)), float(stats.skew(f.log_votes_given))]}
RES['price_z_check'] = {t: float(f.loc[tr & (d.type == t), 'price_z_in_type'].mean()) for t in ['facewash', 'serum', 'patch']}
RES['price_z_check'] = {k: (None if v != v else v) for k, v in RES['price_z_check'].items()}
out = f.copy(); out['review_id'] = d.author_id.astype(str) + '_' + d.product_id; out['rating'] = d.rating; out['low_rating'] = d.low; out['split'] = d.split
out.to_csv(BASE / 'real_features.csv', index=False)

# ------------------------------------------------------------ PART C: does feature engineering help? target = 1-2 star
y = d.low.values; groups_ = d.product_id.values
A = ['type_patch', 'type_serum', 'type_facewash', 'skin_combination', 'skin_dry', 'skin_normal', 'skin_oily', 'skin_unknown', 'log_price', 'log_words']
Bf = A + ['price_z_in_type', 'price_tier', 'log_product_reviews', 'log_title_len', 'skin_missing', 'text_missing', 'sens_x_nonpatch', 'oily_x_serum',
          'month_sin', 'month_cos', 'days_since_start', 'n_exclaim', 'caps_ratio', 'fragranced', 'fragrance_unknown', 'n_ingredients', 'sens_x_fragranced', 'log_loves'] + [c_ for c_ in f.columns if c_.startswith('act_')]
Cf = Bf + ['pos_count', 'neg_count', 'sentiment_score', 'negation_count'] + [c for c in f.columns if c.startswith('flag_')] + ['patch_x_deep', 'sens_x_smell']
sets = {'A. Raw structured': A, 'B. + engineered structured': Bf, 'C. + keyword and sentiment features': Cf}
def models():
    return {'Logistic regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
            'Gradient boosting': HistGradientBoostingClassifier(max_depth=4, learning_rate=.06, max_iter=150, random_state=42)}
gkf = GroupKFold(n_splits=5); perf = {}
for sname, cols in sets.items():
    X = f[cols].values
    for mname in models():
        aucs = []
        for tri, tei in gkf.split(X, y, groups_):
            m = models()[mname].fit(X[tri], y[tri]); aucs.append(roc_auc_score(y[tei], m.predict_proba(X[tei])[:, 1]))
        m = models()[mname].fit(X[tr.values], y[tr.values]); te = ~tr.values
        perf[f'{sname} | {mname}'] = {'cv_mean': float(np.mean(aucs)), 'cv_sd': float(np.std(aucs)), 'test_auc': float(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1])), 'n_features': len(cols)}
# D. TF-IDF text baseline (words only)
tf = TfidfVectorizer(min_df=5, ngram_range=(1, 2), max_features=20000, sublinear_tf=True)
aucs = []
for tri, tei in gkf.split(d, y, groups_):
    Xtr = tf.fit_transform(d.text.values[tri]); Xte = tf.transform(d.text.values[tei])
    m = LogisticRegression(max_iter=2000, C=2).fit(Xtr, y[tri]); aucs.append(roc_auc_score(y[tei], m.predict_proba(Xte)[:, 1]))
perf['D. TF-IDF text only | Logistic regression'] = {'cv_mean': float(np.mean(aucs)), 'cv_sd': float(np.std(aucs)), 'n_features': 20000}
Xtr = tf.fit_transform(d.text[tr]); m = LogisticRegression(max_iter=2000, C=2).fit(Xtr, y[tr.values])
perf['D. TF-IDF text only | Logistic regression']['test_auc'] = float(roc_auc_score(y[~tr.values], m.predict_proba(tf.transform(d.text[~tr]))[:, 1]))
RES['model_perf'] = perf; RES['low_prevalence'] = float(d.low.mean())
# top coefficients of the logistic regression on set C
mC = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(f.loc[tr, Cf], y[tr.values])
co = pd.Series(mC[-1].coef_[0], index=Cf).sort_values(); RES['top_coefs_C'] = {'most_negative': co.head(5).round(2).to_dict(), 'most_positive': co.tail(6).round(2).to_dict()}

# ------------------------------------------------------------ figures
def save(fig, n): fig.savefig(BASE / n, bbox_inches='tight', facecolor='#fcfcfb'); plt.close(fig)
fig, ax = plt.subplots(1, 3, figsize=(12, 3.3), gridspec_kw={'wspace': .75})
rs = d.rating.value_counts().sort_index(); ax[0].bar(rs.index, rs.values, color=BLUE, width=.7)
for x_, v in rs.items(): ax[0].text(x_, v, f'{v / len(d) * 100:.0f}%', ha='center', va='bottom', fontsize=8)
ax[0].set_xticks(range(1, 6)); ax[0].set_title('Rating distribution'); ax[0].set_xlabel('Stars'); ax[0].set_ylabel('Reviews')
yy = d.groupby('year').size(); ax[1].bar(yy.index, yy.values, color=BLUE); ax[1].set_title('Reviews per year'); ax[1].tick_params(axis='x', rotation=60)
mm = d[['skin_type', 'is_recommended', 'helpfulness', 'review_text']].replace('', np.nan).isna().mean() * 100
ax[2].barh(mm.index[::-1], mm.values[::-1], color=BLUE); ax[2].set_title('Missing values (%)')
for i, v in enumerate(mm.values[::-1]): ax[2].text(v, i, f' {v:.1f}%', va='center', fontsize=8)
save(fig, 'fig_r1_overview.png')

fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
rt = pd.crosstab(d.type, d.rating, normalize='index') * 100; w = .27
for i, t in enumerate(['facewash', 'serum', 'patch']): ax[0].bar(np.arange(1, 6) + (i - 1) * w, rt.loc[t].values, w, color=COL[t], label=f'{t} (n={tp.loc[t, "n"]})')
ax[0].legend(frameon=False, fontsize=8); ax[0].set_title('Rating mix by product type'); ax[0].set_xlabel('Stars'); ax[0].set_ylabel('% of reviews')
bp = ax[1].boxplot([d.loc[d.type == t, 'price_usd'] for t in ['facewash', 'serum', 'patch']], tick_labels=['facewash', 'serum', 'patch'], patch_artist=True, showfliers=False)
for b_, t in zip(bp['boxes'], ['facewash', 'serum', 'patch']): b_.set_facecolor(COL[t]); b_.set_alpha(.85)
ax[1].set_title('Price (USD) by type'); save(fig, 'fig_r2_type.png')

fig, ax = plt.subplots(figsize=(8.5, 2.8)); order = ['irritation_or_smell', 'breakout', 'not_effective', 'deep_lesion', 'other']
cp = ct.div(ct.sum(1), axis=0).reindex(columns=order).fillna(0) * 100; left = np.zeros(3)
cols_ = [ORANGE, '#c9a227', BLUE, AQUA, '#d0d0d0']
for o, c_ in zip(order, cols_):
    v = cp.loc[['facewash', 'serum', 'patch'], o].values; ax.barh([f'{t} (n={ct.loc[t].sum()})' for t in ['facewash', 'serum', 'patch']], v, left=left, color=c_, label=o.replace('_', ' '))
    for i, (l_, v_) in enumerate(zip(left, v)):
        if v_ > 6: ax.text(l_ + v_ / 2, i, f'{v_:.0f}%', ha='center', va='center', fontsize=8, color='white' if c_ != '#d0d0d0' else 'black')
    left += v
ax.legend(ncol=5, frameon=False, fontsize=7, loc='upper center', bbox_to_anchor=(.5, 1.25)); ax.set_xlabel('% of 1 to 2 star reviews'); save(fig, 'fig_r3_reasons.png')

fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
yv = d.groupby('year').rating.mean(); ax[0].plot(yv.index, yv.values, color=BLUE, marker='o'); ax[0].set_ylim(3.5, 5); ax[0].set_xticks(range(2008, 2024, 3)); ax[0].set_title('Mean rating by year')
ax[1].bar(['rated 1-2\nand recommend', 'rated 5\nnot recommend'], [RES['rating_vs_recommend']['rating1_rec'], RES['rating_vs_recommend']['rating5_not_rec']], color=[ORANGE, ORANGE]); ax[1].set_title('Rating vs recommendation conflicts (count)')
save(fig, 'fig_r4_time_conflict.png')

fig, ax = plt.subplots(figsize=(6.2, 5.2)); cl = corr.iloc[1:, :-1]; mask = np.triu(np.ones_like(cl, dtype=bool), k=1)
sns.heatmap(cl, mask=mask, cmap=sns.diverging_palette(250, 15, s=75, l=50, as_cmap=True, center='light'), vmin=-.8, vmax=.8, center=0, annot=True, fmt='.2f', annot_kws={'size': 7}, linewidths=.5, cbar_kws={'shrink': .7}, ax=ax)
save(fig, 'fig_r5_corr.png')

fig, ax = plt.subplots(2, 3, figsize=(10, 4.6))
for j, (ttl, raw_s, log_s, rk, lk) in enumerate([('Price (USD)', d.price_usd, f.log_price, *RES['skew']['price']), ('Review length (words)', d.n_words, f.log_words, *RES['skew']['words']), ('Feedback votes', d.total_feedback_count, f.log_votes_given, *RES['skew']['votes'])]):
    ax[0, j].hist(raw_s.dropna().clip(upper=raw_s.quantile(.99)), bins=30, color=GREY); ax[0, j].set_title(f'{ttl}: raw, skew {rk:.2f}', fontsize=9)
    ax[1, j].hist(log_s.dropna(), bins=30, color=BLUE); ax[1, j].set_title(f'after log1p, skew {lk:.2f}', fontsize=9)
fig.tight_layout(); save(fig, 'fig_r6_transform.png')

fig, ax = plt.subplots(figsize=(7.5, 3.4)); names = list(perf); vals = [perf[n]['cv_mean'] for n in names]; errs = [perf[n]['cv_sd'] for n in names]
colr = [BLUE if 'Logistic' in n else ORANGE for n in names]; lab = [n.replace(' | ', '\n') for n in names]
ax.bar(range(len(names)), vals, yerr=errs, color=colr, capsize=3, width=.65); ax.set_ylim(.5, 1.0)
for i, (v, e_) in enumerate(zip(vals, errs)): ax.text(i, v + e_ + .01, f'{v:.3f}', ha='center', fontsize=8)
ax.set_xticks(range(len(names))); ax.set_xticklabels([n.split(' | ')[0].split('.')[0] + ('\nlogistic' if 'Logistic' in n else '\nboosting') for n in names], fontsize=8)
ax.set_ylabel('ROC AUC (5-fold, grouped by product)'); ax.set_title('Predicting 1 to 2 star reviews'); save(fig, 'fig_r7_models.png')

json.dump(RES, open(BASE / 'real_results.json', 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
print('done', RES['n_rows'], RES['n_features'])
