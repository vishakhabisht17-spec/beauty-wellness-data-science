"""Cleaning & preprocessing pipeline for skincare review data (Week 3).

Run:  python cleaning_pipeline.py
Reads  raw_reviews.csv  (synthetic simulation data) and writes:
  clean_reviews.csv, cleaning_log.json, results.json
Every step logs rows before/after and what it changed, for reproducibility.
"""
import html
import json
import re
import unicodedata

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

BASE = Path(__file__).resolve().parent
SEED = 42
FX_TO_USD = {'USD': 1.0, 'INR': 0.012, 'EUR': 1.08}      # illustrative fixed rates
LOG = []


def log(step, df_before, df_after, note, changed=0):
    LOG.append(dict(step=step, rows_before=int(len(df_before)), rows_after=int(len(df_after)),
                    cells_changed=int(changed), note=note))


# ------------------------------------------------------------------ S1
def standardise_columns(df):
    out = df.copy()
    out.columns = [re.sub(r'\W+', '_', c.strip().lower()).strip('_') for c in out.columns]
    for c in out.select_dtypes(include=['object', 'string']).columns:
        out[c] = out[c].astype('object').map(lambda v: v.strip() if isinstance(v, str) else v)
    log('S1 standardise columns', df, out, 'snake_case names; trimmed whitespace in text columns')
    return out


# ------------------------------------------------------------------ S2
NULL_TOKENS = {'', 'n/a', 'na', 'none', 'null', 'unknown', 'not sure', 'nan', '-'}


def to_null(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    if isinstance(v, str) and v.strip().lower() in NULL_TOKENS:
        return np.nan
    return v


def parse_rating(v):
    v = to_null(v)
    if pd.isna(v):
        return np.nan
    m = re.search(r'-?\d+(\.\d+)?', str(v))
    if not m:
        return np.nan
    x = float(m.group())
    return x if 1 <= x <= 5 else np.nan             # impossible values become missing, not guessed


DATE_PATTERNS = [('%Y-%m-%d', r'^\d{4}-\d{2}-\d{2}$'), ('%d/%m/%Y', r'^\d{1,2}/\d{1,2}/\d{4}$'),
                 ('%b %d, %Y', r'^[A-Za-z]{3} \d{1,2}, \d{4}$'), ('%d %b %y', r'^\d{1,2} [A-Za-z]{3} \d{2}$')]


def parse_date(v, today=pd.Timestamp('2026-10-28')):
    v = to_null(v)
    if pd.isna(v):
        return pd.NaT
    for fmt, rx in DATE_PATTERNS:
        if re.match(rx, str(v)):
            d = pd.to_datetime(str(v), format=fmt, errors='coerce')   # slash dates read day-first (stated assumption)
            return d if (pd.notna(d) and d <= today) else pd.NaT       # future dates are invalid
    return pd.NaT


def parse_price(v):
    """Return (amount_usd, currency). Handles $, rupee, EUR, decimal comma, thousands comma."""
    v = to_null(v)
    if pd.isna(v):
        return np.nan, None
    s = str(v)
    cur = 'INR' if '₹' in s or 'INR' in s.upper() else 'EUR' if ('EUR' in s.upper() or '€' in s) else 'USD'
    num = re.sub(r'[^\d,.]', '', s)
    if not num:
        return np.nan, cur
    if ',' in num and '.' not in num and re.search(r',\d{2}$', num):
        num = num.replace(',', '.')                 # 12,99 -> 12.99
    else:
        num = num.replace(',', '')                  # 1,099 -> 1099
    try:
        return float(num) * FX_TO_USD[cur], cur
    except ValueError:
        return np.nan, cur


BOOL_MAP = {'y': True, 'yes': True, 'true': True, '1': True, 'n': False, 'no': False, 'false': False, '0': False}


def coerce_types(df):
    out = df.copy()
    before_valid = out['rating'].notna().sum()
    out['rating'] = out['rating'].map(parse_rating)
    out['review_date'] = out['review_date'].map(parse_date)
    pr = out['price'].map(parse_price)
    out['price_usd'] = pr.map(lambda t: t[0])
    out['price_currency'] = pr.map(lambda t: t[1])
    out['verified_purchase'] = out['verified_purchase'].map(
        lambda v: BOOL_MAP.get(str(to_null(v)).strip().lower(), np.nan)).astype('object')
    out['helpful_votes'] = pd.to_numeric(out['helpful_votes'], errors='coerce')
    out = out.drop(columns=['price'])
    log('S2 coerce types', df, out,
        f'rating to float 1-5 (invalid to NaN); dates parsed with explicit formats; prices converted to USD; booleans mapped',
        changed=int(len(out)))
    return out


# ------------------------------------------------------------------ S3 / S4
def map_product_type(row):
    for source in (row.get('product_type'), row.get('product_name')):
        s = re.sub(r'[^a-z ]', ' ', str(source).lower()) if isinstance(source, str) else ''
        if 'patch' in s: return 'patch'
        if 'serum' in s: return 'serum'
        if 'wash' in s or 'cleanser' in s: return 'facewash'
    return 'other'


def standardise_categories(df):
    out = df.copy()
    n_missing_type = out['product_type'].map(to_null).isna().sum()
    out['product_type_clean'] = out.apply(map_product_type, axis=1)
    out['brand'] = out['brand'].str.strip().str.title()
    out['skin_type_raw'] = out['skin_type'].map(to_null)
    def norm_skin(v):
        if pd.isna(v): return np.nan
        s = str(v).lower()
        for key in ('sensitive', 'oily', 'dry', 'comb'):
            if key in s: return 'combination' if key == 'comb' else key
        return np.nan
    out['skin_type_clean'] = out['skin_type_raw'].map(norm_skin)
    log('S3-S4 standardise categories', df, out,
        f'product_type mapped by keyword rules with product-name fallback ({n_missing_type} missing types recovered); brand title-cased; skin_type unified',
        changed=int(len(out)))
    return out


# ------------------------------------------------------------------ S5 duplicates
def normalise_text_key(s):
    s = unicodedata.normalize('NFKC', str(s)).lower()
    return re.sub(r'[^a-z0-9 ]', '', s).strip()


def drop_duplicates(df):
    out = df.copy()
    n0 = len(out)
    out = out.drop_duplicates(subset=['review_id'], keep='first')
    exact = n0 - len(out)
    # key = product + normalised text + date + rating + helpful votes (all must match; avoids merging genuine look-alike reviews)
    # NOTE (pandas 3): astype(str) keeps missing values as NaN, which would make every row with a missing
    # field share one NaN key and be wrongly collapsed. Missing parts are therefore written out explicitly as 'NA'.
    part = lambda col: out[col].map(lambda x: 'NA' if pd.isna(x) else str(x))
    out['_key'] = (part('product_id') + '|' + out['review_text'].map(normalise_text_key) + '|' + part('review_date')
                   + '|' + part('rating') + '|' + part('helpful_votes'))
    n1 = len(out)
    out = out.drop_duplicates(subset=['_key'], keep='first').drop(columns=['_key'])
    near = n1 - len(out)
    log('S5 duplicates', df, out, f'exact duplicate IDs removed: {exact}; near-duplicates (same product, normalised text, date, rating, votes) removed: {near}')
    return out, exact, near


# ------------------------------------------------------------------ S6 text cleaning
EMOJI_MAP = {'\U0001F60D': ' love ', '\U0001F44D': ' good ', '\U0001F621': ' angry ', '\U0001F615': ' unhappy ',
             ':)': ' happy ', ':(': ' sad '}
PROMO_RX = re.compile(r'(free (product|sample)|in exchange for|received (this|it) (product )?(free|for free)|discount code)', re.I)


def clean_text(s):
    if not isinstance(s, str):
        return ''
    s = html.unescape(s)
    s = re.sub(r'<[^>]+>', ' ', s)
    s = re.sub(r'https?://\S+|www\.\S+', ' ', s)
    for k, v in EMOJI_MAP.items():
        s = s.replace(k, v)
    s = s.lower()
    s = s.replace("didn't", 'did not').replace("doesn't", 'does not').replace("don't", 'do not').replace("won't", 'will not')
    s = re.sub(r'(.)\1{2,}', r'\1\1', s)             # sooo -> soo ; keeps normal double letters
    s = re.sub(r'[^a-z0-9 .,!?\']', ' ', s)          # drop remaining symbols; negations stay intact
    return re.sub(r'\s+', ' ', s).strip()


def process_text(df):
    out = df.copy()
    out['promo_flag'] = out['review_text'].map(lambda s: bool(PROMO_RX.search(s)) if isinstance(s, str) else False)
    out['text_clean'] = out['review_text'].map(clean_text)
    out['n_words'] = out['text_clean'].str.split().str.len()
    n0 = len(out)
    out = out[out['n_words'] >= 3].copy()
    log('S6 text cleaning', df, out,
        f'HTML/URL/emoji handled, negations kept, repeated letters squeezed, promo reviews flagged ({int(out.promo_flag.sum())}); reviews under 3 words removed: {n0 - len(out)}')
    return out


# ------------------------------------------------------------------ S7 missing skin type from text
SKIN_RX = {'sensitive': r'\bsensitive skin\b|\bmy sensitive\b', 'oily': r'\boily skin\b|\bmy oily\b',
           'dry': r'\bdry skin\b|\bmy dry\b', 'combination': r'\bcombination skin\b|\bcombo skin\b'}


def recover_skin_type(df):
    out = df.copy()
    def from_text(t):
        hits = [k for k, rx in SKIN_RX.items() if re.search(rx, t)]
        return hits[0] if len(hits) == 1 else np.nan   # ambiguous mentions stay missing
    miss = out['skin_type_clean'].isna()
    guess = out.loc[miss, 'text_clean'].map(from_text)
    out['skin_type_source'] = np.where(out['skin_type_clean'].notna(), 'reported', 'missing')
    out.loc[guess.dropna().index, 'skin_type_clean'] = guess.dropna()
    out.loc[guess.dropna().index, 'skin_type_source'] = 'text_derived'
    out['skin_type_final'] = out['skin_type_clean'].fillna('unknown')
    log('S7 missing skin type', df, out,
        f'skin type recovered from review text for {int(guess.notna().sum())} of {int(miss.sum())} missing; remaining kept as an explicit unknown class',
        changed=int(guess.notna().sum()))
    return out, int(miss.sum()), int(guess.notna().sum())


# ------------------------------------------------------------------ S8 outliers
def iqr_bounds(x, k=1.5):
    q1, q3 = np.nanpercentile(x, [25, 75])
    return q1 - k * (q3 - q1), q3 + k * (q3 - q1)


def handle_outliers(df):
    out = df.copy()
    # (a) price: compare each row with the product's own median price (robust to product-type price ranges)
    med = out.groupby('product_id')['price_usd'].transform('median')
    ratio = out['price_usd'] / med
    out['price_outlier'] = (out['price_usd'] <= 0) | (ratio > 5) | (ratio < 0.2)
    n_price = int(out['price_outlier'].sum())
    out.loc[out['price_outlier'], 'price_usd'] = np.nan
    prod_price = out.groupby('product_id')['price_usd'].transform('median')
    out['price_usd'] = out['price_usd'].fillna(prod_price)              # repair from the product's own median
    still = out['price_usd'].isna()
    type_med = out.groupby('product_type_clean')['price_usd'].transform('median')
    out['price_imputed_by_type'] = still
    out['price_usd'] = out['price_usd'].fillna(type_med)
    # (b) review length: cap at the upper IQR fence on log scale, keep a flag
    lw = np.log1p(out['n_words'])
    lo, hi = iqr_bounds(lw, k=3.0)   # 3 x IQR = 'extreme' fence, so only truly abnormal lengths are capped
    cap = int(np.expm1(hi))
    out['length_outlier'] = out['n_words'] > cap
    n_len = int(out['length_outlier'].sum())
    out['n_words_capped'] = out['n_words'].clip(upper=cap)
    out['text_clean'] = np.where(out['length_outlier'], out['text_clean'].str.split().str[:cap].str.join(' '), out['text_clean'])
    # (c) helpful votes: winsorise at the 99th percentile
    p99 = out['helpful_votes'].quantile(0.99)
    out['helpful_votes_w'] = out['helpful_votes'].clip(upper=p99)
    out['helpful_votes_w'] = out['helpful_votes_w'].fillna(0)
    log('S8 outliers', df, out,
        f'price outliers repaired from product median: {n_price}; review length capped at {cap} words ({n_len} flagged); helpful votes winsorised at p99={p99:.0f}',
        changed=n_price + n_len)
    return out, n_price, n_len, cap


# ------------------------------------------------------------------ S9 ingredients / fragrance
FRAG_WORDS = re.compile(r'parfum|fragrance|aroma', re.I)
ALLERGENS = {'limonene', 'linalool', 'citronellol', 'geraniol', 'eugenol', 'coumarin', 'citral', 'farnesol'}
ACTIVES = {'salicylic_acid': r'salicylic', 'niacinamide': r'niacinamide', 'hydrocolloid': r'hydrocolloid',
           'benzoyl_peroxide': r'benzoyl peroxide', 'tea_tree': r'tea tree'}


def split_ingredients(s):
    """Split on commas/semicolons that are NOT inside parentheses."""
    if not isinstance(s, str):
        return []
    parts, depth, cur = [], 0, ''
    for ch in s:
        depth += (ch == '(') - (ch == ')')
        if ch in ',;' and depth == 0:
            parts.append(cur); cur = ''
        else:
            cur += ch
    parts.append(cur)
    return [re.sub(r'\s+', ' ', p).strip().lower() for p in parts if p.strip()]


def process_ingredients(df):
    out = df.copy()
    out['ingredients'] = out['ingredients'].map(to_null)
    out['ing_list'] = out['ingredients'].map(split_ingredients)
    known = out['ing_list'].str.len() > 0
    has_frag_word = out['ing_list'].map(lambda L: any(FRAG_WORDS.search(i) for i in L))
    has_allergen = out['ing_list'].map(lambda L: any(any(a in i for a in ALLERGENS) for i in L))
    out['fragrance_free'] = np.where(known, ~(has_frag_word | has_allergen), np.nan)
    claim = out['claim'].map(to_null).map(lambda v: isinstance(v, str) and 'fragrance free' in v.lower())
    out['claims_fragrance_free'] = claim
    out['claim_label_mismatch'] = claim & (out['fragrance_free'] == 0)
    for k, rx in ACTIVES.items():
        out[f'active_{k}'] = out['ing_list'].map(lambda L: any(re.search(rx, i) for i in L))
    out['n_ingredients'] = out['ing_list'].str.len()
    out['ingredients_known'] = known
    log('S9 ingredients', df, out,
        f"ingredient lists parsed (commas inside brackets respected); fragrance_free derived from parfum + EU allergen names; "
        f"{int((~known).sum())} rows have no ingredient list (kept as NaN, not guessed); claim-vs-label mismatches flagged: {int(out.claim_label_mismatch.sum())}")
    return out


# ------------------------------------------------------------------ S10 feature flags
FLAG_RX = {
    'lesion_depth_mention': r'\b(?:cyst|cystic|deep|under the skin|painful|sore)\b',
    'barrier_damage_flag': r'\b(?:burn|burning|stinging|peeling|redness|irritat\w*|red)\b',
    'smell_complaint': r'\b(?:smell|smells|perfume|fragrance|chemical|chemicals)\b',
    'efficacy_positive': r'\b(?:flattened|cleared|works great|calmer|gentle)\b',
    'not_effective': r'\b(?:did not work|no difference|not effective|did nothing|useless|nothing happened)\b',
}


NEGATED_SIDE_EFFECT = re.compile(
    r"\b(?:no|not|never|without|zero|free of)\s+(?:\w+\s+){0,2}(?:irritat\w*|burn\w*|redness|sting\w*|peel\w*|smell\w*|fragrance)\b")


def remove_negated_side_effects(t):
    """'does not irritate my skin' / 'no irritation at all' must not count as a complaint."""
    return NEGATED_SIDE_EFFECT.sub(' ', t)


def add_flags(df):
    out = df.copy()
    side_text = out['text_clean'].map(remove_negated_side_effects)
    for k, rx in FLAG_RX.items():
        src = side_text if k in ('barrier_damage_flag', 'smell_complaint') else out['text_clean']
        out[k] = src.str.contains(rx, regex=True)
    # rule-based first-pass reason label (to be replaced by human-annotated labels later)
    neg = out['rating'] <= 2
    out['reason_rule'] = np.select(
        [out.smell_complaint | out.barrier_damage_flag, neg & out.lesion_depth_mention & (out.product_type_clean == 'patch'),
         neg | out.not_effective],
        ['side_effect', 'misapplication', 'failure'], default='none')
    log('S10 feature flags', df, out, 'regex flags for lesion depth, barrier damage, smell, efficacy (negation-aware: "no irritation" is not a complaint); rule-based first-pass reason label')
    return out


# ------------------------------------------------------------------ S11 validation
def validate(df):
    checks = {
        'review_id unique': df['review_id'].is_unique,
        'rating within 1-5 or NaN': bool(df['rating'].dropna().between(1, 5).all()),
        'no future dates': bool((df['review_date'].dropna() <= pd.Timestamp('2026-10-28')).all()),
        'price_usd positive and present': bool((df['price_usd'] > 0).all()),
        'product_type in allowed set': bool(df['product_type_clean'].isin(['patch', 'serum', 'facewash', 'other']).all()),
        'skin_type in allowed set': bool(df['skin_type_final'].isin(['oily', 'dry', 'combination', 'sensitive', 'unknown']).all()),
        'text_clean has >= 3 words': bool((df['text_clean'].str.split().str.len() >= 3).all()),
    }
    LOG.append(dict(step='S11 validation', rows_before=len(df), rows_after=len(df), cells_changed=0,
                    note='; '.join(f"{k}: {'PASS' if v else 'FAIL'}" for k, v in checks.items())))
    return checks


# ------------------------------------------------------------------ S12 split
def grouped_split(df):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr, te = next(gss.split(df, groups=df['product_id']))
    df = df.copy()
    df['split'] = 'train'
    df.iloc[te, df.columns.get_loc('split')] = 'test'
    overlap = set(df.loc[df.split == 'train', 'product_id']) & set(df.loc[df.split == 'test', 'product_id'])
    LOG.append(dict(step='S12 grouped split', rows_before=len(df), rows_after=len(df), cells_changed=0,
                    note=f'80/20 split grouped by product_id; products appearing in both sets: {len(overlap)}'))
    return df, len(overlap)


# ------------------------------------------------------------------ main
def main():
    np.random.seed(SEED)
    raw = pd.read_csv(str(BASE / 'raw_reviews.csv'), dtype=str, keep_default_na=False, na_values=[])
    raw = raw.replace({'': np.nan})
    raw['Helpful Votes'] = pd.to_numeric(raw['Helpful Votes'], errors='coerce')
    raw_rows = len(raw)

    # profile of the raw data (after only trivial blank-to-NaN conversion)
    def nulls(d, cols):
        return {c: float(d[c].map(to_null).isna().mean()) for c in cols}
    raw_profile = nulls(raw, ['Review Text', 'Rating ', 'review_date', 'Skin Type', 'Verified Purchase', 'Product Type', 'Ingredients'])

    df = standardise_columns(raw)
    df = coerce_types(df)
    df = standardise_categories(df)
    df, n_exact, n_near = drop_duplicates(df)
    rows_after_dedup = int(len(df))
    df = process_text(df)
    df, n_miss_skin, n_rec_skin = recover_skin_type(df)
    df, n_price_out, n_len_out, cap = handle_outliers(df)
    df = process_ingredients(df)
    df = add_flags(df)
    checks = validate(df)
    df, overlap = grouped_split(df)

    keep = ['review_id', 'product_id', 'brand', 'product_name', 'product_type_clean', 'price_usd', 'price_currency',
            'skin_type_final', 'skin_type_source', 'rating', 'review_date', 'verified_purchase', 'helpful_votes_w',
            'text_clean', 'n_words_capped', 'promo_flag', 'price_outlier', 'length_outlier', 'fragrance_free',
            'claims_fragrance_free', 'claim_label_mismatch', 'n_ingredients', 'ingredients_known',
            'active_salicylic_acid', 'active_niacinamide', 'active_hydrocolloid', 'lesion_depth_mention',
            'barrier_damage_flag', 'smell_complaint', 'efficacy_positive', 'not_effective', 'reason_rule', 'split']
    clean = df[keep].copy()
    clean.to_csv(str(BASE / 'clean_reviews.csv'), index=False)

    # ---------- score against hidden ground truth ----------
    gt = pd.read_csv(str(BASE / 'ground_truth.csv')).set_index('review_id')
    j = clean.join(gt, on='review_id', rsuffix='_gt')
    type_acc = float((j['product_type_clean'] == j['type_true']).mean())
    skin_known = j['skin_type_final'] != 'unknown'
    skin_acc = float((j.loc[skin_known, 'skin_type_final'] == j.loc[skin_known, 'skin_true']).mean())
    rating_ok = j['rating'].notna()
    rating_acc = float((j.loc[rating_ok, 'rating'] == j.loc[rating_ok, 'rating_true']).mean())
    price_err = float((abs(j['price_usd'] - j['price_usd_true']) / j['price_usd_true']).median())
    price_within_10 = float(((abs(j['price_usd'] - j['price_usd_true']) / j['price_usd_true']) < 0.10).mean())
    fk = j['fragrance_free'].notna()
    frag_acc = float(((j.loc[fk, 'fragrance_free'] == 1) == (~j.loc[fk, 'has_frag_true'].astype(bool))).mean())
    rule_acc = float((j['reason_rule'] == j['reason_true']).mean())
    prec = {}
    for lab in ['side_effect', 'misapplication', 'failure', 'none']:
        tp = ((j.reason_rule == lab) & (j.reason_true == lab)).sum()
        pp = (j.reason_rule == lab).sum(); ap = (j.reason_true == lab).sum()
        prec[lab] = dict(precision=float(tp / pp) if pp else 0.0, recall=float(tp / ap) if ap else 0.0)

    # raw rating "naive" parse for comparison: what a careless pipeline would do
    naive = pd.to_numeric(raw['Rating '].astype(str).str.extract(r'(-?\d+\.?\d*)')[0], errors='coerce')
    naive_bad = float(((naive < 1) | (naive > 5)).mean())

    res = dict(
        raw_rows=raw_rows, rows_after_dedup=rows_after_dedup, clean_rows=int(len(clean)), exact_dups=n_exact, near_dups=n_near,
        raw_profile=raw_profile,
        clean_profile={'rating': float(clean['rating'].isna().mean()), 'review_date': float(clean['review_date'].isna().mean()),
                       'skin_type (unknown)': float((clean['skin_type_final'] == 'unknown').mean()),
                       'verified_purchase': float(clean['verified_purchase'].isna().mean()),
                       'product_type (other)': float((clean['product_type_clean'] == 'other').mean()),
                       'ingredients': float((~clean['ingredients_known']).mean()),
                       'price_usd': float(clean['price_usd'].isna().mean())},
        skin_missing=n_miss_skin, skin_recovered=n_rec_skin,
        price_outliers=n_price_out, length_outliers=n_len_out, length_cap=cap,
        promo=int(clean.promo_flag.sum()), mismatches=int(clean.claim_label_mismatch.sum()),
        invalid_rating_naive_share=naive_bad,
        accuracy=dict(product_type=type_acc, skin_type=skin_acc, rating=rating_acc, price_median_rel_err=price_err,
                      price_within_10pct=price_within_10, fragrance_flag=frag_acc, reason_rule_overall=rule_acc, reason_pr=prec),
        checks=checks, split=dict(train=int((clean.split == 'train').sum()), test=int((clean.split == 'test').sum()), overlap=overlap),
        type_counts=clean['product_type_clean'].value_counts().to_dict(),
        skin_counts=clean['skin_type_final'].value_counts().to_dict(),
        reason_counts=clean['reason_rule'].value_counts().to_dict(),
        rating_before=raw['Rating '].map(parse_rating).value_counts(dropna=False).sort_index().rename(lambda x: str(x)).to_dict(),
        rating_after=clean['rating'].value_counts(dropna=False).sort_index().rename(lambda x: str(x)).to_dict(),
        n_cols_raw=int(raw.shape[1]), n_cols_clean=int(clean.shape[1]),
    )
    json.dump(LOG, open(str(BASE / 'cleaning_log.json'), 'w'), indent=1, default=str)
    json.dump(res, open(str(BASE / 'results.json'), 'w'), indent=1, default=str)
    for r in LOG:
        print(f"{r['step']:32s} {r['rows_before']:>5d} -> {r['rows_after']:>5d}  {r['note'][:110]}")
    print(json.dumps({k: res[k] for k in ['accuracy', 'checks', 'split', 'skin_missing', 'skin_recovered']}, indent=1, default=str))


if __name__ == '__main__':
    main()
