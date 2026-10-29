"""Generate a SYNTHETIC raw skincare review dataset with realistic quality problems.

All data here is invented for simulation. Hidden columns starting with '_true' hold the
ground truth so the cleaning pipeline can be scored; the pipeline never reads them.
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
rng = np.random.default_rng(42)

N_PRODUCTS = 60
N_REVIEWS = 3000

# ---------------- products ----------------
TYPE_NAMES = {
    'patch': ['Overnight Pimple Patch', 'Hydro Acne Patch', 'Spot Patch Stars', 'Zit Patch Clear'],
    'serum': ['Clear Skin Serum', 'BHA Acne Serum', 'Niacinamide Serum', 'Blemish Control Serum'],
    'facewash': ['Gentle Face Wash', 'Acne Foaming Cleanser', 'Daily Face Wash', 'Oil Control Cleanser'],
}
TYPE_RAW_VARIANTS = {
    'patch': ['Pimple Patch', 'pimple patches', 'Acne PATCH', 'hydrocolloid patch', 'Patch'],
    'serum': ['Serum', 'acne serum ', 'SERUM', 'Face Serum'],
    'facewash': ['Face Wash', 'facewash', 'FACE-WASH', 'Cleanser', 'face cleanser'],
}
BRANDS = ['ClearLeaf', 'DermaGlow', 'PureSkin', 'Nuvo', 'Kaya Labs', 'SkinTrue', 'Aura']
ING_SERUM = ['Aqua', 'Salicylic Acid', 'Niacinamide (Vitamin B3)', 'Glycerin', 'Zinc PCA', 'Panthenol']
ING_WASH = ['Aqua', 'Cocamidopropyl Betaine', 'Salicylic Acid', 'Glycerin', 'Sodium Lauroyl Sarcosinate']
FRAG = ['Parfum', 'Fragrance/Parfum', 'Limonene', 'Linalool', 'Citronellol']
CURRENCY = {'USD': 1.0, 'INR': 0.012, 'EUR': 1.08}   # illustrative fixed rates

rows = []
for i in range(N_PRODUCTS):
    t = rng.choice(['patch', 'serum', 'facewash'], p=[0.3, 0.35, 0.35])
    price_usd = {'patch': rng.uniform(4, 12), 'serum': rng.uniform(8, 45), 'facewash': rng.uniform(5, 25)}[t]
    has_frag = (rng.random() < (0.02 if t == 'patch' else 0.45))
    if t == 'patch':
        ing = ['Hydrocolloid (Pectin, Gelatin, Carboxymethylcellulose)']
    elif t == 'serum':
        ing = list(rng.choice(ING_SERUM, size=4, replace=False))
    else:
        ing = list(rng.choice(ING_WASH, size=4, replace=False))
    if has_frag:
        ing += list(rng.choice(FRAG, size=rng.integers(1, 3), replace=False))
    claim = None
    if rng.random() < 0.5:
        claim = 'fragrance free' if (not has_frag or rng.random() < 0.2) else None  # some false claims
    rows.append(dict(product_id=f'P{i+1:03d}', brand=rng.choice(BRANDS), t=t,
                     name=f"{rng.choice(TYPE_NAMES[t])} {rng.integers(10, 99)}",
                     price_usd=price_usd, currency=rng.choice(['USD', 'INR', 'EUR'], p=[0.5, 0.3, 0.2]),
                     ingredients=ing, has_frag=has_frag, claim=claim,
                     missing_ing=(rng.random() < 0.12)))
prod = pd.DataFrame(rows)

# ---------------- review text templates ----------------
POS = ['Love this, my skin feels so clean', 'Works great and feels gentle', 'Flattened my pimple overnight, amazing',
       'Gentle and no irritation at all', 'Cleared my breakouts in 3 weeks', 'Does not irritate my skin, will buy again',
       'Really happy with this, skin looks calmer']
FAIL = ['Did not work at all, waste of money', 'No difference after a month', 'Not effective, my acne is still there',
        'Patch fell off and the pimple stayed the same', 'Honestly it did nothing for me']
MISAPP = ['Tried it on a deep painful cyst and nothing happened', 'Useless on my under the skin bump, still painful',
          'Did nothing for my cystic acne, very deep and sore']
SIDE = ['Burning and redness after one use', 'Smells like chemicals, strong fragrance', 'Caused redness and peeling',
        'Stinging, my face got irritated', 'The perfume smell is awful']
SKINS = ['oily', 'dry', 'combination', 'sensitive']
SKIN_RAW = {'oily': ['Oily', 'oily skin', 'OILY '], 'dry': ['Dry', 'dry skin', 'DRY'],
            'combination': ['Combination', 'Combo', 'combination skin'], 'sensitive': ['Sensitive', 'sensitive skin', 'SENSITIVE ']}
EMOJI = [' \U0001F60D', ' \U0001F44D', ' \U0001F621', ' \U0001F615', ' :)', ' :(']

recs = []
for k in range(N_REVIEWS):
    p = prod.sample(1, random_state=int(rng.integers(1e9))).iloc[0]
    skin = rng.choice(SKINS, p=[0.3, 0.2, 0.25, 0.25])
    side_p = 0.05 + (0.18 if p.has_frag else 0.0) + (0.12 if skin == 'sensitive' and p.t != 'patch' else 0.0)
    r = rng.random()
    if r < side_p:
        reason, text, rating = 'side_effect', rng.choice(SIDE), int(rng.integers(1, 4))
    elif r < side_p + (0.22 if p.t == 'patch' else 0.03):
        reason, text, rating = 'misapplication', rng.choice(MISAPP), int(rng.integers(1, 3))
    elif r < side_p + (0.22 if p.t == 'patch' else 0.03) + 0.10:
        reason, text, rating = 'failure', rng.choice(FAIL), int(rng.integers(1, 3))
    else:
        reason, text, rating = 'none', rng.choice(POS), int(rng.integers(4, 6))
    if rng.random() < 0.25:
        text += f" I have {skin} skin."
    # text noise
    if rng.random() < 0.08: text = text.upper()
    if rng.random() < 0.10: text += rng.choice(EMOJI)
    if rng.random() < 0.05: text = text.replace('so', 'sooo').replace('really', 'reallyyy')
    if rng.random() < 0.04: text += ' <br>Check https://bit.ly/xyz123'
    if rng.random() < 0.03: text = text.replace("Did not", "Didn't") + ' &amp; more'
    if rng.random() < 0.03: text = 'I received this product free in exchange for an honest review. ' + text
    if rng.random() < 0.03: text = ''
    elif rng.random() < 0.02: text = rng.choice(['ok', 'good', 'bad', '.'])
    if rng.random() < 0.004: text = (text + ' ') * 60   # extreme length outlier
    date = pd.Timestamp('2021-01-01') + pd.Timedelta(days=int(rng.integers(0, 1000)))
    recs.append(dict(review_id=f'R{k+1:05d}', product_id=p.product_id, skin_true=skin, rating_true=rating,
                     reason_true=reason, text=text, date=date))
rev = pd.DataFrame(recs)

# ---------------- raw field formats / corruption ----------------
def fmt_date(d):
    u = rng.random()
    if u < 0.40: return d.strftime('%Y-%m-%d')
    if u < 0.65: return d.strftime('%d/%m/%Y')
    if u < 0.85: return d.strftime('%b %d, %Y')
    if u < 0.95: return d.strftime('%d %b %y')
    if u < 0.975: return rng.choice(['N/A', '32/13/2022', 'yesterday', ''])
    return '2031-' + d.strftime('%m-%d')     # impossible future date

def fmt_rating(r):
    u = rng.random()
    if u < 0.55: return str(r)
    if u < 0.70: return f'{r} stars'
    if u < 0.82: return f'{r}.0'
    if u < 0.86: return rng.choice(['0', '6', '99', '-1'])
    if u < 0.88: return ''
    return str(r)

def fmt_skin(s):
    u = rng.random()
    if u < 0.12: return rng.choice([None, '', 'N/A', 'unknown', 'not sure'])
    return rng.choice(SKIN_RAW[s])

def fmt_price(row):
    p = prod.set_index('product_id').loc[row.product_id]
    usd = p.price_usd
    if rng.random() < 0.015:          # decimal-point typo: 12.99 -> 1299
        usd = usd * 100
    local = usd / CURRENCY[p.currency]
    if p.currency == 'USD':
        s = f'${usd:.2f}' if rng.random() < 0.7 else f'{usd:.2f}'
    elif p.currency == 'INR':
        s = f'\u20B9 {local:,.0f}'
    else:
        s = f'{local:.2f}'.replace('.', ',') + ' EUR'
    if rng.random() < 0.004: s = '0'
    return s

def fmt_bool(_):
    return rng.choice(['Y', 'yes', 'TRUE', '1', 'N', 'no', 'false', None], p=[.25, .15, .15, .1, .08, .07, .05, .15])

raw = pd.DataFrame({
    'Review ID': rev.review_id,
    'Product ID': rev.product_id,
    'Review Text': rev.text,
    'Rating ': [fmt_rating(r) for r in rev.rating_true],
    'review_date': [fmt_date(d) for d in rev.date],
    'Skin Type': [fmt_skin(s) for s in rev.skin_true],
    'Verified Purchase': [fmt_bool(0) for _ in range(len(rev))],
    'Helpful Votes': [int(x) if rng.random() > 0.05 else np.nan for x in np.round(rng.gamma(0.7, 3, len(rev)))],
})
raw['Price'] = [fmt_price(r) for r in rev.itertuples()]
pm = prod.set_index('product_id')
raw['Brand'] = rev.product_id.map(pm.brand).map(lambda b: rng.choice([b, b.upper(), b.lower(), b + ' ']) if rng.random() < 0.3 else b)
raw['Product Name'] = rev.product_id.map(pm.name)
raw['Product Type'] = [rng.choice(TYPE_RAW_VARIANTS[pm.loc[x, 't']]) if rng.random() > 0.04 else rng.choice([None, 'Other', 'N/A', ''])
                       for x in rev.product_id]
def ing_str(pid):
    p = pm.loc[pid]
    if p.missing_ing: return rng.choice([None, 'N/A', ''])
    sep = rng.choice([', ', '; ', ' , '])
    s = sep.join(p.ingredients)
    return s.lower() if rng.random() < 0.3 else s
raw['Ingredients'] = rev.product_id.map(ing_str)
raw['Claim'] = rev.product_id.map(pm.claim)

# duplicates: 4% exact copies (new row, same id) and 2% near-duplicates (new id, same content)
exact = raw.sample(frac=0.04, random_state=1)
near = raw.sample(frac=0.02, random_state=2).copy()
near['Review ID'] = ['R9' + str(i).zfill(4) for i in range(len(near))]
near['Review Text'] = near['Review Text'].map(lambda s: (s or '').lower() + ' ')
raw = pd.concat([raw, exact, near], ignore_index=True).sample(frac=1, random_state=3).reset_index(drop=True)

# keep ground truth in a separate file so the pipeline cannot see it
truth = pd.concat([rev, rev.sample(frac=0.04, random_state=1), rev.sample(frac=0.02, random_state=2)], ignore_index=True)
truth_by_id = rev.set_index('review_id')
gt = pd.DataFrame({
    'review_id': rev.review_id, 'product_id': rev.product_id,
    'type_true': rev.product_id.map(pm.t), 'skin_true': rev.skin_true, 'rating_true': rev.rating_true,
    'reason_true': rev.reason_true, 'price_usd_true': rev.product_id.map(pm.price_usd),
    'has_frag_true': rev.product_id.map(pm.has_frag),
})
raw.to_csv(str(BASE / 'raw_reviews.csv'), index=False)
gt.to_csv(str(BASE / 'ground_truth.csv'), index=False)
print('raw rows:', len(raw), 'products:', raw['Product ID'].nunique())
