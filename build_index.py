"""
Smart Library Recommender - index builder

Run this ONCE before starting the web app. It loads the enriched book data,
deduplicates editions, builds both the TF-IDF matrix and the SBERT embeddings,
and saves everything to artifacts/ so the web app starts instantly.

    python build_index.py

SBERT is optional. If sentence-transformers is not installed, the index still
builds with TF-IDF only and the app runs with that model alone.
"""

import os
import re
import ast
import time
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

DATA = 'books_full_with_description.csv'
COVER_SOURCE = 'Books.csv'          # optional, for publisher cover URLs
RATINGS = 'Ratings.csv'             # optional, enables collaborative filtering
OUT = 'artifacts'

MIN_USER_RATINGS = 3                # drop users with fewer interactions
CF_FACTORS = 48                     # latent dimensions per book
CF_ITERATIONS = 20

GENERIC = {'Fiction', 'Nonfiction', 'Novels', 'Literature', 'Audiobook', 'Adult',
           'Contemporary', 'Classics', 'Book Club', 'Young Adult', 'Childrens',
           'Literary Fiction', 'American', 'Unfinished', 'Owned', 'Favorites',
           'Series', 'Adult Fiction', '21st Century', '20th Century'}

STOP = set("""a about above after again against all am an and any are as at be because been before
being below between both but by can did do does doing down during each few for from further had has
have having he her here hers herself him himself his how i if in into is it its itself just me more
most my myself no nor not now of off on once only or other our ours ourselves out over own same she
should so some such than that the their theirs them themselves then there these they this those
through to too under until up very was we were what when where which while who whom why will with
you your yours yourself yourselves""".split())


def clean_text(t):
    t = str(t).lower()
    t = re.sub(r'[^a-z\s]', ' ', t)
    return ' '.join(w for w in t.split() if w not in STOP and len(w) > 2)


def parse_genres(x):
    try:
        lst = ast.literal_eval(x) if isinstance(x, str) else []
    except Exception:
        lst = []
    return [g for g in lst if g not in GENERIC][:5]


def main():
    os.makedirs(OUT, exist_ok=True)

    if not os.path.exists(DATA):
        raise SystemExit(f'Missing {DATA}. Put it next to this script and run again.')

    print('Loading', DATA)
    df = pd.read_csv(DATA, low_memory=False)
    df = df.dropna(subset=['description']).reset_index(drop=True)
    print('  rows with descriptions:', len(df))

    # --- deduplicate editions -------------------------------------------------
    df['dedupe_key'] = (
        df['Book-Title'].str.lower().str.replace(r'[^a-z0-9]', '', regex=True) + '|' +
        df['Book-Author'].fillna('').str.lower().str.replace(r'[^a-z]', '', regex=True))
    before = len(df)
    df = df.drop_duplicates(subset='dedupe_key').reset_index(drop=True)
    print(f'  after edition dedupe: {len(df)} (removed {before - len(df)})')

    df['genres'] = df['genres_raw'].apply(parse_genres)

    # ISBNs lose their leading zeros when the CSV is parsed as numbers
    # ("0440234743" comes back as "440234743"). Zero-pad to recover them -
    # without this, the cover lookup matches 2,672 books instead of 14,884.
    df['ISBN'] = (df['ISBN'].astype(str).str.strip().str.upper()
                  .str.replace(r'\.0$', '', regex=True).str.zfill(10))

    # --- optional publisher cover URLs ---------------------------------------
    df['cover_url'] = ''
    if os.path.exists(COVER_SOURCE):
        src = pd.read_csv(COVER_SOURCE, encoding='latin-1',
                          on_bad_lines='skip', low_memory=False)
        src['ISBN'] = src['ISBN'].astype(str).str.strip().str.upper()
        m = dict(zip(src['ISBN'], src['Image-URL-M']))
        df['cover_url'] = df['ISBN'].map(m).fillna('')
        print('  books with a publisher cover URL:', (df['cover_url'] != '').sum())
    else:
        print(f'  {COVER_SOURCE} not found - covers will come from Open Library only')

    # --- shared text field ----------------------------------------------------
    df['text'] = (df['Book-Title'].fillna('') + '. '
                  + df['Book-Author'].fillna('') + '. '
                  + df['description'].fillna(''))
    df['clean'] = df['text'].apply(clean_text)

    # --- TF-IDF ---------------------------------------------------------------
    print('Building TF-IDF')
    t0 = time.time()
    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)
    T = normalize(tfidf.fit_transform(df['clean'])).astype(np.float32)
    print(f'  matrix {T.shape}  nnz/row {T.nnz / T.shape[0]:.1f}  {time.time() - t0:.1f}s')

    sp.save_npz(f'{OUT}/tfidf.npz', T)
    with open(f'{OUT}/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf, f)

    # --- SBERT (optional) -----------------------------------------------------
    try:
        from sentence_transformers import SentenceTransformer
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            device = 'cpu'

        print(f'Building SBERT embeddings on {device}')
        t0 = time.time()
        enc = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        E = enc.encode(df['text'].tolist(), batch_size=128,
                       show_progress_bar=True, normalize_embeddings=True,
                       convert_to_numpy=True).astype(np.float32)
        np.save(f'{OUT}/sbert.npy', E)
        print(f'  embeddings {E.shape}  {time.time() - t0:.1f}s')
    except ImportError:
        print('SBERT skipped - sentence-transformers not installed.')
        print('  Install with: pip install sentence-transformers')

    # --- Collaborative filtering (ALS on implicit feedback) -------------------
    # Content models read text. This arm reads behaviour: which books the same
    # readers borrowed. It needs Ratings.csv.
    if os.path.exists(RATINGS):
        print('Training collaborative filtering model')
        import scipy.sparse as sps
        from als import train_als

        ratings = pd.read_csv(RATINGS, encoding='latin-1', on_bad_lines='skip')
        ratings.columns = ['user', 'isbn', 'rating']
        # Same zero-padding fix as above - without it almost nothing joins.
        ratings['isbn'] = (ratings['isbn'].astype(str).str.strip().str.upper()
                           .str.replace(r'\.0$', '', regex=True).str.zfill(10))

        pos = {b: i for i, b in enumerate(df['ISBN'])}
        ratings = ratings[ratings['isbn'].isin(pos)]

        # Users with very few interactions add noise, not signal.
        counts = ratings['user'].value_counts()
        ratings = ratings[ratings['user'].isin(counts[counts >= MIN_USER_RATINGS].index)]

        users = ratings['user'].unique()
        u2i = {u: i for i, u in enumerate(users)}
        Cui = sps.csr_matrix(
            (np.ones(len(ratings)),
             (ratings['user'].map(u2i).values, ratings['isbn'].map(pos).values)),
            shape=(len(users), len(df)))
        print(f'  interaction matrix {Cui.shape}, {Cui.nnz} interactions')

        t0 = time.time()
        _, item_factors = train_als(Cui, factors=CF_FACTORS, reg=0.05,
                                    alpha=40.0, iterations=CF_ITERATIONS)
        item_factors = normalize(item_factors).astype(np.float32)

        # A book nobody interacted with has no behavioural signal. Zero it and
        # record the fact, so the app can fall back to content for those.
        has_cf = np.asarray(Cui.sum(0)).ravel() > 0
        item_factors[~has_cf] = 0.0

        np.save(f'{OUT}/cf_items.npy', item_factors)
        np.save(f'{OUT}/cf_has.npy', has_cf)
        print(f'  item factors {item_factors.shape}  {time.time() - t0:.1f}s')
        print(f'  books with behavioural signal: {has_cf.sum()} / {len(df)}')
    else:
        print(f'CF skipped - {RATINGS} not found. Put it next to this script')
        print('  to enable the collaborative filtering and hybrid models.')

    # --- metadata -------------------------------------------------------------
    meta = df[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication',
               'Publisher', 'description', 'cover_url']].copy()
    meta.columns = ['isbn', 'title', 'author', 'year', 'publisher',
                    'description', 'cover_url']
    meta['genres'] = df['genres'].apply(lambda g: '|'.join(g))
    # CSV keeps dependencies to pandas alone (parquet would need pyarrow)
    meta.to_csv(f'{OUT}/meta.csv', index=False)

    print(f'\nDone. Artifacts written to {OUT}/')
    print('Next:  python app.py')


if __name__ == '__main__':
    main()
