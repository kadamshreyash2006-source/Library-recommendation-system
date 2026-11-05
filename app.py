"""
Smart Library Recommender - web app

Run build_index.py first, then:

    python app.py

Then open http://127.0.0.1:5000 in a browser.

Models served, depending on which artifacts exist:
  tfidf   content, word overlap
  sbert   content, meaning
  hybrid  content fused with collaborative filtering (ALS on borrowing data)

On top of those models the website adds user accounts, star ratings, a
personal library and a "For you" feed. The recommendation logic below
(content_scores, model_scores, rank and /api/recommend) is the original
app.py logic, unchanged: "find similar books" returns exactly the original
model output. Reader preferences are used only for the "For you" page,
which combines the same model scores for the books a reader has rated or
saved (see "personalisation" further down).
"""

import os
import re
import pickle
import secrets
from functools import wraps
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import scipy.sparse as sp
from flask import (Flask, abort, g, jsonify, redirect, render_template, request,
                   session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash

import db

ART = 'artifacts'

# How much the hybrid leans on content vs behaviour.
# 1.0 = content only, 0.0 = collaborative only.
# 0.9 measured best on the genre-match evaluation (NDCG@10 0.477 vs 0.453).
HYBRID_ALPHA = float(os.environ.get('HYBRID_ALPHA', 0.9))

app = Flask(__name__)

# ---------------------------------------------------------------- load once
if not os.path.exists(f'{ART}/meta.csv'):
    raise SystemExit('No artifacts found. Run:  python build_index.py')

# dtype=str on isbn is essential: pandas would otherwise read "0440234743"
# back as a number and strip the leading zero again, breaking cover lookups.
META = pd.read_csv(f'{ART}/meta.csv', low_memory=False, dtype={'isbn': str})
META['genres'] = META['genres'].fillna('')
META['cover_url'] = META['cover_url'].fillna('')
META['author'] = META['author'].fillna('Unknown')
META['description'] = META['description'].fillna('')

T = sp.load_npz(f'{ART}/tfidf.npz').tocsr()
with open(f'{ART}/tfidf_vectorizer.pkl', 'rb') as f:
    VECTORIZER = pickle.load(f)

E = np.load(f'{ART}/sbert.npy') if os.path.exists(f'{ART}/sbert.npy') else None

CF = HAS_CF = None
if os.path.exists(f'{ART}/cf_items.npy'):
    CF = np.load(f'{ART}/cf_items.npy')
    HAS_CF = np.load(f'{ART}/cf_has.npy')

# Lazily loaded SBERT encoder, so a free-text query that is not in the
# catalogue can still be embedded. Loading costs a second, so it only
# happens on the first free-text query.
_ENCODER = None
_ENCODER_FAILED = False


def get_encoder():
    global _ENCODER, _ENCODER_FAILED
    if _ENCODER is not None or _ENCODER_FAILED:
        return _ENCODER
    try:
        from sentence_transformers import SentenceTransformer
        print('Loading SBERT encoder for free-text queries...')
        _ENCODER = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as exc:
        print(f'Could not load SBERT encoder ({exc}); free-text queries will '
              f'use TF-IDF only.')
        _ENCODER_FAILED = True
    return _ENCODER


MODELS = {}
if CF is not None:
    MODELS['hybrid'] = 'Hybrid'
if E is not None:
    MODELS['sbert'] = 'SBERT'
MODELS['tfidf'] = 'TF-IDF'

print(f'Loaded {len(META)} books. Models: {", ".join(MODELS.values())}')
if E is None:
    print('  No SBERT embeddings - pip install sentence-transformers, then rerun build_index.py')
if CF is None:
    print('  No collaborative filtering - put Ratings.csv next to build_index.py and rerun')
else:
    print(f'  Books with behavioural signal: {int(HAS_CF.sum())} / {len(META)}')

TITLES_LOWER = META['title'].astype(str).str.lower()
GENRE_SETS = [set(g.split('|')) - {''} for g in META['genres']]

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


# ---------------------------------------------------------------- scoring
def content_scores(model, idx=None, text=None):
    """Similarity of every book to a query, given either a catalogue row or
    free text. Returns None when the query cannot be represented."""
    if model == 'sbert':
        if E is None:
            return None
        if idx is not None:
            return E @ E[idx]
        enc = get_encoder()
        if enc is None:
            return None
        v = enc.encode([text], normalize_embeddings=True,
                       convert_to_numpy=True).astype(np.float32)[0]
        return E @ v
    # tfidf
    if idx is not None:
        return (T @ T[idx].T).toarray().ravel()
    from sklearn.preprocessing import normalize as _n
    qv = _n(VECTORIZER.transform([clean_text(text)]))
    if qv.nnz == 0:
        return None
    return (T @ qv.T).toarray().ravel()


def model_scores(model, idx=None, text=None):
    """Scores for one model. Returns (scores, note)."""
    if model == 'hybrid':
        base = 'sbert' if E is not None else 'tfidf'
        c = content_scores(base, idx, text)
        if c is None:
            return None, None
        c = np.asarray(c, dtype=np.float64).ravel()
        # A book outside the catalogue has no behavioural vector, and one
        # nobody has borrowed has no signal either: fall back to content.
        if idx is None:
            return c, 'content only - this book is not in the borrowing data'
        if not HAS_CF[idx]:
            return c, 'content only - no borrowing history for this book'
        b = CF @ CF[idx]
        b = np.asarray(b, dtype=np.float64).ravel()
        b[~HAS_CF] = 0.0
        return HYBRID_ALPHA * c + (1 - HYBRID_ALPHA) * b, None

    s = content_scores(model, idx, text)
    if s is None:
        return None, None
    return np.asarray(s, dtype=np.float64).ravel(), None


def rank(scores, k, exclude=None):
    s = scores.copy()
    if exclude is not None:
        s[exclude] = -np.inf
    k = min(k, len(s) - 1)
    part = np.argpartition(-s, k)[:k]
    return part[np.argsort(-s[part])]


def row_payload(i, score=None, query_genres=None):
    r = META.iloc[i]
    genres = [g for g in str(r['genres']).split('|') if g]
    shared = sorted(set(genres) & query_genres) if query_genres else []
    desc = str(r['description'])
    return {
        'index': int(i),
        'isbn': str(r['isbn']),
        'title': str(r['title']),
        'author': str(r['author']),
        'year': str(r['year']),
        'publisher': str(r['publisher']),
        'genres': genres,
        'shared_genres': shared,
        'is_hit': len(shared) >= 2,
        'cover_url': str(r['cover_url']),
        'blurb': desc[:260] + ('...' if len(desc) > 260 else ''),
        'score': round(float(score), 4) if score is not None else None,
    }


# ================================================================ website
# Everything below is the website layer: accounts, ratings, library, and
# personalisation built from the model scores above.

def _secret_key():
    if os.environ.get('SECRET_KEY'):
        return os.environ['SECRET_KEY']
    path = '.secret_key'
    if not os.path.exists(path):
        with open(path, 'w') as fh:
            fh.write(secrets.token_hex(32))
    with open(path) as fh:
        return fh.read().strip()


app.config.update(
    SECRET_KEY=_secret_key(),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    # Render serves the site over HTTPS (it sets RENDER=true), so cookies can be HTTPS-only there.
    SESSION_COOKIE_SECURE=bool(os.environ.get('RENDER')),
    PERMANENT_SESSION_LIFETIME=60 * 60 * 24 * 30,
)
db.init_db()
app.teardown_appcontext(db.close_db)

ISBN_TO_IDX = {isbn: i for i, isbn in enumerate(META['isbn'].astype(str))}
N_BOOKS = len(META)
AUTHORS_LOWER = META['author'].astype(str).str.lower().str.strip().values
# Title without series/edition notes, so "Dr. Franklin's Island (Readers
# Circle)" and "Dr. Franklin's Island" count as one book in a feed.
TITLE_KEYS = (META['title'].astype(str).str.lower()
              .str.replace(r'\s*[\(\[:].*$', '', regex=True)
              .str.replace(r'[^a-z0-9]', '', regex=True).values)

# --- genre matrix: one row per book, one column per genre, for fast affinity
_genre_counts = pd.Series([gname for s in GENRE_SETS for gname in s]).value_counts()
GENRES = list(_genre_counts.index)
G2COL = {gname: j for j, gname in enumerate(GENRES)}
_rows = [i for i, s in enumerate(GENRE_SETS) for _ in s]
_cols = [G2COL[gname] for s in GENRE_SETS for gname in s]
GENRE_MATRIX = sp.csr_matrix((np.ones(len(_rows)), (_rows, _cols)),
                             shape=(N_BOOKS, len(GENRES)))
GENRE_NORM = np.sqrt(np.maximum(np.asarray(GENRE_MATRIX.sum(1)).ravel(), 1))
# Genres offered when a reader picks favourites: the common, specific ones.
PICKABLE_GENRES = [gname for gname in GENRES[:60]
                   if gname not in {'Historical', 'Science Fiction Fantasy',
                                    'Mystery Thriller', 'British Literature',
                                    'School', 'Juvenile', 'France',
                                    'The United States Of America',
                                    'Biography Memoir'}][:44]


# --- popularity from the Book-Crossing interactions, cached in artifacts/
def _load_popularity():
    path = f'{ART}/popularity.npz'
    if os.path.exists(path):
        z = np.load(path)
        if len(z['count']) == N_BOOKS:
            return z['count'], z['avg']
    count = np.zeros(N_BOOKS)
    avg = np.zeros(N_BOOKS)
    if os.path.exists('Ratings.csv'):
        print('Computing book popularity from Ratings.csv (one time)...')
        r = pd.read_csv('Ratings.csv', encoding='latin-1', on_bad_lines='skip')
        r.columns = ['user', 'isbn', 'rating']
        r['isbn'] = (r['isbn'].astype(str).str.strip().str.upper()
                     .str.replace(r'\.0$', '', regex=True).str.zfill(10))
        r['idx'] = r['isbn'].map(ISBN_TO_IDX)
        r = r.dropna(subset=['idx'])
        r['idx'] = r['idx'].astype(int)
        c = r.groupby('idx').size()
        count[c.index.values] = c.values
        explicit = r[r['rating'] > 0]
        a = explicit.groupby('idx')['rating'].mean()
        avg[a.index.values] = a.values
        np.savez(path, count=count, avg=avg)
    return count, avg


POP_COUNT, POP_AVG = _load_popularity()
# Log-scaled to 0..1, nudged by average rating so well-liked books edge
# ahead of merely well-known ones.
_pop = np.log1p(POP_COUNT)
POPULARITY = _pop / (_pop.max() or 1.0)
POPULARITY = POPULARITY * (0.85 + 0.15 * np.clip(POP_AVG / 10.0, 0, 1))


# ---------------------------------------------------------------- personalisation
# A reader's taste is built from the same model scores that power "similar
# books": every book they rated or saved is used as a query, its score vector
# is weighted by how much they liked it, and the vectors are summed. Disliked
# books get negative weight, which pushes their look-alikes down.
#
#   taste(j)  = sum_i w_i * model_scores(TASTE_MODEL, idx=i)[j] / sum_i |w_i|
#   genre(j)  = how well book j's genres match the reader's genre weights
#   for_you   = taste + GENRE_BETA * genre + POP_BETA * popularity
#
# Readers with no history yet fall back to their chosen genres plus
# popularity, so the feed works from the moment they sign up.

TASTE_MODEL = next(iter(MODELS))           # hybrid when available, else best content model
GENRE_BETA = float(os.environ.get('GENRE_BETA', 0.15))
POP_BETA = 0.05
PERSONAL_LAMBDA = float(os.environ.get('PERSONAL_LAMBDA', 0.35))
MAX_SEEDS = 60
SHELF_WEIGHT = {'want': 0.4, 'reading': 0.6, 'read': 0.5}


def rating_weight(stars):
    """1 star -> -0.6, 3 stars -> +0.2, 5 stars -> +1.0"""
    return (stars - 2.5) / 2.5


class Profile:
    def __init__(self, uid):
        self.uid = uid
        self.ratings = db.ratings_for(uid)        # {isbn: (stars, ts)}
        self.library = db.library_for(uid)        # {isbn: (shelf, ts)}
        self.fav_genres = db.genres_for(uid)

        seeds = {}
        for isbn, (stars, ts) in self.ratings.items():
            if isbn in ISBN_TO_IDX:
                seeds[ISBN_TO_IDX[isbn]] = (rating_weight(stars), ts)
        for isbn, (shelf, ts) in self.library.items():
            i = ISBN_TO_IDX.get(isbn)
            # An explicit rating says more than a save, so it wins.
            if i is not None and i not in seeds:
                seeds[i] = (SHELF_WEIGHT[shelf], ts)
        # Most recent first, capped so a long history stays fast.
        ordered = sorted(seeds.items(), key=lambda kv: kv[1][1], reverse=True)[:MAX_SEEDS]
        self.seed_idx = np.array([i for i, _ in ordered], dtype=int)
        self.seed_w = np.array([w for _, (w, _) in ordered], dtype=np.float64)
        self.seen = set(ISBN_TO_IDX[b] for b in list(self.ratings) + list(self.library)
                        if b in ISBN_TO_IDX)

        # Genre weights: explicit favourites, plus genres of rated/saved books.
        gw = np.zeros(len(GENRES))
        for gname in self.fav_genres:
            if gname in G2COL:
                gw[G2COL[gname]] += 1.0
        for i, w in zip(self.seed_idx, self.seed_w):
            for gname in GENRE_SETS[i]:
                gw[G2COL[gname]] += 0.5 * w
        top = np.abs(gw).max()
        self.genre_weights = gw / top if top > 0 else gw

        self._taste = None
        self._seed_scores = None

    @property
    def has_history(self):
        return len(self.seed_idx) > 0

    @property
    def is_empty(self):
        return not self.has_history and not self.fav_genres

    def seed_scores(self):
        """(n_seeds, n_books): the model's score vector for each seed book."""
        if self._seed_scores is None:
            rows = [model_scores(TASTE_MODEL, idx=int(i))[0] for i in self.seed_idx]
            self._seed_scores = (np.vstack(rows) if rows
                                 else np.zeros((0, N_BOOKS)))
        return self._seed_scores

    def taste(self):
        if self._taste is None:
            if not self.has_history:
                self._taste = np.zeros(N_BOOKS)
            else:
                self._taste = (self.seed_w @ self.seed_scores()) / np.abs(self.seed_w).sum()
        return self._taste

    def genre_affinity(self):
        if not self.genre_weights.any():
            return np.zeros(N_BOOKS)
        a = (GENRE_MATRIX @ self.genre_weights) / GENRE_NORM
        return np.clip(a, -1, 1)

    def affinity(self):
        """How much this reader is likely to enjoy each book, used to re-rank
        similar-book results. Roughly on the same scale as a similarity."""
        if self.has_history:
            return self.taste() + GENRE_BETA * self.genre_affinity()
        return 2 * GENRE_BETA * self.genre_affinity()

    def for_you_scores(self):
        if self.has_history:
            return (self.taste() + GENRE_BETA * self.genre_affinity()
                    + POP_BETA * POPULARITY)
        if self.fav_genres:
            return self.genre_affinity() + 0.3 * POPULARITY
        return POPULARITY.copy()

    def reason(self, j):
        """A short 'why this book' for the For you feed."""
        liked = sorted((g for g in GENRE_SETS[j]
                        if g in G2COL and self.genre_weights[G2COL[g]] > 0.25),
                       key=lambda g: -self.genre_weights[G2COL[g]])
        if self.has_history:
            contrib = self.seed_w * self.seed_scores()[:, j]
            best = int(np.argmax(contrib))
            if contrib[best] > 0:
                seed = int(self.seed_idx[best])
                isbn = META.iloc[seed]['isbn']
                verb = 'liked' if isbn in self.ratings else 'saved'
                return {'text': f'Because you {verb} {META.iloc[seed]["title"]}',
                        'seed': seed, 'genres': liked[:3]}
        if liked:
            return {'text': 'Popular in ' + ', '.join(liked[:2]), 'seed': None,
                    'genres': liked[:3]}
        return {'text': 'Popular with readers', 'seed': None, 'genres': []}


def current_user():
    if 'user' not in g:
        uid = session.get('uid')
        g.user = db.user_by_id(uid) if uid else None
        if uid and g.user is None:
            session.clear()
    return g.user


def current_profile():
    if 'profile' not in g:
        u = current_user()
        g.profile = Profile(u['id']) if u else None
    return g.profile


def annotate(items):
    """Add the signed-in reader's rating and shelf to book payloads, plus
    how many Book-Crossing readers interacted with each book."""
    p = current_profile()
    for b in items:
        i = b['index']
        if i < 0:
            continue
        b['readers'] = int(POP_COUNT[i])
        if p is not None:
            r = p.ratings.get(b['isbn'])
            s = p.library.get(b['isbn'])
            b['user_rating'] = r[0] if r else 0
            b['shelf'] = s[0] if s else None
    return items


def diverse_top(scores, k, exclude=(), per_author=2):
    """Top-k by score, skipping excluded rows and capping books per author
    so one prolific writer does not fill the whole feed."""
    s = scores.copy()
    if exclude:
        s[list(exclude)] = -np.inf
    pool = rank(s, min(len(s) - 1, k * 6))
    out, per = [], {}
    titles = {TITLE_KEYS[i] for i in exclude}
    for j in pool:
        if not np.isfinite(s[j]):
            break
        a = AUTHORS_LOWER[j]
        if per.get(a, 0) >= per_author or TITLE_KEYS[j] in titles:
            continue
        per[a] = per.get(a, 0) + 1
        titles.add(TITLE_KEYS[j])
        out.append(int(j))
        if len(out) == k:
            break
    return out


# ---------------------------------------------------------------- auth helpers
def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if current_user() is None:
            if request.path.startswith('/api/'):
                return jsonify({'error': 'Please sign in first.', 'login': True}), 401
            return redirect(url_for('login', next=request.full_path.rstrip('?')))
        return view(*args, **kwargs)
    return wrapped


def csrf_token():
    if 'csrf' not in session:
        session['csrf'] = secrets.token_urlsafe(24)
    return session['csrf']


@app.before_request
def protect_writes():
    """Forms carry a CSRF token; API writes must be JSON, which a page on
    another site cannot send without the browser asking first."""
    if request.method in ('POST', 'PUT', 'DELETE'):
        if request.path.startswith('/api/'):
            if not request.is_json:
                abort(415)
        elif request.form.get('csrf') != session.get('csrf'):
            abort(400, 'Form expired. Go back, reload the page and try again.')


@app.context_processor
def inject_globals():
    u = current_user()
    return {'user': u, 'csrf_token': csrf_token, 'book_count': N_BOOKS,
            'models': MODELS, 'shelves': db.SHELVES}


def safe_next(target):
    if target and urlparse(target).netloc == '' and target.startswith('/') \
            and not target.startswith('//'):
        return target
    return url_for('home')


USERNAME_RE = re.compile(r'^[A-Za-z0-9_.-]{3,30}$')


# ---------------------------------------------------------------- pages
@app.route('/')
def home():
    return render_template('home.html')


@app.route('/discover')
def discover():
    return render_template('discover.html', start_index=request.args.get('index', ''))


@app.route('/book/<int:idx>')
def book(idx):
    if idx < 0 or idx >= N_BOOKS:
        abort(404)
    b = annotate([row_payload(idx)])[0]
    b['description'] = str(META.iloc[idx]['description'])
    b['avg'] = round(float(POP_AVG[idx]) / 2, 1) if POP_AVG[idx] else None
    return render_template('book.html', book=b)


@app.route('/library')
@login_required
def library_page():
    return render_template('library.html')


@app.route('/profile')
@login_required
def profile_page():
    return render_template('profile.html', onboarding=False)


@app.route('/welcome')
@login_required
def welcome():
    return render_template('profile.html', onboarding=True)


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        name = request.form.get('username', '').strip()
        pw = request.form.get('password', '')
        u = db.user_by_name(name)
        if u and check_password_hash(u['password_hash'], pw):
            session.clear()
            session['uid'] = u['id']
            session.permanent = True
            return redirect(safe_next(request.args.get('next')))
        error = 'That username and password do not match.'
    return render_template('auth.html', mode='login', error=error,
                           next=request.args.get('next', ''))


@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    form = {}
    if request.method == 'POST':
        form = request.form
        name = form.get('username', '').strip()
        display = form.get('display_name', '').strip() or name
        pw = form.get('password', '')
        if not USERNAME_RE.match(name):
            error = 'Usernames are 3-30 letters, numbers, dots, dashes or underscores.'
        elif len(pw) < 6:
            error = 'Use a password of at least 6 characters.'
        elif pw != form.get('confirm', ''):
            error = 'The two passwords do not match.'
        elif db.user_by_name(name):
            error = 'That username is taken.'
        else:
            uid = db.create_user(name, display[:40], generate_password_hash(pw))
            session.clear()
            session['uid'] = uid
            session.permanent = True
            return redirect(url_for('welcome'))
    return render_template('auth.html', mode='register', error=error, form=form,
                           next='')


@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return redirect(url_for('home'))


# ---------------------------------------------------------------- API: core
@app.route('/api/search')
def search():
    q = request.args.get('q', '').strip().lower()
    if len(q) < 2:
        return jsonify([])
    mask = TITLES_LOWER.str.startswith(q)
    hits = list(META.index[mask][:8])
    if len(hits) < 8:
        mask2 = TITLES_LOWER.str.contains(re.escape(q), na=False)
        extra = [i for i in META.index[mask2] if i not in set(hits)]
        hits += extra[:8 - len(hits)]
    return jsonify([{'index': int(i),
                     'title': str(META.iloc[i]['title']),
                     'author': str(META.iloc[i]['author'])} for i in hits])


@app.route('/api/recommend')
def recommend():
    raw = request.args.get('index')
    k = min(int(request.args.get('k', 6)), 20)
    text = request.args.get('title', '').strip()

    idx = None
    if raw is not None and raw.isdigit() and int(raw) < len(META):
        idx = int(raw)
    elif text:
        exact = META.index[TITLES_LOWER == text.lower()]
        if len(exact):
            idx = int(exact[0])
        else:
            partial = META.index[TITLES_LOWER.str.contains(re.escape(text.lower()),
                                                           na=False)]
            if len(partial):
                idx = int(partial[0])
    else:
        return jsonify({'error': 'Enter a book title or a description to search.'}), 400

    # Free text that matches no catalogue title is still a valid query: it gets
    # vectorised the same way the books were, and compared against them.
    free_text = idx is None
    query_genres = GENRE_SETS[idx] if idx is not None else set()

    results, notes = {}, {}
    shown = dict(MODELS)
    if free_text and 'hybrid' in shown:
        # With no catalogue row there is no behavioural vector, so the hybrid
        # would just repeat its content half. Drop the column and say why.
        shown.pop('hybrid')
        notes['_query'] = ('Collaborative filtering needs a book that readers have '
                           'borrowed. For a free-text search only the content '
                           'models can answer.')

    for key in shown:
        scores, note = model_scores(key, idx=idx, text=text)
        if scores is None:
            continue
        top = rank(scores, k, exclude=idx)
        # annotate() only adds the reader's own rating/shelf for display;
        # the books and their order are exactly the original model output.
        results[key] = annotate([row_payload(j, scores[j], query_genres) for j in top])
        if note:
            notes[key] = note

    if not results:
        return jsonify({'error': f'Could not interpret "{text}". Try other words, '
                                 f'or pick a title from the suggestions.'}), 404

    if free_text:
        query = {'title': text, 'author': 'Free-text query', 'genres': [],
                 'blurb': 'Not a catalogue book - matched against book descriptions.',
                 'cover_url': '', 'isbn': '', 'publisher': '', 'year': '',
                 'index': -1, 'shared_genres': [], 'is_hit': False, 'score': None}
    else:
        query = annotate([row_payload(idx)])[0]

    return jsonify({'query': query, 'models': shown, 'results': results,
                    'notes': notes, 'free_text': free_text})


@app.route('/api/random')
def random_book():
    i = int(np.random.default_rng().integers(0, len(META)))
    return jsonify(annotate([row_payload(i)])[0])


# ---------------------------------------------------------------- API: website
@app.route('/api/foryou')
@login_required
def for_you():
    k = min(int(request.args.get('k', 18)), 48)
    p = current_profile()
    scores = p.for_you_scores()
    top = diverse_top(scores, k, exclude=p.seen)
    items = []
    for j in top:
        b = row_payload(j)
        b['reason'] = p.reason(j)
        items.append(b)
    annotate(items)

    # "Because you liked X" shelves for the reader's three most recent loves.
    shelves = []
    loved = [int(i) for i, w in zip(p.seed_idx, p.seed_w) if w >= 0.6][:3]
    shown = set(top) | p.seen
    for i in loved:
        s, _ = model_scores(TASTE_MODEL, idx=i)
        s = s + PERSONAL_LAMBDA * p.affinity()
        picks = diverse_top(s, 8, exclude=shown | {i})
        shown |= set(picks)
        shelves.append({'seed': annotate([row_payload(i)])[0],
                        'books': annotate([row_payload(j) for j in picks])})

    return jsonify({'books': items, 'shelves': shelves,
                    'has_history': p.has_history,
                    'fav_genres': p.fav_genres,
                    'counts': {'rated': len(p.ratings), 'saved': len(p.library)}})


@app.route('/api/popular')
def popular():
    k = min(int(request.args.get('k', 12)), 48)
    genre = request.args.get('genre', '')
    s = POPULARITY.copy()
    if genre:
        if genre not in G2COL:
            return jsonify([])
        mask = np.asarray(GENRE_MATRIX[:, G2COL[genre]].todense()).ravel() > 0
        s[~mask] = -np.inf
    p = current_profile()
    exclude = p.seen if p else ()
    return jsonify(annotate([row_payload(j) for j in diverse_top(s, k, exclude)]))


@app.route('/api/genres')
def genres():
    return jsonify(PICKABLE_GENRES)


def _book_from_body():
    body = request.get_json(silent=True) or {}
    try:
        idx = int(body.get('index'))
    except (TypeError, ValueError):
        abort(400, 'index is required')
    if idx < 0 or idx >= N_BOOKS:
        abort(404)
    return idx, str(META.iloc[idx]['isbn']), body


@app.route('/api/rate', methods=['POST'])
@login_required
def rate():
    idx, isbn, body = _book_from_body()
    try:
        stars = int(body.get('rating', 0))
    except (TypeError, ValueError):
        stars = -1
    if stars < 0 or stars > 5:
        return jsonify({'error': 'Ratings are 1 to 5 stars (0 removes it).'}), 400
    db.set_rating(current_user()['id'], isbn, stars)
    # Rating a book means you have read it; file it on the Read shelf
    # unless it is already somewhere in the library.
    if stars and isbn not in db.library_for(current_user()['id']):
        db.save_book(current_user()['id'], isbn, 'read')
    g.pop('profile', None)
    return jsonify(annotate([row_payload(idx)])[0])


@app.route('/api/library', methods=['GET'])
@login_required
def library_list():
    p = current_profile()
    items = []
    for isbn, (shelf, ts) in p.library.items():
        i = ISBN_TO_IDX.get(isbn)
        if i is None:
            continue
        b = row_payload(i)
        b['added_at'] = ts
        items.append(b)
    annotate(items)
    rated = []
    for isbn, (stars, ts) in p.ratings.items():
        i = ISBN_TO_IDX.get(isbn)
        if i is not None:
            b = row_payload(i)
            b['rated_at'] = ts
            rated.append(b)
    annotate(rated)
    return jsonify({'library': items, 'rated': rated, 'shelves': db.SHELVES})


@app.route('/api/library', methods=['POST'])
@login_required
def library_save():
    idx, isbn, body = _book_from_body()
    shelf = body.get('shelf', 'want')
    if shelf not in db.SHELVES:
        return jsonify({'error': 'Unknown shelf.'}), 400
    db.save_book(current_user()['id'], isbn, shelf)
    g.pop('profile', None)
    return jsonify(annotate([row_payload(idx)])[0])


@app.route('/api/library/remove', methods=['POST'])
@login_required
def library_remove():
    idx, isbn, _ = _book_from_body()
    db.remove_book(current_user()['id'], isbn)
    g.pop('profile', None)
    return jsonify(annotate([row_payload(idx)])[0])


@app.route('/api/preferences', methods=['GET', 'POST'])
@login_required
def preferences():
    uid = current_user()['id']
    if request.method == 'POST':
        body = request.get_json(silent=True) or {}
        chosen = [x for x in body.get('genres', []) if x in G2COL][:20]
        db.set_genres(uid, chosen)
        g.pop('profile', None)
    p = current_profile()
    # The reader's taste as the recommender sees it: strongest genre weights.
    order = np.argsort(-p.genre_weights)
    top = [{'genre': GENRES[j], 'weight': round(float(p.genre_weights[j]), 3)}
           for j in order[:10] if p.genre_weights[j] > 0]
    stars = [v[0] for v in p.ratings.values()]
    return jsonify({
        'genres': p.fav_genres,
        'taste': top,
        'stats': {
            'rated': len(p.ratings),
            'saved': len(p.library),
            'avg_rating': round(sum(stars) / len(stars), 2) if stars else None,
            'shelves': {s: sum(1 for v in p.library.values() if v[0] == s)
                        for s in db.SHELVES},
        },
    })


@app.errorhandler(404)
def not_found(e):
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Not found.'}), 404
    return render_template('error.html', message='That page is not on the shelves.'), 404


@app.errorhandler(400)
def bad_request(e):
    if request.path.startswith('/api/'):
        return jsonify({'error': getattr(e, 'description', 'Bad request.')}), 400
    return render_template('error.html', message=getattr(e, 'description', 'Bad request.')), 400


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f'\n  Open http://127.0.0.1:{port} in your browser\n')
    app.run(host='0.0.0.0', port=port, debug=False)
