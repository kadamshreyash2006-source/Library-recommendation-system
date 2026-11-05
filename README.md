# Smart Library Recommender — web app

A book recommender website: readers create an account, rate books (1–5 stars),
save books to a personal library (Want to read / Reading / Read), and get a
**For you** feed ranked by their own taste. Underneath is the same
content-based + hybrid recommender as before. Enter a book you liked,
get similar books back with cover thumbnails, genres and similarity scores.
Three models run side by side so you can see where they disagree: TF-IDF,
SBERT, and a hybrid that fuses semantic similarity with collaborative
filtering trained on real borrowing data.

## Files

| File | What it does |
|---|---|
| `build_index.py` | One-time: builds the TF-IDF matrix, SBERT embeddings and the ALS collaborative model, saves to `artifacts/` |
| `als.py` | Implicit-feedback ALS in NumPy — no extra dependency |
| `app.py` | Flask server: recommendation models (unchanged), accounts, ratings, library, personalisation |
| `db.py` | SQLite storage for users, ratings, saved books and genre preferences (`library.db`, created automatically) |
| `templates/` | Pages: home / For you, discover, book, library, profile & onboarding, sign in / register |
| `static/` | `style.css` and `app.js` shared by every page (stars, save menu, covers, search) |
| `requirements.txt` | Dependencies |

## Data you need

Put these in the same folder as the scripts:

- **`books_full_with_description.csv`** — required. 14,884 books with descriptions and genres.
- **`Books.csv`** — optional. Book-Crossing metadata, used for a second cover-image
  source. Covers work without it (Open Library is tried first).
- **`Ratings.csv`** — optional but recommended. Book-Crossing's 1.1M interactions.
  Without it the hybrid model is unavailable and only the content models run.

## How to run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

To include SBERT (recommended — it is the better model):

```bash
pip install sentence-transformers
```

Without it, everything still runs with TF-IDF alone.

### 2. Build the index — once

```bash
python build_index.py
```

Takes about 10 seconds for TF-IDF, plus 2–5 minutes for SBERT the first time
(it downloads a ~90 MB model). You only repeat this if the data changes.

Expected output:

```
  rows with descriptions: 14884
  after edition dedupe: 10671 (removed 4213)
  books with a publisher cover URL: 10671
Building TF-IDF
  matrix (10671, 20000)  nnz/row 64.1
```

### 3. Start the app

```bash
python app.py
```

Open **http://127.0.0.1:5000** in a browser.

To use a different port: `PORT=8080 python app.py`

## Running in Google Colab

Colab cannot open `localhost` directly, so tunnel it:

```python
!pip install flask pyngrok sentence-transformers -q
# upload books_full_with_description.csv, build_index.py, app.py,
# and templates/index.html (keep index.html inside a templates/ folder)
!python build_index.py

from pyngrok import ngrok
# free token from https://dashboard.ngrok.com/get-started/your-authtoken
ngrok.set_auth_token("YOUR_TOKEN")
print(ngrok.connect(5000))
!python app.py
```

Click the printed `https://....ngrok-free.app` link.

## The website

| Page | What it does |
|---|---|
| `/` | Signed out: landing page and popular books. Signed in: **For you** feed, "Because you liked …" rows, browse by genre |
| `/register`, `/login` | Accounts. Passwords are hashed (werkzeug); sessions last 30 days |
| `/welcome` | After sign-up: pick favourite genres, then rate a few popular books from them |
| `/discover` | The original three-model comparison (exact original results), now with rate/save on every result |
| `/book/<index>` | Book details, your rating, your shelf, and similar books (Hybrid model, original results) |
| `/library` | Your saved books by shelf, plus everything you rated; sort by date, title, author or rating |
| `/profile` | Stats, edit favourite genres, and the genre profile the recommender has learned |

### How preferences shape recommendations

The recommendation logic is the original `app.py`, untouched:
`content_scores`, `model_scores`, `rank` and `/api/recommend` return exactly
the same books in the same order as the original app, signed in or not.
Preferences are used only on the **For you** page:

- **Taste vector.** Every book you rated or saved is used as a query to the
  hybrid model, and the score vectors are added up, weighted by how much you
  liked each book: 5★ = +1.0, 4★ = +0.6, 3★ = +0.2, 2★ = −0.2, 1★ = −0.6;
  saved but unrated = +0.4 to +0.6. Books you disliked push their look-alikes down.
- **Genre affinity.** Your chosen genres, plus the genres of books you rated or
  saved, give every book a genre-match score.
- **For you** = taste + 0.15 × genre affinity + a small popularity prior
  (from Book-Crossing interaction counts). Books you already rated or saved are
  hidden; at most two per author and one per title, so the feed stays varied.
  A brand-new reader with only genres gets popular books in those genres.
- **"Because you liked …" rows** on the For you page start from the hybrid
  model's similar books for a book you rated highly, nudged by your taste.
- **Similar-book results** (Discover and the book page) are not personalised:
  they are the original models' output.

Tuning knobs (environment variables): `GENRE_BETA` (default 0.15),
`PERSONAL_LAMBDA` (default 0.35), plus the existing `HYBRID_ALPHA`.

On first start the app reads `Ratings.csv` once to count how many readers
interacted with each book, and caches it to `artifacts/popularity.npz`.

## How to use it

- Start typing a title — suggestions appear after two characters.
  Arrow keys to move, Enter to select, Escape to close.
- **Find similar books** runs the search; **Surprise me** picks a random book.
- Each result shows a cover, the author, genre tags and a similarity bar.
  A highlighted tag is one shared with your query book; two or more shared
  tags is counted as a hit in the project's evaluation.

## Notes

- **Covers** load from Open Library by ISBN, falling back to the publisher URL
  in `Books.csv`, then to a text placeholder. Some older books have no cover
  anywhere — the placeholder is expected, not a bug.
- **ISBNs need zero-padding.** Reading the CSV turns `0440234743` into
  `440234743`, which breaks every cover lookup. `build_index.py` pads them back
  and `app.py` reads the ISBN column as text. Do not remove either.
- **No N×N similarity matrix is built.** At 10,671 books a dense one would be
  about 900 MB. One query row is computed on demand instead, in a few
  milliseconds.
- **Three models.** TF-IDF and SBERT read text only. The hybrid fuses the best
  content model with ALS collaborative filtering at `HYBRID_ALPHA` (default 0.9,
  tuned: NDCG@10 0.477 vs 0.453 for content alone). Override with
  `HYBRID_ALPHA=0.8 python app.py`.
- **Free-text queries work.** Typing something that is not a catalogue title —
  "spaceships and interstellar war" — vectorises the text the same way the books
  were vectorised and ranks against them. The hybrid column is hidden for these,
  because collaborative filtering needs a book readers have actually borrowed.
- **Cold start.** 593 of 10,671 books have no borrowing history. For those the
  hybrid falls back to content alone and says so above the results.

## API

| Endpoint | Returns |
|---|---|
| `GET /api/search?q=hobb` | Title suggestions |
| `GET /api/recommend?title=The+Hobbit&k=6` | Recommendations from every loaded model |
| `GET /api/recommend?index=8182&k=6` | Same, by row index |
| `GET /api/random` | A random book |
| `GET /api/foryou?k=18` | Personal feed + "Because you liked" rows (signed in) |
| `GET /api/popular?genre=Fantasy&k=12` | Most-read books, optionally in one genre |
| `POST /api/rate` `{"index": 1459, "rating": 5}` | Rate 1–5, or 0 to remove |
| `POST /api/library` `{"index": 1459, "shelf": "want"}` | Save / move (`want`, `reading`, `read`) |
| `POST /api/library/remove` `{"index": 1459}` | Remove from library |
| `GET /api/library` | Saved and rated books |
| `GET/POST /api/preferences` `{"genres": [...]}` | Favourite genres, learned taste, stats |

## Troubleshooting

**`No artifacts found`** — run `python build_index.py` first.

**`Missing books_full_with_description.csv`** — the data file must sit in the
same folder as `build_index.py`.

**Port already in use** — `PORT=8080 python app.py`.

**All covers are placeholders** — check the browser console. Open Library may
be unreachable from your network; the app still works, covers just fall back
to text.

**Start over with no users** — stop the app and delete `library.db`.

**Running for others on a network** — set a fixed secret:
`SECRET_KEY=some-long-random-string python app.py`. Otherwise one is generated
and kept in `.secret_key`.

**SBERT column missing** — `pip install sentence-transformers`, then rerun
`build_index.py`. The app prints which models loaded at startup.

## Deploying on Render

The repo includes `render.yaml`, so Render can set everything up from it.

1. Push this folder to a GitHub repository (include the `artifacts/` folder).
2. On render.com: **New > Blueprint**, pick the repository, and click **Apply**.
3. Render installs `requirements.txt`, starts the app with gunicorn, and gives
   you a `https://<name>.onrender.com` link.

User accounts, ratings and saved books are kept in `/var/data/library.db` on a
1 GB persistent disk, which needs Render's paid Starter instance. To use the free
instance instead, change `plan: starter` to `plan: free` and delete the `disk:`
block in `render.yaml`. The site works the same, but all accounts and ratings
are wiped whenever Render restarts or sleeps the app.

The app uses about 230 MB of memory. SBERT free-text search is not installed on
Render (it needs PyTorch, about 1 GB); typed descriptions use TF-IDF instead.
Searching from a catalogue book still uses all three models.
