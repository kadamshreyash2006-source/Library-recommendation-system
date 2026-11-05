"""
User data for the Smart Library website: accounts, ratings, saved books and
genre preferences, kept in a single SQLite file (library.db).

SQLite ships with Python, so this adds no dependency. Books are stored by ISBN
rather than by row index, so user data survives a rebuild of the index.
"""

import os
import sqlite3
from datetime import datetime, timezone

from flask import g

DB_PATH = os.environ.get('LIBRARY_DB', 'library.db')

SHELVES = {
    'want': 'Want to read',
    'reading': 'Reading',
    'read': 'Read',
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT    NOT NULL UNIQUE COLLATE NOCASE,
    display_name  TEXT    NOT NULL,
    password_hash TEXT    NOT NULL,
    created_at    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS ratings (
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    isbn       TEXT    NOT NULL,
    rating     INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    updated_at TEXT    NOT NULL,
    PRIMARY KEY (user_id, isbn)
);

CREATE TABLE IF NOT EXISTS library (
    user_id  INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    isbn     TEXT    NOT NULL,
    shelf    TEXT    NOT NULL CHECK (shelf IN ('want', 'reading', 'read')),
    added_at TEXT    NOT NULL,
    PRIMARY KEY (user_id, isbn)
);

CREATE TABLE IF NOT EXISTS genre_prefs (
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    genre   TEXT    NOT NULL,
    PRIMARY KEY (user_id, genre)
);
"""


def _now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
        g.db.execute('PRAGMA foreign_keys = ON')
    return g.db


def close_db(_exc=None):
    db = g.pop('db', None)
    if db is not None:
        db.close()


def init_db():
    # On Render the database lives on a mounted disk (LIBRARY_DB=/var/data/library.db);
    # create the folder if it is not there yet.
    folder = os.path.dirname(os.path.abspath(DB_PATH))
    os.makedirs(folder, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.executescript(SCHEMA)
    con.commit()
    con.close()


# ---------------------------------------------------------------- users
def create_user(username, display_name, password_hash):
    db = get_db()
    cur = db.execute(
        'INSERT INTO users (username, display_name, password_hash, created_at) '
        'VALUES (?, ?, ?, ?)', (username, display_name, password_hash, _now()))
    db.commit()
    return cur.lastrowid


def user_by_name(username):
    return get_db().execute('SELECT * FROM users WHERE username = ?',
                            (username,)).fetchone()


def user_by_id(uid):
    return get_db().execute('SELECT * FROM users WHERE id = ?', (uid,)).fetchone()


# ---------------------------------------------------------------- ratings
def set_rating(uid, isbn, rating):
    db = get_db()
    if rating:
        db.execute(
            'INSERT INTO ratings (user_id, isbn, rating, updated_at) VALUES (?, ?, ?, ?) '
            'ON CONFLICT(user_id, isbn) DO UPDATE SET rating = excluded.rating, '
            'updated_at = excluded.updated_at', (uid, isbn, int(rating), _now()))
    else:
        db.execute('DELETE FROM ratings WHERE user_id = ? AND isbn = ?', (uid, isbn))
    db.commit()


def ratings_for(uid):
    """{isbn: (rating, updated_at)}"""
    rows = get_db().execute(
        'SELECT isbn, rating, updated_at FROM ratings WHERE user_id = ? '
        'ORDER BY updated_at DESC', (uid,)).fetchall()
    return {r['isbn']: (r['rating'], r['updated_at']) for r in rows}


# ---------------------------------------------------------------- library
def save_book(uid, isbn, shelf='want'):
    if shelf not in SHELVES:
        raise ValueError(f'unknown shelf {shelf}')
    db = get_db()
    db.execute(
        'INSERT INTO library (user_id, isbn, shelf, added_at) VALUES (?, ?, ?, ?) '
        'ON CONFLICT(user_id, isbn) DO UPDATE SET shelf = excluded.shelf',
        (uid, isbn, shelf, _now()))
    db.commit()


def remove_book(uid, isbn):
    db = get_db()
    db.execute('DELETE FROM library WHERE user_id = ? AND isbn = ?', (uid, isbn))
    db.commit()


def library_for(uid):
    """{isbn: (shelf, added_at)}"""
    rows = get_db().execute(
        'SELECT isbn, shelf, added_at FROM library WHERE user_id = ? '
        'ORDER BY added_at DESC', (uid,)).fetchall()
    return {r['isbn']: (r['shelf'], r['added_at']) for r in rows}


# ---------------------------------------------------------------- genres
def set_genres(uid, genres):
    db = get_db()
    db.execute('DELETE FROM genre_prefs WHERE user_id = ?', (uid,))
    db.executemany('INSERT OR IGNORE INTO genre_prefs (user_id, genre) VALUES (?, ?)',
                   [(uid, gname) for gname in genres])
    db.commit()


def genres_for(uid):
    return [r['genre'] for r in get_db().execute(
        'SELECT genre FROM genre_prefs WHERE user_id = ? ORDER BY genre', (uid,))]
