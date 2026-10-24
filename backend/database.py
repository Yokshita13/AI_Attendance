"""SQLite data layer for the Smart Attendance System."""
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime

from werkzeug.security import generate_password_hash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# All runtime data (database, face samples, model, secret key) lives in DATA_DIR.
# On a server, point DATA_DIR at a persistent disk/volume so data survives redeploys.
DATA_DIR = os.environ.get("DATA_DIR", BASE_DIR)
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "attendance.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS admins (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    roll_no TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    department TEXT DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    date TEXT NOT NULL,          -- YYYY-MM-DD
    time TEXT NOT NULL,          -- HH:MM:SS
    confidence REAL,
    UNIQUE(student_id, date)     -- one entry per student per day (no duplicates)
);

CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance(date);
"""


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_db() as db:
        db.executescript(SCHEMA)
        if db.execute("SELECT COUNT(*) FROM admins").fetchone()[0] == 0:
            user = os.environ.get("ADMIN_USER", "admin")
            pwd = os.environ.get("ADMIN_PASSWORD")
            if not pwd:
                if os.environ.get("PRODUCTION") == "1":
                    raise RuntimeError("Set the ADMIN_PASSWORD environment variable before first start.")
                pwd = "admin123"  # local development only
            db.execute(
                "INSERT INTO admins (username, password_hash) VALUES (?, ?)",
                (user, generate_password_hash(pwd)),
            )


# ---------- students ----------
def add_student(roll_no, name, department=""):
    with get_db() as db:
        cur = db.execute(
            "INSERT INTO students (roll_no, name, department, created_at) VALUES (?,?,?,?)",
            (roll_no, name, department, datetime.now().isoformat(timespec="seconds")),
        )
        return cur.lastrowid


def get_student(student_id):
    with get_db() as db:
        return db.execute("SELECT * FROM students WHERE id=?", (student_id,)).fetchone()


def get_student_by_roll(roll_no):
    with get_db() as db:
        return db.execute("SELECT * FROM students WHERE roll_no=?", (roll_no,)).fetchone()


def list_students():
    with get_db() as db:
        return db.execute("SELECT * FROM students ORDER BY roll_no").fetchall()


def delete_student(student_id):
    with get_db() as db:
        db.execute("DELETE FROM students WHERE id=?", (student_id,))


# ---------- attendance ----------
def mark_attendance(student_id, confidence):
    """Returns (marked_now: bool, time_string). Duplicate scans on the same day are ignored."""
    now = datetime.now()
    d, t = now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")
    with get_db() as db:
        row = db.execute(
            "SELECT time FROM attendance WHERE student_id=? AND date=?", (student_id, d)
        ).fetchone()
        if row:
            return False, row["time"]
        db.execute(
            "INSERT INTO attendance (student_id, date, time, confidence) VALUES (?,?,?,?)",
            (student_id, d, t, confidence),
        )
        return True, t


def attendance_for_date(date):
    with get_db() as db:
        return db.execute(
            """SELECT s.roll_no, s.name, s.department, a.time, a.confidence
               FROM attendance a JOIN students s ON s.id = a.student_id
               WHERE a.date=? ORDER BY a.time""",
            (date,),
        ).fetchall()


def absentees_for_date(date):
    with get_db() as db:
        return db.execute(
            """SELECT roll_no, name, department FROM students
               WHERE id NOT IN (SELECT student_id FROM attendance WHERE date=?)
               ORDER BY roll_no""",
            (date,),
        ).fetchall()


def monthly_summary(month):
    """month = 'YYYY-MM'. Percentage is against the days on which attendance was taken
    (days with at least one record) in that month."""
    with get_db() as db:
        total_days = db.execute(
            "SELECT COUNT(DISTINCT date) FROM attendance WHERE date LIKE ?", (month + "-%",)
        ).fetchone()[0]
        rows = db.execute(
            """SELECT s.roll_no, s.name, s.department,
                      COUNT(a.id) AS present_days
               FROM students s
               LEFT JOIN attendance a
                 ON a.student_id = s.id AND a.date LIKE ?
               GROUP BY s.id ORDER BY s.roll_no""",
            (month + "-%",),
        ).fetchall()
    result = []
    for r in rows:
        pct = round(100 * r["present_days"] / total_days, 1) if total_days else 0.0
        result.append({**dict(r), "total_days": total_days, "percentage": pct})
    return total_days, result


def dashboard_stats():
    today = datetime.now().strftime("%Y-%m-%d")
    with get_db() as db:
        total = db.execute("SELECT COUNT(*) FROM students").fetchone()[0]
        present = db.execute(
            "SELECT COUNT(*) FROM attendance WHERE date=?", (today,)
        ).fetchone()[0]
        trend = db.execute(
            """SELECT date, COUNT(*) AS n FROM attendance
               GROUP BY date ORDER BY date DESC LIMIT 7"""
        ).fetchall()
    return {
        "total": total,
        "present": present,
        "absent": max(total - present, 0),
        "rate": round(100 * present / total, 1) if total else 0.0,
        "trend": [dict(r) for r in reversed(trend)],
    }