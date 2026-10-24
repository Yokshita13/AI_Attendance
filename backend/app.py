"""AI-Based Smart Attendance System - Flask application."""
import csv
import hmac
import io
import os
import secrets
from datetime import datetime
from functools import wraps

from flask import (Flask, Response, flash, jsonify, redirect, render_template,
                   request, session, url_for)
from werkzeug.security import check_password_hash

import database as db
import face_engine as fe

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR", BASE_DIR)
PRODUCTION = os.environ.get("PRODUCTION") == "1"
# Optional: if set, only devices that know this code (or logged-in admins) can mark attendance.
KIOSK_CODE = os.environ.get("KIOSK_CODE", "")


def _secret_key():
    key = os.environ.get("SECRET_KEY")
    if key:
        return key
    if PRODUCTION:
        raise RuntimeError("Set the SECRET_KEY environment variable in production.")
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, ".secret_key")
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(secrets.token_hex(32))
    with open(path) as f:
        return f.read().strip()


FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend")
app = Flask(
    __name__,
    template_folder=os.path.join(FRONTEND_DIR, "templates"),
    static_folder=os.path.join(FRONTEND_DIR, "static"),
)
app.secret_key = _secret_key()
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=PRODUCTION,   # cookies only over HTTPS when deployed
    PERMANENT_SESSION_LIFETIME=60 * 60 * 12,
)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # registration uploads several frames

db.init_db()
fe.load_model()

MIN_SAMPLES = 10


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("admin"):
            if request.path.startswith("/api/"):
                return jsonify(error="Login required"), 401
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)
    return wrapped


def kiosk_required(view):
    """Scan endpoints: open if no KIOSK_CODE is configured, else need admin login or the code."""
    @wraps(view)
    def wrapped(*args, **kwargs):
        if KIOSK_CODE and not (session.get("admin") or session.get("kiosk")):
            if request.path.startswith("/api/"):
                return jsonify(status="locked"), 401
            return redirect(url_for("kiosk_access"))
        return view(*args, **kwargs)
    return wrapped


@app.route("/kiosk-access", methods=["GET", "POST"])
def kiosk_access():
    if request.method == "POST":
        if KIOSK_CODE and hmac.compare_digest(request.form.get("code", ""), KIOSK_CODE):
            session["kiosk"] = True
            session.permanent = True
            return redirect(url_for("scan"))
        flash("Wrong access code", "error")
    return render_template("kiosk.html")


@app.get("/healthz")
def healthz():
    return "ok"


# ---------------- auth ----------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        with db.get_db() as conn:
            row = conn.execute(
                "SELECT * FROM admins WHERE username=?", (request.form.get("username", ""),)
            ).fetchone()
        if row and check_password_hash(row["password_hash"], request.form.get("password", "")):
            session["admin"] = row["username"]
            session.permanent = True
            nxt = request.args.get("next", "")
            return redirect(nxt if nxt.startswith("/") and not nxt.startswith("//") else url_for("dashboard"))
        flash("Invalid username or password", "error")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("scan"))


# ---------------- public kiosk ----------------
@app.route("/")
@kiosk_required
def scan():
    return render_template("scan.html")


@app.route("/api/recognize", methods=["POST"])
@kiosk_required
def api_recognize():
    data = request.get_json(silent=True) or {}
    try:
        img = fe.decode_image(data.get("image", ""))
    except Exception:
        return jsonify(status="error", message="Bad image"), 400

    result = fe.recognize(img)
    if result["status"] != "match":
        return jsonify(result)

    student = db.get_student(result["student_id"])
    if student is None:  # stale model entry for a deleted student
        return jsonify(status="unknown")
    marked, time_str = db.mark_attendance(student["id"], result["confidence"])
    return jsonify(
        status="marked" if marked else "already",
        name=student["name"],
        roll_no=student["roll_no"],
        department=student["department"],
        time=time_str,
        confidence=result["confidence"],
    )


@app.route("/api/today")
@kiosk_required
def api_today():
    today = datetime.now().strftime("%Y-%m-%d")
    rows = db.attendance_for_date(today)
    return jsonify([{"roll_no": r["roll_no"], "name": r["name"], "time": r["time"]} for r in rows][::-1][:8])


# ---------------- admin pages ----------------
@app.route("/dashboard")
@login_required
def dashboard():
    today = datetime.now().strftime("%Y-%m-%d")
    return render_template(
        "dashboard.html",
        stats=db.dashboard_stats(),
        recent=db.attendance_for_date(today),
        today=today,
    )


@app.route("/students")
@login_required
def students():
    return render_template("students.html", students=db.list_students())


@app.route("/students/<int:student_id>/delete", methods=["POST"])
@login_required
def delete_student(student_id):
    db.delete_student(student_id)
    fe.delete_samples(student_id)
    fe.train()
    flash("Student removed and model retrained.", "ok")
    return redirect(url_for("students"))


@app.route("/register")
@login_required
def register():
    return render_template("register.html", min_samples=MIN_SAMPLES)


@app.route("/api/register", methods=["POST"])
@login_required
def api_register():
    data = request.get_json(silent=True) or {}
    roll_no = (data.get("roll_no") or "").strip()
    name = (data.get("name") or "").strip()
    department = (data.get("department") or "").strip()
    if not roll_no or not name:
        return jsonify(error="Roll number and name are required."), 400
    if db.get_student_by_roll(roll_no):
        return jsonify(error=f"Roll number {roll_no} is already registered."), 409

    faces = []
    for url in data.get("images", []):
        try:
            face = fe.extract_face(fe.decode_image(url))
        except Exception:
            continue
        if face is not None:
            faces.append(face)
    if len(faces) < MIN_SAMPLES:
        return jsonify(error=f"Only {len(faces)} usable face images captured "
                             f"(need {MIN_SAMPLES}). Improve lighting and face the camera."), 422

    # Block the same face being enrolled under a second identity.
    probe = fe.recognize_face_crop(faces[len(faces) // 2])
    if probe["status"] == "match":
        other = db.get_student(probe["student_id"])
        if other:
            return jsonify(error=f"This face already matches {other['name']} ({other['roll_no']})."), 409

    sid = db.add_student(roll_no, name, department)
    fe.save_samples(sid, faces)
    fe.train()
    return jsonify(ok=True, student_id=sid, samples=len(faces))


# ---------------- reports ----------------
@app.route("/reports")
@login_required
def reports():
    now = datetime.now()
    date = request.args.get("date") or now.strftime("%Y-%m-%d")
    month = request.args.get("month") or now.strftime("%Y-%m")
    total_days, summary = db.monthly_summary(month)
    return render_template(
        "reports.html",
        date=date, month=month,
        present=db.attendance_for_date(date),
        absent=db.absentees_for_date(date),
        total_days=total_days, summary=summary,
    )


def _csv_response(rows, header, filename):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    w.writerows(rows)
    return Response(
        buf.getvalue(), mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.route("/reports/daily.csv")
@login_required
def daily_csv():
    date = request.args.get("date") or datetime.now().strftime("%Y-%m-%d")
    rows = [(r["roll_no"], r["name"], r["department"], date, r["time"], "Present")
            for r in db.attendance_for_date(date)]
    rows += [(r["roll_no"], r["name"], r["department"], date, "", "Absent")
             for r in db.absentees_for_date(date)]
    return _csv_response(rows, ["Roll No", "Name", "Department", "Date", "Time", "Status"],
                         f"attendance_{date}.csv")


@app.route("/reports/monthly.csv")
@login_required
def monthly_csv():
    month = request.args.get("month") or datetime.now().strftime("%Y-%m")
    _, summary = db.monthly_summary(month)
    rows = [(r["roll_no"], r["name"], r["department"], r["present_days"],
             r["total_days"], r["percentage"]) for r in summary]
    return _csv_response(rows, ["Roll No", "Name", "Department", "Days Present",
                                "Days Held", "Percentage"], f"attendance_{month}.csv")


if __name__ == "__main__":
    # Local development only. In production run:  gunicorn app:app  (see Procfile)
    app.run(host="127.0.0.1", port=5000, debug=False)