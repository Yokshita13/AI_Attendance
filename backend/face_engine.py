"""Face detection (OpenCV Haar cascade) + recognition (LBPH, implemented with NumPy so it works
on every OpenCV version - the cv2.face contrib module was removed in OpenCV 5)."""
import base64
import os
import shutil
import threading

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR", BASE_DIR)
DATASET_DIR = os.path.join(DATA_DIR, "dataset")
MODEL_PATH = os.path.join(DATA_DIR, "trainer.npz")

FACE_SIZE = (200, 200)
# LBPH confidence is a distance: LOWER = better match. Tune for your camera/lighting.
CONFIDENCE_THRESHOLD = float(os.environ.get("FACE_THRESHOLD", 50))

_lock = threading.Lock()
if not hasattr(cv2, "CascadeClassifier"):
    raise RuntimeError(
        "This OpenCV version has no Haar cascade support (OpenCV 5 removed it). "
        "Run: pip uninstall -y opencv-python && pip install \"opencv-python>=4.8,<5\""
    )
_detector = cv2.CascadeClassifier(
    os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
)
if _detector.empty():
    raise RuntimeError("Could not load the Haar cascade file bundled with opencv-python 4.x.")
_recognizer = None  # trained model, or None when no students are enrolled


def decode_image(data_url):
    """base64 data URL (from the browser canvas) -> BGR numpy image."""
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    buf = np.frombuffer(base64.b64decode(data_url), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image data")
    return img


def extract_face(img):
    """Detect the largest face; return a normalised 200x200 grayscale crop, or None."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = _detector.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=6, minSize=(80, 80)
    )
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face = cv2.resize(gray[y:y + h, x:x + w], FACE_SIZE)
    return cv2.equalizeHist(face)


# ---------- dataset ----------
def save_samples(student_id, faces):
    folder = os.path.join(DATASET_DIR, str(student_id))
    os.makedirs(folder, exist_ok=True)
    for i, face in enumerate(faces):
        cv2.imwrite(os.path.join(folder, f"{i:03d}.png"), face)


def delete_samples(student_id):
    shutil.rmtree(os.path.join(DATASET_DIR, str(student_id)), ignore_errors=True)


# ---------- LBPH recognizer (Local Binary Pattern Histograms) ----------
GRID = 8  # 8x8 cells, each with a 256-bin histogram


def _lbp_histogram(face):
    """200x200 grayscale -> concatenated, per-cell L1-normalised LBP histograms (float32)."""
    f = face.astype(np.int16)
    c = f[1:-1, 1:-1]
    code = np.zeros(c.shape, np.uint8)
    # 8 neighbours, clockwise from top-left
    shifts = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0)]
    for bit, (dy, dx) in enumerate(shifts):
        nb = f[dy:dy + c.shape[0], dx:dx + c.shape[1]]
        code |= ((nb >= c).astype(np.uint8) << bit)
    h, w = code.shape
    ch, cw = h // GRID, w // GRID
    hist = np.zeros((GRID, GRID, 256), np.float32)
    for gy in range(GRID):
        for gx in range(GRID):
            cell = code[gy * ch:(gy + 1) * ch, gx * cw:(gx + 1) * cw]
            hist[gy, gx] = np.bincount(cell.ravel(), minlength=256)
    hist /= hist.sum(axis=2, keepdims=True)
    return hist.ravel()


class LBPHModel:
    """Nearest-neighbour over stored histograms using chi-square distance
    (same definition OpenCV's LBPH uses: lower = better match)."""

    def __init__(self, hists, labels):
        self.hists = hists          # (N, GRID*GRID*256)
        self.labels = labels        # (N,)

    @classmethod
    def fit(cls, faces, labels):
        return cls(np.stack([_lbp_histogram(f) for f in faces]), np.asarray(labels, np.int32))

    def predict(self, face):
        q = _lbp_histogram(face)
        d = ((self.hists - q) ** 2 / (self.hists + q + 1e-10)).sum(axis=1)
        i = int(np.argmin(d))
        return int(self.labels[i]), float(d[i])

    def save(self, path):
        np.savez_compressed(path, hists=self.hists, labels=self.labels)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["hists"], z["labels"])


# ---------- model ----------
# ---------- model ----------
def sample_path(student_id):
    """Path of a middle face sample (used so an admin can check a pending sign-up), or None."""
    folder = os.path.join(DATASET_DIR, str(student_id))
    if not os.path.isdir(folder):
        return None
    files = sorted(os.listdir(folder))
    return os.path.join(folder, files[len(files) // 2]) if files else None


def match_among(face, student_ids):
    """True if `face` matches any stored sample of the given students (used to catch the same
    face being submitted twice among pending sign-ups). Pending sets are small, so no caching."""
    images, labels = [], []
    for sid in student_ids:
        folder = os.path.join(DATASET_DIR, str(sid))
        if not os.path.isdir(folder):
            continue
        for fn in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, fn), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                labels.append(int(sid))
    if not images:
        return False
    _, distance = LBPHModel.fit(images, labels).predict(face)
    return distance <= CONFIDENCE_THRESHOLD


def train(allowed_ids=None):
    """Retrain the LBPH model from dataset/<student_id>/ (only ids in allowed_ids if given)."""
    global _recognizer
    images, labels = [], []
    if os.path.isdir(DATASET_DIR):
        for sid in os.listdir(DATASET_DIR):
            folder = os.path.join(DATASET_DIR, sid)
            if not (sid.isdigit() and os.path.isdir(folder)):
                continue
            if allowed_ids is not None and int(sid) not in allowed_ids:
                continue
            for fn in os.listdir(folder):
                img = cv2.imread(os.path.join(folder, fn), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)
                    labels.append(int(sid))
    with _lock:
        if not images:
            _recognizer = None
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)
            return 0
        rec = LBPHModel.fit(images, labels)
        rec.save(MODEL_PATH)
        _recognizer = rec
    return len(images)


def load_model():
    global _recognizer
    if os.path.exists(MODEL_PATH):
        rec = LBPHModel.load(MODEL_PATH)
        with _lock:
            _recognizer = rec


def recognize(img):
    """Returns dict: {status: 'no_face'|'no_model'|'unknown'|'match', student_id, confidence}."""
    face = extract_face(img)
    if face is None:
        return {"status": "no_face"}
    return recognize_face_crop(face)


def recognize_face_crop(face):
    """Match an already-extracted 200x200 face crop against the trained model."""
    with _lock:
        rec = _recognizer
        if rec is None:
            return {"status": "no_model"}
        label, distance = rec.predict(face)
    if distance <= CONFIDENCE_THRESHOLD:
        return {"status": "match", "student_id": int(label), "confidence": round(distance, 1)}
    return {"status": "unknown", "confidence": round(distance, 1)}
