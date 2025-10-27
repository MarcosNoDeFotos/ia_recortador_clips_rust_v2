import os
import cv2
import torch
import numpy as np
import threading
from tqdm import tqdm
from flask import Blueprint, jsonify, render_template, request
from transformers import CLIPProcessor, CLIPModel
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib
import multiprocessing as mp
from datetime import datetime

# === CONFIGURACIÓN DE RUTAS ===
app = Blueprint("generar_modelo", __name__, template_folder="templates")

CURRENT_PATH = os.path.dirname(__file__).replace("\\", "/") + "/"
DATASET_DIR = os.path.join(CURRENT_PATH, "../dataset/")
OUTPUT_MODEL = os.path.join(CURRENT_PATH, "../modelos/modelo_rust_svm.pkl")
LOG_FILE = os.path.join(CURRENT_PATH, "../modelos/classification_report.log")
FRAME_SAMPLE_RATE = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === CONFIGURACIÓN DE MÁSCARAS ===
MASK_REGIONS = [
    {"x_ratio": 0.01, "y_ratio": 0.48, "width_ratio": 0.25, "height_ratio": 0.25},
    {"x_ratio": 0.844, "y_ratio": 0.87, "width_ratio": 0.5, "height_ratio": 0.5},
    {"x_ratio": 0.31, "y_ratio": 0, "width_ratio": 0.39, "height_ratio": 0.04},
    {"x_ratio": 0, "y_ratio": 0.95, "width_ratio": 0.1, "height_ratio": 0.05},
]

# === VARIABLES GLOBALES DE PROGRESO ===
progress_data = {
    "status": "idle",
    "progress": 0,
    "total": 0,
    "message": "",
    "report": "",
}


# === FUNCIONES AUXILIARES ===
def mask_face_region(frame):
    h, w, _ = frame.shape
    for maskRegion in MASK_REGIONS:
        x1 = int(w * maskRegion["x_ratio"])
        y1 = int(h * maskRegion["y_ratio"])
        x2 = int(x1 + w * maskRegion["width_ratio"])
        y2 = int(y1 + h * maskRegion["height_ratio"])
        frame[y1:y2, x1:x2] = 0
    return frame


def extract_frames(video_path, num_frames=FRAME_SAMPLE_RATE):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return frames
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = mask_face_region(frame)
            frames.append(frame)
    cap.release()
    return frames


def get_video_embedding(video_path, model, processor):
    frames = extract_frames(video_path)
    if not frames:
        raise ValueError(f"No se pudieron leer frames de {video_path}")
    inputs = processor(images=frames, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        img_feats = model.get_image_features(**inputs)
    img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
    return img_feats.mean(dim=0).cpu().numpy()


def process_video(args):
    cls, file, label, model, processor = args
    if not file.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
        return None
    path = os.path.join(DATASET_DIR, cls, file)
    try:
        emb = get_video_embedding(path, model, processor)
        return emb, label
    except Exception as e:
        print(f"⚠️ Error procesando {file}: {e}")
        return None


# === ENDPOINT PRINCIPAL (HTML) ===
@app.route("/")
def generar_modelo_index():
    return render_template("generar_modelo_index.html")


# === LISTAR CLASES ===
@app.route("/listar_clases")
def listar_clases():
    clases = []
    for cls in sorted(os.listdir(DATASET_DIR)):
        cls_path = os.path.join(DATASET_DIR, cls)
        if not os.path.isdir(cls_path):
            continue
        count = len([f for f in os.listdir(cls_path) if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))])
        clases.append({"clase": cls, "videos": count})
    return jsonify(clases)


# === GENERAR MODELO (WEB) ===
@app.route("/generar_modelo", methods=["POST"])
def generar_modelo():
    def generar_modelo_thread():
        progress_data.update({
            "status": "running",
            "progress": 0,
            "message": "Cargando modelo CLIP...",
            "report": ""
        })

        try:
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", weights_only=False).to(DEVICE)
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        except Exception as e:
            progress_data.update({"status": "error", "message": f"Error cargando CLIP: {e}"})
            return

        classes = sorted(os.listdir(DATASET_DIR))
        tasks = []
        for label, cls in enumerate(classes):
            cls_path = os.path.join(DATASET_DIR, cls)
            if os.path.isdir(cls_path):
                for file in os.listdir(cls_path):
                    if file.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                        tasks.append((cls, file, label, model, processor))

        total = len(tasks)
        progress_data["total"] = total
        if total == 0:
            progress_data.update({"status": "done", "message": "No hay vídeos para entrenar."})
            return

        progress_data["message"] = f"Procesando {total} vídeos..."
        mp.set_start_method("spawn", force=True)
        results = []
        with mp.Pool(processes=min(4, mp.cpu_count())) as pool:
            for i, r in enumerate(pool.imap_unordered(process_video, tasks), 1):
                if r:
                    results.append(r)
                progress_data["progress"] = i
                progress_data["message"] = f"Extrayendo embeddings ({i}/{total})..."

        if not results:
            progress_data.update({"status": "error", "message": "No se pudieron generar embeddings."})
            return

        X, y = zip(*results)
        X = np.stack(X)
        y = np.array(y)

        unique, counts = np.unique(y, return_counts=True)
        valid_classes = [cls for cls, count in zip(unique, counts) if count >= 2]
        mask = np.isin(y, valid_classes)
        X, y = X[mask], y[mask]
        classes = [c for i, c in enumerate(classes) if i in valid_classes]

        progress_data["message"] = "Entrenando modelo SVM..."
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        param_grid = {"C": [0.1, 1, 10], "gamma": ["scale", 0.01, 0.001], "kernel": ["rbf"]}
        grid = GridSearchCV(SVC(probability=True), param_grid, cv=3, n_jobs=-1, verbose=0)
        grid.fit(X_train, y_train)
        clf = grid.best_estimator_

        label_to_class = {label: cls for label, cls in zip(np.unique(y_train), classes)}
        class_labels = [label_to_class[i] for i in clf.classes_]

        y_pred = clf.predict(X_test)
        report = classification_report(
            y_test, y_pred,
            labels=clf.classes_,
            target_names=class_labels,
            zero_division=0
        )

        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n=== Reporte generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            f.write(report)
            f.write("\n" + "=" * 80 + "\n")

        joblib.dump({"clf": clf, "classes": class_labels}, OUTPUT_MODEL)

        progress_data.update({
            "status": "done",
            "message": "Modelo generado correctamente.",
            "report": report
        })

    threading.Thread(target=generar_modelo_thread, daemon=True).start()
    return jsonify({"status": "started"})


# === PROGRESO ===
@app.route("/progreso")
def progreso():
    return jsonify(progress_data)


# === LEER LOG ===
@app.route("/leer_log")
def leer_log():
    if not os.path.exists(LOG_FILE):
        return jsonify({"log": "No hay registros aún."})
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        return jsonify({"log": f.read()})
