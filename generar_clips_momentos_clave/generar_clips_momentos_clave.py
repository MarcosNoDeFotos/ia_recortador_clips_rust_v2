import os
import json
import sqlite3
from flask import Blueprint, render_template, request, jsonify, send_from_directory, make_response, url_for
from datetime import datetime
from multiprocessing import Process, Queue
import tkinter as tk
from tkinter import filedialog
import subprocess

app = Blueprint("generar_clips_momentos_clave", __name__, template_folder="templates")

CURRENT_PATH = os.path.dirname(__file__).replace("\\", "/") + "/"
MODEL_PATH = CURRENT_PATH + "../modelos/modelo_rust_svm.pkl"
ACTIONS_JSON = CURRENT_PATH + "../dataset/acciones.json"
VIDEOS_INPUT_DIR = CURRENT_PATH + "../videos_entrada/"
OUTPUT_DIR = CURRENT_PATH + "../clips_generados/"
FRAMES_CACHE_DIR = os.path.join(OUTPUT_DIR, "frames_cache")
DB_PATH = CURRENT_PATH + "../database.db"
os.makedirs(VIDEOS_INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FRAMES_CACHE_DIR, exist_ok=True)

PROGRESS_FILE = os.path.join(OUTPUT_DIR, "progreso_momentos_clave.json")

MASK_REGIONS = [
    {"x_ratio": 0.01, "y_ratio": 0.48, "width_ratio": 0.25, "height_ratio": 0.25},
    {"x_ratio": 0.844, "y_ratio": 0.87, "width_ratio": 0.5, "height_ratio": 0.5},
    {"x_ratio": 0.31, "y_ratio": 0.0, "width_ratio": 0.39, "height_ratio": 0.04},
    {"x_ratio": 0.0, "y_ratio": 0.95, "width_ratio": 0.1, "height_ratio": 0.05},
]

progress_data = {"log": "Esperando inicio...", "done": False, "progress": 0}

def get_clip_model_and_processor():
    import torch
    from transformers import CLIPProcessor, CLIPModel
    import joblib
    from sklearn.preprocessing import normalize

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", weights_only=False).to(DEVICE)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    data = joblib.load(MODEL_PATH)
    clf, classes = data["clf"], data["classes"]
    return model, processor, clf, classes, DEVICE, normalize

def get_actions_map():
    with open(ACTIONS_JSON, "r", encoding="utf-8") as f:
        raw_actions = json.load(f)
    actions_map = {}
    for cls, info in raw_actions.items():
        if isinstance(info, dict):
            actions_map[cls] = info.get("frases", [])
        else:
            actions_map[cls] = info
    return actions_map

def mask_face_region(frame):
    h, w, _ = frame.shape
    for m in MASK_REGIONS:
        x1 = int(w * m["x_ratio"])
        y1 = int(h * m["y_ratio"])
        x2 = int(x1 + w * m["width_ratio"])
        y2 = int(y1 + h * m["height_ratio"])
        frame[y1:y2, x1:x2] = 0
    return frame

def extract_frames_streaming(video_path, fps=1, batch_size=200):
    import cv2
    import numpy as np
    name = os.path.splitext(os.path.basename(video_path))[0]
    cache_dir = os.path.join(FRAMES_CACHE_DIR, name)
    os.makedirs(cache_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(fps_video // fps)) if fps_video > 0 else 1
    batch_frames, batch_times = [], []

    for i in range(0, total, step):
        log_progress(f"Extrayendo frame {i}/{total}", done=False, progress=i*100/total)
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = mask_face_region(frame)
        t = i / fps_video if fps_video > 0 else 0.0

        np.save(os.path.join(cache_dir, f"{i:08d}.npy"), np.array([frame, t], dtype=object))
        batch_frames.append(frame)
        batch_times.append(t)

        if len(batch_frames) >= batch_size:
            yield batch_frames, batch_times
            batch_frames, batch_times = [], []

    if batch_frames:
        yield batch_frames, batch_times

    cap.release()

def read_cached_frames_streaming(cache_dir, batch_size=200):
    import numpy as np
    cached = sorted([f for f in os.listdir(cache_dir) if f.endswith(".npy")])
    batch_frames, batch_times = [], []
    for i, f in enumerate(cached):
        log_progress(f"Leyendo frame de caché {i}/{cached.__len__()}", done=False, progress=i*100/cached.__len__())
        data = np.load(os.path.join(cache_dir, f), allow_pickle=True)
        batch_frames.append(data[0])
        batch_times.append(data[1])
        if len(batch_frames) >= batch_size:
            yield batch_frames, batch_times
            batch_frames, batch_times = [], []
    if batch_frames:
        yield batch_frames, batch_times

def log_progress(message, done=False, progress=None):
    global progress_data
    progress_data["log"] = message
    progress_data["done"] = done
    if progress is not None:
        progress_data["progress"] = progress

def detectar_momentos_clave(video_path, threshold=0.25, min_segment=2):
    import numpy as np
    import torch
    model, processor, clf, classes, DEVICE, normalize = get_clip_model_and_processor()
    actions_map = get_actions_map()

    name = os.path.splitext(os.path.basename(video_path))[0]
    cache_dir = os.path.join(FRAMES_CACHE_DIR, name)
    os.makedirs(cache_dir, exist_ok=True)

    import cv2
    cap = cv2.VideoCapture(video_path)
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    frame_count_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if fps_video and fps_video > 0 else 0
    cap.release()

    step = max(1, int(fps_video // 1)) if fps_video and fps_video > 0 else 1
    total_frames = max(1, frame_count_video // step) if frame_count_video > 0 else 1

    use_cache = any(f.endswith(".npy") for f in os.listdir(cache_dir))
    if use_cache:
        log_progress("🔄 Leyendo frames cacheados del disco...", progress=1)
    else:
        log_progress("🖼️ Extrayendo frames del vídeo y aplicando máscaras...", progress=1)
    frame_stream = read_cached_frames_streaming(cache_dir) if use_cache else extract_frames_streaming(video_path)

    idx_classes = {cls: classes.index(cls) for cls in classes if cls in actions_map}
    timestamps = []
    probs_by_class = {cls: [] for cls in idx_classes}

    frame_counter = 0
    log_progress("⏳ Preparando análisis de momentos clave...", progress=2)

    def process_batch(batch_frames):
        
        with torch.no_grad():
            inputs = processor(images=batch_frames, return_tensors="pt", padding=True).to(DEVICE)
            feats = model.get_image_features(**inputs)
            feats = normalize(feats.cpu().numpy())
            proba = clf.predict_proba(feats)  # shape: (batch, n_classes)
        return proba

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        batch_times_list = []
        batch_idx = 0
        for batch_frames, batch_times in frame_stream:
            batch_idx += 1
            log_progress(f"🔢 Procesando lote de frames {batch_idx}/{frame_stream.__sizeof__()}", progress=min(5 + batch_idx, 20))
            futures.append(executor.submit(process_batch, batch_frames))
            batch_times_list.append(batch_times)
        log_progress(f"🧠 Extrayendo características visuales y clasificando {len(futures)} lotes de frames...", progress=25)
        for i, f in enumerate(futures):
            proba = f.result()
            batch_times = batch_times_list[i]
            timestamps.extend(batch_times)
            for cls, idx in idx_classes.items():
                probs_by_class[cls].extend(proba[:, idx])
            frame_counter += len(batch_times)
            percent = min(89, round((frame_counter / total_frames) * 65 + 25, 1))
            log_progress(f"🧠 Clasificando frames: {frame_counter}/{total_frames} ({percent}%)", progress=percent)
        del batch_frames, batch_times, proba

    log_progress("🧩 Detectando segmentos en las predicciones...", progress=90)
    segmentos = {}
    with sqlite3.connect(DB_PATH) as db:
        for cls, probs in probs_by_class.items():
            segs, start, segment_probs = [], None, []
            for i, prob in enumerate(probs):
                if prob > threshold and start is None:
                    start = timestamps[i]
                    segment_probs = [prob]
                elif prob > threshold:
                    segment_probs.append(prob)
                elif prob <= threshold and start is not None:
                    end = timestamps[i]
                    avg_prob = float(np.mean(segment_probs))
                    if end - start >= min_segment:
                        segs.append((start, end, avg_prob))
                        db.execute(
                            "INSERT INTO clips(video_name, start, end, fecha_generacion, accuracy, prompt) VALUES (?, ?, ?, ?, ?, ?)",
                            (video_path, start, end - start, None, round(avg_prob, 3), cls)
                        )
                    start = None
                    segment_probs = []
            if start and segment_probs:
                end = timestamps[-1]
                avg_prob = float(np.mean(segment_probs))
                if end - start >= min_segment:
                    segs.append((start, end, avg_prob))
                    db.execute(
                        "INSERT INTO clips(video_name, start, end, fecha_generacion, accuracy, prompt) VALUES (?, ?, ?, ?, ?, ?)",
                        (video_path, start, end - start, None, round(avg_prob, 3), cls)
                    )
            segmentos[cls] = segs
        db.commit()
    log_progress("✅ Segmentos detectados y guardados en la base de datos", done=True, progress=100)
    return segmentos

def seleccionar_videos(queue):
    root = tk.Tk()
    root.withdraw()
    paths = filedialog.askopenfilenames(title="Seleccionar vídeos", filetypes=[("Vídeos", "*.mp4;*.mov;*.avi;*.mkv")])
    queue.put(paths)

@app.route('/')
def index():
    return render_template('generar_clips_momentos_clave_index.html')

@app.route('/agregar_videos', methods=['GET'])
def agregar_videos():
    q = Queue()
    p = Process(target=seleccionar_videos, args=(q,))
    p.start()
    p.join()
    paths = q.get()
    if not paths:
        return jsonify({"success": False})
    import shutil
    for pth in paths:
        shutil.move(pth, os.path.join(VIDEOS_INPUT_DIR, os.path.basename(pth)))
    return jsonify({"success": True})

@app.route('/listar_videos')
def listar_videos():
    vids = [f for f in os.listdir(VIDEOS_INPUT_DIR) if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))]
    return jsonify(vids)

@app.route('/video_original')
def video_original():
    video = request.args.get("video")
    if not video:
        return "No video", 400
    safe = os.path.basename(video)
    if not os.path.exists(os.path.join(VIDEOS_INPUT_DIR, safe)):
        return "Not found", 404
    return send_from_directory(VIDEOS_INPUT_DIR, safe)

@app.route('/detectar_momentos', methods=['POST'])
def detectar_momentos():
    data = request.get_json()
    video = data.get("video")
    if not video:
        return jsonify({"success": False, "error": "No video"})
    video_path = os.path.join(VIDEOS_INPUT_DIR, video)
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
    segmentos = detectar_momentos_clave(video_path)
    # Formato: [{clase, start, end, avg_prob}]
    result = []
    for cls, segs in segmentos.items():
        for s, e, avg in segs:
            result.append({
                "clase": cls,
                "start": float(s),
                "end": float(e),
                "avg_prob": float(avg),
                "video_name": video
            })
    return jsonify({"success": True, "segmentos": result})

@app.route('/progreso')
def progreso():
    # Solo lee la variable global, no el archivo
    return make_response(jsonify(progress_data))

@app.route('/video/<path:filename>')
def video_file(filename):
    return send_from_directory(OUTPUT_DIR, filename)

@app.route('/segmentos_por_video')
def segmentos_por_video():
    """Devuelve un dict {video_basename: cantidad_segmentos}"""
    with sqlite3.connect(DB_PATH) as db:
        rows = db.execute("SELECT video_name, COUNT(*) FROM clips GROUP BY video_name").fetchall()
    result = {}
    for video_name, count in rows:
        base = os.path.basename(video_name)
        result[base] = count
    return jsonify(result)

@app.route('/segmentos_de_video')
def segmentos_de_video():
    """Devuelve los segmentos para un vídeo concreto (por basename)"""
    video = request.args.get("video")
    if not video:
        return jsonify([])
    with sqlite3.connect(DB_PATH) as db:
        rows = db.execute(
            "SELECT clase, start, end, accuracy, prompt FROM clips WHERE video_name LIKE ? ORDER BY start",
            (f"%{video}",)
        ).fetchall()
    result = []
    for clase, start, end, accuracy, prompt in rows:
        result.append({
            "clase": prompt if prompt else clase,
            "start": float(start),
            "end": float(start + end),
            "avg_prob": float(accuracy) if accuracy is not None else 0.0,
            "video_name": video
        })
    return jsonify(result)

@app.route('/generar_clip', methods=['POST'])
def generar_clip():
    """
    Genera el clip .mp4 para el segmento dado.
    POST JSON: { "video_name": str, "start": float, "end": float, "clase": str }
    """
    data = request.get_json()
    video_name = data.get("video_name")
    start = float(data.get("start"))
    end = float(data.get("end"))
    clase = data.get("clase", "")

    video_path = None
    # Buscar ruta completa en DB
    with sqlite3.connect(DB_PATH) as db:
        row = db.execute("SELECT video_name FROM clips WHERE video_name LIKE ? LIMIT 1", (f"%{video_name}",)).fetchone()
        if row:
            video_path = row[0]
        else:
            # fallback: buscar en videos_entrada
            candidate = os.path.join(VIDEOS_INPUT_DIR, video_name)
            if os.path.exists(candidate):
                video_path = candidate
    if not video_path or not os.path.exists(video_path):
        return jsonify({"success": False, "error": "Video no encontrado"}), 404

    basename_out = f"{os.path.splitext(os.path.basename(video_path))[0]}_{clase}_{int(start)}_{int(end)}.mp4"
    output_path = os.path.join(OUTPUT_DIR, basename_out)

    try:
        duration = end - start
        cmd = [
            "ffmpeg",
            "-accurate_seek",
            "-ss", str(start),
            "-t", str(duration),
            "-i", video_path,
            "-map", "0",
            "-c", "copy",
            "-avoid_negative_ts", "1",
            "-fflags", "+genpts",
            "-reset_timestamps", "1",
            "-y",
            output_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Opcional: guardar en DB si quieres registrar el clip generado
        return jsonify({"success": True, "generated": basename_out, "url": url_for("generar_clips_momentos_clave.video_file", filename=basename_out)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/eliminar_segmento', methods=['POST'])
def eliminar_segmento():
    """
    Elimina un segmento individual de la base de datos.
    POST JSON: { "video_name": str, "clase": str, "start": float, "end": float }
    """
    data = request.get_json()
    video_name = data.get("video_name")
    clase = data.get("clase")
    start = float(data.get("start"))
    end = float(data.get("end"))
    with sqlite3.connect(DB_PATH) as db:
        db.execute(
            "DELETE FROM clips WHERE video_name LIKE ? AND (prompt=? OR clase=?) AND ABS(start-?)<0.01 AND ABS(start+end-?)<0.01",
            (f"%{video_name}", clase, clase, start, end)
        )
        db.commit()
    return jsonify({"success": True})

@app.route('/eliminar_clase', methods=['POST'])
def eliminar_clase():
    """
    Elimina todos los segmentos de una clase para un vídeo.
    POST JSON: { "video_name": str, "clase": str }
    """
    data = request.get_json()
    video_name = data.get("video_name")
    clase = data.get("clase")
    with sqlite3.connect(DB_PATH) as db:
        db.execute(
            "DELETE FROM clips WHERE video_name LIKE ? AND (prompt=? OR clase=?)",
            (f"%{video_name}", clase, clase)
        )
        db.commit()
    return jsonify({"success": True})
