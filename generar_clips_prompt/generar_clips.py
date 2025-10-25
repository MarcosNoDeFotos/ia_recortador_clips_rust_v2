import os
import json
import re
import cv2
import torch
import numpy as np
from flask import Blueprint, render_template, request, jsonify, send_from_directory, make_response
from sklearn.preprocessing import normalize
from transformers import CLIPProcessor, CLIPModel
import joblib
import subprocess
from multiprocessing import  Process, Queue
import tkinter as tk
from tkinter import filedialog

# === CONFIG ===
app = Blueprint("generar_clips", __name__, template_folder="templates")

CURRENT_PATH = os.path.dirname(__file__).replace("\\", "/") + "/"
MODEL_PATH = CURRENT_PATH + "../modelos/modelo_rust_svm.pkl"
ACTIONS_JSON = CURRENT_PATH + "../dataset/acciones.json"
VIDEOS_INPUT_DIR = CURRENT_PATH + "../videos_entrada/"
OUTPUT_DIR = CURRENT_PATH + "../clips_generados/"
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "progreso_generar_clips.json")
FRAMES_CACHE_DIR = os.path.join(OUTPUT_DIR, "frames_cache")

os.makedirs(VIDEOS_INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FRAMES_CACHE_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === CARGAR MODELO ===
print(f"🔹 Cargando modelo CLIP ({DEVICE}) y clasificador SVM...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", weights_only=False).to(DEVICE)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
data = joblib.load(MODEL_PATH)
clf, classes = data["clf"], data["classes"]

with open(ACTIONS_JSON, "r", encoding="utf-8") as f:
    ACTIONS_MAP = json.load(f)


MASK_REGIONS = [
    {"x_ratio": 0.01, "y_ratio": 0.48, "width_ratio": 0.25, "height_ratio": 0.25},
    {"x_ratio": 0.844, "y_ratio": 0.87, "width_ratio": 0.5, "height_ratio": 0.5},
    {"x_ratio": 0.31, "y_ratio": 0.0, "width_ratio": 0.39, "height_ratio": 0.04},
    {"x_ratio": 0.0, "y_ratio": 0.95, "width_ratio": 0.1, "height_ratio": 0.05},
]


def log_progress(message, done=False, progress=None):
    """Escribe progreso en archivo JSON para que el frontend lo lea en tiempo real"""
    progress_data = {"log": message, "done": done}
    if progress is not None:
        progress_data["progress"] = progress
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress_data, f, ensure_ascii=False, indent=2)


def mask_face_region(frame):
    h, w, _ = frame.shape
    for m in MASK_REGIONS:
        x1 = int(w * m["x_ratio"])
        y1 = int(h * m["y_ratio"])
        x2 = int(x1 + w * m["width_ratio"])
        y2 = int(y1 + h * m["height_ratio"])
        frame[y1:y2, x1:x2] = 0
    return frame


def get_frame_embedding(frame):
    inputs = processor(images=[frame], return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
    feats = normalize(feats.cpu().numpy())
    return feats[0]


def parse_duration_from_prompt(prompt):
    prompt = prompt.lower()
    m = re.search(r"(\d+(?:[\.,]\d+)?)\s*(min|minute|minuto|seg|segundo|s)", prompt)
    if not m:
        return 60
    val = float(m.group(1).replace(",", "."))
    unit = m.group(2)
    return int(val * 60) if unit.startswith("min") else int(val)


def find_most_similar_class(prompt, min_similarity=0.25):
    text_list, lookup = [], []
    for cls, phrases in ACTIONS_MAP.items():
        for p in phrases:
            text_list.append(p)
            lookup.append(cls)
    inputs = processor(text=text_list, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        text_feats = model.get_text_features(**inputs)
    text_feats = text_feats / text_feats.norm(p=2, dim=-1, keepdim=True)
    prompt_inputs = processor(text=prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        prompt_feat = model.get_text_features(**prompt_inputs)
    prompt_feat = prompt_feat / prompt_feat.norm(p=2, dim=-1, keepdim=True)
    sims = (prompt_feat @ text_feats.T).cpu().numpy()[0]
    idx = np.argmax(sims)
    cls = lookup[idx]
    phrase = text_list[idx]
    score = float(sims[idx])
    if score < min_similarity:
        return None, score, phrase
    return cls, score, phrase


def extract_frames_streaming(video_path, fps=1, batch_size=200):
    """Extrae frames en streaming, guarda en caché y rinde por lotes."""
    name = os.path.splitext(os.path.basename(video_path))[0]
    cache_dir = os.path.join(FRAMES_CACHE_DIR, name)
    os.makedirs(cache_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(fps_video // fps))
    batch_frames, batch_times = [], []

    for i in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = mask_face_region(frame)
        t = i / fps_video

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
    """Lee frames del caché en streaming por lotes."""
    cached = sorted([f for f in os.listdir(cache_dir) if f.endswith(".npy")])
    batch_frames, batch_times = [], []
    for f in cached:
        data = np.load(os.path.join(cache_dir, f), allow_pickle=True)
        batch_frames.append(data[0])
        batch_times.append(data[1])
        if len(batch_frames) >= batch_size:
            yield batch_frames, batch_times
            batch_frames, batch_times = [], []
    if batch_frames:
        yield batch_frames, batch_times


def generar_clips(video_path, prompt, threshold=0.25):
    clip_duration = parse_duration_from_prompt(prompt)
    cls, score, phrase = find_most_similar_class(prompt)
    if not cls:
        return []

    log_progress(f"🎯 Clase detectada: {cls} (similaridad {round(score,3)})")
    log_progress(f"📽️ Analizando {os.path.basename(video_path)}")

    idx_class = classes.index(cls)
    name = os.path.splitext(os.path.basename(video_path))[0]
    cache_dir = os.path.join(FRAMES_CACHE_DIR, name)
    os.makedirs(cache_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    frame_count_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    step = max(1, int(fps_video // 1))  # fps de extracción
    total_frames = max(1, frame_count_video // step)

    use_cache = any(f.endswith(".npy") for f in os.listdir(cache_dir))
    frame_stream = (
        read_cached_frames_streaming(cache_dir)
        if use_cache else extract_frames_streaming(video_path)
    )

    timestamps, target_probs = [], []
    frame_counter = 0

    # === Análisis de frames ===
    for batch_frames, batch_times in frame_stream:
        with torch.no_grad():
            inputs = processor(images=batch_frames, return_tensors="pt", padding=True).to(DEVICE)
            feats = model.get_image_features(**inputs)
            feats = normalize(feats.cpu().numpy())
            probs = clf.predict_proba(feats)[:, idx_class]

        timestamps.extend(batch_times)
        target_probs.extend(probs)
        frame_counter += len(batch_frames)

        percent = min(100, round((frame_counter / total_frames) * 100, 1))
        log_progress(f"⏱️ Procesados {frame_counter}/{total_frames} frames ({percent}%)", progress=percent)

        del batch_frames, batch_times, feats, probs

    # === Detección de segmentos ===
    log_progress("🧩 Detectando segmentos relevantes...", progress=85)
    segments, start, segment_probs = [], None, []

    for i, prob in enumerate(target_probs):
        if prob > threshold and start is None:
            start = timestamps[i]
            segment_probs = [prob]
        elif prob > threshold:
            segment_probs.append(prob)
        elif prob <= threshold and start is not None:
            end = timestamps[i]
            avg_prob = np.mean(segment_probs)
            if end - start >= clip_duration / 6:
                segments.append((start, end, avg_prob))
            start = None
            segment_probs = []

    if start and segment_probs:
        end = timestamps[-1]
        avg_prob = np.mean(segment_probs)
        if end - start >= clip_duration / 6:
            segments.append((start, end, avg_prob))

    # === Generar clips ===
    total_segments = len(segments)
    clips = []
    for i, (start, end, avg) in enumerate(segments):
        mid = (start + end) / 2
        clip_start = max(0, mid - clip_duration / 2)
        name = f"{os.path.splitext(os.path.basename(video_path))[0]}_{cls}_{i+1}_{round(avg,3)}.mp4"
        output_path = os.path.join(OUTPUT_DIR, name)

        progress_est = 85 + round((i / max(1, total_segments)) * 15, 1)
        log_progress(f"🎞️ Generando clip {i+1}/{total_segments} → {name}", progress=progress_est)

        cmd = ["ffmpeg", "-y", "-ss", str(clip_start), "-i", video_path,
               "-t", str(clip_duration), "-map", "0", "-c", "copy", output_path]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        clips.append({"nombre": name, "prompt": prompt})

    log_progress(f"✅ {len(clips)} clips generados desde {os.path.basename(video_path)}", done=True, progress=100)
    return clips


def proceso_generar(prompt, videos):
    all_clips = []
    log_progress("🚀 Iniciando generación de clips...", progress=0)
    for idx, v in enumerate(videos):
        vpath = os.path.join(VIDEOS_INPUT_DIR, v)
        if os.path.exists(vpath):
            log_progress(f"▶️ Procesando video {idx+1}/{len(videos)}: {v}")
            clips = generar_clips(vpath, prompt)
            all_clips.extend(clips)
    log_progress("✅ Proceso completado.", done=True, progress=100)

    log_file = os.path.join(OUTPUT_DIR, "clips_generados.json")
    try:
        existing = []
        if os.path.exists(log_file):
            existing = json.load(open(log_file, "r", encoding="utf-8"))
        existing.extend(all_clips)
        json.dump(existing, open(log_file, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    except Exception as e:
        print("Error guardando clips_generados.json:", e)
def detectar_clase_test(prompt):
    cls, score, _ = find_most_similar_class(prompt)
    if not cls:
        cls = "none :("
        score = 0
    return cls, round(score,3)

def seleccionar_videos(queue):
    root = tk.Tk()
    root.withdraw()
    paths = filedialog.askopenfilenames(title="Seleccionar vídeos", filetypes=[("Vídeos", "*.mp4;*.mov;*.avi;*.mkv")])
    queue.put(paths)

# === RUTAS WEB ===
@app.route('/')
def index():
    return render_template('generar_clips_index.html')

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


@app.route('/listar_clips')
def listar_clips():
    clips_json = os.path.join(OUTPUT_DIR, "clips_generados.json")
    if os.path.exists(clips_json):
        with open(clips_json, "r", encoding="utf-8") as f:
            clips = json.load(f)
    else:
        clips = []
    return jsonify(clips)


@app.route('/video/<path:filename>')
def video_file(filename):
    return send_from_directory(OUTPUT_DIR, filename)


@app.route('/generar', methods=['POST'])
def generar():
    data = request.get_json()
    prompt = data.get("prompt", "")
    videos = data.get("videos", [])
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
    p = Process(target=proceso_generar, args=(prompt, videos))
    p.start()
    return jsonify({"success": True})
@app.route('/detectar_clase', methods=['POST'])
def detectar_clase():
    data = request.get_json()
    prompt = data.get("prompt", "")

    clase, similaridad = detectar_clase_test(prompt)
    return jsonify({"success": True, "clase": clase, "similaridad": similaridad})


@app.route('/progreso')
def progreso():
    if not os.path.exists(PROGRESS_FILE):
        resp = make_response(jsonify({"log": "Esperando inicio...", "done": False}))
    else:
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            resp = make_response(jsonify(json.load(f)))
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp
