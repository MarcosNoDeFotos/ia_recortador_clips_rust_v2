import os
import json
import re
import cv2
import torch
import numpy as np
from tqdm import tqdm
from flask import Blueprint, render_template, request, jsonify, send_from_directory
from sklearn.preprocessing import normalize
from transformers import CLIPProcessor, CLIPModel
import joblib
import subprocess
from multiprocessing import Pool, cpu_count, Process, Queue
import tkinter as tk
from tkinter import filedialog

# === CONFIG ===
app = Blueprint("generar_clips", __name__, template_folder="templates")

CURRENT_PATH = os.path.dirname(__file__).replace("\\", "/") + "/"
MODEL_PATH = CURRENT_PATH + "../modelos/modelo_rust_svm.pkl"
ACTIONS_JSON = CURRENT_PATH + "../dataset/acciones.json"
VIDEOS_INPUT_DIR = CURRENT_PATH + "../videos_entrada/"
OUTPUT_DIR = CURRENT_PATH + "../clips_generados/"
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

def mask_face_region(frame):
    h, w, _ = frame.shape
    for m in MASK_REGIONS:
        x1 = int(w * m["x_ratio"])
        y1 = int(h * m["y_ratio"])
        x2 = int(x1 + w * m["width_ratio"])
        y2 = int(y1 + h * m["height_ratio"])
        frame[y1:y2, x1:x2] = 0
    return frame

# === FRAMES & FEATURES ===
def extract_frames(video_path, fps=1):
    name = os.path.splitext(os.path.basename(video_path))[0]
    cache_dir = os.path.join(FRAMES_CACHE_DIR, name)
    os.makedirs(cache_dir, exist_ok=True)
    cached = sorted([f for f in os.listdir(cache_dir) if f.endswith(".npy")])
    if cached:
        frames, times = [], []
        for f in cached:
            data = np.load(os.path.join(cache_dir, f), allow_pickle=True)
            frames.append(data[0])
            times.append(data[1])
        return frames, times
    cap = cv2.VideoCapture(video_path)
    frames, times = [], []
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(fps_video // fps))
    for i in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = mask_face_region(frame)
        frames.append(frame)
        t = i / fps_video
        times.append(t)
        np.save(os.path.join(cache_dir, f"{i:08d}.npy"), np.array([frame, t], dtype=object))
    cap.release()
    return frames, times

def get_frame_embedding(frame):
    inputs = processor(images=[frame], return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
    feats = normalize(feats.cpu().numpy())
    return feats[0]

def get_embeddings_parallel(frames):
    with Pool(processes=min(cpu_count(), 4)) as pool:
        embeddings = list(pool.imap(get_frame_embedding, frames))
    return np.array(embeddings)

def parse_duration_from_prompt(prompt):
    prompt = prompt.lower()
    m = re.search(r"(\d+(?:[\.,]\d+)?)\s*(min|minute|minuto|seg|segundo|s)", prompt)
    if not m: return 60
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

def generar_clips(video_path, prompt, threshold=0.25):
    clip_duration = parse_duration_from_prompt(prompt)
    cls, score, phrase = find_most_similar_class(prompt)
    if not cls: return []
    frames, timestamps = extract_frames(video_path, fps=2)
    embeddings = get_embeddings_parallel(frames)
    probs = clf.predict_proba(embeddings)
    idx_class = classes.index(cls)
    target_probs = probs[:, idx_class]

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

    clips = []
    for i, (start, end, avg) in enumerate(segments):
        mid = (start + end) / 2
        clip_start = max(0, mid - clip_duration / 2)
        name = f"{os.path.splitext(os.path.basename(video_path))[0]}_{cls}_{i+1}_{round(avg,3)}.mp4"
        output_path = os.path.join(OUTPUT_DIR, name)
        cmd = ["ffmpeg", "-y", "-ss", str(clip_start), "-i", video_path, "-t", str(clip_duration), "-map", "0", "-c", "copy", output_path]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        clips.append(name)
    return clips

# === TK para añadir vídeos ===
def seleccionar_videos(queue):
    root = tk.Tk()
    root.withdraw()
    paths = filedialog.askopenfilenames(title="Seleccionar vídeos", filetypes=[("Vídeos", "*.mp4;*.mov;*.avi;*.mkv")])
    queue.put(paths)

# === RUTAS WEB ===
@app.route('/')
def index():
    return render_template('generar_clips_index.html')

@app.route('/listar_videos')
def listar_videos():
    vids = [f for f in os.listdir(VIDEOS_INPUT_DIR) if f.lower().endswith((".mp4",".mov",".avi",".mkv"))]
    return jsonify(vids)

@app.route('/listar_clips')
def listar_clips():
    clips = []
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith(".mp4"):
            clips.append({"nombre": f, "prompt": "Desconocido"})  # puedes mejorar esto guardando prompts
    return jsonify(clips)

@app.route('/video/<path:filename>')
def video_file(filename):
    return send_from_directory(OUTPUT_DIR, filename)

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

@app.route('/generar', methods=['POST'])
def generar():
    data = request.get_json()
    prompt = data.get("prompt", "")
    videos = data.get("videos", [])
    all_clips = []
    for v in videos:
        vpath = os.path.join(VIDEOS_INPUT_DIR, v)
        if os.path.exists(vpath):
            clips = generar_clips(vpath, prompt)
            all_clips.extend(clips)
    return jsonify({"success": True, "clips": all_clips})
