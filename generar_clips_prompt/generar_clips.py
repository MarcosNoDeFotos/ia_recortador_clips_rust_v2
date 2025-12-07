import os
import json
import re
import sqlite3
from flask import Blueprint, render_template, request, jsonify, send_from_directory, make_response, url_for
from datetime import datetime
from multiprocessing import Process, Queue
import tkinter as tk
from tkinter import filedialog
import subprocess

# === CONFIG ===
app = Blueprint("generar_clips", __name__, template_folder="templates")

CURRENT_PATH = os.path.dirname(__file__).replace("\\", "/") + "/"
MODEL_PATH = CURRENT_PATH + "../modelos/modelo_rust_svm.pkl"
ACTIONS_JSON = CURRENT_PATH + "../dataset/acciones.json"
VIDEOS_INPUT_DIR = CURRENT_PATH + "../videos_entrada/"
OUTPUT_DIR = CURRENT_PATH + "../clips_generados/"
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "progreso_generar_clips.json")
FRAMES_CACHE_DIR = os.path.join(OUTPUT_DIR, "frames_cache")
DB_PATH = CURRENT_PATH + "../database.db"
os.makedirs(VIDEOS_INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FRAMES_CACHE_DIR, exist_ok=True)

MASK_REGIONS = [
    {"x_ratio": 0.01, "y_ratio": 0.48, "width_ratio": 0.25, "height_ratio": 0.25},
    {"x_ratio": 0.844, "y_ratio": 0.87, "width_ratio": 0.5, "height_ratio": 0.5},
    {"x_ratio": 0.31, "y_ratio": 0.0, "width_ratio": 0.39, "height_ratio": 0.04},
    {"x_ratio": 0.0, "y_ratio": 0.95, "width_ratio": 0.1, "height_ratio": 0.05},
]

progress_data = {"log": "Esperando inicio...", "done": False, "progress": 0}

def log_progress(message, done=False, progress=None):
    global progress_data
    progress_data["log"] = message
    progress_data["done"] = done
    if progress is not None:
        progress_data["progress"] = progress


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


def get_actions_map_and_equivalents():
    with open(ACTIONS_JSON, "r", encoding="utf-8") as f:
        raw_actions = json.load(f)
    actions_map = {}
    equivalent_classes = {}
    for cls, info in raw_actions.items():
        if isinstance(info, dict):
            actions_map[cls] = info.get("frases", [])
            equivalent_classes[cls] = info.get("similares", [])
        else:
            actions_map[cls] = info
            equivalent_classes[cls] = []
    return actions_map, equivalent_classes


def mask_face_region(frame):
    h, w, _ = frame.shape
    for m in MASK_REGIONS:
        x1 = int(w * m["x_ratio"])
        y1 = int(h * m["y_ratio"])
        x2 = int(x1 + w * m["width_ratio"])
        y2 = int(y1 + h * m["height_ratio"])
        frame[y1:y2, x1:x2] = 0
    return frame


def get_frame_embedding(frame, processor, model, DEVICE, normalize):
    import torch
    inputs = processor(images=[frame], return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
    feats = normalize(feats.cpu().numpy())
    return feats[0]


def resolve_equivalent_class(cls, equivalent_classes):
    for main, similars in equivalent_classes.items():
        if cls == main or cls in similars:
            return main
    return cls


def parse_duration_from_prompt(prompt):
    prompt = prompt.lower()
    m = re.search(r"(\d+(?:[\.,]\d+)?)\s*(min|minute|minuto|seg|segundo|s)", prompt)
    if not m:
        return 60
    val = float(m.group(1).replace(",", "."))
    unit = m.group(2)
    return int(val * 60) if unit.startswith("min") else int(val)


def find_most_similar_class(prompt, processor, model, actions_map, min_similarity=0.25, DEVICE=None):
    import torch
    import numpy as np
    text_list, lookup = [], []
    for cls, phrases in actions_map.items():
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
    idx = int(np.argmax(sims))
    cls = lookup[idx]
    phrase = text_list[idx]
    score = float(sims[idx])
    if score < min_similarity:
        return None, score, phrase
    return cls, score, phrase


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
    for f in cached:
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


def generar_clips(video_path, prompt, threshold=0.25):
    import numpy as np
    model, processor, clf, classes, DEVICE, normalize = get_clip_model_and_processor()
    actions_map, equivalent_classes = get_actions_map_and_equivalents()
    clip_duration = parse_duration_from_prompt(prompt)
    cls, score, phrase = find_most_similar_class(prompt, processor, model, actions_map, DEVICE=DEVICE)
    cls = resolve_equivalent_class(cls, equivalent_classes)
    if not cls:
        return []

    log_progress(f"🎯 Clase detectada: {cls} (similaridad {round(score,3)})")
    log_progress(f"📽️ Analizando {os.path.basename(video_path)}")

    idx_class = classes.index(cls)
    name = os.path.splitext(os.path.basename(video_path))[0]
    cache_dir = os.path.join(FRAMES_CACHE_DIR, name)
    os.makedirs(cache_dir, exist_ok=True)

    import cv2
    cap = cv2.VideoCapture(video_path)
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    frame_count_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if fps_video and fps_video>0 else 0
    cap.release()

    step = max(1, int(fps_video // 1)) if fps_video and fps_video>0 else 1
    total_frames = max(1, frame_count_video // step) if frame_count_video>0 else 1

    use_cache = any(f.endswith(".npy") for f in os.listdir(cache_dir))
    frame_stream = read_cached_frames_streaming(cache_dir) if use_cache else extract_frames_streaming(video_path)

    timestamps, target_probs = [], []
    frame_counter = 0

    # === Análisis de frames ===
    for batch_frames, batch_times in frame_stream:
        import torch
        with torch.no_grad():
            inputs = processor(images=batch_frames, return_tensors="pt", padding=True).to(DEVICE)
            feats = model.get_image_features(**inputs)
            feats = normalize(feats.cpu().numpy())
            probs = clf.predict_proba(feats)[:, idx_class]

        timestamps.extend(batch_times)
        target_probs.extend(probs)
        frame_counter += len(batch_frames)

        percent = min(99, round((frame_counter / total_frames) * 100, 1))
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
            avg_prob = float(np.mean(segment_probs))
            if end - start >= clip_duration / 6:
                segments.append((start, end, avg_prob))
            start = None
            segment_probs = []

    if start and segment_probs:
        end = timestamps[-1]
        avg_prob = float(np.mean(segment_probs))
        if end - start >= clip_duration / 6:
            segments.append((start, end, avg_prob))

    # === Guardar segmentos detectados en DB ===
    total_segments = len(segments)
    with sqlite3.connect(DB_PATH) as db:
        for i, (start, end, avg) in enumerate(segments):
            try:
                mid = (start + end) / 2
                clip_start = max(0, mid - clip_duration / 2)
                # guardamos start como clip_start y duracion en "end" (compatible con antes)
                # Nota: aquí mantenemos "start" (segundos) y "end" (duración) como antes del diseño original
                db.execute(
                    "INSERT INTO clips(video_name, start, end, fecha_generacion, accuracy, prompt) values (?, ?, ?, ?, ?, ?)",
                    (video_path, clip_start, clip_duration, None, round(float(avg), 3), prompt)
                )
            except Exception as e:
                print("Error guardando segmento en DB:", e)
        db.commit()

    log_progress(f"✅ {total_segments} clips detectados", done=True, progress=100)
    return segments


def proceso_generar(prompt, videos):
    log_progress("🚀 Iniciando generación de clips...", progress=0)
    for idx, v in enumerate(videos):
        vpath = os.path.join(VIDEOS_INPUT_DIR, v)
        if os.path.exists(vpath):
            log_progress(f"▶️ Procesando video {idx+1}/{len(videos)}: {v}")
            generar_clips(vpath, prompt)
    log_progress("✅ Proceso completado.", done=True, progress=100)


def detectar_clase_test(prompt):
    model, processor, clf, classes, DEVICE, normalize = get_clip_model_and_processor()
    actions_map, equivalent_classes = get_actions_map_and_equivalents()
    cls, score, _ = find_most_similar_class(prompt, processor, model, actions_map, DEVICE=DEVICE)
    if not cls:
        cls = "none :("
        score = 0
    return cls, round(score, 3)


def seleccionar_videos(queue):
    root = tk.Tk()
    root.withdraw()
    paths = filedialog.askopenfilenames(title="Seleccionar vídeos", filetypes=[("Vídeos", "*.mp4;*.mov;*.avi;*.mkv")])
    queue.put(paths)


def get_clips_detectados(min_accuracy=0.0):
    clips = []
    with sqlite3.connect(DB_PATH) as db:
        rows = db.execute("SELECT id, video_name, start, end, fecha_generacion, accuracy, prompt, generated_path, generated_at from clips WHERE accuracy >= ? ORDER BY prompt, accuracy DESC", (min_accuracy,)).fetchall()
        for row in rows:
            clips.append({
                "id": row[0],
                "video_name": row[1],
                "start": row[2],
                "end": row[3],
                "fecha_generacion": row[4],
                "accuracy": row[5],
                "prompt": row[6],
                "generated_path": row[7],
                "generated_at": row[8],
            })
    return clips


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


@app.route('/video_original')
def video_original():
    """Serve original videos from videos_entrada (safe basename)"""
    video = request.args.get("video")
    if not video:
        return "No video", 400
    safe = os.path.basename(video)
    if not os.path.exists(os.path.join(VIDEOS_INPUT_DIR, safe)):
        return "Not found", 404
    return send_from_directory(VIDEOS_INPUT_DIR, safe)


@app.route('/listar_clips')
def listar_clips():
    min_acc = float(request.args.get("min_accuracy", 0.0))
    return jsonify(get_clips_detectados(min_accuracy=min_acc))


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
    # Solo lee la variable global, no el archivo
    return make_response(jsonify(progress_data))


@app.route('/generar_clip', methods=['POST'])
def generar_clip():
    """
    Genera el clip .mp4 para el segmento dado.
    POST JSON: { "clip_id": int, "start": float, "end": float }
    """
    data = request.get_json()
    clip_id = int(data.get("clip_id"))
    start = float(data.get("start"))
    end = float(data.get("end"))

    # Buscar clip en DB
    with sqlite3.connect(DB_PATH) as db:
        row = db.execute("SELECT id, video_name FROM clips WHERE id = ?", (clip_id,)).fetchone()
        if not row:
            return jsonify({"success": False, "error": "Clip no encontrado"}), 404
        video_path = row[1]

    basename_out = f"{os.path.splitext(os.path.basename(video_path))[0]}_clip_{clip_id}_{int(start)}_{int(end)}.mp4"
    output_path = os.path.join(OUTPUT_DIR, basename_out)

    # Construir comando ffmpeg
    # Para precisión y que no haya pérdida visible usamos recodificación de vídeo con crf bajo y -an (sin audio)
    # Usamos "-i input -ss START -to END" para seek preciso con recodificación.
    try:
        duration = end - start
        cmd = [
            "ffmpeg",
            "-accurate_seek",                     # precisión total
            "-ss", str(start),
            "-t", str(duration),
            "-i", video_path,                     # input después de -i para precisión
            "-map", "0", # Copia todas las pistas de vídeo y audio
            "-c", "copy", # Copia sin recodificar ni cambiar la calidad
            "-avoid_negative_ts", "1", # corrige timestamps negativos
            "-fflags", "+genpts", # recalcula los PTS (Presentation Timestamps) del vídeo
            "-reset_timestamps", "1", # fuerza que todos los streams comiencen en 0
            "-y",
            output_path
        ]

        log_progress(f"🎬 Generando clip {basename_out} ...")
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Guardar generated_path y timestamp en DB
        with sqlite3.connect(DB_PATH) as db:
            db.execute("UPDATE clips SET generated_path = ?, generated_at = ? WHERE id = ?", (basename_out, datetime.now(), clip_id))
            db.commit()

        log_progress(f"✅ Clip generado: {basename_out}", done=True, progress=100)
        # Devolver ruta relativa que puede abrirse con /generar_clips/video/<filename>
        return jsonify({"success": True, "generated": basename_out, "url": url_for("generar_clips.video_file", filename=basename_out)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/eliminar_clip', methods=['POST'])
def eliminar_clip():
    """
    Elimina un clip de la tabla clips por su ID.
    POST JSON: { "clip_id": int }
    """
    data = request.get_json()
    clip_id = int(data.get("clip_id"))
    
    try:
        with sqlite3.connect(DB_PATH) as db:
            db.execute("DELETE FROM clips WHERE id = ?", (clip_id,))
            db.commit()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
