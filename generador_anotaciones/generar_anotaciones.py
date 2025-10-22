import os
import json
import time
import cv2
from flask import Flask, render_template, request, jsonify, make_response, send_file, Blueprint
from multiprocessing import Process, Queue
import tkinter as tk
from tkinter import filedialog
from collections import Counter
import shutil


app = Blueprint("generador_anotaciones", __name__, template_folder="templates", static_folder="static", static_url_path="/anotaciones/static")

videoPath = None
SEGMENT_DURATION = 10
CURRENT_PATH = os.path.dirname(__file__).replace("\\", "/") + "/"
OUTPUT_DIR = CURRENT_PATH+"../anotaciones"
VIDEOS_DIR = CURRENT_PATH + "../videos_para_anotar/"
DATASET_DIR = CURRENT_PATH + "../dataset/"
os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def seleccionar_video(queue):
    root = tk.Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(
        title="Seleccionar vídeo",
        filetypes=[("Archivos de vídeo", "*.mp4;*.mov;*.avi;*.mkv")]
    )
    queue.put(video_path)


def sec_to_hhmmss(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def get_segments(video_path, segment_duration):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("No se pudo abrir el vídeo")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_length = total_frames / fps

    segments = []
    start_time = 0.0
    while start_time < video_length:
        end_time = min(start_time + segment_duration, video_length)
        segments.append({
            "start": start_time,
            "end": end_time,
            "start_str": sec_to_hhmmss(start_time),
            "end_str": sec_to_hhmmss(end_time)
        })
        start_time += segment_duration

    cap.release()
    return segments

def get_most_common_labels(max_labels=15):
    labels = []

    # 1️⃣ Extraer labels de los JSON de anotaciones
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith("_annotations.json"):
            try:
                with open(os.path.join(OUTPUT_DIR, f), "r", encoding="utf-8") as jf:
                    data = json.load(jf)
                    for a in data:
                        if isinstance(a, dict) and "label" in a and a["label"]:
                            labels.append(a["label"])
            except Exception:
                pass

    # Contar las labels más comunes
    counter = Counter(labels)
    most_common = [label for label, _ in counter.most_common(max_labels)]

    # 2️⃣ Añadir nombres de clases que hay en la carpeta dataset/
    if os.path.exists(DATASET_DIR):
        for cls in os.listdir(DATASET_DIR):
            if os.path.isdir(os.path.join(DATASET_DIR, cls)):
                most_common.append(cls)

    # 3️⃣ Eliminar duplicados manteniendo el orden
    seen = set()
    final_labels = []
    for label in most_common:
        if label not in seen:
            final_labels.append(label)
            seen.add(label)

    return final_labels



@app.route('/abrir_video', methods=['GET'])
def abrir_video():
    global videoPath
    queue = Queue()
    p = Process(target=seleccionar_video, args=(queue,))
    p.start()
    p.join()
    selected_path = queue.get()

    if not selected_path:
        return jsonify({"error": "No se seleccionó ningún archivo"}), 400

    videoPath = selected_path
    base_name = os.path.basename(videoPath)
    annotation_file = os.path.join(OUTPUT_DIR, f"{base_name}_annotations.json")

    annotations = []
    if os.path.exists(annotation_file):
        try:
            with open(annotation_file, "r", encoding="utf-8") as f:
                annotations = json.load(f)
        except Exception as e:
            print(f"⚠️ Error al leer anotaciones previas: {e}")

    tags = get_most_common_labels()

    return jsonify({
        "video_url": f"/anotaciones/video?ts={int(time.time())}",
        "annotations": annotations,
        "tags": tags
    })


@app.route('/video')
def video_file():
    video_name = request.args.get("video")
    if not video_name:
        return "No hay vídeo indicado", 404
    video_path = os.path.join(VIDEOS_DIR, video_name)
    if not os.path.exists(video_path):
        return "Vídeo no encontrado", 404

    response = make_response(send_file(video_path))
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response



@app.route('/get_segments', methods=['POST'])
def get_segments_route():
    global videoPath
    if not videoPath:
        return jsonify({"error": "No hay vídeo abierto"}), 400

    data = request.json
    segment_duration = float(data.get("segment_duration", SEGMENT_DURATION))
    segments = get_segments(videoPath, segment_duration)
    return jsonify(segments)


@app.route("/save_annotations", methods=["POST"])
def save_annotations():
    global videoPath
    if not videoPath:
        return jsonify({"error": "No hay vídeo abierto"}), 400

    data = request.json
    annotations = data.get("annotations", [])
    annotations = [a for a in annotations if a.get("label")]

    base_name = os.path.basename(videoPath)
    output_file = os.path.join(OUTPUT_DIR, f"{base_name}_annotations.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)

    # Devolver etiquetas actualizadas
    updated_tags = get_most_common_labels()

    return jsonify({"status": "ok", "saved_to": output_file, "tags": updated_tags})


@app.route("/eliminar", methods=["POST"])
def eliminar_anotacion():
    data = request.get_json()
    video_file = data.get("video")
    index = data.get("index")

    annotations_path = os.path.join(OUTPUT_DIR, f"{video_file}_annotations.json")
    if not os.path.exists(annotations_path):
        return jsonify({"status": "error", "message": "Archivo no encontrado"})

    try:
        with open(annotations_path, "r", encoding="utf-8") as f:
            annotations = json.load(f)

        if 0 <= index < len(annotations):
            annotations.pop(index)
            with open(annotations_path, "w", encoding="utf-8") as f:
                json.dump(annotations, f, indent=2, ensure_ascii=False)
            return jsonify({"status": "ok"})
        else:
            return jsonify({"status": "error", "message": "Índice fuera de rango"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})




@app.route("/listar_videos")
def listar_videos():
    videos = [f for f in os.listdir(VIDEOS_DIR) if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))]
    return jsonify({"videos": videos})

@app.route("/abrir_video_modal")
def abrir_video_modal():
    global videoPath
    video_name = request.args.get("video")
    if not video_name:
        return jsonify({"error": "No se indicó ningún vídeo"}), 400
    video_path = os.path.join(VIDEOS_DIR, video_name)
    if not os.path.exists(video_path):
        return jsonify({"error": "Vídeo no encontrado"}), 404
    videoPath = video_path

    base_name = os.path.basename(videoPath)
    annotation_file = os.path.join(OUTPUT_DIR, f"{base_name}_annotations.json")
    annotations = []
    if os.path.exists(annotation_file):
        try:
            with open(annotation_file, "r", encoding="utf-8") as f:
                annotations = json.load(f)
        except:
            pass
    tags = get_most_common_labels()
    return jsonify({"video_url": f"/anotaciones/video?video={video_name}&ts={int(time.time())}", "annotations": annotations, "tags": tags})

@app.route("/get_annotation_count")
def get_annotation_count():
    video_name = request.args.get("video")
    if not video_name:
        return jsonify({"count": 0})

    annotation_file = os.path.join(OUTPUT_DIR, f"{video_name}_annotations.json")
    if os.path.exists(annotation_file):
        try:
            with open(annotation_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return jsonify({"count": len(data)})
        except:
            return jsonify({"count": 0})
    return jsonify({"count": 0})


@app.route("/agregar_video")
def agregar_video():
    # Abrir Tkinter para seleccionar vídeo
    queue = Queue()
    p = Process(target=seleccionar_video, args=(queue,))
    p.start()
    p.join()
    selected_path = queue.get()
    if not selected_path:
        return jsonify({"success": False, "error": "No se seleccionó ningún vídeo"})

    # Mover el vídeo a la carpeta
    dest = os.path.join(VIDEOS_DIR, os.path.basename(selected_path))
    shutil.move(selected_path, dest)
    return jsonify({"success": True})



@app.route('/')
def index():
    return render_template('anotaciones_index.html')


# if __name__ == "__main__":
#     app.run(debug=True)
