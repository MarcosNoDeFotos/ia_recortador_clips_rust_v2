import os
import json
import time
import cv2
from flask import Flask, render_template, request, jsonify, make_response, send_file
from multiprocessing import Process, Queue
import tkinter as tk
from tkinter import filedialog
from collections import Counter

app = Flask(__name__)

videoPath = None
SEGMENT_DURATION = 10
OUTPUT_DIR = "anotaciones"
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
    counter = Counter(labels)
    most_common = [label for label, _ in counter.most_common(max_labels)]
    return most_common


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
        "video_url": f"/video?ts={int(time.time())}",
        "annotations": annotations,
        "tags": tags
    })


@app.route('/video')
def video_file():
    global videoPath
    if not videoPath:
        return "No hay vídeo seleccionado", 404

    response = make_response(send_file(videoPath))
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


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
