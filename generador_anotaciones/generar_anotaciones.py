import os
import json
import time
import sqlite3
from flask import render_template, request, jsonify, make_response, send_file, Blueprint
from multiprocessing import Process, Queue
import tkinter as tk
from tkinter import filedialog
from collections import Counter
import shutil


app = Blueprint("generador_anotaciones", __name__, template_folder="templates", static_folder="static", static_url_path="/anotaciones/static")

videoPath = None
SEGMENT_DURATION = 10
CURRENT_PATH = os.path.dirname(__file__).replace("\\", "/") + "/"
# OUTPUT_DIR = CURRENT_PATH+"../anotaciones"
DB_PATH = CURRENT_PATH + "../database.db"
DATASET_DIR = CURRENT_PATH + "../dataset/"


videos_origen_dir = CURRENT_PATH + "../videos_para_anotar/"
os.makedirs(videos_origen_dir, exist_ok=True)
# os.makedirs(OUTPUT_DIR, exist_ok=True)


def seleccionar_video(queue):
    root = tk.Tk()
    root.withdraw()
    video_paths = filedialog.askopenfilenames(
        title="Seleccionar vídeo",
        filetypes=[("Archivos de vídeo", "*.mp4;*.mov;*.avi;*.mkv")]
    )
    queue.put(video_paths)

# === 🎬 Selección de vídeo con Tkinter ===
def trg_seleccionar_ruta_videos(queue):
    root = tk.Tk()
    root.withdraw()
    selected = filedialog.askdirectory(
        title="Selecciona una ruta para generar anotaciones"
    )
    root.destroy()
    queue.put(selected)


def sec_to_hhmmss(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def get_most_common_labels(max_labels=50):
    labels = []
    # Obtener labels de las anotaciones generadas en la db
    with sqlite3.connect(DB_PATH) as db:
        rows = db.execute("SELECT DISTINCT label FROM anotaciones").fetchall()
        for row in rows:
            labels.append(row[0])

    # 1️⃣ Extraer labels de los JSON de anotaciones
    # for f in os.listdir(OUTPUT_DIR):
    #     if f.endswith("_annotations.json"):
    #         try:
    #             with open(os.path.join(OUTPUT_DIR, f), "r", encoding="utf-8") as jf:
    #                 data = json.load(jf)
    #                 for a in data:
    #                     if isinstance(a, dict) and "label" in a and a["label"]:
    #                         labels.append(a["label"])
    #         except Exception:
    #             pass
    
    # Contar las labels más comunes
    # counter = Counter(labels)
    # most_common = [label for label, _ in counter.most_common(max_labels)]

    # 2️⃣ Añadir nombres de clases que hay en la carpeta dataset/
    if os.path.exists(DATASET_DIR):
        for cls in os.listdir(DATASET_DIR):
            if os.path.isdir(os.path.join(DATASET_DIR, cls)) and cls not in labels:
                labels.append(cls)
    labels.sort()
    # 3️⃣ Eliminar duplicados manteniendo el orden
    # seen = set()
    # final_labels = []
    # for label in most_common:
    #     if label not in seen:
    #         final_labels.append(label)
    #         seen.add(label)

    return labels

def get_anotaciones_video(videoName):
    anotaciones = [] 
    with sqlite3.connect(DB_PATH) as db:
        video_path = os.path.join(videos_origen_dir, videoName)
        rows = db.execute("SELECT id, start, end, start_str, end_str, label from anotaciones where video_name = ? order by start", (video_path,)).fetchall()
        for row in rows:
            anotaciones.append({
                "id": row[0],
                "start": row[1],
                "end": row[2],
                "start_str": row[3],
                "end_str": row[4],
                "label": row[5],
            })
    return anotaciones
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
    # base_name = os.path.basename(videoPath)
    annotations = get_anotaciones_video(videoPath)
    # annotation_file = os.path.join(OUTPUT_DIR, f"{base_name}_annotations.json")

    # if os.path.exists(annotation_file):
    #     try:
    #         with open(annotation_file, "r", encoding="utf-8") as f:
    #             annotations = json.load(f)
    #     except Exception as e:
    #         print(f"⚠️ Error al leer anotaciones previas: {e}")
    


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
    video_path = os.path.join(videos_origen_dir, video_name)
    if not os.path.exists(video_path):
        return "Vídeo no encontrado", 404

    response = make_response(send_file(video_path))
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response




@app.route("/save_annotations", methods=["POST"])
def save_annotations():
    global videoPath
    if not videoPath:
        return jsonify({"error": "No hay vídeo abierto"}), 400

    data = request.json
    annotations = data.get("annotations", [])
    annotations = [a for a in annotations if a.get("label")]

    # base_name = os.path.basename(videoPath)
    # output_file = os.path.join(OUTPUT_DIR, f"{base_name}_annotations.json")

    # with open(output_file, "w", encoding="utf-8") as f:
    #     json.dump(annotations, f, indent=2, ensure_ascii=False)

    video_full_path = os.path.join(videos_origen_dir, videoPath)
    with sqlite3.connect(DB_PATH) as db:
        for anotacion in annotations:
            db.execute("INSERT INTO anotaciones(video_name, start, end, start_str, end_str, label) values (?,?,?,?,?,?)", (video_full_path, anotacion["start"], anotacion["end"], anotacion["start_str"], anotacion["end_str"], anotacion["label"]))
        db.commit()
    # Devolver etiquetas actualizadas
    updated_tags = get_most_common_labels()

    return jsonify({"status": "ok", "tags": updated_tags})


@app.route("/eliminar_anotacion", methods=["POST"])
def eliminar_anotacion():
    data = request.get_json()
    video_file = data.get("video")
    index = data.get("index")
    video_full_path = os.path.join(videos_origen_dir, video_file)
    with sqlite3.connect(DB_PATH) as db:
        db.execute("DELETE FROM anotaciones WHERE video_name = ? and id = ?", (video_full_path, index))
        db.commit()
    
    annotations = get_anotaciones_video(video_file)
    return jsonify({"status": "ok", "annotations": annotations})

    # annotations_path = os.path.join(OUTPUT_DIR, f"{video_file}_annotations.json")
    # if not os.path.exists(annotations_path):
    #     return jsonify({"status": "error", "message": "Archivo no encontrado"})

    # try:
    #     with open(annotations_path, "r", encoding="utf-8") as f:
    #         annotations = json.load(f)

    #     if 0 <= index < len(annotations):
    #         annotations.pop(index)
    #         with open(annotations_path, "w", encoding="utf-8") as f:
    #             json.dump(annotations, f, indent=2, ensure_ascii=False)
    #         return jsonify({"status": "ok", "annotations": annotations})
    #     else:
    #         return jsonify({"status": "error", "message": "Índice fuera de rango"})
    # except Exception as e:
    #     return jsonify({"status": "error", "message": str(e)})




@app.route("/listar_videos")
def listar_videos():
    videos = [f for f in os.listdir(videos_origen_dir) if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))]
    return jsonify({"videos": videos})

@app.route("/abrir_video_modal")
def abrir_video_modal():
    global videoPath
    video_name = request.args.get("video")
    if not video_name:
        return jsonify({"error": "No se indicó ningún vídeo"}), 400
    video_path = os.path.join(videos_origen_dir, video_name)
    if not os.path.exists(video_path):
        return jsonify({"error": "Vídeo no encontrado"}), 404
    videoPath = video_path

    # base_name = os.path.basename(videoPath)
    # annotation_file = os.path.join(OUTPUT_DIR, f"{base_name}_annotations.json")
    # annotations = []
    # if os.path.exists(annotation_file):
    #     try:
    #         with open(annotation_file, "r", encoding="utf-8") as f:
    #             annotations = json.load(f)
    #     except:
    #         pass
    annotations = get_anotaciones_video(videoPath)
    tags = get_most_common_labels()
    return jsonify({"video_url": f"/anotaciones/video?video={video_name}&ts={int(time.time())}", "annotations": annotations, "tags": tags})

@app.route("/get_annotation_count")
def get_annotation_count():
    video_name = request.args.get("video")
    if not video_name:
        return jsonify({"count": 0})
    video_name = os.path.join(videos_origen_dir, video_name)
    # annotation_file = os.path.join(OUTPUT_DIR, f"{video_name}_annotations.json")
    # if os.path.exists(annotation_file):
    #     try:
    #         with open(annotation_file, "r", encoding="utf-8") as f:
    #             data = json.load(f)
    #             return jsonify({"count": len(data)})
    #     except:
    #         return jsonify({"count": 0})
    annotations = 0
    with sqlite3.connect(DB_PATH) as db:
        anotacionesDB = db.execute("SELECT count(*) from anotaciones where video_name = ?", (video_name,)).fetchone()
        if anotacionesDB and anotacionesDB.__len__() != 0:
            annotations = int(anotacionesDB[0])
    return jsonify({"count": annotations})


@app.route("/seleccionar_ruta_videos")
def seleccionar_ruta_videos():
    # Abrir Tkinter para seleccionar vídeo
    queue = Queue()
    p = Process(target=trg_seleccionar_ruta_videos, args=(queue,))
    p.start()
    p.join()
    selected_path = queue.get()
    if not selected_path:
        return jsonify({"success": False, "error": "No se seleccionó ningún vídeo"})
    global videos_origen_dir
    videos_origen_dir = selected_path
    return jsonify({"success": True, "path": selected_path})



@app.route('/')
def index():
    return render_template('anotaciones_index.html')


# if __name__ == "__main__":
#     app.run(debug=True)
