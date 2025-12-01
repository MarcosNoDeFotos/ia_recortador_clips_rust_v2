import os
import json
import subprocess
import threading
import sqlite3
from flask import Flask, jsonify, render_template, request, Blueprint

app = Blueprint("generar_clases", __name__, template_folder="templates")

# === CONFIGURACIÓN DE RUTAS ===
CURRENT_PATH = os.path.dirname(__file__).replace("\\", "/") + "/"
DB_PATH = CURRENT_PATH + "../database.db"
DATASET_DIR = os.path.join(CURRENT_PATH, "../dataset/")
COMPLETED_FILE = os.path.join(CURRENT_PATH, "videos_completados.json")

os.makedirs(DATASET_DIR, exist_ok=True)

# === VARIABLES GLOBALES ===
progress_data = {
    "status": "idle",
    "progress": 0,
    "total": 0,
    "message": "",
    "created_classes": {}
}

# === ENDPOINT PRINCIPAL (HTML) ===
@app.route("/")
def generar_clases_index():
    return render_template("generar_clases_index.html")


# === LISTAR VÍDEOS Y SUS ANOTACIONES ===
@app.route("/listar_videos")
def listar_videos():
    videos = []
   
    with sqlite3.connect(DB_PATH) as db:
        rows = db.execute("SELECT video_name, count(*) as cantidad from anotaciones group by video_name").fetchall()
        for row in rows: 
            videos.append({
                "video": row[0],
                "annotations": row[1]
            })
    return jsonify(videos)


# === GENERAR CLASES ===
@app.route("/generar_clases", methods=["POST"])
def generar_clases():
    def proceso_generacion():
        progress_data.update({
            "status": "running",
            "progress": 0,
            "message": "Iniciando generación...",
            "created_classes": {}
        })

        # video_files = [f for f in os.listdir(VIDEOS_DIR) if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))]
        anotaciones_crear = []
        with sqlite3.connect(DB_PATH) as db:
            rows = db.execute("SELECT id, start, end, start_str, end_str, label, video_name from anotaciones order by video_name, start").fetchall()
            for row in rows:
                anotaciones_crear.append({
                    "id": row[0],
                    "start": row[1],
                    "end": row[2],
                    "start_str": row[3],
                    "end_str": row[4],
                    "label": row[5],
                    "video_name" : row[6]
                })
        # pendientes = [v for v in video_files if v not in completed]
        # total_videos = len(pendientes)
        progress_data["total"] = anotaciones_crear.__len__()

        # if total_videos == 0:
        #     progress_data.update({
        #         "status": "done",
        #         "message": "No hay vídeos nuevos para procesar.",
        #         "created_classes": {}
        #     })
        #     return

        processed_videos = 0  

        for i, anotacion_data in enumerate(anotaciones_crear, 1):
            try:
                video_path = anotacion_data["video_name"]
                progress_data["message"] = f"Procesando vídeo {i}/{anotaciones_crear.__len__()}: {video_path}"
                # === Si llega aquí, sí se procesa el vídeo ===
                label = anotacion_data["label"]
                start = anotacion_data["start"]
                end = anotacion_data["end"]
                start_str = anotacion_data["start_str"].replace(":", "-")
                end_str = anotacion_data["end_str"].replace(":", "-")

                if not label or start is None or end is None:
                    continue

                class_dir = os.path.join(DATASET_DIR, label)
                os.makedirs(class_dir, exist_ok=True)

                clip_name = f"{os.path.basename(video_path)}_{start_str}_{end_str}.mp4"
                clip_path = os.path.join(class_dir, clip_name)
                print("Salida: "+clip_path)
                if not os.path.exists(clip_path):
                    cmd = [
                        "ffmpeg",
                        "-accurate_seek",                     # precisión total
                        "-ss", str(start),
                        "-to", str(end),
                        "-i", video_path,                     # input después de -i para precisión
                        "-an",                                # eliminar audio
                        "-y",
                        clip_path
                    ]
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                    progress_data["created_classes"].setdefault(label, 0)
                    progress_data["created_classes"][label] += 1
                processed_videos += 1
                progress_data["progress"] = processed_videos

            except Exception as e:
                progress_data["message"] = f"Error procesando {anotacion_data}: {e}"
                print(e)

        # === Al finalizar ===
        progress_data.update({
            "status": "done",
            "progress": progress_data["total"],  # 👈 fuerza barra al 100%
            "message": f"Se han creado {len(progress_data['created_classes'])} clases.",
        })

    threading.Thread(target=proceso_generacion, daemon=True).start()
    return jsonify({"status": "started"})


# === PROGRESO ===
@app.route("/progreso")
def progreso():
    return jsonify(progress_data)
