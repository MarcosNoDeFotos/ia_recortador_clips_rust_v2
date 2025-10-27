import os
import json
import subprocess
import threading
from flask import Flask, jsonify, render_template, request, Blueprint

app = Blueprint("generar_clases", __name__, template_folder="templates")

# === CONFIGURACIÓN DE RUTAS ===
CURRENT_PATH = os.path.dirname(__file__).replace("\\", "/") + "/"
VIDEOS_DIR = os.path.join(CURRENT_PATH, "../videos_para_anotar/")
ANNOTATIONS_DIR = os.path.join(CURRENT_PATH, "../anotaciones/")
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


# === FUNCIONES AUXILIARES ===
def load_completed_videos():
    if os.path.exists(COMPLETED_FILE):
        try:
            with open(COMPLETED_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_completed_videos(completed):
    with open(COMPLETED_FILE, "w", encoding="utf-8") as f:
        json.dump(completed, f, indent=2, ensure_ascii=False)


# === ENDPOINT PRINCIPAL (HTML) ===
@app.route("/")
def generar_clases_index():
    return render_template("generar_clases_index.html")


# === LISTAR VÍDEOS Y SUS ANOTACIONES ===
@app.route("/listar_videos")
def listar_videos():
    completed = load_completed_videos()
    videos = []
    for file in os.listdir(VIDEOS_DIR):
        if not file.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
            continue
        ann_path = os.path.join(ANNOTATIONS_DIR, f"{file}_annotations.json")
        annotations = 0
        if os.path.exists(ann_path):
            try:
                with open(ann_path, "r", encoding="utf-8") as f:
                    annotations = len(json.load(f))
            except Exception:
                pass
        videos.append({
            "video": file,
            "annotations": annotations,
            "completed": file in completed
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

        completed = load_completed_videos()
        video_files = [f for f in os.listdir(VIDEOS_DIR) if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))]
        pendientes = [v for v in video_files if v not in completed]
        total_videos = len(pendientes)
        progress_data["total"] = total_videos

        if total_videos == 0:
            progress_data.update({
                "status": "done",
                "message": "No hay vídeos nuevos para procesar.",
                "created_classes": {}
            })
            return

        processed_videos = 0  # 👈 contador de vídeos realmente procesados

        for i, video_file in enumerate(pendientes, 1):
            try:
                progress_data["message"] = f"Procesando vídeo {i}/{total_videos}: {video_file}"
                video_path = os.path.join(VIDEOS_DIR, video_file)
                annotation_path = os.path.join(ANNOTATIONS_DIR, f"{video_file}_annotations.json")

                if not os.path.exists(annotation_path):
                    progress_data["message"] = f"No hay anotaciones para {video_file}, se omite."
                    continue

                with open(annotation_path, "r", encoding="utf-8") as f:
                    annotations = json.load(f)

                if not annotations:
                    progress_data["message"] = f"Sin anotaciones en {video_file}, se omite."
                    continue

                # === Si llega aquí, sí se procesa el vídeo ===
                for ann in annotations:
                    label = ann.get("label")
                    start = ann.get("start")
                    end = ann.get("end")
                    start_str = ann.get("start_str", "").replace(":", "-")
                    end_str = ann.get("end_str", "").replace(":", "-")

                    if not label or start is None or end is None:
                        continue

                    class_dir = os.path.join(DATASET_DIR, label)
                    os.makedirs(class_dir, exist_ok=True)

                    clip_name = f"{os.path.splitext(video_file)[0]}_{start_str}_{end_str}.mp4"
                    clip_path = os.path.join(class_dir, clip_name)

                    cmd = [
                        "ffmpeg",
                        "-i", video_path,                     # input después de -i para precisión
                        "-ss", str(start),
                        "-to", str(end),
                        "-accurate_seek",                     # precisión total
                        "-an",                                # eliminar audio
                        "-c:v", "libx264",                    # codec compatible
                        "-crf", "0",                          # CRF=0 => lossless (sin pérdida)
                        "-preset", "ultrafast",               # velocidad de compresión alta (sin afectar calidad)
                        "-pix_fmt", "yuv420p",                # formato compatible y estándar
                        "-y",
                        clip_path
                    ]
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                    progress_data["created_classes"].setdefault(label, 0)
                    progress_data["created_classes"][label] += 1

                completed.append(video_file)
                save_completed_videos(completed)

                processed_videos += 1
                progress_data["progress"] = processed_videos / total_videos * 100  # 👈 porcentaje real

            except Exception as e:
                progress_data["message"] = f"Error procesando {video_file}: {e}"

        # === Al finalizar ===
        progress_data.update({
            "status": "done",
            "progress": 100,  # 👈 fuerza barra al 100%
            "message": f"Se han creado {len(progress_data['created_classes'])} clases.",
        })

    threading.Thread(target=proceso_generacion, daemon=True).start()
    return jsonify({"status": "started"})


# === PROGRESO ===
@app.route("/progreso")
def progreso():
    return jsonify(progress_data)


# === REINICIAR ESTADO ===
@app.route("/reiniciar_estado", methods=["POST"])
def reiniciar_estado():
    save_completed_videos([])
    return jsonify({"status": "ok", "message": "Estado reiniciado. Todos los vídeos podrán procesarse de nuevo."})
