import os
import base64
import json
import re
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from flask import Blueprint, render_template, request, jsonify, send_from_directory, url_for, make_response
import cv2
import numpy as np

app = Blueprint("generar_clips_directo", __name__, template_folder="templates")


CAPTURAR_MULTIPLES_SEGMENTOS_SIN_TERMINAR = True # Cuando está capturando en directo, permite capturar múltiples segmentos solapados sin esperar a que termine el anterior
CURRENT_PATH = os.path.dirname(__file__).replace("\\", "/") + "/"
MODEL_PATH = CURRENT_PATH + "../modelos/modelo_rust_svm.pkl"
ACTIONS_JSON = CURRENT_PATH + "../dataset/acciones.json"
OUTPUT_DIR = CURRENT_PATH + "../clips_generados/"
DB_PATH = CURRENT_PATH + "../database.db"
os.makedirs(OUTPUT_DIR, exist_ok=True)

progress_data = {"log": "Esperando inicio...", "done": False, "progress": 0}
capture_thread = None
capture_running = False
last_segment_end = None

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

def list_camera_devices():
    # Devuelve una lista de dispositivos de cámara disponibles junto con un frame de prueba de cada dispositivo
    devices = []

    # Listar dispositivos detectados por OpenCV
    for i in range(10):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                cv2.imshow('frame', frame)
                _, buffer_img= cv2.imencode('.jpg', frame)
                data = base64.b64encode(buffer_img)
                devices.append({"id": i, "name": f"Dispositivo {i}", "frame_base64": data.decode('utf-8')})
                cap.release()
    return devices

def log_progress(message, done=False, progress=None):
    global progress_data
    progress_data["log"] = message
    progress_data["done"] = done
    if progress is not None:
        progress_data["progress"] = progress

def detect_segments_live(cam_id, grabaciones_dir):
    global capture_running, last_segment_end
    model, processor, clf, classes, DEVICE, normalize = get_clip_model_and_processor()
    actions_map = get_actions_map()
    idx_classes = {cls: classes.index(cls) for cls in classes if cls in actions_map}
    cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        log_progress("No se pudo abrir la cámara seleccionada.", done=True)
        return

    log_progress("Capturando en directo...", done=False, progress=0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = int(fps)  # 1 frame por segundo
    frame_count = 0
    last_segment_end = None
    detected_segments = []
    last_segment_id = None
    last_segment_clase = None
    accuracy_mismo_segmento = []
    last_detection_time = None
    while capture_running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        frame_count += 1
        if frame_count % frame_interval != 0:
            continue  # Solo 1 fps

        # Preprocesado
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = processor(images=[rgb], return_tensors="pt").to(DEVICE)
        with threading.Lock():
            import torch
            with torch.no_grad():
                feats = model.get_image_features(**inputs)
                feats = normalize(feats.cpu().numpy())
                proba = clf.predict_proba(feats)[0]

        now = datetime.now()
        for cls, idx in idx_classes.items():
            prob = proba[idx]
            if prob > 0.25:
                videoGrabando = [f for f in os.listdir("G:/videos") if f.lower().endswith(".mp4") and f.startswith(now.strftime("%Y-%m-%d"))][-1].replace(".mp4", "")
                videoTime = datetime.strptime(videoGrabando, "%Y-%m-%d %H-%M-%S")
                elapsed = (now - videoTime).total_seconds() # El tiempo que pasa desde que empezó la grabación hasta el momento en el que se detecta la acción
                # Si estamos dentro de un segmento activo, ignorar
                # if last_segment_end and elapsed < last_segment_end:
                if not CAPTURAR_MULTIPLES_SEGMENTOS_SIN_TERMINAR and last_segment_end and now < last_segment_end:
                    break
                start = elapsed - 60 # now - timedelta(minutes=1)
                if start < 0:
                    start = 0
                end = elapsed + 120
                # print(f"🎬 Momento clave detectado: {cls} (prob={prob:.3f}) en {now.strftime('%Y-%m-%d %H:%M:%S')}, segmento {start:.1f}s - {end:.1f}s")
                last_segment_end = end
                accuracy_mismo_segmento.append(float(prob))
                # Guardar en DB
                with sqlite3.connect(DB_PATH) as db:
                    mediaAccuracy = sum(accuracy_mismo_segmento)/len(accuracy_mismo_segmento)
                    if last_segment_clase == cls and last_segment_id is not None and last_detection_time and (now - last_detection_time).total_seconds() < 10:
                        last_detection_time = datetime.now()
                        # Actualizar el segmento existente
                        db.execute(
                            "UPDATE clips SET end = ?, accuracy = ? WHERE id = ?",
                            (
                                end,
                                float(mediaAccuracy),
                                last_segment_id
                            )
                        )
                        break
                    else:
                        accuracy_mismo_segmento.clear()
                        accuracy_mismo_segmento.append(float(prob))
                        stat = db.execute(
                            "INSERT INTO clips(video_name, start, end, fecha_generacion, accuracy, prompt) VALUES (?, ?, ?, ?, ?, ?) returning id",
                            (
                                "directo",  # video_name
                                start,
                                end,
                                now.strftime("%Y-%m-%d %H:%M:%S"),
                                float(mediaAccuracy),
                                cls
                            )
                        )
                        last_segment_id = stat.fetchone()[0]
                    db.commit()
                last_detection_time = datetime.now()
                last_segment_clase = cls
                log_progress(f"Momento clave detectado: {cls} ({now.strftime('%Y-%m-%d %H:%M:%S')})", done=False)
                break
        time.sleep(0.1)
    cap.release()
    log_progress("Captura detenida.", done=True)

@app.route('/')
def index():
    return render_template('generar_clips_directo_index.html')

@app.route('/listar_camaras')
def listar_camaras():
    return jsonify(list_camera_devices())

@app.route('/iniciar_captura', methods=['POST'])
def iniciar_captura():
    global capture_thread, capture_running
    data = request.get_json()
    cam_id = int(data.get("cam_id", 0))
    grabaciones_dir = data.get("grabaciones_dir", "G:/videos")
    if capture_running:
        return jsonify({"success": False, "error": "Ya está en marcha"})
    capture_running = True
    capture_thread = threading.Thread(target=detect_segments_live, args=(cam_id, grabaciones_dir), daemon=True)
    capture_thread.start()
    return jsonify({"success": True})

@app.route('/parar_captura', methods=['POST'])
def parar_captura():
    global capture_running
    capture_running = False
    return jsonify({"success": True})

@app.route('/progreso')
def progreso():
    return make_response(jsonify(progress_data))



@app.route('/segmentos_detectados')
def segmentos_detectados():
    # Agrupa por fecha y ordena por fecha de detección ascendente
    with sqlite3.connect(DB_PATH) as db:
        rows = db.execute(
            "SELECT id, start, end, accuracy, prompt, fecha_generacion FROM clips WHERE video_name = ? ORDER BY start DESC",
            ("directo",)
        ).fetchall()
    grupos = {}
    for row in rows:
        id_, start, end, accuracy, prompt, fecha_generacion = row
        match = re.match(r"[0-9]+[-][0-9]+[-][0-9]+\s?[0-9]+[:]", fecha_generacion)
        fecha_str = match.group(0)+"00:00"
        if fecha_str not in grupos:
            grupos[fecha_str] = []
        grupos[fecha_str].append({
            "id": id_,
            "start": start,
            "end": end,
            "accuracy": accuracy,
            "clase": prompt,
            "fecha": fecha_str
        })
    # Devuelve [{fecha, segmentos:[...]}]
    result = []
    for fecha, segs in sorted(grupos.items()):
        result.append({"fecha": fecha, "segmentos": segs})
    return jsonify(result)

@app.route('/eliminar_segmento', methods=['POST'])
def eliminar_segmento():
    data = request.get_json()
    seg_id = data.get("id")
    with sqlite3.connect(DB_PATH) as db:
        db.execute("DELETE FROM clips WHERE id = ?", (seg_id,))
        db.commit()
    return jsonify({"success": True})

@app.route('/video_original')
def video_original():
    # Busca el último vídeo grabado en la ruta de grabaciones
    grabaciones_dir = request.args.get("grabaciones_dir", "G:/videos")
    fecha = request.args.get("fecha")
    if not fecha:
        return "No fecha", 400
    # Busca archivos con formato "YYYY-MM-DD *.mp4"
    date_prefix = fecha.split(" ")[0]
    files = [f for f in os.listdir(grabaciones_dir) if f.startswith(date_prefix) and f.lower().endswith(".mp4")]
    print(files)
    if not files:
        return jsonify({"success": False, "error": "No hay grabación disponible"}), 404
    files.sort()
    video_file = files[-1]
    path = os.path.join(grabaciones_dir, video_file)
    if not os.path.exists(path):
        return jsonify({"success": False, "error": "El archivo de vídeo no está disponible"}), 404
    return send_from_directory(grabaciones_dir, video_file)
