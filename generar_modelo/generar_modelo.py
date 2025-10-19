import os
import cv2
import json
import numpy as np
from tqdm import tqdm

# === CONFIGURACIÓN ===
CURRENT_PATH = os.path.dirname(__file__).replace("\\", "/") + "/"
VIDEOS_DIR = CURRENT_PATH + "videos_para_anotar/"
OUTPUT_DIR = CURRENT_PATH + "anotaciones/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEGMENT_DURATION = 10  # segundos por segmento
FRAME_SAMPLE = 1       # número de frames a mostrar por segmento (representativo)


MASK_REGIONS = [
    { # Cámara
        "x_ratio": 0.01,
        "y_ratio": 0.48,
        "width_ratio": 0.25,
        "height_ratio": 0.25
    },
    { # Salud, comida, agua
        "x_ratio": 0.844,
        "y_ratio": 0.87,
        "width_ratio": 0.5,
        "height_ratio": 0.5
    },
    { # brújula
        "x_ratio": 0.31,
        "y_ratio": 0,
        "width_ratio": 0.39,
        "height_ratio": 0.04
    },
    { # ping, fps
        "x_ratio": 0,
        "y_ratio": 0.95,
        "width_ratio": 0.1,
        "height_ratio": 0.05
    },
]



# === FUNCIONES ===
def mask_face_region(frame):
    """Aplica máscaras sobre HUD o cámara si es necesario."""
    h, w, _ = frame.shape
    for maskRegion in MASK_REGIONS:
        x1 = int(w * maskRegion["x_ratio"])
        y1 = int(h * maskRegion["y_ratio"])
        x2 = int(x1 + w * maskRegion["width_ratio"])
        y2 = int(y1 + h * maskRegion["height_ratio"])
        frame[y1:y2, x1:x2] = 0
    return frame

def extract_segments(video_path, segment_duration=SEGMENT_DURATION):
    print("Extrayendo segmentos...")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_length = total_frames / fps

    segments = []
    start_time = 0.0
    while start_time < video_length:
        end_time = min(start_time + segment_duration, video_length)
        frame_idx = int(start_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = mask_face_region(frame_rgb)
        segments.append({
            "start": start_time,
            "end": end_time,
            "frame": frame_rgb
        })
        start_time += segment_duration
    cap.release()
    return segments

def annotate_video(video_path, output_json):
    print(f"\n📹 Anotando vídeo: {os.path.basename(video_path)}")
    segments = extract_segments(video_path)
    annotations = []

    for seg in tqdm(segments, desc="Segmentos"):
        frame_bgr = cv2.cvtColor(seg["frame"], cv2.COLOR_RGB2BGR)
        cv2.imshow("Segmento", cv2.resize(frame_bgr, (640, 360)))
        print("Pulsa 's' para etiquetar este segmento...")
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        # Espera a que abras la ventana y puedas escribir la etiqueta
        label = input(f"Etiqueta para segmento {seg['start']:.1f}-{seg['end']:.1f}s: ").strip()

        annotations.append({
            "start": seg["start"],
            "end": seg["end"],
            "label": label
        })

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Anotaciones guardadas en: {output_json}")

# === MAIN ===
if __name__ == "__main__":
    video_files = [f for f in os.listdir(VIDEOS_DIR) if f.lower().endswith((".mp4", ".mov", ".avi"))]
    if not video_files:
        print("❌ No se encontraron vídeos en la carpeta videos_para_anotar/")
    for vf in video_files:
        video_path = os.path.join(VIDEOS_DIR, vf)
        output_json = os.path.join(OUTPUT_DIR, os.path.splitext(vf)[0] + "_annotations.json")
        annotate_video(video_path, output_json)
