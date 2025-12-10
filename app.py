from flask import Flask, render_template
import os
import sqlite3
import json
if __name__ == "__main__": 
    print("Cargando apps...")
    from generador_anotaciones.generar_anotaciones import app as anotacionesApp
    from generar_clases.generar_clases import app as generarClasesApp
    from generar_modelo.generar_modelo_svm import app as generarModeloSVMApp
    from generar_clips_prompt.generar_clips import app as generarClipsApp
    from generar_clips_momentos_clave.generar_clips_momentos_clave import app as generarClipsMomentosClaveApp
    from generar_clips_directo.generar_clips_directo import app as generarClipsDirectoApp
    print("Apps cargadas")

CURRENT_PATH = os.path.dirname(__file__).replace("\\", "/") + "/"


def init_db():
    with sqlite3.connect(CURRENT_PATH+"database.db") as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS anotaciones (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_name TEXT,
                    start DECIMAL,
                    end DECIMAL,
                    start_str TEXT,
                    end_str TEXT,
                    label TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS clips (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_name TEXT,
                    start DECIMAL,
                    end DECIMAL,
                    fecha_generacion TIMESTAMP,
                    accuracy DECIMAL,
                    prompt TEXT,
                    generated_path TEXT,
                    generated_at TIMESTAMP,
                    clase TEXT,
                    label TEXT
                )
            """)
            conn.commit()


def create_app():
    app = Flask(__name__)

    # Registrar los blueprints de cada módulo
    app.register_blueprint(anotacionesApp, url_prefix="/anotaciones")
    app.register_blueprint(generarClasesApp, url_prefix="/generar_clases")
    app.register_blueprint(generarModeloSVMApp, url_prefix="/generar_modelo")
    app.register_blueprint(generarClipsApp, url_prefix="/generar_clips")
    app.register_blueprint(generarClipsMomentosClaveApp, url_prefix="/generar_clips_momentos_clave")
    app.register_blueprint(generarClipsDirectoApp, url_prefix="/generar_clips_directo")

    @app.route("/")
    def index():
        return render_template("index.html")


    return app

if __name__ == "__main__":
    init_db()
    app = create_app()
    host= "127.0.0.1"
    port = 5000
    serverConfigPath =CURRENT_PATH+"serverConfig.json"
    try:
        with open(serverConfigPath, encoding="utf-8") as serverConfig:
            serverConfigData = json.loads(serverConfig.read())
            host = serverConfigData["host"]
            port = serverConfigData["port"]
    except Exception as e:
        print(f"Error al abrir el archivo de configuración del servidor en {serverConfigPath}. \n{e}")
    app.run(debug=True, use_reloader=False, host=host, port=port)