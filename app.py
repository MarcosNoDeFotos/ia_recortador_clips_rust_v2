from flask import Flask, render_template
import os
import json
if __name__ == "__main__": 
    print("Cargando apps...")
    from generador_anotaciones.generar_anotaciones import app as anotacionesApp
    from generar_clases.generar_clases import app as generarClasesApp
    from generar_modelo.generar_modelo_svm import app as generarModeloSVMApp
    from generar_clips_prompt.generar_clips import app as generarClipsApp
    print("Apps cargadas")

CURRENT_PATH = os.path.dirname(__file__).replace("\\", "/") + "/"

def create_app():
    app = Flask(__name__)

    # Registrar los blueprints de cada módulo
    app.register_blueprint(anotacionesApp, url_prefix="/anotaciones")
    app.register_blueprint(generarClasesApp, url_prefix="/generar_clases")
    app.register_blueprint(generarModeloSVMApp, url_prefix="/generar_modelo")
    app.register_blueprint(generarClipsApp, url_prefix="/generar_clips")

    @app.route("/")
    def index():
        return render_template("index.html")


    return app

if __name__ == "__main__":
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