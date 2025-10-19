from flask import Flask, render_template
from generador_anotaciones.generar_anotaciones import app as anotacionesApp
# from app2.routes import app2_bp

def create_app():
    app = Flask(__name__)

    # Registrar los blueprints de cada módulo
    app.register_blueprint(anotacionesApp, url_prefix="/anotaciones")
    # app.register_blueprint(app2_bp, url_prefix="/traduccion")

    @app.route("/")
    def index():
        return render_template("index.html")


    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)