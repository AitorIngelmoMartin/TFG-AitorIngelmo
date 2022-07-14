from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from flaskwebgui import FlaskUI # import FlaskUI
import os
from funciones import limpiar_directorio
app = Flask(__name__)

directorio_almacenamiento_datos = "./static/uploads"
app.config["UPLOAD_FOLDER"] = "./static/uploads"

limpiar_directorio(directorio_almacenamiento_datos)

#Defino la ruda upload, que solo admite método POST
@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        archivo_subido = request.files["archivo_subido"] #El argumento es el nombre del input en HTML
        nombre_archivo = secure_filename(archivo_subido.filename)
        archivo_subido.save(os.path.join(app.config['UPLOAD_FOLDER'], nombre_archivo))
        return "Archivo subido con éxito"

@app.route('/')
def index():
    return render_template('index.html')


ui = FlaskUI(app, width=500, height=500) # add app and parameters
if __name__ == '__main__':
    app.run(debug=True)
    #ui.run()
    