from flask import Flask, render_template
from flaskwebgui import FlaskUI # import FlaskUI


app = Flask(__name__)
ui = FlaskUI(app, width=500, height=500) # add app and parameters

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    #app.run()
    ui.run()