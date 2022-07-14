from flaskwebgui import FlaskUI
from app import app

FlaskUI(app, width=600, height=500).run()
