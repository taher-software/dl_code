from flask import Flask
from pathlib import Path
import sys

service_path = Path(__file__).parent.parent
sys.path.append(str(service_path))
from src.config import Config
from flask_sqlalchemy import SQLAlchemy
db = SQLAlchemy()
from flask_mail import Mail


app = Flask(__name__)
app.config.from_object(Config)

db.init_app(app)

    

mail = Mail(app)


import src.migrations_models

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=Config.PORT, debug=Config.DEBUG == '1')