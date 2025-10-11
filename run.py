import os
from flask import Flask
from app.routes import routes
from app.models import AppConfig
from prometheus_flask_exporter import PrometheusMetrics

template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app', 'templates')

app = Flask(__name__, template_folder=template_dir)
app.secret_key = AppConfig().flask_secret_key
app.register_blueprint(routes)

metrics = PrometheusMetrics(app, path='/metrics')
metrics.info('app_info', 'Application info', version='1.0.0', app_name='Net4CleanAir Literature Review Explorer')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
