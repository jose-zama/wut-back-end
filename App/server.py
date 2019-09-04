from flask import Flask
from flask import request, jsonify
from ..Conf.container import Controllers


app = Flask(__name__)
app_controller = Controllers.app()


@app.route('/healthcheck')
def health_check():
    return app_controller.healthcheck()


@app.route('/classification/movements', methods=['POST'])
def check_movements():
    # TODO: Validate input
    result = app_controller.check_movements(request.get_json())
    return jsonify(result)


@app.errorhandler(404)
def error_handler(error):
    return '404 Not Found', 404

