from flask import Flask
from flask import request, jsonify
from flask_json_schema import JsonSchema, JsonValidationError
from ..Conf.container import Controllers
from .validation import check_movements_schema


app = Flask(__name__)
schema = JsonSchema(app)
app_controller = Controllers.app()


@app.route('/healthcheck')
def health_check():
    return app_controller.healthcheck()


@app.route('/classification/movements', methods=['POST'])
@schema.validate(check_movements_schema)
def check_movements():
    result = app_controller.check_movements(request.get_json())
    return jsonify(result)

@app.errorhandler(404)
def error_handler(error):
    return '404 Not Found', 404


@app.errorhandler(JsonValidationError)
def validation_error(e):
    return jsonify({'error': e.message, 'errors': [error.message for error in e.errors]}), 400


