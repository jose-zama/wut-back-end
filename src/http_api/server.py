from flask import Flask
from flask import request, jsonify
from flask_json_schema import JsonSchema, JsonValidationError
from flask_cors import CORS
from src.ModelWrapper import predict

app = Flask(__name__)
# app.config['SERVER_NAME'] = 'ec2-3-87-106-62.compute-1.amazonaws.com:5000'
schema = JsonSchema(app)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/healthcheck')
def health_check():
    return 'OK'


@app.route('/classification/movements', methods=['POST'])
# @schema.validate(check_movements_schema)
def check_movements():
    result = predict(request.data.decode("utf-8"))
    return result


@app.errorhandler(404)
def error_handler(error):
    return '404 Not Found', 404


@app.errorhandler(JsonValidationError)
def validation_error(e):
    return jsonify({'error': e.message, 'errors': [error.message for error in e.errors]}), 400
