import logging

# Configure logging
logging.basicConfig(level=logging.ERROR, filename='api_errors.log')

@app.errorhandler(APIException)
def handle_api_exception(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response

@app.errorhandler(Exception)
def handle_generic_exception(error):
    logging.error(f"Unexpected error: {str(error)}", exc_info=True)
    response = jsonify({
        'error': {
            'message': 'An unexpected error occurred',
            'code': 500
        }
    })
    response.status_code = 500
    return response