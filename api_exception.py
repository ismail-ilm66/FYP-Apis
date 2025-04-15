from werkzeug.exceptions import HTTPException

class APIException(HTTPException):
    def __init__(self, message, status_code, payload=None):
        super().__init__()
        self.message = message
        self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['error'] = {
            'message': self.message,
            'code': self.status_code
        }
        return rv

class BadRequest(APIException):
    def __init__(self, message="Invalid request"):
        super().__init__(message, 400)

class InternalServerError(APIException):
    def __init__(self, message="Internal server error"):
        super().__init__(message, 500)