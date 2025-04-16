
class APIException(Exception):
    def __init__(self, message, status_code):
        self.message = message
        self.status_code = status_code

    def to_dict(self):
        return {
            'error': {
                'message': self.message,
                'code': self.status_code
            }
        }
class BadRequest(APIException):
    def __init__(self, message="Invalid request"):
        super().__init__(message, 400)

class InternalServerError(APIException):
    def __init__(self, message="Internal server error"):
        super().__init__(message, 500)