import logging
import json
import uuid
from datetime import datetime
from flask import g, request

class StructuredFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }

        # Add request context if available
        if hasattr(g, 'request_id'):
            log_data['request_id'] = g.request_id
        if hasattr(g, 'route'):
            log_data['route'] = g.route
        if hasattr(g, 'method'):
            log_data['method'] = g.method
        if hasattr(g, 'status_code'):
            log_data['status_code'] = g.status_code
        if hasattr(g, 'latency_ms'):
            log_data['latency_ms'] = g.latency_ms

        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_data)

def setup_logging(app):
    """Configure structured JSON logging for the Flask app."""
    handler = logging.StreamHandler()
    handler.setFormatter(StructuredFormatter())

    app.logger.handlers = [handler]
    app.logger.setLevel(logging.INFO)

    # Add request ID middleware
    @app.before_request
    def assign_request_id():
        g.request_id = str(uuid.uuid4())
        g.route = request.path
        g.method = request.method
        g.start_time = datetime.utcnow()

    @app.after_request
    def log_request(response):
        g.status_code = response.status_code
        g.latency_ms = (datetime.utcnow() - g.start_time).total_seconds() * 1000
        app.logger.info(f"Request completed: {g.method} {g.route} -> {g.status_code} ({g.latency_ms:.2f}ms)")
        return response
