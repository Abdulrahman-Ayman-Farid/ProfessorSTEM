import pytest
import os
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False
    with app.test_client() as client:
        yield client

@pytest.fixture
def runner():
    app.config['TESTING'] = True
    with app.test_request_context():
        yield app.test_cli_runner()
