import pytest

def test_home_page(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'ProfessorSTEM' in response.data

def test_signup_page(client):
    response = client.get('/signup')
    assert response.status_code == 200

def test_signin_page(client):
    response = client.get('/signin')
    assert response.status_code == 200

def test_404_page(client):
    response = client.get('/nonexistent')
    assert response.status_code == 404

def test_user_training_page(client):
    response = client.get('/user_training')
    assert response.status_code == 200
