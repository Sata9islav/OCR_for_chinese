import os
import requests


def test_status_app():
    response = requests.get("http://0.0.0.0:8001/")
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'


def test_method_get_for_recognition():
    response = requests.get("http://0.0.0.0:8001/recognition")
    assert response.json()['message'] == "Please use POST request to submit the file."


def test_method_post_for_recognition():
    cwd = os.getcwd()
    path_to_imgs = os.path.join(cwd, 'images')
    test_files = [f for f in os.listdir(path_to_imgs) if f.endswith('.png')]
    for test_file in test_files:
        file_path = os.path.join(path_to_imgs, test_file)
        file = {'file': open(file_path, 'rb')}
        response = requests.post("http://0.0.0.0:8001/recognition", files=file)
        assert response.status_code == 200
        assert 'rows' in response.json()
        assert isinstance(response.json()['rows'][0], str)
