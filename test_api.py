import requests

url = "http://127.0.0.1:5000/recognize"

files = {
    'image': open('test_images/omkar_test.jpg', 'rb')
}

response = requests.post(url, files=files)
print(response.json())
