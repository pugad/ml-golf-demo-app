import requests
import sys
import json

def post_predict(host:str):
    '''
    Sends an HTTP POST request to the /predict endpoint.
    '''
    proto = ''
    if not host.startswith('http'):
        proto = 'http://'
    
    with open('manualtests/manual_predict_data.json', 'r') as f:
        data = json.loads(f.read())
        response = requests.post(proto + host, json=data)
    
    return response

if __name__ == '__main__':
    args = sys.argv
    print('Requesting for a prediction...')
    response = post_predict(host=args[1])
    if response.ok:
        print(' ...✅ OK:', response.json())
    else:
        print(' ...❌ failed:', response.content)