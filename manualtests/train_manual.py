import requests
import sys
import json

def post_train(host:str):
    '''
    Sends an HTTP POST request to the /train endpoint.
    '''
    proto = ''
    if not host.startswith('http'):
        proto = 'http://'
    
    with open('manualtests/manual_train_data.json', 'r') as f:
        data = json.loads(f.read())
        response = requests.post(proto + host, json={'data':data})
    
    return response

if __name__ == '__main__':
    args = sys.argv
    print('Sending training data...')
    response = post_train(host=args[1])
    if response.ok:
        print('...✅ OK:', response.json())
    else:
        print('... ❌ failed:', response.content)