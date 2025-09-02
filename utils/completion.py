import requests
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv() # tự tìm file .env trong thư mục hiện tại

EURON_API_KEY = os.getenv("EURON_API_KEY2")

def generate_completion(prompt, model='gpt-4.1-nano'):
    url = 'https://api.euron.one/api/v1/euri/chat/completions'
    headers ={
        "Authorization": f"Bearer {EURON_API_KEY}",
        'Content-Type':'application/json'
    }
    
    payload ={
        'model':model,
        'messages':[{'role':'user', 'content':prompt}],
        'temperature':0.3,
        'max_tokens':500
    }

    response = requests.post(url, headers=headers, json=payload)
    return response.json()['choices'][0]['message']['content']