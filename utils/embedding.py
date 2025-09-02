import requests
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv() # nó tự tìm file .env trong folder

EURON_API_KEY = os.getenv("EURON_API_KEY2")

print(f"DEBUG: EURON_API_KEY hiện tại: {EURON_API_KEY[:5]}...{EURON_API_KEY[-5:] if EURON_API_KEY else 'Không có'}")

def get_embedding(text, model="text-embedding-3-small"): 
    url = 'https://api.euron.one/api/v1/euri/embeddings'
    headers = {
        'Authorization':'Bearer {}'.format(EURON_API_KEY),
        'Content-Type':'application/json'
    }
    payload = {
        'input':text,
        'model':model
    }

    response = requests.post(url, headers=headers, json=payload)

    # return luôn về 1 mảng chứa các giá trị embedding
    # return np.array(response.json()['data'][0]['embedding'])
    # Thêm đoạn mã này để kiểm tra phản hồi
    try:
        response_json = response.json()
        if 'data' not in response_json:
            print(f"Lỗi: Không tìm thấy khóa 'data' trong phản hồi API. Phản hồi đầy đủ: {response_json}")
            # Hoặc raise một ngoại lệ tùy chỉnh để xử lý tốt hơn
            raise KeyError("'data' key not found in Euron API response")
        return np.array(response_json['data'][0]['embedding'])
    except requests.exceptions.JSONDecodeError:
        print(f"Lỗi: Phản hồi từ API không phải là JSON. Phản hồi văn bản: {response.text}")
        raise # Re-raise để lỗi gốc vẫn được hiển thị
    except Exception as e:
        print(f"Lỗi không xác định khi xử lý phản hồi API: {e}")
        print(f"Phản hồi đầy đủ: {response.text}")
        raise