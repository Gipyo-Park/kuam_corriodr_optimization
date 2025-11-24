import requests
import json
import datetime

# SERVER = "http://10.74.11.159:8000" # Insert your server address here
SERVER = "http://127.0.0.1:8000"
# SERVER = "https://bff54cdf0390.ngrok-free.app"  

# Original test coordinates 
# payload = {
#     "start_point": [35.603336, 129.077692, 0.0],
#     "end_point": [35.603336, 129.077692, 0.0]
# }

# Using test coordinates from path_engine.py with altitude 0
payload = {
    "start_point": [35.5845361, 129.1076472, 500.0],
    "end_point": [35.6249583, 129.1335528, 500.0]
}

def post_and_log(endpoint, payload):
    url = f"{SERVER}{endpoint}"
    try:
        resp = requests.post(url, json=payload, timeout=10)
        body = resp.json()
        log_obj = {
            "ts": datetime.datetime.now().isoformat(),
            "endpoint": endpoint,
            "request": payload,
            "status": resp.status_code,
            "response": body
        }
        print(json.dumps(log_obj, indent=2, ensure_ascii=False))
        # print(body)
        resp.raise_for_status()
        return body
    except Exception as e:
        print(f"Error: {e}")
        return {}

if __name__ == "__main__":
    post_and_log("/path-with-ground-risk", payload)
