import requests

URL = "https://huggingface.co/api/datasets/Aneeshers/tennis-sackmann-archive/tree/main/wta"

resp = requests.get(URL, timeout=30)
resp.raise_for_status()

for item in resp.json():
    print(f"{item['type']:9} {item.get('size', 0):>10,} {item['path']}")