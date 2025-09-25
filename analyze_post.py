import os, requests
fp = r"graph_test_2.png"
with open(fp, 'rb') as f:
    r = requests.post('http://127.0.0.1:8087/api/graph/analyze', files={'file': (os.path.basename(fp), f, 'image/png')})
print(r.status_code)
print(r.text)
