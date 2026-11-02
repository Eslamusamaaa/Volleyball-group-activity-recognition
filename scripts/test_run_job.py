import requests, json, time

res = requests.post('http://localhost:8000/api/library/analyze', json={
    'category': 'from internet',
    'filename': 'Best Volleyball Rally at VNL 2024.mp4'
})
print('Response:', res.status_code, res.text)
if res.ok:
    job_id = res.json()['job_id']
    print(f'Started job: {job_id}')
    for _ in range(8):
        time.sleep(2)
        r = requests.get(f'http://localhost:8000/api/jobs/{job_id}')
        j = r.json()
        print(f"  Status: {j['status']}, Progress: {j['progress']}%, Phase: {j['phase']}, Error: {j.get('error')}")

