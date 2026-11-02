"""Test analysis via server API using smallest library video."""
import requests, json, time

# Pick smallest dataset video from library
r = requests.get('http://localhost:8000/api/library')
lib = r.json()
plays = lib['dataset_plays']
plays.sort(key=lambda x: x['size_mb'])
print('Smallest dataset videos:')
for p in plays[:3]:
    print(f"  {p['filename']} ({p['size_mb']} MB, {p['duration_sec']}s)")

small = plays[0]
print(f"\nAnalyzing: {small['filename']}")

# Submit analysis job
t0 = time.time()
res = requests.post('http://localhost:8000/api/library/analyze', json={
    'category': small['category'],
    'filename': small['filename']
})
job = res.json()
job_id = job['job_id']
print(f'Job started: {job_id}')

# Poll for completion
while True:
    r2 = requests.get(f'http://localhost:8000/api/jobs/{job_id}')
    j = r2.json()
    elapsed = time.time() - t0
    print(f"  [{elapsed:.0f}s] {j['status']} {j['progress']}% - {j['phase']}")
    if j['status'] in ('done', 'error'):
        break
    time.sleep(3)

print(f"\nFinal: {j['status']}")
if j['status'] == 'done':
    print(f"Events: {len(j['events'])}")
    print(f"Duration: {j['duration_sec']}s")
    print(f"Annotated URL: {j['annotated_video_url']}")
    print(f"Total time: {time.time()-t0:.1f}s")
    for ev in j['events'][:5]:
        print(f"  {ev['start_sec']}s-{ev['end_sec']}s  {ev['team']} {ev['event']} ({ev['confidence']*100:.0f}%)")
else:
    print(f"Error: {j['error']}")

