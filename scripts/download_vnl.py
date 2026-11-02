import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.inference.download_youtube_matches import download_url

url = 'https://www.youtube.com/watch?v=0lN1HfFAYUY'
old_file = os.path.join('production Data', 'from internet', 'Best Volleyball Rally at VNL 2024.mp4')
if os.path.exists(old_file):
    try:
        os.remove(old_file)
        print('Removed old 1-minute file')
    except Exception as e:
        print('Could not remove old file:', e)

def prog(d):
    pct = d.get('percent', 0)
    speed = d.get('speed', '')
    eta = d.get('eta', '')
    if int(pct) % 10 == 0:
        print(f"Downloading: {pct}% | Speed: {speed} | ETA: {eta}")

print('Starting full download of: Best Volleyball Rally at VNL 2024 (10 minutes)...')
res = download_url(url, out_name='Best Volleyball Rally at VNL 2024.mp4', max_seconds=None, progress_callback=prog)
print('Finished successfully:', res)
