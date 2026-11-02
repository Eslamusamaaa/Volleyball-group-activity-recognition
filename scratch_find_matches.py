import sys
import yt_dlp

sys.stdout.reconfigure(encoding='utf-8')

queries = [
    'Volleyball Full Match VNL',
    'Volleyball Full Match Men Olympic',
    'Volleyball Full Match CEV Champions League',
    'NCAA Volleyball Full Match broadcast'
]
ydl_opts = {'quiet': True, 'extract_flat': True}

print("Searching for official full match broadcasts on YouTube...")
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    for q in queries:
        try:
            res = ydl.extract_info(f'ytsearch5:{q}', download=False)
            for e in res.get('entries', []):
                dur = e.get('duration') or 0
                title = e.get('title')
                vid = e.get('id')
                # Looking for full match broadcast (> 20 minutes)
                if dur > 1200:
                    mins = dur // 60
                    print(f"[{mins} min] {title} | https://www.youtube.com/watch?v={vid}")
        except Exception as err:
            print("Err:", err)

