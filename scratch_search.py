import sys
import yt_dlp

sys.stdout.reconfigure(encoding='utf-8')

queries = [
    'longest volleyball rallies non stop multi attacks',
    'volleyball crazy extended rallies 4 minutes',
    'volleyball defense digs and counter attacks',
    'volleyball long rallies compilation 4 minutes',
    'epic volleyball long rallies 3 minutes'
]
ydl_opts = {'quiet': True, 'extract_flat': True}
candidates = []
seen = set()

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    for q in queries:
        try:
            res = ydl.extract_info(f'ytsearch8:{q}', download=False)
            for e in res.get('entries', []):
                vid = e.get('id')
                if not vid or vid in seen:
                    continue
                seen.add(vid)
                dur = e.get('duration')
                # 2.5 to 5.5 min: 150 to 330s
                if dur and 150 <= dur <= 330:
                    candidates.append({
                        'duration': dur,
                        'title': e.get('title'),
                        'url': f'https://www.youtube.com/watch?v={vid}',
                        'id': vid
                    })
        except Exception as err:
            print("Query err:", err)

print(f"Total matching multi-rally videos: {len(candidates)}")
for c in candidates:
    m = c['duration'] // 60
    s = c['duration'] % 60
    print(f"{m}m {s:02d}s | {c['title']} | {c['url']}")
