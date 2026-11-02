"""Download real volleyball match clips / rallies from YouTube into:
D:\\volleyball datasets\\production Data\\from internet

Can be used as a CLI tool or imported by the backend server.
"""
import os
import sys
import argparse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VENV_SCRIPTS = os.path.join(ROOT, ".venv", "Scripts")
if os.path.isdir(VENV_SCRIPTS):
    os.environ["PATH"] = VENV_SCRIPTS + os.pathsep + os.environ.get("PATH", "")

import yt_dlp

OUT_DIR = os.path.join(ROOT, "production Data", "from internet")

# Curated high-quality indoor match clips / rallies from international tournaments
CURATED_MATCHES = [
    {
        "name": "Japan_Epic_Volleyball_Rally.mp4",
        "query": "volleyball long rally Japan Nishida",
        "title": "Japan National Team Epic Rally",
    },
    {
        "name": "Olympic_Men_Volleyball_Rally.mp4",
        "query": "Olympic volleyball longest rally men highlights",
        "title": "Men's Olympic Volleyball Championship Rally",
    },
    {
        "name": "VNL_Championship_Rally.mp4",
        "query": "VNL volleyball best rally 2024",
        "title": "Volleyball Nations League Best Rally",
    },
    {
        "name": "France_vs_Poland_Olympic_Final.mp4",
        "query": "France vs Poland Olympic volleyball final rally",
        "title": "France vs Poland Olympic Final Rally",
    },
]


def get_ydl_opts(out_template, max_seconds=60, progress_callback=None):
    opts = {
        # Best mp4 video up to 1080p + best audio, or best single file
        "format": "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4][height<=1080]/best",
        "outtmpl": out_template,
        "quiet": False,
        "no_warnings": False,
        "merge_output_format": "mp4",
        "ffmpeg_location": VENV_SCRIPTS if os.path.isdir(VENV_SCRIPTS) else None,
        # Limit to first section if video is long to ensure rapid download of match action
        "download_ranges": yt_dlp.utils.download_range_func(None, [(0, max_seconds)]),
        "force_keyframes_at_cuts": True,
    }
    if progress_callback:
        def _hook(d):
            status = d.get("status")
            if status == "downloading":
                total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                downloaded = d.get("downloaded_bytes", 0)
                pct = round((downloaded / total) * 100, 1) if total > 0 else 0.0
                speed = d.get("_speed_str", "").strip() or "-- KB/s"
                eta = d.get("_eta_str", "").strip() or "--"
                progress_callback({
                    "status": "downloading",
                    "percent": pct,
                    "speed": speed,
                    "eta": eta,
                    "downloaded": downloaded,
                    "total": total,
                })
            elif status == "finished":
                progress_callback({
                    "status": "processing",
                    "percent": 99.0,
                    "speed": "Merging audio & video",
                    "eta": "Finalizing",
                })
        opts["progress_hooks"] = [_hook]
    return opts


def download_url(url, out_name=None, out_dir=OUT_DIR, max_seconds=60, progress_callback=None):
    os.makedirs(out_dir, exist_ok=True)
    if out_name:
        if not out_name.endswith(".mp4"):
            out_name += ".mp4"
        out_template = os.path.join(out_dir, out_name)
    else:
        out_template = os.path.join(out_dir, "%(title).50s.%(ext)s")

    opts = get_ydl_opts(out_template, max_seconds=max_seconds, progress_callback=progress_callback)
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        base, _ = os.path.splitext(filename)
        mp4_file = base + ".mp4"
        if os.path.exists(mp4_file):
            filename = mp4_file
        return {
            "title": info.get("title", "Match Video"),
            "duration": min(info.get("duration", max_seconds) or max_seconds, max_seconds),
            "file": filename,
            "filename": os.path.basename(filename),
        }


def search_and_download(query, out_name=None, out_dir=OUT_DIR, max_seconds=60, progress_callback=None):
    os.makedirs(out_dir, exist_ok=True)
    if out_name:
        if not out_name.endswith(".mp4"):
            out_name += ".mp4"
        out_template = os.path.join(out_dir, out_name)
    else:
        out_template = os.path.join(out_dir, "%(title).50s.%(ext)s")

    opts = get_ydl_opts(out_template, max_seconds=max_seconds, progress_callback=progress_callback)
    search_str = f"ytsearch1:{query}"
    with yt_dlp.YoutubeDL(opts) as ydl:
        res = ydl.extract_info(search_str, download=True)
        entries = res.get("entries", [])
        if not entries:
            raise RuntimeError(f"No results found for query: {query}")
        info = entries[0]
        filename = ydl.prepare_filename(info)
        base, _ = os.path.splitext(filename)
        mp4_file = base + ".mp4"
        if os.path.exists(mp4_file):
            filename = mp4_file
        return {
            "title": info.get("title"),
            "duration": min(info.get("duration", max_seconds) or max_seconds, max_seconds),
            "file": filename,
            "filename": os.path.basename(filename),
        }


def download_curated_matches(out_dir=OUT_DIR, count=3, max_seconds=60):
    os.makedirs(out_dir, exist_ok=True)
    downloaded = []
    for item in CURATED_MATCHES[:count]:
        out_path = os.path.join(out_dir, item["name"])
        if os.path.exists(out_path) and os.path.getsize(out_path) > 100000:
            print(f"Already exists: {item['name']}")
            downloaded.append({"title": item["title"], "file": out_path, "filename": item["name"]})
            continue
        try:
            print(f"Downloading curated match rally: {item['title']} ...")
            res = search_and_download(item["query"], out_name=item["name"], out_dir=out_dir, max_seconds=max_seconds)
            downloaded.append(res)
            print(f"Done: {item['name']}")
        except Exception as e:
            print(f"Failed to download {item['title']}: {e}")
    return downloaded


def main():
    parser = argparse.ArgumentParser(description="Download real volleyball matches from YouTube.")
    parser.add_argument("--url", type=str, default=None, help="Specific YouTube video URL")
    parser.add_argument("--query", type=str, default=None, help="Search query for volleyball match")
    parser.add_argument("--name", type=str, default=None, help="Output filename")
    parser.add_argument("--curated", action="store_true", help="Download curated sample match rallies")
    parser.add_argument("--count", type=int, default=3, help="Number of curated matches to download")
    parser.add_argument("--max-seconds", type=int, default=60, help="Max duration in seconds")
    args = parser.parse_args()

    if args.url:
        res = download_url(args.url, out_name=args.name, max_seconds=args.max_seconds)
        print(f"Successfully downloaded: {res['filename']}")
    elif args.query:
        res = search_and_download(args.query, out_name=args.name, max_seconds=args.max_seconds)
        print(f"Successfully downloaded: {res['filename']}")
    else:
        print("Downloading curated volleyball match clips to:", OUT_DIR)
        download_curated_matches(count=args.count, max_seconds=args.max_seconds)


if __name__ == "__main__":
    main()
