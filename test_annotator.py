"""Quick speed test on small video with new annotator."""
import sys, os, time, glob
sys.path.insert(0, r"D:\volleyball datasets")
os.chdir(r"D:\volleyball datasets")

# Pick the SMALLEST dataset video
videos = sorted(glob.glob(r"production Data\from our dataset\*.mp4"),
                key=lambda p: os.path.getsize(p))
print(f"Found {len(videos)} videos")
for v in videos[:5]:
    print(f"  {os.path.basename(v)}: {os.path.getsize(v)/1024:.1f} KB")

test_vid = videos[0]
print(f"\nTesting with smallest: {test_vid}")

from src.inference.annotator import render_annotated_pipeline

t0 = time.time()
def progress_cb(pct, msg):
    elapsed = time.time() - t0
    print(f"  [{pct:3d}%] {msg}  ({elapsed:.1f}s)")

try:
    result = render_annotated_pipeline(
        test_vid,
        "outputs/test_fast.mp4",
        progress_callback=progress_cb
    )
    elapsed = time.time() - t0
    print(f"\nSUCCESS in {elapsed:.1f}s")
    print(f"Events: {len(result['events'])}")
    print(f"Duration: {result['duration_sec']}s, Frames: {result['total_frames']}")
except Exception as e:
    import traceback
    print(f"\nFAILED:")
    traceback.print_exc()
