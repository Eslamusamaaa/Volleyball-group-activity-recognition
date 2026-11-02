"""Final patch: fix remaining impl detail strings in index.html."""
html_path = r"D:\volleyball datasets\frontend\index.html"
with open(html_path, "r", encoding="utf-8") as f:
    content = f.read()

# Strip carriage returns for clean processing
content = content.replace('\r\n', '\n').replace('\r', '\n')

# Fix remaining hero desc
content = content.replace(
    '          Automated end-to-end pipeline reproducing & extending CVPR hierarchical models: fine-tuned YOLOv8n player detection, ByteTrack multi-player tracking, ResNet50 spatial backbone, and B8-mt Hierarchical LSTM multi-task classification.',
    '          Upload any volleyball match video to get back a fully annotated output — players tracked and labeled, team plays identified, and every key moment timestamped with a broadcast-style overlay.'
)

# Fix hero title
content = content.replace(
    '<h1 class="hero-title">Real-Time Volleyball Group Activity Recognition & Video Intelligence</h1>',
    '<h1 class="hero-title">AI-Powered Volleyball Match Analysis & Play Recognition</h1>'
)

# Fix KPI labels
content = content.replace('Scene Accuracy (B8-mt)', 'Scene Recognition Accuracy')
content = content.replace('Player Action Head', 'Player Action Accuracy')

# Remove config-row with sliding window details
import re
content = re.sub(
    r'            <div class="config-row">.*?</div>\n',
    '            <div class="config-row">\n'
    '              <label>Mode: <strong>AI Video Analysis</strong></label>\n'
    '              <label>Output: <strong>Annotated Video + Event Timeline</strong></label>\n'
    '            </div>\n',
    content,
    flags=re.DOTALL,
    count=1
)

# Fix button text (html entity version)
content = content.replace('⚡ Analyze &amp; Render Annotated Video', '⚡ Analyze Video')
content = content.replace('⚡ Analyze & Render Annotated Video', '⚡ Analyze Video')

# Fix nav Architecture -> How It Works
content = content.replace('<a href="#tech">Architecture</a>', '<a href="#tech">How It Works</a>')

# Write back
with open(html_path, "w", encoding="utf-8", newline='\n') as f:
    f.write(content)

print("Patched OK")

# Verify
checks = ["B8-mt", "ByteTrack", "ResNet50", "LSTM1", "Sliding Window", "Stride: 9", "ffmpeg remux",
          "Hierarchical Deep Temporal", "Hierarchical LSTM multi-task"]
for c in checks:
    status = "STILL PRESENT" if c in content else "removed OK"
    print(f"  {status}: {c}")

