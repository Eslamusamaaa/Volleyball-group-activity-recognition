"""Patch frontend/index.html to remove implementation details."""
import re

html_path = r"D:\volleyball datasets\frontend\index.html"
with open(html_path, "r", encoding="utf-8") as f:
    html = f.read()

original = html

# 1. Remove hero description with model names
html = html.replace(
    '        <p class="hero-desc">\n'
    '          Automated end-to-end pipeline reproducing &amp; extending CVPR hierarchical models: fine-tuned YOLOv8n player detection, ByteTrack multi-player tracking, ResNet50 spatial backbone, and B8-mt Hierarchical LSTM multi-task classification.\n'
    '        </p>',
    '        <p class="hero-desc">\n'
    '          Upload any volleyball match video to get back a fully annotated output — players tracked and labeled, team plays identified, and every key moment timestamped with a broadcast-style overlay.\n'
    '        </p>'
)

# 2. Remove badge-pill "Hierarchical Deep Temporal Vision System"
html = html.replace(
    '        <div class="badge-pill">Hierarchical Deep Temporal Vision System</div>',
    '        <div class="badge-pill">AI-Powered Match Analysis</div>'
)

# 3. Remove KPI label technical names
html = html.replace('Scene Accuracy (B8-mt)', 'Scene Recognition Accuracy')
html = html.replace('Player Action Head', 'Player Action Accuracy')
html = html.replace('Tracker Identity Match', 'Player Tracking Accuracy')
html = html.replace('Player Detection mAP50', 'Player Detection Accuracy')

# 4. Remove config-row (Sliding Window, Stride, Output labels)
html = html.replace(
    '            <div class="config-row">\n'
    '              <label>Sliding Window: <strong>9 Frames</strong></label>\n'
    '              <label>Stride: <strong>9 Frames</strong></label>\n'
    '              <label>Output: <strong>H.264 Annotated Video + JSON Timeline</strong></label>\n'
    '            </div>',
    '            <div class="config-row">\n'
    '              <label>Mode: <strong>AI Video Analysis</strong></label>\n'
    '              <label>Output: <strong>Annotated Video + Event Timeline</strong></label>\n'
    '            </div>'
)

# 5. Simplify the button text
html = html.replace('⚡ Analyze &amp; Render Annotated Video', '⚡ Analyze Video')

# 6. Remove entire Technical Architecture section and replace with clean "How It Works"
tech_section_old = '''  <!-- Technical Specs &amp; Pipeline Section -->
  <section id="tech" class="tech-section">
    <div class="container">
      <div class="section-header">
        <h2>Neural Architecture &amp; Inference Pipeline</h2>
        <p>Hierarchical multi-stage design combining modern object detection, multi-object tracking, and dual-level LSTM models.</p>
      </div>

      <div class="pipeline-flow">
        <div class="pipeline-step">
          <div class="step-num">01</div>
          <h4>Raw Video Input</h4>
          <p>Extracts per-video native FPS via OpenCV; sliding 9-frame window over continuous match footage.</p>
        </div>
        <div class="pipeline-step">
          <div class="step-num">02</div>
          <h4>Player Detector</h4>
          <p>Fine-tuned YOLOv8n on volleyball ground truth (98.0% mAP50, 95.3% precision, 94.2% recall).</p>
        </div>
        <div class="pipeline-step">
          <div class="step-num">03</div>
          <h4>ByteTrack Tracker</h4>
          <p>97.7% identity preservation match across frames. Rejects staff/spectators and maintains track consistency.</p>
        </div>
        <div class="pipeline-step">
          <div class="step-num">04</div>
          <h4>Spatial Backbone</h4>
          <p>ResNet50 Stage B3-A crops player boxes into 2048-dim feature vectors per tracked player per frame.</p>
        </div>
        <div class="pipeline-step">
          <div class="step-num">05</div>
          <h4>B8-mt Temporal LSTM</h4>
          <p>Hierarchical temporal model: LSTM1 tracks player trajectories, LSTM2 aggregates team context for scene + player action classification.</p>
        </div>
        <div class="pipeline-step">
          <div class="step-num">06</div>
          <h4>H.264 Video Render</h4>
          <p>Two-pass hardware-accelerated ffmpeg remuxing with burned-in player bounding boxes and broadcast scoreboard HUD.</p>
        </div>
      </div>
    </div>
  </section>'''

tech_section_new = '''  <!-- How It Works Section -->
  <section id="tech" class="tech-section">
    <div class="container">
      <div class="section-header">
        <h2>How It Works</h2>
        <p>Upload a volleyball match video and the AI engine processes it in real-time to produce a fully annotated broadcast-quality output.</p>
      </div>

      <div class="pipeline-flow">
        <div class="pipeline-step">
          <div class="step-num">01</div>
          <h4>Upload Match Video</h4>
          <p>Upload an MP4, AVI, MOV, or MKV file. Supports clips from 10 seconds up to full 90-minute matches.</p>
        </div>
        <div class="pipeline-step">
          <div class="step-num">02</div>
          <h4>Player Detection</h4>
          <p>Every player on court is detected in each frame with 98% accuracy, filtering out staff, referees, and spectators.</p>
        </div>
        <div class="pipeline-step">
          <div class="step-num">03</div>
          <h4>Multi-Player Tracking</h4>
          <p>Players are continuously tracked across frames with consistent IDs, even through occlusions and fast movements.</p>
        </div>
        <div class="pipeline-step">
          <div class="step-num">04</div>
          <h4>Action Recognition</h4>
          <p>The AI classifies every player's action — spike, block, set, pass — and the team's overall play at every moment.</p>
        </div>
        <div class="pipeline-step">
          <div class="step-num">05</div>
          <h4>Event Timeline</h4>
          <p>Key match events are assembled into a timestamped timeline you can click to jump directly to any moment.</p>
        </div>
        <div class="pipeline-step">
          <div class="step-num">06</div>
          <h4>Annotated Video Output</h4>
          <p>The result is streamed back as a broadcast-quality video with bounding boxes and live HUD overlay burned in.</p>
        </div>
      </div>
    </div>
  </section>'''

# Also try the non-escaped version
tech_section_old2 = tech_section_old.replace('&amp;', '&')
if tech_section_old2 in html:
    html = html.replace(tech_section_old2, tech_section_new)
    print("Replaced tech section (unescaped)")
elif tech_section_old in html:
    html = html.replace(tech_section_old, tech_section_new)
    print("Replaced tech section (escaped)")
else:
    print("WARNING: Could not find tech section for replacement, using regex fallback")
    # Regex fallback
    html = re.sub(
        r'<!-- Technical Specs.*?</section>',
        tech_section_new,
        html,
        flags=re.DOTALL
    )
    print("Applied regex replacement")

# 7. Remove "Architecture" nav link -> rename to "How It Works"
html = html.replace('<a href="#tech">Architecture</a>', '<a href="#tech">How It Works</a>')

# 8. Check for hero title simplification (optional but clean)
html = html.replace(
    '<h1 class="hero-title">Real-Time Volleyball Group Activity Recognition &amp; Video Intelligence</h1>',
    '<h1 class="hero-title">AI-Powered Volleyball Match Analysis &amp; Play Recognition</h1>'
)

with open(html_path, "w", encoding="utf-8") as f:
    f.write(html)

changed = html != original
print(f"File updated: {changed}")
print(f"Length: {len(html)} chars")

# Verify removal
checks = ["Sliding Window", "Stride: 9", "B8-mt", "ByteTrack", "ResNet50", "LSTM1", "LSTM2", "ffmpeg remux"]
for c in checks:
    if c in html:
        print(f"STILL PRESENT: {c}")
    else:
        print(f"OK removed: {c}")

