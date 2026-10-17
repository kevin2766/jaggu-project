"""
SatyaDrishti Flask Web Application.
Cyber Samvad Hackathon 2026 - Problem Statement #1:
'Is This Video Real?' — Indian Deepfake Detection Platform
"""

import os
import hashlib
import time
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from detector import analyze_video_file

app = Flask(__name__)
app.config['SECRET_KEY'] = 'satyadrishti_cyber_samvad_2026_secure'
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['DEMO_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'demo_samples')
app.config['MAX_CONTENT_LENGTH'] = 120 * 1024 * 1024  # 120 MB max

ALLOWED_EXTENSIONS = {'mp4', 'webm', 'mov', 'avi', 'mkv'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def compute_sha256(filepath):
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    # Check if user provided an uploaded file or a demo selection
    demo_choice = request.form.get('demo_choice')
    video_path = None
    original_filename = "demo.mp4"

    if demo_choice:
        valid_demos = {
            'real': 'demo_real.mp4',
            'deepfake': 'demo_deepfake.mp4',
            'lipsync': 'demo_lipsync.mp4',
            'low_quality': 'demo_low_quality.mp4'
        }
        if demo_choice in valid_demos:
            filename = valid_demos[demo_choice]
            video_path = os.path.join(app.config['DEMO_FOLDER'], filename)
            original_filename = filename
            web_video_url = f"/static/demo_samples/{filename}"
        else:
            return jsonify({"status": "error", "message": "Invalid demo selection"}), 400
    else:
        if 'video' not in request.files:
            return jsonify({"status": "error", "message": "No video file provided"}), 400

        file = request.files['video']
        if file.filename == '':
            return jsonify({"status": "error", "message": "Empty file name"}), 400

        if not allowed_file(file.filename):
            return jsonify({"status": "error", "message": f"Unsupported format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

        sec_name = secure_filename(file.filename)
        filename = f"{int(time.time())}_{sec_name}"
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)
        original_filename = sec_name
        web_video_url = f"/uploads/{filename}"

    # Compute SHA-256 for forensic integrity audit
    sha256_hash = compute_sha256(video_path)
    file_size_mb = round(os.path.getsize(video_path) / (1024 * 1024), 2)

    # Run multi-signal detection engine
    result = analyze_video_file(video_path)

    # Attach forensic dossier metadata
    result['file_info'] = {
        "filename": original_filename,
        "video_url": web_video_url,
        "sha256": sha256_hash,
        "size_mb": file_size_mb,
        "analyzed_at": time.strftime("%Y-%m-%d %H:%M:%S UTC")
    }

    return jsonify(result)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/api/helpline_info')
def helpline_info():
    return jsonify({
        "national_portal": "https://cybercrime.gov.in",
        "toll_free_helpline": "1930",
        "organization": "National Cyber Crime Reporting Portal (MHA, Government of India)",
        "advisory": "Do not transfer money or share OTPs based on emergency video calls or deepfake voice clips."
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"SatyaDrishti Platform starting on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
