# 👁️ SatyaDrishti AI (सत्य दृष्टि)
### Indian Deepfake Detection & Media Forensic Platform
**Cyber Samvad Hackathon 2026 — Problem Statement #1: "Is This Video Real?"**

---

## 🌟 Executive Summary

In India's fast-paced digital ecosystem of 900+ million internet users, deepfakes pose critical national security, electoral, and cyber-fraud challenges. From manipulated political speeches and fake celebrity investment scams to cyber extortion ("digital arrest" calls), malicious actors weaponize synthetic media to cause harm.

**SatyaDrishti** (*"Vision of Truth"*) is a multi-signal media forensics platform created to give citizens, journalists, law enforcement, and fact-checkers fast warning signs before they trust or forward a video.

---

## 🎯 Key Hackathon Features & Compliance

| Requirement from Problem Statement | SatyaDrishti Implementation |
| :--- | :--- |
| **Upload video or select demo file** | Drag-and-drop video upload (MP4/WebM/MOV) + 4 instant one-click demo scenarios. |
| **3 Official Hackathon Verdicts** | 🔴 **"Possible manipulation detected"**<br>🟢 **"No strong signs detected"**<br>🟡 **"Unable to assess"** (Avoids false accusations on degraded media). |
| **Highlight forensic clues** | • Boundary blending seam analysis (Laplacian edge discontinuity)<br>• Biological eye-blink tracking (12-20 blinks/min baseline vs frozen gaze)<br>• 2D FFT spectral decay analysis (GAN/Diffusion high-frequency grid residue)<br>• Chrominance shift (YCrCb face vs neck mismatch)<br>• Inter-frame motion warp jitter |
| **Confidence & Risk Level** | Quantitative Manipulation Risk Index (0–100%) and Forensic Confidence Metric (0–100%). |
| **Plain-Language Explanations** | Multi-lingual plain explanations in **7 Indian Languages**: English, Hindi (हिंदी), Marathi (मराठी), Gujarati (ગુજરાતી), Bengali (বাংলা), Tamil (தமிழ்), Telugu (తెలుగు). |
| **Ethical Guardrails** | Prominent warning: *"Automated results can be wrong and should not be treated as final proof. Do not label a real person as a liar or claim a video is definitely fake based only on an AI score."* |
| **WhatsApp Virality Protection** | 3-step citizen verification protocol + direct helpline links to **1930** and `cybercrime.gov.in`. |
| **Audit Dossier Export** | One-click printable / PDF Forensic Audit Dossier with cryptographic SHA-256 file fingerprint. |
| **Built-in Pitch Presentation Deck** | Interactive 6-slide presentation modal built directly into the UI for hackathon jury evaluation. |

---

## 📂 Project Structure

```
satya_drishti_deepfake_detector/
├── app.py                     # Flask web server and REST API endpoints
├── detector.py                # Multi-modal forensic detection engine
├── generate_samples.py        # Generates realistic synthetic benchmark video samples
├── test_api.py                # End-to-end API test suite
├── README.md                  # Complete technical documentation
├── HACKATHON_PITCH.md         # Hackathon jury presentation deck & speech script
├── models/                    # Cascade & feature tracking models
│   ├── haarcascade_frontalface_default.xml
│   └── haarcascade_eye.xml
├── static/
│   ├── css/style.css          # Cyber-defense dark theme & responsive UI
│   ├── js/app.js              # Client interactivity, video sync, language toggles
│   └── demo_samples/          # Preloaded test videos
│       ├── demo_real.mp4          # Authentic address
│       ├── demo_deepfake.mp4      # AI Face-swap manipulation
│       ├── demo_lipsync.mp4       # Audio-visual lip desynchronization
│       └── demo_low_quality.mp4   # Degraded WhatsApp forward (Unable to assess)
├── templates/
│   └── index.html             # Command center interface & slide deck modal
└── uploads/                   # Uploaded video staging
```

---

## 🚀 Quickstart Guide

### 1. Requirements
- Python 3.10+
- Dependencies: `flask`, `opencv-python`, `numpy`, `scipy`, `pillow`

```bash
pip install flask opencv-python scipy pillow numpy
```

### 2. Generate Sample Media (Optional - Pre-generated)
```bash
python generate_samples.py
```

### 3. Run the Platform
```bash
python app.py
```
Open your browser and navigate to: **`http://localhost:5000`**

---

## 🔬 Multi-Signal Forensic Architecture

SatyaDrishti avoids brittle single-model failures by aggregating orthogonal forensic signals:

```
Video Stream ──┬──> [1. Quality Assessor] ──> (Sharpness < 25 / Res < 360p) ──> "Unable to assess"
               │
               ├──> [2. Spatial Seam Analyzer] ──> Laplacian boundary variance
               │
               ├──> [3. 2D FFT Spectral Engine] ──> Radial frequency roll-off & grid check
               │
               ├──> [4. Biological Dynamics] ──> Eye blink cadence (0 blinks or >55 blinks/min)
               │
               └──> [5. Temporal Motion Tracker] ──> Inter-frame facial warp jitter
                                   │
                                   ▼
                    [Weighted Multi-Signal Synthesizer]
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        ▼                          ▼                          ▼
"Possible manipulation"    "No strong signs"       "Unable to assess"
   (Risk: 50% - 100%)       (Risk: 0% - 35%)     (Inconclusive / Blur)
```

---

## 🛡️ Limitations (Honest Hackathon Disclosure)

1. **Extreme Compression**: Videos forwarded 5+ times through WhatsApp at 144p resolution lose high-frequency spatial clues; SatyaDrishti responsibly classifies these as *"Unable to assess"* to prevent false accusations.
2. **Whole-Frame Diffusion (Sora / Kling)**: Seamless whole-frame generative models without face splicing require complementary temporal diffusion transformers and audio spectrogram alignment.
3. **Screen Re-recording**: Filming a monitor with a phone introduces moiré patterns that can obscure subtle synthesis artifacts.

---

## 🇮🇳 Societal Impact & Alignment with Digital India

- **Election Integrity**: Rapid verification of viral political videos during state and general elections.
- **Cybercrime Prevention**: Countering "Digital Arrest" scams and impersonation extortion before citizens transfer funds.
- **Vernacular Literacy**: Bridging the gap for non-English speakers through 7 regional Indian languages.
- **Emergency Escalation**: Immediate one-click dialing of the National Cyber Fraud Helpline **1930** and reporting to **cybercrime.gov.in**.
