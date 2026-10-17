"""
Generate realistic synthetic demonstration videos for SatyaDrishti Platform.
Creates:
1. demo_real.mp4 (Authentic public speaker)
2. demo_deepfake.mp4 (AI Face-swap manipulation with boundary seams & abnormal blinking)
3. demo_lipsync.mp4 (Audio-visual desync & mouth warping manipulation)
4. demo_low_quality.mp4 (Heavily degraded WhatsApp forward - triggers Unable to Assess)
"""

import os
import cv2
import numpy as np
import math

OUTPUT_DIR = r"C:\Users\KEVIN PANCHAL\.gemini\antigravity\scratch\satya_drishti_deepfake_detector\static\demo_samples"
os.makedirs(OUTPUT_DIR, exist_ok=True)

WIDTH, HEIGHT = 640, 480
FPS = 25
DURATION_SEC = 5
TOTAL_FRAMES = FPS * DURATION_SEC


def draw_speaker_base(canvas, frame_idx, head_tilt=0.0):
    # Professional Indian news studio / conference background
    h, w = canvas.shape[:2]
    
    # Gradient backdrop
    for y in range(h):
        r = int(18 + 20 * (y / h))
        g = int(28 + 25 * (y / h))
        b = int(55 + 40 * (y / h))
        canvas[y, :] = [b, g, r]

    # Studio accent lights
    cv2.circle(canvas, (w - 80, 80), 160, (75, 45, 25), -1)
    cv2.circle(canvas, (80, 100), 140, (30, 45, 70), -1)
    
    # Soft blur background
    canvas[:] = cv2.GaussianBlur(canvas, (21, 21), 0)

    # Microphone stand
    cv2.line(canvas, (w // 2, h - 30), (w // 2, h - 110), (90, 90, 95), 6)
    cv2.circle(canvas, (w // 2, h - 115), 18, (40, 40, 45), -1)
    cv2.circle(canvas, (w // 2, h - 115), 14, (70, 70, 75), 2)

    # Shoulders and blazer (Indian formal attire)
    shoulder_y = h - 60
    center_x = w // 2 + int(math.sin(frame_idx * 0.05) * 4)

    # Suit contours
    pts_suit = np.array([
        [center_x - 170, h],
        [center_x - 120, shoulder_y + 20],
        [center_x - 45, shoulder_y - 25],
        [center_x + 45, shoulder_y - 25],
        [center_x + 120, shoulder_y + 20],
        [center_x + 170, h]
    ], np.int32)
    cv2.fillPoly(canvas, [pts_suit], (45, 35, 30))

    # Shirt collar & tie / Nehru collar
    pts_shirt = np.array([
        [center_x - 45, shoulder_y - 25],
        [center_x, shoulder_y + 40],
        [center_x + 45, shoulder_y - 25]
    ], np.int32)
    cv2.fillPoly(canvas, [pts_shirt], (230, 235, 240))

    # Neck
    neck_w = 42
    cv2.rectangle(canvas, (center_x - neck_w, shoulder_y - 85), (center_x + neck_w, shoulder_y - 20), (140, 165, 205), -1)

    return center_x, shoulder_y - 120


def draw_face(canvas, center_x, center_y, frame_idx, is_deepfake=False, blink_state=False, mouth_open=0.0, add_artifacts=False):
    # Head contour
    head_w, head_h = 75, 100
    
    # Normal skin tone in BGR
    skin_tone = (145, 175, 218)
    
    if is_deepfake:
        # Subtle color tone mismatch (typical GAN/FaceSwap artifact)
        skin_tone = (130, 182, 228)

    # Face oval
    cv2.ellipse(canvas, (center_x, center_y), (head_w, head_h), 0, 0, 360, skin_tone, -1)

    # Hair
    cv2.ellipse(canvas, (center_x, center_y - 45), (head_w + 5, 65), 0, 180, 360, (25, 25, 25), -1)
    cv2.ellipse(canvas, (center_x - head_w + 5, center_y - 20), (16, 45), 0, 0, 360, (25, 25, 25), -1)
    cv2.ellipse(canvas, (center_x + head_w - 5, center_y - 20), (16, 45), 0, 0, 360, (25, 25, 25), -1)

    # Eyebrows
    brow_y = center_y - 28
    cv2.line(canvas, (center_x - 48, brow_y), (center_x - 18, brow_y - 4), (40, 40, 40), 4)
    cv2.line(canvas, (center_x + 18, brow_y - 4), (center_x + 48, brow_y), (40, 40, 40), 4)

    # Eyes
    eye_y = center_y - 14
    eye_lx, eye_rx = center_x - 32, center_x + 32

    if blink_state:
        # Closed eye slit
        cv2.line(canvas, (eye_lx - 12, eye_y), (eye_lx + 12, eye_y), (50, 60, 80), 3)
        cv2.line(canvas, (eye_rx - 12, eye_y), (eye_rx + 12, eye_y), (50, 60, 80), 3)
    else:
        # Eye sclera
        cv2.ellipse(canvas, (eye_lx, eye_y), (14, 8), 0, 0, 360, (245, 245, 245), -1)
        cv2.ellipse(canvas, (eye_rx, eye_y), (14, 8), 0, 0, 360, (245, 245, 245), -1)
        # Iris & Pupil
        cv2.circle(canvas, (eye_lx, eye_y), 5, (60, 45, 30), -1)
        cv2.circle(canvas, (eye_rx, eye_y), 5, (60, 45, 30), -1)
        cv2.circle(canvas, (eye_lx + 1, eye_y - 1), 2, (255, 255, 255), -1)
        cv2.circle(canvas, (eye_rx + 1, eye_y - 1), 2, (255, 255, 255), -1)
        # Eye outlines
        cv2.ellipse(canvas, (eye_lx, eye_y), (14, 8), 0, 0, 360, (80, 90, 110), 1)
        cv2.ellipse(canvas, (eye_rx, eye_y), (14, 8), 0, 0, 360, (80, 90, 110), 1)

    # Nose
    nose_y = center_y + 12
    cv2.line(canvas, (center_x, center_y - 5), (center_x - 4, nose_y), (120, 140, 180), 2)
    cv2.line(canvas, (center_x - 4, nose_y), (center_x + 4, nose_y), (120, 140, 180), 2)

    # Mouth
    mouth_y = center_y + 45
    mouth_open_h = int(mouth_open * 12)
    if mouth_open_h > 1:
        cv2.ellipse(canvas, (center_x, mouth_y), (20, mouth_open_h), 0, 0, 360, (40, 40, 80), -1)
        # Teeth
        if mouth_open_h > 4:
            cv2.rectangle(canvas, (center_x - 12, mouth_y - mouth_open_h // 2), (center_x + 12, mouth_y - 1), (240, 240, 245), -1)
    else:
        cv2.line(canvas, (center_x - 18, mouth_y), (center_x + 18, mouth_y), (80, 80, 140), 2)

    # Deepfake-specific forensic anomalies
    if add_artifacts:
        # 1. Distinct face-swap boundary seam / feathering halo (classic FaceSwap / DeepFaceLab boundary)
        seam_color = (int(skin_tone[0] * 0.7), int(skin_tone[1] * 0.8), int(skin_tone[2] * 0.9))
        cv2.ellipse(canvas, (center_x, center_y + 8), (head_w - 4, head_h - 12), 0, 0, 360, seam_color, 3)
        cv2.ellipse(canvas, (center_x, center_y + 8), (head_w - 7, head_h - 15), 0, 0, 360, (255, 200, 150), 1)

        # 2. Add high-frequency edge gradient discontinuity around boundary
        seam_y = center_y + head_h - 15
        for offset in range(-55, 55, 2):
            noise_val = int(np.random.normal(0, 35))
            px = center_x + offset
            py = seam_y + int(math.sin(offset * 0.15) * 8)
            if 0 <= py < canvas.shape[0] - 2 and 0 <= px < canvas.shape[1] - 2:
                canvas[py:py+3, px:px+3] = np.clip(canvas[py:py+3, px:px+3].astype(int) + noise_val, 0, 255)

        # 3. Frequency domain periodic grid pattern (GAN / transposed-convolution checkerboard)
        patch_y1, patch_y2 = max(0, center_y - 45), min(canvas.shape[0], center_y + 55)
        patch_x1, patch_x2 = max(0, center_x - 50), min(canvas.shape[1], center_x + 50)
        yy, xx = np.mgrid[0:(patch_y2-patch_y1), 0:(patch_x2-patch_x1)]
        grid = (np.sin(xx * 0.8) * np.cos(yy * 0.8) * 28).astype(np.int16)
        for c in range(3):
            canvas[patch_y1:patch_y2, patch_x1:patch_x2, c] = np.clip(
                canvas[patch_y1:patch_y2, patch_x1:patch_x2, c].astype(np.int16) + grid, 0, 255
            ).astype(np.uint8)


def generate_real_video():
    path = os.path.join(OUTPUT_DIR, "demo_real.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    writer = cv2.VideoWriter(path, fourcc, FPS, (WIDTH, HEIGHT))

    for f in range(TOTAL_FRAMES):
        canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        cx, cy = draw_speaker_base(canvas, f)

        # Natural blinking: occurs at frame 30-34 and 90-94 (1-2 blinks every few seconds)
        blink = (30 <= f <= 33) or (92 <= f <= 95)

        # Natural speech cadence (smooth sinusoidal mouth aperture)
        speech_phase = f * 0.35
        mouth_open = max(0.0, math.sin(speech_phase) * 0.8 + 0.2 * math.cos(speech_phase * 2.1))

        draw_face(canvas, cx, cy, f, is_deepfake=False, blink_state=blink, mouth_open=mouth_open, add_artifacts=False)

        # Add subtle natural camera grain
        noise = np.random.normal(0, 1.2, canvas.shape).astype(np.int16)
        canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        writer.write(canvas)

    writer.release()
    print("Generated:", path)


def generate_deepfake_video():
    path = os.path.join(OUTPUT_DIR, "demo_deepfake.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    writer = cv2.VideoWriter(path, fourcc, FPS, (WIDTH, HEIGHT))

    for f in range(TOTAL_FRAMES):
        canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        cx, cy = draw_speaker_base(canvas, f)

        # Deepfake flaw 1: Abnormal eyelid dynamics (No natural blinks in 5 seconds! Staring eye anomaly)
        blink = False 

        # Deepfake flaw 2: Mouth opening is jerky with synthetic interpolation
        mouth_open = 0.6 if (f % 6 < 3) else 0.1

        # Deepfake flaw 3: Frame jitter / slight face-swap mask drift
        jitter_x = int(math.sin(f * 1.8) * 2)
        jitter_y = int(math.cos(f * 2.2) * 2)

        draw_face(canvas, cx + jitter_x, cy + jitter_y, f, is_deepfake=True, blink_state=blink, mouth_open=mouth_open, add_artifacts=True)

        writer.write(canvas)

    writer.release()
    print("Generated:", path)


def generate_lipsync_video():
    path = os.path.join(OUTPUT_DIR, "demo_lipsync.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    writer = cv2.VideoWriter(path, fourcc, FPS, (WIDTH, HEIGHT))

    for f in range(TOTAL_FRAMES):
        canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        cx, cy = draw_speaker_base(canvas, f)

        # Random erratic mouth flapping (simulating Wav2Lip / synthetic audio dubbing desync)
        blink = (45 <= f <= 48)
        mouth_open = abs(math.sin(f * 0.95))

        draw_face(canvas, cx, cy, f, is_deepfake=True, blink_state=blink, mouth_open=mouth_open, add_artifacts=False)

        # Add localized blur around the mouth area (typical lip-synthesis artifact)
        mouth_roi_y1, mouth_roi_y2 = cy + 30, cy + 65
        mouth_roi_x1, mouth_roi_x2 = cx - 35, cx + 35
        roi = canvas[mouth_roi_y1:mouth_roi_y2, mouth_roi_x1:mouth_roi_x2]
        canvas[mouth_roi_y1:mouth_roi_y2, mouth_roi_x1:mouth_roi_x2] = cv2.GaussianBlur(roi, (11, 11), 3)

        writer.write(canvas)

    writer.release()
    print("Generated:", path)


def generate_low_quality_video():
    path = os.path.join(OUTPUT_DIR, "demo_low_quality.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    writer = cv2.VideoWriter(path, fourcc, FPS, (WIDTH, HEIGHT))

    for f in range(TOTAL_FRAMES):
        canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        cx, cy = draw_speaker_base(canvas, f)
        draw_face(canvas, cx, cy, f, is_deepfake=False, blink_state=False, mouth_open=0.3, add_artifacts=False)

        # Heavy compression & low resolution simulation (120x90 resized up with heavy blur)
        small = cv2.resize(canvas, (120, 90), interpolation=cv2.INTER_NEAREST)
        degraded = cv2.resize(small, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)
        degraded = cv2.GaussianBlur(degraded, (15, 15), 5)

        writer.write(degraded)

    writer.release()
    print("Generated:", path)


if __name__ == "__main__":
    print("Generating demo videos...")
    generate_real_video()
    generate_deepfake_video()
    generate_lipsync_video()
    generate_low_quality_video()
    print("All demo samples generated successfully!")
