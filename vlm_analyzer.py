"""
VLM Scene Analyzer using Moondream Vision Model.
Performs deep visual understanding: reads text on held items (banknotes, cards, books),
recognizes custom objects, describes actions, posture, and room environment.
"""
import base64
import cv2
import requests
import config


def analyze_scene_vlm(frame, prompt="Describe what the person is holding in their hand, doing, wearing, and everything visible in detail."):
    """Sends video frame to Moondream VLM for deep visual scene analysis."""
    if not getattr(config, "ENABLE_VLM_VISION", True):
        return None

    try:
        # Resize frame for optimal VLM processing speed
        h, w = frame.shape[:2]
        if w > 640:
            scale = 640.0 / w
            frame = cv2.resize(frame, (640, int(h * scale)))

        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        img_b64 = base64.b64encode(buffer).decode("utf-8")

        vlm_model = getattr(config, "VLM_MODEL", "moondream")

        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": vlm_model,
                "prompt": prompt,
                "images": [img_b64],
                "stream": False
            },
            timeout=8
        )

        if resp.status_code == 200:
            description = resp.json().get("response", "").strip()
            if description and len(description) > 5:
                return description
    except Exception:
        pass

    return None
