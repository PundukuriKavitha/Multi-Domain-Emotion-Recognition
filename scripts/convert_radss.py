import os
import shutil

# =========================
# PATHS (EDIT IF NEEDED)
# =========================
RAVDESS_INPUT = "dataset1"
RAVDESS_OUTPUT = "data/source/RAVDESS"

# =========================
# EMOTION CODE MAPPING
# =========================
# RAVDESS filename format:
# 03-01-05-01-01-01-01.wav
#        ↑
#     Emotion code
emotion_map = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry"
}

# =========================
# CREATE OUTPUT FOLDERS
# =========================
os.makedirs(RAVDESS_OUTPUT, exist_ok=True)
for emotion in emotion_map.values():
    os.makedirs(os.path.join(RAVDESS_OUTPUT, emotion), exist_ok=True)

# =========================
# PROCESS FILES
# =========================
for actor in os.listdir(RAVDESS_INPUT):
    actor_path = os.path.join(RAVDESS_INPUT, actor)

    if not os.path.isdir(actor_path):
        continue

    for file in os.listdir(actor_path):
        if not file.endswith(".wav"):
            continue

        parts = file.split("-")
        if len(parts) < 3:
            continue

        emotion_code = parts[2]

        if emotion_code in emotion_map:
            emotion = emotion_map[emotion_code]
            src = os.path.join(actor_path, file)
            dst = os.path.join(RAVDESS_OUTPUT, emotion, file)
            shutil.copy(src, dst)

print("✅ RAVDESS converted to emotion-wise folders")
