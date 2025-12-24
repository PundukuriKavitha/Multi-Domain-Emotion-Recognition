import os
import shutil

# =========================
# PATHS (EDIT IF NEEDED)
# =========================
TESS_INPUT = "TESS Toronto emotional speech set data"
TESS_OUTPUT = "data/target/TESS"

# =========================
# EMOTIONS TO KEEP
# =========================
valid_emotions = {
    "angry": "angry",
    "happy": "happy",
    "sad": "sad",
    "neutral": "neutral"
}

# =========================
# CREATE OUTPUT FOLDERS
# =========================
os.makedirs(TESS_OUTPUT, exist_ok=True)
for emotion in valid_emotions.values():
    os.makedirs(os.path.join(TESS_OUTPUT, emotion), exist_ok=True)

# =========================
# PROCESS FILES
# =========================
for folder in os.listdir(TESS_INPUT):
    folder_path = os.path.join(TESS_INPUT, folder)

    if not os.path.isdir(folder_path):
        continue

    folder_lower = folder.lower()

    for key in valid_emotions:
        if key in folder_lower:
            emotion = valid_emotions[key]

            for file in os.listdir(folder_path):
                if file.endswith(".wav"):
                    src = os.path.join(folder_path, file)
                    dst = os.path.join(TESS_OUTPUT, emotion, file)
                    shutil.copy(src, dst)

print("✅ TESS converted to emotion-wise folders")