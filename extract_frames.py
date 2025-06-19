import os
import cv2
import json
import easyocr
from fuzzywuzzy import fuzz
from skimage.metrics import structural_similarity as ssim

currentdir = os.getcwd()

videopath = os.path.join(currentdir, "Videos")
framespath = os.path.join(currentdir, "Frames")

os.makedirs(videopath, exist_ok=True)
os.makedirs(framespath, exist_ok=True)

getAllVideos = os.listdir(videopath)

reader = easyocr.Reader(['en'])

for video in getAllVideos:
    if video.endswith(".mp4"):
        video_full_path = os.path.join(videopath, video)
        vidcap = cv2.VideoCapture(video_full_path)

        fps = vidcap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            print(f"Could not get FPS for {video}. Skipping.")
            continue

        frame_interval = int(round(fps / 1))  # extract 1 frame per second

        count = 0
        saved_frame_count = 0
        success, image = vidcap.read()

        imagetextdic = {}
        seen_texts = []  # Store tuples: (normalized_text, full_text)

        while success:
            if count % frame_interval == 0:
                frame_filename = os.path.join(framespath, f"{os.path.splitext(video)[0]}_frame{saved_frame_count}.jpg")
                cv2.imwrite(frame_filename, image)

                results = reader.readtext(image)
                textList = [text for _, text, _ in results]
                full_text = ', '.join(textList).strip()
                normalized_text = full_text.lower()

                # Text-based deduplication
                is_unique_text = True
                for existing_norm, _ in seen_texts:
                    if fuzz.ratio(normalized_text, existing_norm) >= 90:
                        is_unique_text = False
                        os.remove(frame_filename)
                        break

                if is_unique_text and normalized_text:
                    seen_texts.append((normalized_text, full_text))
                    imagetextdic[f"{os.path.splitext(video)[0]}_frame{saved_frame_count}"] = full_text
                    saved_frame_count += 1
                else:
                    try:
                        os.remove(frame_filename)
                    except:
                        pass

            success, image = vidcap.read()
            count += 1

        vidcap.release()
        print(f"Extracted {saved_frame_count} frames with <90% similar OCR from {video}.")

        # --- IMAGE-BASED DEDUPLICATION ---
        frame_files = sorted(
            [f for f in os.listdir(framespath) if f.startswith(os.path.splitext(video)[0])],
            key=lambda x: int(x.split("frame")[-1].replace(".jpg", ""))
        )

        unique_frames = []

        for i, frame_file in enumerate(frame_files):
            current_frame_path = os.path.join(framespath, frame_file)
            current_img = cv2.imread(current_frame_path)
            current_gray = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)

            is_unique_image = True

            for prev_frame in unique_frames:
                prev_frame_path = os.path.join(framespath, prev_frame)
                prev_img = cv2.imread(prev_frame_path)
                prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)

                # Resize if needed to ensure same dimensions
                if current_gray.shape != prev_gray.shape:
                    prev_gray = cv2.resize(prev_gray, (current_gray.shape[1], current_gray.shape[0]))

                similarity_index, _ = ssim(current_gray, prev_gray, full=True)

                if similarity_index >= 0.55:  # 70% similarity threshold
                    is_unique_image = False
                    os.remove(current_frame_path)
                    break

            if is_unique_image:
                unique_frames.append(frame_file)

        print(f"Removed visually similar frames. Remaining: {len(unique_frames)}")

        # Update JSON to keep only the final unique frames
        final_imagetextdic = {
            k: v for k, v in imagetextdic.items()
            if f"{k}.jpg" in unique_frames
        }

        output_json = f"{os.path.splitext(video)[0]}_final_unique_text.json"
        with open(output_json, "w", encoding="utf-8") as exportdic:
            json.dump(final_imagetextdic, exportdic, indent=2)

        print(f"Final JSON saved with only unique frames: {output_json}")

# this code is for remove duplicate frames from in video so we are using text extraction from an image and taking the similar text and removing the duplicate frames from which has similar text or accuracy above 70% and then we are using ssim to check the similarity between the frames.