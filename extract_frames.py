import os, json, easyocr, cv2

currentdir = os.getcwd()

if not os.path.isdir(os.path.join(currentdir, "Videos")):
    os.mkdir(os.path.join(currentdir, "Videos"))
if not os.path.isdir(os.path.join(currentdir, "Frames")):
    os.mkdir(os.path.join(currentdir, "Frames"))

videopath = os.path.join(currentdir, "Videos")
framespath = os.path.join(currentdir, "Frames")
getAllVideos = os.listdir(videopath)

for video in getAllVideos:
    if video.endswith(".mp4"):
        video_full_path = os.path.join(videopath, video)
        vidcap = cv2.VideoCapture(video_full_path)

        fps = vidcap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            print(f"Could not get FPS for {video}. Skipping.")
            continue

        frame_interval = int(round(fps / 1))

        count = 0
        saved_frame_count = 0
        success, image = vidcap.read()

        imagetextdic = {}
        while success:
            if count % frame_interval == 0:
                frame_filename = os.path.join(framespath, f"{os.path.splitext(video)[0]}_frame{saved_frame_count}.jpg")
                cv2.imwrite(frame_filename, image)
                saved_frame_count += 1

            image_path = frame_filename
            image = cv2.imread(image_path)

            reader = easyocr.Reader(['en'])

            while success:
                if count % frame_interval == 0:
                    frame_filename = os.path.join(framespath, f"{os.path.splitext(video)[0]}_frame{saved_frame_count}.jpg")
                    cv2.imwrite(frame_filename, image)

                    results = reader.readtext(image)

                    textList = [text for _, text, _ in results]

                    print('textList: ', textList)

                    imagetextdic[f"{os.path.splitext(video)[0]}_frame{saved_frame_count}"] = ', '.join(textList)

                    saved_frame_count += 1

                success, image = vidcap.read()
                count += 1
        vidcap.release()
        print(f"Extracted {saved_frame_count} frames from {video} at 1 fps.")

        with open(f"{os.path.splitext(video)[0]}_text.json", "w", encoding="utf-8") as exportdic:
            json.dump(imagetextdic, exportdic, indent=2)
