import cv2
import subprocess
from rec_lib import FaceRecognition

def get_youtube_stream(url):
    cmd = [
        "yt-dlp",
        "-f", "best[ext=mp4]/best",
        "-g", url  # -g kiirja a video urlt
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout.strip()

def main():
    fr = FaceRecognition(db_file="faces.db")
    url = "https://www.youtube.com/watch?v=gFRtAAmiFbE"

    stream_url = get_youtube_stream(url)
    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        print("E: Could not open YouTube stream.")
        return

    print("q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame") #gatya
            break

        results = fr.process_frame(frame)

        for res in results:
            x1, y1, x2, y2 = res["bbox"]
            label = res["label"]
            score = res["score"]


            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)
            print(label)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()