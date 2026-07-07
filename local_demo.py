import cv2
from rec_lib import FaceRecognition

def main():
    fr = FaceRecognition(db_file="faces.db")

    cap = cv2.VideoCapture(0)  #kamera
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        results = fr.process_frame(frame)

        for res in results:
            x1, y1, x2, y2 = res["bbox"]
            label = res["label"]
            score = res["score"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame, f"{label}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0), 2
            )
        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()