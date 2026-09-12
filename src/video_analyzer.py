import cv2


def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    duration = frame_count / fps if fps > 0 else 0

    cap.release()

    return {
        "frame_count": frame_count,
        "fps": fps,
        "duration": duration
    }
if __name__ == "__main__":
    result = analyze_video("data/payasam_test.mp4")
    print(result)
def extract_middle_frame(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    middle_frame = frame_count // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)

    success, frame = cap.read()
    cap.release()

    if success:
        cv2.imwrite("data/middle_frame.jpg", frame)
        return True

    return False
if __name__ == "__main__":
    result = analyze_video("data/payasam_test.mp4")
    print(result)

    frame_result = extract_middle_frame("data/payasam_test.mp4")
    print("Middle frame extracted:", frame_result)