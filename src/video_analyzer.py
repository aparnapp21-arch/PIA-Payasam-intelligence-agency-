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


def calculate_motion(frame1, frame2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    difference = cv2.absdiff(gray1, gray2)

    motion_score = difference.mean()

    return motion_score


if __name__ == "__main__":
    video_path = "data/payasam_test.mp4"

    result = analyze_video(video_path)
    print(result)

    frame_result = extract_middle_frame(video_path)
    print("Middle frame extracted:", frame_result)

    cap = cv2.VideoCapture(video_path)

    motion_scores = []

    if cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, 200)

        success, previous_frame = cap.read()

        if success:
            for i in range(10):
                success, current_frame = cap.read()

                if success:
                    motion = calculate_motion(
                        previous_frame,
                        current_frame
                    )

                    motion_scores.append(motion)
                    previous_frame = current_frame

        cap.release()

    print("Motion scores:", motion_scores)

    if motion_scores:
        average_motion = sum(motion_scores) / len(motion_scores)
        print("Average motion:", average_motion)