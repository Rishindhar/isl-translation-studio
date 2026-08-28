import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from process_video import process_isl_video

def create_multi_gesture_video(output_path="isl_multi_gesture_input.mp4", duration_sec=6):
    """
    Creates a multi-gesture test video containing 4 sequential gesture movements:
    1. 'hello' wave
    2. 'thank_you' forward-down motion
    3. 'yes' fist nod
    4. 'help' hand elevation
    """
    fps = 30
    total_frames = duration_sec * fps
    width, height = 640, 480

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"\n1. Creating multi-gesture input video '{output_path}' ({total_frames} frames)...")

    for frame_idx in range(total_frames):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:] = (20, 20, 25)  # Dark theme background

        t = (frame_idx % 45) / 45.0  # 45 frames per gesture sequence
        gesture_phase = frame_idx // 45

        center_x = 320
        center_y = 240

        if gesture_phase == 0:
            # Gesture 1: 'hello' (side-to-side oscillation)
            center_x += int(100 * np.sin(4 * np.pi * t))
            label_text = "Input Sequence 1: Hello Waving Gesture"
        elif gesture_phase == 1:
            # Gesture 2: 'thank_you' (moving down & forward)
            center_y += int(80 * t)
            center_x += int(40 * t)
            label_text = "Input Sequence 2: Thank You Gesture"
        elif gesture_phase == 2:
            # Gesture 3: 'yes' (vertical nodding)
            center_y += int(60 * np.sin(3 * np.pi * t))
            label_text = "Input Sequence 3: Yes Nodding Gesture"
        else:
            # Gesture 4: 'help' (upward lifting motion)
            center_y -= int(100 * t)
            label_text = "Input Sequence 4: Help Lifting Gesture"

        # Draw skin-tone hand shape with fingers
        cv2.circle(img, (center_x, center_y), 45, (180, 210, 240), -1)
        cv2.circle(img, (center_x, center_y + 70), 20, (180, 210, 240), -1)

        for fx in [-30, -10, 10, 30]:
            cv2.line(img, (int(center_x + fx), int(center_y)),
                     (int(center_x + fx * 1.2), int(center_y - 75)), (180, 210, 240), 10)
        cv2.line(img, (int(center_x - 30), int(center_y + 10)),
                 (int(center_x - 65), int(center_y - 20)), (180, 210, 240), 10)

        # On-screen input watermark
        cv2.putText(img, label_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(img, f"Frame {frame_idx + 1}/{total_frames}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        out.write(img)

    out.release()
    print(f"   Created video: '{output_path}'")

def main():
    in_video = "isl_multi_gesture_input.mp4"
    out_video = "isl_multi_gesture_translated.mp4"

    create_multi_gesture_video(in_video, duration_sec=6)

    print(f"\n2. Running ISL Video Translation Pipeline on '{in_video}'...")
    process_isl_video(in_video, out_video, lang="en")

    print("\n3. Running Hindi Language Video Translation Pipeline...")
    out_video_hi = "isl_multi_gesture_translated_hindi.mp4"
    process_isl_video(in_video, out_video_hi, lang="hi")

    print("\n=======================================================")
    print("  Video Input Processing Run Completed Successfully!")
    print(f"  English Annotated Output: '{out_video}'")
    print(f"  Hindi Annotated Output  : '{out_video_hi}'")
    print("=======================================================\n")

if __name__ == "__main__":
    main()
