import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def create_button(image, text, position, size):
    overlay = image.copy()
    cv2.rectangle(overlay, position, (position[0] + size[0], position[1] + size[1]), (0, 0, 255), -1)
    cv2.putText(overlay, text, (position[0] + 5, position[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.addWeighted(overlay, 0.4, image, 0.6, 0, image)

def get_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def finger_is_closed(finger_tip, finger_base, palm_center, threshold=0.1):
    tip_distance = np.linalg.norm(np.array([finger_tip.x, finger_tip.y]) - np.array([palm_center.x, palm_center.y]))
    base_distance = np.linalg.norm(np.array([finger_base.x, finger_base.y]) - np.array([palm_center.x, palm_center.y]))
    return tip_distance < base_distance + threshold

def recognize_letter(landmarks):
    thumb_tip, thumb_ip, thumb_mcp = landmarks[4], landmarks[3], landmarks[2]
    index_tip, index_pip, index_mcp = landmarks[8], landmarks[7], landmarks[5]
    middle_tip, middle_pip, middle_mcp = landmarks[12], landmarks[11], landmarks[9]
    ring_tip, ring_pip, ring_mcp = landmarks[16], landmarks[15], landmarks[13]
    pinky_tip, pinky_pip, pinky_mcp = landmarks[20], landmarks[19], landmarks[17]
    wrist = landmarks[0]

    def finger_is_closed(tip, pip, mcp, threshold=0.05):
        return distance(tip, wrist) < distance(pip, wrist) + threshold

    def distance(p1, p2):
        return np.linalg.norm(np.array([p1.x, p1.y, p1.z]) - np.array([p2.x, p2.y, p2.z]))

    def is_horizontal(p1, p2, threshold=0.1):
        return abs(p1.y - p2.y) < threshold

    def is_vertical(p1, p2, threshold=0.1):
        return abs(p1.x - p2.x) < threshold
    
def main():
    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands()
    button_position = (10, 10)
    button_size = (100, 40)

    fps_frames = deque(maxlen=30)  
    fps = 0

    while True:
        frame_start_time = time.time()

        success, image = cap.read()
        if not success:
            print("webcam?!?!?!?")
            break

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_label = "Left" if handedness.classification[0].label == "Left" else "Right"
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS)

                landmarks = hand_landmarks.landmark
                letter = recognize_letter(landmarks)

                x_coords = [landmark.x for landmark in landmarks]
                y_coords = [landmark.y for landmark in landmarks]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                h, w, _ = image.shape
                bbox_start = (int(x_min * w), int(y_min * h))
                bbox_end = (int(x_max * w), int(y_max * h))
                cv2.rectangle(image, bbox_start, bbox_end, (0, 255, 0), 2)
                cv2.putText(image, f"{hand_label} Hand: Letter {letter}", (bbox_start[0], bbox_start[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        create_button(image, "fixe", button_position, button_size)

        
        frame_end_time = time.time()
        fps_frames.append(frame_end_time - frame_start_time)
        if len(fps_frames) == 30:
            fps = int(30 / sum(fps_frames))

        # Display FPS
        cv2.putText(image, f"FPS: {fps}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('camera', image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        if cv2.getWindowProperty('camera', cv2.WND_PROP_VISIBLE) >= 1:
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                mouse = cv2.getWindowProperty('camera', cv2.WND_PROP_ASPECT_RATIO)
                if mouse != -1:
                    x, y = int(mouse) & 0xffff, (int(mouse) >> 16) & 0xffff
                    if button_position[0] < x < button_position[0] + button_size[0] and \
                       button_position[1] < y < button_position[1] + button_size[1]:
                        break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

if __name__ == "__main__":
    main()