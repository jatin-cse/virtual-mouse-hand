import cv2
import mediapipe as mp
import pyautogui
import time
import math

# Screen size
screen_w, screen_h = pyautogui.size()

# Camera
cap = cv2.VideoCapture(0)

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Smooth movement
prev_x, prev_y = 0, 0
smoothening = 7

# Timing control
last_left_click = 0
last_right_click = 0
last_scroll_time = 0

click_delay = 1
scroll_delay = 0.3

def distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            lm = hand.landmark

            # Landmark positions
            ix, iy = int(lm[8].x * w), int(lm[8].y * h)     # Index tip
            mx, my = int(lm[12].x * w), int(lm[12].y * h)  # Middle tip
            tx, ty = int(lm[4].x * w), int(lm[4].y * h)    # Thumb tip

            # Mouse movement (index finger)
            screen_x = int(lm[8].x * screen_w)
            screen_y = int(lm[8].y * screen_h)

            curr_x = prev_x + (screen_x - prev_x) / smoothening
            curr_y = prev_y + (screen_y - prev_y) / smoothening

            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

            # Distances
            thumb_index_dist = distance(tx, ty, ix, iy)
            thumb_middle_dist = distance(tx, ty, mx, my)

            current_time = time.time()

            # LEFT CLICK (Thumb + Index)
            if thumb_index_dist < 30 and current_time - last_left_click > click_delay:
                pyautogui.click()
                last_left_click = current_time
                cv2.putText(frame, "LEFT CLICK", (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 3)

            # RIGHT CLICK (Thumb + Middle)
            if thumb_middle_dist < 30 and current_time - last_right_click > click_delay:
                pyautogui.rightClick()
                last_right_click = current_time
                cv2.putText(frame, "RIGHT CLICK", (30, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 0, 0), 3)

            # SCROLL (Index & Middle fingers)
            if abs(iy - my) < 30 and current_time - last_scroll_time > scroll_delay:
                if iy < h // 2:
                    pyautogui.scroll(40)   # Scroll up
                    cv2.putText(frame, "SCROLL UP", (30, 160),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 255), 3)
                else:
                    pyautogui.scroll(-40)  # Scroll down
                    cv2.putText(frame, "SCROLL DOWN", (30, 160),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 255), 3)

                last_scroll_time = current_time

            # Draw landmarks
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Gesture Mouse", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
