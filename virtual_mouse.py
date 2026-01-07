import cv2
import mediapipe as mp
import pyautogui
import time

# Screen size
screen_w, screen_h = pyautogui.size()

# Camera
cap = cv2.VideoCapture(0)

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Smoothing
prev_x, prev_y = 0, 0
smoothening = 7

# Click control
last_click_time = 0
click_delay = 1  # seconds

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

            # Index finger tip
            x1 = int(lm[8].x * w)
            y1 = int(lm[8].y * h)

            # Thumb tip
            x2 = int(lm[4].x * w)
            y2 = int(lm[4].y * h)

            # Convert camera coords to screen coords
            screen_x = int(lm[8].x * screen_w)
            screen_y = int(lm[8].y * screen_h)

            # Smooth mouse movement
            curr_x = prev_x + (screen_x - prev_x) / smoothening
            curr_y = prev_y + (screen_y - prev_y) / smoothening

            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

            # Draw landmarks
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            # Distance between thumb & index (for click)
            distance = ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5

            # Click when pinched
            current_time = time.time()
            if distance < 30 and current_time - last_click_time > click_delay:
                pyautogui.click()
                last_click_time = current_time
                cv2.putText(frame, "CLICK", (40, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 3)

    cv2.imshow("Hand Gesture Mouse", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
