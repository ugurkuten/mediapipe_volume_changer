import cv2
import mediapipe as mp
import pyautogui as pygui

hands = mp.solutions.hands.Hands()
draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def volume(frame, hand):
    pos1=hand.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
    pos2=hand.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
    dist= pos1.y-pos2.y
    cv2.line(frame, (int(pos1.x*frame.shape[1]), int(pos1.y*frame.shape[0])), (int(pos2.x*frame.shape[1]), int(pos2.y*frame.shape[0])), (0,0,255), 3)
    if dist<0.1:
        pygui.press("volumedown")
    if dist>0.1: 
        pygui.press("volumeup")   
    print("Distance" , dist)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            draw.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
            volume(frame,hand)
    
    cv2.imshow("Hands", frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
