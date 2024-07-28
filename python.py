import cv2
import mediapipe as mp
import pyautogui
import numpy as np


# Initialize the webcam
webcam = cv2.VideoCapture(0)
my_hands = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils

x1 = y1 = x2 = y2 = 0

while True:
    # Read frame from the webcam
    _, image = webcam.read()
    image= cv2.flip(image, 1)
    frame_height, frame_width, _ = image.shape

    # Convert the image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and find hands
    output = my_hands.process(rgb_image)
    hands = output.multi_hand_landmarks

    # Draw hand landmarks
    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(image, hand)
            landmarks = hand.landmark
            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                if id == 8:
                    cv2.circle(img=image, center=(x, y), radius=8, color=(0, 255, 255), thickness=3)
                    x1 = x
                    y1 = y
                if id == 4:
                    cv2.circle(img=image, center=(x, y), radius=8, color=(0, 0, 255), thickness=3)
                    x2 = x
                    y2 = y

                    # Draw a line between the two points
                    cv2.line(img=image, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=3)

            # Calculate the distance between the points
            dist = np.linalg.norm(np.array([x2, y2]) - np.array([x1, y1]))
            print(f"Distance: {dist}")

            # Volume control
            if dist > 50:
                pyautogui.press("volumeup")
            else:
                pyautogui.press("volumedown")

    # Display the image with landmarks
    cv2.imshow('Volume control using hand gesture', image)

    # Exit loop when 'Esc' is pressed
    key = cv2.waitKey(10)
    if key == 27:  # 27 is the ASCII code for the Esc key
        break

# Release the webcam and close the window
webcam.release()
cv2.destroyAllWindows()


