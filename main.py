import os
import cv2
import mediapipe as mp
import math
import numpy as np


def bool_array_to_decimal(arr):
    binary_str = ''.join(['1' if x else '0' for x in arr])
    return int(binary_str, 2)


def is_palm_facing_forwards(landmarks, left):
    # Extract three palm points
    wrist = np.array([landmarks.landmark[0].x, landmarks.landmark[0].y,
                      landmarks.landmark[0].z])
    thumb = np.array([landmarks.landmark[5].x, landmarks.landmark[5].y,
                      landmarks.landmark[5].z])
    index = np.array([landmarks.landmark[17].x, landmarks.landmark[17].y,
                      landmarks.landmark[17].z])

    # Calculate the normal vector of the plane using the cross product of two vectors on the plane
    v1 = thumb - wrist
    v2 = index - wrist
    normal = np.cross(v1, v2)

    # Check the sign of the dot product of the normal vector and a vector pointing forwards
    forwards = np.array([0, 0, -1])  # Assuming forward direction is in the negative z direction
    dot_product = np.dot(normal, forwards)
    if left:
        return dot_product >= 0  # Palm is facing forwards
    else:
        return dot_product < 0  # Palm is facing backwards


def get_raised_fingers(hand_landmarks, left, forward):
    # Define the finger joint indices
    finger_joints = [[20, 19, 18], [16, 15, 14], [12, 11, 10], [8, 7, 6]]
    thumb_joint = [[4, 3]]

    # Initialize list of raised fingers
    rised_figers = [False] * 5

    # Calculate the angle between the thumb joints
    joint1 = hand_landmarks.landmark[thumb_joint[0][0]]
    joint2 = hand_landmarks.landmark[thumb_joint[0][1]]
    angle = abs(math.degrees(math.atan2(joint1.y - joint2.y, joint1.x - joint2.x)))

    # Check if the thumb is raised based on the angle
    if ((left and forward) or (not left and not forward)) and angle < 80:
        rised_figers[4] = True
    elif ((not left and forward) or (left and not forward)) and angle > 120:
        rised_figers[0] = True

    # Check if each finger is raised or not
    for finger_idx in range(4):
        # Calculate the angle between the finger joints
        joint1 = hand_landmarks.landmark[finger_joints[finger_idx][0]]
        joint2 = hand_landmarks.landmark[finger_joints[finger_idx][1]]
        joint3 = hand_landmarks.landmark[finger_joints[finger_idx][2]]
        angle = math.degrees(math.atan2(joint3.y - joint2.y, joint3.x - joint2.x) -
                             math.atan2(joint1.y - joint2.y, joint1.x - joint2.x))
        # Check if the finger is raised based on the angle
        if angle > 0:
            if (left and forward) or (not left and not forward):
                rised_figers[finger_idx] = True
            elif (not left and forward) or (left and not forward):
                rised_figers[4 - finger_idx] = True

    return rised_figers


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Video capturer
cap = cv2.VideoCapture(0)

# Hand model params
with mp_hands.Hands(
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.9,
        static_image_mode=True) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        # To improve performance, optionally mark the image as not writeable to pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # If hand detected
        if results.multi_hand_landmarks:
            # For each hand
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Get the handedness of the current hand
                handedness = results.multi_handedness[i]
                left = handedness.classification[0].label == "Left"
                # Get the orientation of the hand
                forward = is_palm_facing_forwards(hand_landmarks, left)
                # Get rised fingers
                rised = get_raised_fingers(hand_landmarks, left, forward)
                # Print landmarks
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # Print the result
                y = int(image.shape[0] - 10)
                for j in range(5):
                    x = int(image.shape[1] / 2 - 50 * (j - 2))
                    if not left:
                        x += 120
                    else:
                        x -= 180
                    txt = "0"
                    if rised[4 - j]:
                        txt = "1"
                    cv2.putText(image, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
                x = int(image.shape[1] / 2 + 40 * 3)
                if not left:
                    x += 120
                else:
                    x -= 180
                cv2.putText(image, "=" + str(bool_array_to_decimal(rised)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (255, 255, 255), 1, cv2.LINE_AA)
        # Display the image
        cv2.imshow('Binary counter', image)
        # While !ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
