import mediapipe as mp
import cv2
import numpy as np
import os
import json

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

video_folder = 'word videos/'

json_file = open('reference.json', 'w')

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    data = {}

    for idx, filename in enumerate(os.listdir(video_folder)):
        if filename.endswith(".mp4"):
            video_file = os.path.join(video_folder, filename)

            file_name = filename.split('.')[0]
            data[file_name] = []

            cap = cv2.VideoCapture(video_file)

            frame_number = 0

            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    break

                frame = cv2.resize(frame, (800, 750))

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                results = hands.process(image_rgb)

                if results.multi_hand_landmarks:
                    hand_coordinates = []

                    for hand_landmarks in results.multi_hand_landmarks:
                        hand = hand_landmarks.landmark

                        if hand[mp_hands.HandLandmark.WRIST].x < hand[mp_hands.HandLandmark.THUMB_CMC].x:
                            handedness = "Left"
                        else:
                            handedness = "Right"

                        for joint_id, landmark in enumerate(hand):
                            x, y, z = landmark.x, landmark.y, landmark.z
                            joint_data = {
                                "Joint Index": joint_id,
                                "Coordinates": [x, y, z]
                            }
                            hand_coordinates.append(joint_data)

                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                                  mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2,
                                                                         circle_radius=4),
                                                  mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2,
                                                                         circle_radius=2))

                    data[file_name].append({
                        "Frame": frame_number,
                        "Left Hand Coordinates": hand_coordinates if handedness == "Left" else [],
                        "Right Hand Coordinates": hand_coordinates if handedness == "Right" else []
                    })

                    frame_number += 1

                cv2.imshow('Video', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()

        print(f"Processing video {idx + 1} of {len(os.listdir(video_folder))}: {file_name}")

    json.dump(data, json_file)

json_file.close()

json_file = open('reference.json', 'r')
data = json.load(json_file)
json_file.close()

for word in data:
    frames = data[word]
    num_frames = len(frames)

    if num_frames > 1:
        interpolated_frames = []

        for i in range(num_frames - 1):
            current_frame = frames[i]
            next_frame = frames[i + 1]

            if next_frame["Frame"] - current_frame["Frame"] > 1:
                gap = next_frame["Frame"] - current_frame["Frame"]

                for j in range(1, gap):
                    interpolation_ratio = j / gap

                    interpolated_coordinates = []

                    for joint_data in current_frame["Left Hand Coordinates"]:
                        current_coordinates = joint_data["Coordinates"]
                        next_coordinates = next_frame["Left Hand Coordinates"][joint_data["Joint Index"]]["Coordinates"]

                        interpolated_coordinates.append({
                            "Joint Index": joint_data["Joint Index"],
                            "Coordinates": [
                                current_coordinates[0] + (next_coordinates[0] - current_coordinates[0]) * interpolation_ratio,
                                current_coordinates[1] + (next_coordinates[1] - current_coordinates[1]) * interpolation_ratio,
                                current_coordinates[2] + (next_coordinates[2] - current_coordinates[2]) * interpolation_ratio
                            ]
                        })

                    interpolated_frames.append({
                        "Frame": current_frame["Frame"] + j,
                        "Left Hand Coordinates": interpolated_coordinates,
                        "Right Hand Coordinates": []
                    })

        frames.extend(interpolated_frames)

json_file = open('reference.json', 'w')
json.dump(data, json_file)
json_file.close()