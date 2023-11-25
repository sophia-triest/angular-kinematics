import mediapipe as mp
import cv2
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import os


def getAngle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
        annotated_image,
        pose_landmarks_proto,
        solutions.pose.POSE_CONNECTIONS,
        solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image


BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
model_path = 'pose_landmarker_heavy.task'

# Create a pose landmarker instance with the video mode:
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO, min_tracking_confidence=0.9)

filename = './5.png'
frameCount = 0
with PoseLandmarker.create_from_options(options) as landmarker:
    img = cv2.imread(filename)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    detect = landmarker.detect_for_video(mp_image, frameCount)

    labeled = draw_landmarks_on_image(img, detect)
    scale_percent = 50  # percent of original size
    width = int(labeled.shape[1] * scale_percent / 100)
    height = int(labeled.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(labeled, dim, interpolation=cv2.INTER_AREA)
    cv2.imshow('video', resized)

    point1 = detect.pose_landmarks[0][24]
    point2 = detect.pose_landmarks[0][26]
    point3 = detect.pose_landmarks[0][28]
    angle_in_degrees = getAngle(
        [point1.x, point1.y, point1.z],
        [point2.x, point2.y, point2.z],
        [point3.x, point3.y, point3.z]
    )
    print(angle_in_degrees)
    cv2.waitKey()

