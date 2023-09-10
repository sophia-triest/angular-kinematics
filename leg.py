# STEP 1: Import the necessary modules.
import mediapipe as mp
import cv2
from mediapipe.python.solutions.drawing_utils import draw_landmarks
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

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

  # Loop through the detected hands to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the hand landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto)

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in pose_landmarks]
    y_coordinates = [landmark.y for landmark in pose_landmarks]



  return annotated_image


base_options = python.BaseOptions(model_asset_path='pose_landmarker_heavy.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)



# STEP 3: Load the input image.
image = mp.Image.create_from_file("SDFGHJK.jpeg.png")

# STEP 4: Detect hand landmarks from the input image.
detection_result = detector.detect(image)



# STEP 5: Process the classification result. In this case, visualize it.
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)


#segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
#visualized_mask = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
#cv2.imshow(visualized_mask)


angle1 = detection_result.pose_landmarks[0][23]
angle2 = detection_result.pose_landmarks[0][25]
angle3 = detection_result.pose_landmarks[0][27]

angle_in_degrees = getAngle(
    [angle1.x, angle1.y, angle1.z],
    [angle2.x, angle2.y, angle2.z],
    [angle3.x, angle3.y, angle3.z]
)

print( " " + str(angle_in_degrees))

cv2.imshow("landabc", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
#cv2.destroyAllWindows()