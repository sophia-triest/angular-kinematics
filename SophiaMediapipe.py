# STEP 1: Import the necessary modules.
import mediapipe as mp
import cv2
from mediapipe.python.solutions.drawing_utils import draw_landmarks
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

#obsolete code
def getAngle(a, b, c):
  ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]]
  bc = [c[0] - b[0], c[1] - b[1], c[2] - b[2]]

  abVec = math.sqrt(ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2])
  bcVec = math.sqrt(bc[0] * bc[0] + bc[1] * bc[1] + bc[2] * bc[2])

  abNorm = [ab[0] / abVec, ab[1] / abVec, ab[2] / abVec]
  bcNorm = [bc[0] / bcVec, bc[1] / bcVec, bc[2] / bcVec]

  res = abNorm[0] * bcNorm[0] + abNorm[1] * bcNorm[1] + abNorm[2] * bcNorm[2]

  return math.acos(res) * 180.0 / 3.141592653589793;


def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image

# STEP 2: Create an HandLandmarker object.
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# STEP 3: Load the input image.
image = mp.Image.create_from_file("mediapipehandtest1.jpg")

# STEP 4: Detect hand landmarks from the input image.
detection_result = detector.detect(image)

# STEP 5: Process the classification result. In this case, visualize it.
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
#cv2.imshow("landabc",cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))


#cv2.waitKey(0)
#cv2.destroyAllWindows()

import math

angle1 = detection_result.hand_landmarks[0][1]
angle2 = detection_result.hand_landmarks[0][2]
angle3 = detection_result.hand_landmarks[0][3]

angle_in_degrees = getAngle(
  [angle1.x, angle1.y, angle1.z],
  [angle2.x, angle2.y, angle2.z],
  [angle3.x, angle3.y, angle3.z],
)


getAngle(angle1, angle2, angle3)

