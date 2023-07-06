import cv2
import mediapipe as mp
from math import hypot
import screen_brightness_control as sbc
import numpy as np

mpHands = mp.solutions.hands
hands = mpHands.Hands(
	static_image_mode=False,
	model_complexity=1,
	min_detection_confidence=0.75,
	min_tracking_confidence=0.75,
	max_num_hands=2)

Draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
	# Read video frame by frame
	_, frame = cap.read()

	frame = cv2.flip(frame, 1)

	# Convert BGR image to RGB image
	frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# Process the RGB image
	Process = hands.process(frameRGB)

	landmarkList = []
	# if hands are present in image(frame)
	if Process.multi_hand_landmarks:
		# detect handmarks
		for handlm in Process.multi_hand_landmarks:
			for _id, landmarks in enumerate(handlm.landmark):
				# store height and width of image
				height, width, color_channels = frame.shape

				# calculate and append x, y coordinates
				# of handmarks from image(frame) to lmList
				x, y = int(landmarks.x*width), int(landmarks.y*height)
				landmarkList.append([_id, x, y])

			# draw Landmarks
			Draw.draw_landmarks(frame, handlm,
								mpHands.HAND_CONNECTIONS)

	if landmarkList != []:
		x1, y1 = landmarkList[4][1], landmarkList[4][2] #thumb
		x2, y2 = landmarkList[8][1], landmarkList[8][2] #index finger
  		#For circle at the tips of thumb and index finger
		cv2.circle(frame, (x1, y1), 7, (0, 255, 0), cv2.FILLED) #image #fingers #radius #rgb
		cv2.circle(frame, (x2, y2), 7, (0, 255, 0), cv2.FILLED) #image #fingers #radius #rgb
		cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3) #Line between tips of index finger and thumb

		L = hypot(x2-x1, y2-y1) #distance between tips using hypotenuse

		# Hand range 15 - 220, Brightness range 0 - 100
		b_level = np.interp(L, [15, 220], [0, 100])

		# set brightness
		sbc.set_brightness(int(b_level))
  

	cv2.imshow('Image', frame)
	if cv2.waitKey(1) & 0xff == ord('q'):
		break
