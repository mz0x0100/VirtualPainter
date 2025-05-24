import cv2
import time
import os
import numpy as np
import requests
import detectlib as dlib

###############################
cam_width = 1280
cam_height = 720

# red by default
brush_color = (0, 0, 255)
brush_thickness = 20
eraser_thickness = 150
################################

cap = cv2.VideoCapture(0)
cap.set(3, cam_width)
cap.set(4, cam_height)
# cap.set(cv2.CAP_PROP_FPS, 180)

path = 'graphics'
overlay_list = []
files = sorted(os.listdir(path))

for file in files:
    image = cv2.imread(f'{path}/{file}')
    overlay_list.append(image)

header = overlay_list[0]

detector = dlib.HandDetector(min_detection_confidence=0.5)
img_canvas = np.zeros((cam_height, cam_width, 3), np.uint8)

p_time = 0
xp, yp = 0, 0

# url = 'http://192.168.43.162:8080/shot.jpg'

p_time = 0
while True:
    # 1. Import image
    success, img = cap.read()
    # img = cv2.flip(img, 1)
    # img_res = requests.get(url)
    # img_arr = np.array(bytearray(img_res.content), dtype=np.uint8)
    # img = cv2.imdecode(img_arr, -1)
    img = cv2.resize(img, (cam_width, cam_height))
    img = cv2.flip(img, 1)

    # Set the header image
    img[0:110, 0:cam_width] = header

    # 2. Find Hand Landmarks
    img = detector.detect_hands(img, draw_color=brush_color)
    lmList = detector.find_position(img, draw=False)

    if len(lmList) != 0:

        detector.draw_circle(img, dlib.INDEX_FINGER_TIP, color=brush_color)
        # tip of index and middle fingers
        x1, y1 = lmList[dlib.INDEX_FINGER_TIP][1:]
        x2, y2 = lmList[dlib.MIDDLE_FINGER_TIP][1:]

        # 3. Check which fingers are up
        fingers = detector.fingers_up()

        # 4. When two fingers are up -> Selection Mode
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0

            # Check if at header
            if y1 < 110:
                if 290 < x1 < 420:
                    header = overlay_list[0]
                    brush_color = (0, 0, 255)
                elif 460 < x1 < 590:
                    header = overlay_list[1]
                    brush_color = (0, 255, 0)
                elif 660 < x1 < 790:
                    header = overlay_list[2]
                    brush_color = (255, 0, 0)
                elif 860 < x1 < 990:
                    header = overlay_list[3]
                    brush_color = (255, 0, 255)
                elif 1000 < x1 < 1140:
                    header = overlay_list[4]
                    brush_color = (0, 0, 0)

            cv2.rectangle(img, (x1, y1), (x2, y2), brush_color, cv2.FILLED)

        # 5. When Index Finger is up -> Drawing Mode
        if fingers[1] and not fingers[2]:
            cv2.circle(img, (x1, y1), 15, brush_color, cv2.FILLED)

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if brush_color == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), brush_color, eraser_thickness)
                cv2.line(img_canvas, (xp, yp), (x1, y1), brush_color, eraser_thickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), brush_color, brush_thickness)
                cv2.line(img_canvas, (xp, yp), (x1, y1), brush_color, brush_thickness)
            print('Drawing mode')
            xp, yp = x1, y1

    img_gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
    _, img_inverse = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_inverse = cv2.cvtColor(img_inverse, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, img_inverse)
    img = cv2.bitwise_or(img, img_canvas)

    cv2.imshow("Image", img)
    # cv2.imshow("Canvas", img_canvas)
    cv2.waitKey(1)
