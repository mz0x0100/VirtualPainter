import enum
import time
import cv2
import mediapipe as mp

'''
    Hand landmarks
'''
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_FINGER_MCP = 5
INDEX_FINGER_PIP = 6
INDEX_FINGER_DIP = 7
INDEX_FINGER_TIP = 8
MIDDLE_FINGER_MCP = 9
MIDDLE_FINGER_PIP = 10
MIDDLE_FINGER_DIP = 11
MIDDLE_FINGER_TIP = 12
RING_FINGER_MCP = 13
RING_FINGER_PIP = 14
RING_FINGER_DIP = 15
RING_FINGER_TIP = 16
PINKY_FINGER_MCP = 17
PINKY_FINGER_PIP = 18
PINKY_FINGER_DIP = 19
PINKY_FINGER_TIP = 20


class HandDetector:

    def __init__(self, static_image_mode=False, max_num_hands=2, model_complexity=1, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):

        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode, max_num_hands, model_complexity, min_detection_confidence,
                                         min_tracking_confidence)
        self.mp_draw = mp.solutions.drawing_utils
        self.detected_hands = []
        self.tip_ids = [THUMB_TIP, INDEX_FINGER_TIP, MIDDLE_FINGER_TIP, RING_FINGER_TIP, PINKY_FINGER_TIP]
        self.landmark_list = []

    def detect_hands(self, img, show_hlabel=True, draw_landmarks=False, draw_rect=True, angle_thickness=5, angle_length=20,
                     draw_color=(0, 255, 0), draw_thickness=1):

        # Convert image to rgb and process it
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.detected_hands = self.hands.process(img_rgb)

        hands = []
        landmarks_list = []
        centers = []
        classes = []
        # if any hand landmark is detected
        if self.detected_hands.multi_hand_landmarks:
            for landmarks, hand_class in zip(self.detected_hands.multi_hand_landmarks,
                                             self.detected_hands.multi_handedness):
                ih, iw, ic = img.shape
                lm_list = []
                x_list = []
                y_list = []
                for lm_id, landmark in enumerate(landmarks.landmark):
                    """
                        - Calculate point x by multiplying the landmark coordinate x by the image width
                        - Calculate point y by multiplying the landmark coordinate y by the image height
                        - Calculate point z by multiplying the landmark coordinate z by the image width
                        
                    """
                    px = int(landmark.x * iw)
                    py = int(landmark.y * ih)
                    pz = int(landmark.z * iw)
                    x_list.append(px)
                    y_list.append(py)

                    lm_list.append([lm_id, px, py, pz])

                """
                    #######################################################
                        CALCULATING THE FACE BOUNDING BOX
                    #######################################################
                    
                    1. Find the landmark width maximum value of x and y coordinates
                    2. Find the landmark width minimum value of x and y coordinates
                    3. Find the width bounding box by subtracting max x by min x
                    4. Find the height bounding box by subtracting max y by min y
                """
                x_max, x_min = max(x_list), min(x_list)
                y_max, y_min = max(y_list), min(y_list)
                bbox_width = x_max - x_min
                bbox_height = y_max - y_min

                bounding_box = x_min, y_min, bbox_width, bbox_height

                # draw landmarks
                if draw_landmarks:
                    self.mp_draw.draw_landmarks(img, landmarks, self.mp_hands.HAND_CONNECTIONS)

                if draw_rect:
                    # Increase the size of the bounding box
                    draw_bbox = (bounding_box[0] - 20, bounding_box[1] - 20, (x_max - (bounding_box[0] - 20)) + 20,
                                 (y_max - (bounding_box[1] - 20)) + 20)
                    # x and y coordinates for drawing lines
                    x1, y1, bw, bh = draw_bbox
                    x2, y2 = x1 + bw, y1 + bh

                    cv2.rectangle(img, draw_bbox, draw_color, draw_thickness)

                    # Top-Left Points -> x1, y1
                    cv2.line(img, (x1, y1), (x1 + angle_length, y1), draw_color, angle_thickness)
                    cv2.line(img, (x1, y1), (x1, y1 + angle_length), draw_color, angle_thickness)

                    # Top-Right Points -> x2, y1
                    cv2.line(img, (x2, y1), (x2 - angle_length, y1), draw_color, angle_thickness)
                    cv2.line(img, (x2, y1), (x2, y1 + angle_length), draw_color, angle_thickness)

                    # Bottom-Left Points -> x1, y2
                    cv2.line(img, (x1, y2), (x1, y2 - angle_length), draw_color, angle_thickness)
                    cv2.line(img, (x1, y2), (x1 + angle_length, y2), draw_color, angle_thickness)

                    # Bottom-Right Points -> x2, y2
                    cv2.line(img, (x2, y2), (x2, y2 - angle_length), draw_color, angle_thickness)
                    cv2.line(img, (x2, y2), (x2 - angle_length, y2), draw_color, angle_thickness)

                    # if show_hlabel:

        return img

    def hands_count(self):
        if self.detected_hands.multi_hand_landmarks:
            return len(self.detected_hands.multi_hand_landmarks)
        return 0

    def find_position(self, img, hand_index=0, draw=True, draw_color=(255, 0, 0), draw_thickness=15,
                      draw_size=15):

        self.landmark_list = []
        if self.detected_hands.multi_hand_landmarks:
            # get the landmarks at the hand_index
            hand = self.detected_hands.multi_hand_landmarks[hand_index]
            for lm_id, lm in enumerate(hand.landmark):
                # get the image height, width and channel
                ih, iw, ic = img.shape

                # calculate the x and y coordinate in pixels
                cx = int(lm.x * iw)
                cy = int(lm.y * ih)

                self.landmark_list.append([lm_id, cx, cy])
                if draw:
                    # Draw a circle on that landmark
                    cv2.circle(img, (cx, cy), draw_size, draw_color, draw_thickness)

        return self.landmark_list

    def detected_hands_count(self):
        detected_hands = None
        if self.detected_hands.multi_hand_landmarks:
            detected_hands = len(self.detected_hands.multi_hand_landmarks)

        return detected_hands

    def draw_circle(self, img, landmark_index, hand_index=0, color=(0, 255, 0), thickness=cv2.FILLED, size=15):

        lm_list = self.find_position(img, hand_index, draw=False)
        if len(lm_list) != 0:
            cx = lm_list[landmark_index][1]
            cy = lm_list[landmark_index][2]
            cv2.circle(img, (cx, cy), size, color, thickness)
            cv2.circle(img, (cx, cy), size + 5, color, 2)

    def fingers_up(self):
        """
        Checks which fingers are up by calculating the distances between the tips of the fingers and PIP
        :return:
        Array containing values (0 or 1), 1 means that the finger is up and 0 if finger's down, the array represents
        the state for the THUMB_TIP, INDEX_FINGER_TIP MIDDLE_FINGER_TIP, RING_FINGER_TIP and PINKY_FINGER_TIP
        respectively
        """

        fingers = []

        if self.landmark_list[self.tip_ids[0] - 1][1] > \
                self.landmark_list[self.tip_ids[0]][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for tid in range(1, len(self.tip_ids)):
            if self.landmark_list[self.tip_ids[tid] - 2][2] > self.landmark_list[self.tip_ids[tid]][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    # def is_finger_up(self, finger_tip_id):
    #
    #     # validate that finger_tip_id is a valid tip id
    #     # if finger_tip_id not in self.tip_ids:
    #     #     raise f"{finger_tip_id} is not a recognized finger tip"
    #
    #     # Check if the finger_tip_id belongs to the thumb
    #     if finger_tip_id == THUMB_TIP:
    #         if self.landmark_list[self.tip_ids[0] - 1][1] > \
    #                 self.landmark_list[self.tip_ids[0]][1]:
    #             return True
    #         else:
    #             return False
    #
    #     if self.landmark_list[self.tip_ids[finger_tip_id] - 2][2] > \
    #             self.landmark_list[self.tip_ids[finger_tip_id]][2]:
    #         return True
    #     else:
    #         return False


class FaceDetector:

    def __init__(self, min_detection_confidence=0.5, model_selection=0):
        self.min_detection_confidence = min_detection_confidence
        self.model_selection = model_selection

        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(self.min_detection_confidence, self.model_selection)
        self.detected_faces = []

    def detect_faces(self, img, draw=True):
        """

        :param img: The image to detect faces on
        :param draw: Draw the detected faces
        :return: Image, faces bounding box array
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.detected_faces = self.face_detection.process(img_rgb)

        bboxs = []
        if self.detected_faces.detections:
            for detection_id, detection in enumerate(self.detected_faces.detections):

                ih, iw, ic = img.shape
                # relative bounding box
                r_bbox = detection.location_data.relative_bounding_box

                # calculate the bounding box
                bbox = int(r_bbox.xmin * iw), int(r_bbox.ymin * ih), int(r_bbox.width * iw), int(r_bbox.height * ih)
                score = detection.score

                bboxs.append([detection_id, bbox, score])

                if draw:
                    img = self.fancy_draw(img, bbox)
                    cv2.putText(img, f"{int(score[0] * 100)}%", (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 1,
                                (0, 255, 0), 2)

        return img, bboxs

    @staticmethod
    def fancy_draw(img, bbox, draw_color=(0, 255, 0), rect_thickness=1, angle_size=30, angle_thickness=5):
        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h

        cv2.rectangle(img, bbox, draw_color, rect_thickness)

        # Top-Left x1, y1
        cv2.line(img, (x1, y1), (x1 + angle_size, y1), draw_color, angle_thickness)
        cv2.line(img, (x1, y1), (x1, y1 + angle_size), draw_color, angle_thickness)

        # Top-Right x2, y1
        cv2.line(img, (x2, y1), (x2 - angle_size, y1), draw_color, angle_thickness)
        cv2.line(img, (x2, y1), (x2, y1 + angle_size), draw_color, angle_thickness)

        # Bottom-Left x1, y2
        cv2.line(img, (x1, y2), (x1, y2 - angle_size), draw_color, angle_thickness)
        cv2.line(img, (x1, y2), (x1 + angle_size, y2), draw_color, angle_thickness)

        # Bottom-Right x2, y2
        cv2.line(img, (x2, y2), (x2 - angle_size, y2), draw_color, angle_thickness)
        cv2.line(img, (x2, y2), (x2, y2 - angle_size), draw_color, angle_thickness)

        return img


class PoseDetector:

    def __init__(self, static_image_mode=False, model_complexity=1, smooth_landmarks=True, enable_segmentation=False,
                 smooth_segmentation=True, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode, model_complexity, smooth_landmarks, enable_segmentation,
                                      smooth_segmentation, min_detection_confidence, min_tracking_confidence)
        self.mp_draw = mp.solutions.drawing_utils
        # to hold the detected pose landmarks
        self.detected_landmarks = []

    def detect_landmarks(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.detected_landmarks = self.pose.process(img_rgb)

        if self.detected_landmarks.pose_landmarks and draw:
            self.mp_draw.draw_landmarks(img, self.detected_landmarks.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        return img


