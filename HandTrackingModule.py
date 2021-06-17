import cv2
import time
import mediapipe as mp
import numpy as np

class handDetector:
    def __init__(self, mode=False, maxHands=5, minimumDetectionConfidence=0.5, minimumTrackingConfidence=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.minimumDetectionConfidence = minimumDetectionConfidence
        self.minimumTrackingConfidence = minimumTrackingConfidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands,
                                        min_detection_confidence=self.minimumDetectionConfidence,
                                        min_tracking_confidence=self.minimumTrackingConfidence)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.lmlist = []

    def findHands(self, image, draw=True):
            imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.results = self.hands.process(imageRGB)
            # print(results.multi_hand_landmarks)

            if self.results.multi_hand_landmarks:
                for hand in self.results.multi_hand_landmarks:
                    if draw:
                        self.mpDraw.draw_landmarks(image, hand, self.mpHands.HAND_CONNECTIONS)

            return image

    def findPositions(self, image, handNumber=0, draw=True):
        self.lmlist = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNumber]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                self.lmlist.append([id, cx, cy])

                if draw:
                    cv2.circle(image, (cx, cy), 10, (0, 0, 255), cv2.FILLED)

        return self.lmlist

    def fingersUp(self):
        fingers = []
        # Thumb
        if self.lmlist[self.tipIds[0]][1] > self.lmlist[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers
        for id in range(1, 5):
            if self.lmlist[self.tipIds[id]][2] < self.lmlist[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

            # totalFingers = fingers.count(1)

        return fingers

def main():

    '''This is the Dummy Code... can be Copied in another file/project and it will run the exact same way
        After importing this file to the project'''

    cTime = 0
    pTime = 0
    vid = cv2.VideoCapture(0)

    detector = handDetector()

    while True:
        status, image = vid.read()

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        image = detector.findHands(image)
        lmlist = detector.findPositions(image)

        if len(lmlist) != 0:
            print(lmlist[8])

        cv2.putText(image, 'FPS: ' + str(int(fps)), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 1)

        cv2.imshow('Camera', image)

        cv2.waitKey(1)


if __name__ == '__main__':
    main()
