import cv2
import numpy as np
import os
import time
import HandTrackingModule as htm

folderPath = 'virtualPainter'

headerList = os.listdir(folderPath)
print(headerList)

overlayHeader = []
for header in headerList:
    head = cv2.imread(folderPath + '/' + header)
    overlayHeader.append(head)

header = overlayHeader[3]
print(len(overlayHeader))

detector = htm.handDetector(maxHands=1, minimumDetectionConfidence=0.6, minimumTrackingConfidence=0.5)

print('Video Capturing Initiating...')
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

drawColor = (0, 0, 0)
brushThickness = 15
eraserThickness = 50
xp, yp = 0, 0
drawingCanvas = np.zeros((720, 1280, 3), np.uint8)
cTime = 0
pTime = 0
while True:
    status, frame = cap.read()
    frame = cv2.flip(frame, 1)

    frame = detector.findHands(frame, draw=False)
    lmList = detector.findPositions(frame, draw=False)

    if len(lmList) != 0:
        # xp, yp = 0, 0

        # print(lmList)
        x1, y1 = lmList[8][1:3]
        x2, y2 = lmList[12][1:3]

        fingers = detector.fingersUp()

        # Selection Mode
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            #print("Selection Mode")
            # # Checking for the click
            if y1 < 125:
                if 230 < x1 < 360:
                    header = overlayHeader[0]
                    drawColor = (0, 0, 255)
                elif 450 < x1 < 650:
                    header = overlayHeader[1]
                    drawColor = (0, 255, 0)
                elif 800 < x1 < 950:
                    header = overlayHeader[2]
                    drawColor = (225, 105, 65)
                elif 1050 < x1 < 1200:
                    header = overlayHeader[3]
                    drawColor = (0, 0, 0)
            cv2.rectangle(frame, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        # Drawing Mode
        if fingers[1] and fingers[2] == False:
            cv2.circle(frame, (x1, y1), 15, drawColor, cv2.FILLED)
            # print("Drawing Mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            #
            # cv2.line(frame, (xp, yp), (x1, y1), drawColor, brushThickness)
            # cv2.line(drawingCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            if drawColor == (0, 0, 0):
                cv2.line(frame, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(drawingCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)

            else:
                cv2.line(frame, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(drawingCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1

    canvasGray = cv2.cvtColor(drawingCanvas, cv2.COLOR_BGR2GRAY)
    _, canvasInv = cv2.threshold(canvasGray, 50, 255, cv2.THRESH_BINARY_INV)
    canvasInv = cv2.cvtColor(canvasInv, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, canvasInv)
    frame = cv2.bitwise_or(frame, drawingCanvas)

    frame[0:125, 0:1280] = header

    cTime = time.time()
    fps = int(1/(cTime-pTime))
    pTime = cTime

    cv2.putText(frame, 'FPS: '+str(fps), (15, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

    #img = cv2.addWeighted(frame, 0.5, drawingCanvas, 0.5, 0)

    cv2.imshow('Virtual Painter', frame)
    #cv2.imshow('Virtual Painter - Canvas', drawingCanvas)
    cv2.waitKey(1)

video.release()