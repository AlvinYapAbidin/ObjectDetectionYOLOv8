import math
import cv2
import cvzone
from cvzone.ColorModule import ColorFinder
import numpy as np

# Initialize the Video
cap = cv2.VideoCapture('Videos/vid (4).mp4')

# Create the color Finder object
myColorFinder = ColorFinder(False)
hsvVals = {'hmin': 8, 'smin': 96, 'vmin': 115, 'hmax': 14, 'smax': 255, 'vmax': 255}

# Variables
posListX, posListY = [], []
xList = [item for item in range(0, 1300)]
prediction = False

while True:
    # Grab the image

    success, img = cap.read()
    # img = cv2.imread("Ball.png")
    img = img[0:900, :]

    # Find the Color Ball
    imgColor, mask = myColorFinder.update(img, hsvVals)
    # Find location of the Ball
    imgContours, contours = cvzone.findContours(img, mask, minArea=500)

    if contours:
        posListX.append(contours[0]['center'][0])
        posListY.append(contours[0]['center'][1])

    if posListX:
        # Polynomial Regression y = Ax^2 + Bx + C
        # Find the Coefficients
        A, B, C = np.polyfit(posListX, posListY, 2)

        for i, (posX, posY) in enumerate(zip(posListX, posListY)):
            pos = (posX, posY)
            cv2.circle(imgContours, pos, 10, (0, 255, 0), cv2.FILLED)
            if i == 0:
                cv2.line(imgContours, pos, pos, (0, 255, 0), 5)
            else:
                cv2.line(imgContours, pos, (posListX[i - 1], posListY[i - 1]), (0, 255, 0), 5)

        for x in xList:
            y = int(A * x ** 2 + B * x + C)
            cv2.circle(imgContours, (x, y), 2, (255, 0, 255), cv2.FILLED)

        if len(posListX) < 10:
            # Prediction
            # X values 330 to 430  Y 590
            a = A
            b = B
            c = C - 590

            x = int((-b - math.sqrt(b ** 2 - (4 * a * c))) / (2 * a))
            prediction = 330 < x < 430

        if prediction:
            cvzone.putTextRect(imgContours, "Basket", (50, 150),
                               scale=5, thickness=5, colorR=(0, 200, 0), offset=20)
        else:
            cvzone.putTextRect(imgContours, "No Basket", (50, 150),
                               scale=5, thickness=5, colorR=(0, 0, 200), offset=20)

    # Display
    imgContours = cv2.resize(imgContours, (0, 0), None, 0.7, 0.7)
    # cv2.imshow("Image", img)
    cv2.imshow("ImageColor", imgContours)
    cv2.waitKey(100)

# Code with stop after 10 frames and resume with s key

import cv2
import cvzone
from cvzone.ColorModule import ColorFinder
import numpy as np
import math


cap = cv2.VideoCapture('Videos/vid (4).mp4')
myColorFinder = ColorFinder(False)
hsvVals = {'hmin': 8, 'smin': 124, 'vmin': 13, 'hmax': 24, 'smax': 255, 'vmax': 255}

posListX = []
posListY = []

listX = [item for item in range(0, 1300)]
start = True
prediction = False

while True:

    if start:
        if len(posListX) == 10: start = False
        success, img = cap.read()
        # img = cv2.imread('Ball.png')


        img = img[0:900, :]
        imgPrediction = img.copy()
        imgResult = img.copy()
        imgBall, mask = myColorFinder.update(img, hsvVals)
        imgCon, contours = cvzone.findContours(img, mask, 200)
        if contours:
            posListX.append(contours[0]['center'][0])
            posListY.append(contours[0]['center'][1])

        if posListX:
            if len(posListX) < 18:
                coff = np.polyfit(posListX, posListY, 2)
            for i, (posX, posY) in enumerate(zip(posListX, posListY)):
                pos = (posX, posY)
                cv2.circle(imgCon, pos, 10, (0, 255, 0), cv2.FILLED)
                cv2.circle(imgResult, pos, 10, (0, 255, 0), cv2.FILLED)

                if i == 0:
                    cv2.line(imgCon, pos, pos, (0, 255, 0), 2)
                    cv2.line(imgResult, pos, pos, (0, 255, 0), 2)
                else:
                    cv2.line(imgCon, (posListX[i - 1], posListY[i - 1]), pos, (0, 255, 0), 2)
                    cv2.line(imgResult, (posListX[i - 1], posListY[i - 1]), pos, (0, 255, 0), 2)

            for x in listX:
                y = int(coff[0] * x ** 2 + coff[1] * x + coff[2])
                cv2.circle(imgPrediction, (x, y), 2, (255, 0, 255), cv2.FILLED)
                cv2.circle(imgResult, (x, y), 2, (255, 0, 255), cv2.FILLED)

            # Predict
            if len(posListX) < 10:
                # y = int(coff[0] * x ** 2 + coff[1] * x + coff[2])
                a, b, c = coff
                c = c - 593
                x = int((-b - math.sqrt(b ** 2 - (4 * a * c))) / (2 * a))
                prediction = 300 < x < 430
            if prediction:
                cvzone.putTextRect(imgResult, "Basket", (50, 150), colorR=(0, 200, 0),
                                   scale=5, thickness=10, offset=20)
            else:
                cvzone.putTextRect(imgResult, "No Basket", (50, 150), colorR=(0, 0, 200),
                                   scale=5, thickness=10, offset=20)

        cv2.line(imgCon, (330, 593), (430, 593), (255, 0, 255), 10)
        imgResult = cv2.resize(imgResult, (0, 0), None, 0.7, 0.7)
        # imgStacked = cvzone.stackImages([img,imgCon,imgPrediction,imgResult],2,0.35)

        cv2.imshow("imgCon", imgResult)

    key = cv2.waitKey(100)
    if key == ord("s"):
        start = True
