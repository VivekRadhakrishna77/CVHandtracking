import cv2 as cv
import numpy as np
import time
import mediapipe as mp


class handDetector:
    def __init__(self, mode=False, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.max_hands,self.detection_confidence, self.tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils


    def findHands(self, img, draw = True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLandmark in self.results.multi_hand_landmarks:
                if draw:
                    for id, lm in enumerate(handLandmark.landmark):
                        h, w, ch = img.shape
                        X, Y = int(lm.x * w), int(lm.y * h)
                        print(id, X, Y)

                        if id%4 == 0:
                            cv.circle(img, (X,Y), 15, (255,0,255), -1)

                    self.mpDraw.draw_landmarks(img, handLandmark, self.mpHands.HAND_CONNECTIONS)            

        return img

    

def main():
    pTime, cTime = 0, 0
    cap = cv.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        image = detector.findHands(img)
        

        cv.putText(image,str(int(fps)), (10,70), cv.FONT_HERSHEY_TRIPLEX, 3, (0,255,0), 1)

        cv.imshow("Image", image)
        cv.waitKey(1)

if __name__ == '__main__':
    main()