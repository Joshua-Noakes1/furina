# Dashcam Stuff

import os
import cv2
import imutils

webcam = ''


def main():
    # Get webcam and set to 720p
    webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while (True):
        ret, frame = webcam.read()

        # get frames needed
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 13, 15, 15)
        edged = cv2.Canny(gray, 100, 200)

        #  find contours
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        webcam_contours = frame.copy()
        cv2.drawContours(webcam_contours, contours, -1, (0, 255, 0), 2)
        print("Number of Contours found = " + str(len(contours)))
        
        cv2.imshow("Webcam View", frame)
        cv2.imshow("Edged Image", edged)
        cv2.imshow("Contours", webcam_contours)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


main()
