# Dashcam Stuff

import os
import cv2
import numpy as np
import pytesseract 
if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

webcam = None


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
        
        # dilate the contours
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilatedView = cv2.dilate(edged, kernel, iterations=1)
        contoursWithDilate, _ = cv2.findContours(dilatedView.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                  
        # attempt to find rectangles the size of a license plate
        foundContour = None
        for contour in contoursWithDilate:
            approx = cv2.approxPolyDP(contour, 0.018 * cv2.arcLength(contour, True), True)
            if len(approx) == 4:
                foundContour = approx
                break

        if foundContour is not None:
            cv2.drawContours(frame, [foundContour], -1, (0, 255, 0), 3)
            
            # crop the image to the found license plate
            mask = np.zeros(gray.shape, np.uint8)
            cropedLPImage = cv2.drawContours(mask, [foundContour], 0, 255, -1)
            cropedLPImage = cv2.bitwise_and(frame, frame, mask=mask)
            (x, y) = np.where(mask == 255)
            (topx, topy) = (np.min(x), np.min(y))
            (bottomx, bottomy) = (np.max(x), np.max(y))
            cropedLPImage = frame[topx:bottomx + 1, topy:bottomy + 1]
            
            # attempt to read the license plate
            text = pytesseract.image_to_string(cropedLPImage, config='--psm 11')
            if len(text) > 0:
                print("Detected license plate Number is:", text)
                cv2.imshow("License Plate", cropedLPImage)
            
        # View the webcam
        cv2.imshow("Webcam View", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


main()
