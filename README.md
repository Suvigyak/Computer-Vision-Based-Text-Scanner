# Computer-Vision-Based-Text-Scanner
Develop a computer vision-based text scanner that can scan any text from an image using the optical character recognition algorithm and display the text on your screen.This project, intend to help by providing their user with visual recognition of various text and images. In this project, we discuss a system that uses existing technologies such as the Optical Character Recognition (OCR), and use it to automatically identify and recognize texts and signs in the environment and help the users. The challenge of extracting text from images of documents has traditionally been referred to as Optical Character Recognition (OCR) and has been the focus of much research


import cv2

import numpy as np

import pytesseract

 

img_original = cv2.imread("hhh.jpg")

 

# Defining the end points of the image

pts1 = np.float32([[110,190],[600,75],[725,315],[185,490]])

pts2 = np.float32([[100,100],[700,100],[700,500],[100,500]])

 

# Calculates perspective transform from above points

M = cv2.getPerspectiveTransform(pts1,pts2)

 

# Now apply this transform

dst = cv2.warpPerspective(img_original,M,(800,600))

 

image_work = dst.copy()

 

# Convert image from BGR to RGB

img_original_RGB = cv2.cvtColor(img_original,cv2.COLOR_BGR2RGB)

dst_RGB = cv2.cvtColor(dst,cv2.COLOR_BGR2RGB)

 

# gray scale

dst_gray = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)

 

#Smoothing

dst_gray_smooth = cv2.GaussianBlur(dst_gray,(5,5),0)

 

# Thresholding

ret,dst_gray_smooth_thresh = cv2.threshold(dst_gray_smooth,180,255,cv2.THRESH_BINARY)

 

# Canny edge

dst_gray_smooth_edge = cv2.Canny(dst_gray_smooth_thresh,150,300)

 

contour, heirarchy = cv2.findContours(dst_gray_smooth_edge,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

 

# Now try to find four points of the corner

cnts = sorted(contour,key=cv2.contourArea,reverse=True)[:5]

 

for i in cnts :

    perimeter = cv2.arcLength(i,True)

    approx = cv2.approxPolyDP(i,0.02*perimeter,True)

 

    if len(approx) == 4 :

        (x,y,w,h) = cv2.boundingRect(i)

##        final_image = cv2.rectangle(image_work,(x,y),(x+w,y+h),(255,0,0),3)
##
##        final_image = cv2.cvtColor(final_image,cv2.COLOR_BGR2RGB)
##
##        cv2.imshow('Detected',final_image)

        image_work = cv2.rectangle(image_work,(x,y),(x+w,y+h),(255,0,0),3)

        image_work = cv2.cvtColor(image_work,cv2.COLOR_BGR2RGB)

        cv2.imshow('Detected',image_work)

print(pytesseract.image_to_string(image_work))

 

cv2.imshow('Original',img_original_RGB)

 

cv2.waitKey(0) 


cv2.destroyAllWindows()
