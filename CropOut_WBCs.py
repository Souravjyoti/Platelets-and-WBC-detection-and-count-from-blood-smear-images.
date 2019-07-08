import numpy as np
import cv2

img = cv2.imread('Path to image')

image = cv2.GaussianBlur(img,(5,5),0)
cv2.imshow('Filtered_original',image)

Lab = cv2.cvtColor(image, cv2.COLOR_LBGR2LAB)

#masking/thresholding
lower = np.array([130,146,20])
upper = np.array([255,255,180])

mask = cv2.inRange(Lab, lower, upper)
#cv2.imshow('Masked',mask)

masked = np.ones(image.shape[:2], dtype="uint8") * 255

#morphology
kernal = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernal, iterations = 2)
final = cv2.dilate(opening,kernal,iterations = 3)
#cv2.imshow('Final',final)

im2,contours,hierarchy = cv2.findContours(final, 1, 2)

for (i, c) in enumerate(contours):
    cnt = contours[i]
    #M = cv2.moments(cnt)
    #print( M )
    x,y,w,h = cv2.boundingRect(cnt)
    if h>170 or w>170:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        crop = image[y:y+h, x:x+w]
        #cv2.imwrite('/home/sourav/Documents/Segmentation/Cropped/sample{}.jpg'.format(i+1), crop)
        cv2.imshow('Crop{}'.format(i+1),crop)

#cv2.imshow('Rect',img)

cv2.waitKey(0)
cv2.destroyAllWindows()

