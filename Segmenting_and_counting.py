import cv2
import glob
import numpy as np

filenames = [img for img in glob.glob("Path to the image file")]

filenames.sort() 

for bb,imgs in enumerate(filenames):
    print(bb,imgs)

    img = cv2.imread(imgs)
    image = cv2.GaussianBlur(img,(5,5),0)
    #cv2.imshow('Filtered_original',image)

    Lab = cv2.cvtColor(image, cv2.COLOR_LBGR2LAB)

    #masking/thresholding
    lower = np.array([130,146,70])
    upper = np.array([255,255,180])

    mask = cv2.inRange(Lab, lower, upper)
    #cv2.imshow('Masked',mask)

    masked = np.ones(image.shape[:2], dtype="uint8") * 255
    masked2 = np.ones(image.shape[:2], dtype="uint8") * 255
    masked3 = np.ones(image.shape[:2], dtype="uint8") * 255

    #morphology
    kernal = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernal, iterations = 2)
    final = cv2.dilate(opening,kernal,iterations = 3)
    #cv2.imshow('Final',final)

    # find contours
    (_, contours, _) = cv2.findContours(final, cv2.RETR_EXTERNAL, 
    cv2.CHAIN_APPROX_SIMPLE)

    # print table of contours and sizes
    #print("Found %d objects." % len(contours))
    for (i, c) in enumerate(contours):
        #print("\tSize of contour %d: %d" % (i, len(c)))
        if(len(c)<162):
            #contours.remove(c)
            cv2.drawContours(masked, [c], -1, 0, -1)
            #cv2.drawContours(final, [c], -1, (0,255,0), 3)
    wbc = cv2.bitwise_and(final,final,mask=masked)
    platelets = final-wbc

    #Removing the debris
    (_, contours, _) = cv2.findContours(platelets, cv2.RETR_EXTERNAL, 
    cv2.CHAIN_APPROX_SIMPLE)

    # print table of contours and sizes
    #print("Found %d objects." % len(contours))
    for (i, c) in enumerate(contours):
        #print("\tSize of contour %d: %d" % (i, len(c)))
        if(len(c)<18):
            #contours.remove(c)
            cv2.drawContours(masked2, [c], -1, 0, -1)
            #cv2.drawContours(final, [c], -1, (0,255,0), 3)
    platelets_final = cv2.bitwise_and(platelets,platelets,mask=masked2)

    #Convex hull for wbc
    im2, contours, hierarchy = cv2.findContours(wbc, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    hull = []

    for i in range(len(contours)):
        hull.append(cv2.convexHull(contours[i], False))

    drawing = np.zeros((wbc.shape[0], wbc.shape[1], 3), np.uint8)

    for i in range(len(contours)):
        color_contours = (0,255,0)
        color = (255,255,255)
        cv2.drawContours(drawing, contours, i, color_contours, 2, 8, hierarchy)
        cv2.drawContours(drawing, hull, i, color, 2, 8)
    #cv2.imshow('WBC',wbc)
    #cv2.imshow('WBC Hull',drawing)
    #cv2.imshow('platelets',platelets)

    (_, contours, _) = cv2.findContours(platelets_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for (i, c) in enumerate(contours):
        cnt = contours[i]   
        x,y,z = cv2.minAreaRect(cnt)
        aspect_ratio = y[0]/y[1]
        if(aspect_ratio<0.65 or aspect_ratio>1.35):
            #contours.remove(c)
            cv2.drawContours(masked3, [c], -1, 0, -1)
            #cv2.drawContours(final, [c], -1, (0,255,0), 3)
    platelets_shaped = cv2.bitwise_and(platelets_final,platelets_final,mask=masked3)

    #Convex hull for platelets
    im2, contours, hierarchy = cv2.findContours(platelets_shaped, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    hull2 = []

    for i in range(len(contours)):
        hull2.append(cv2.convexHull(contours[i], False))

    drawing2 = np.zeros((platelets_shaped.shape[0], platelets_shaped.shape[1], 3), np.uint8)

    for i in range(len(contours)):
        color_contours = (0,255,0)
        color = (255,255,255)
        cv2.drawContours(drawing2, contours, i, color_contours, 2, 8, hierarchy)
        cv2.drawContours(drawing2, hull2, i, color, 2, 8)
    
    cv2.imwrite('/home/sourav/Documents/Segmentation/WBC/sample{}.jpg'.format(bb+1), drawing)
    cv2.imwrite('/home/sourav/Documents/Segmentation/PLATELETS/sample{}.jpg'.format(bb+1), drawing2)
    
cv2.waitKey(0)
cv2.destroyAllWindows()
