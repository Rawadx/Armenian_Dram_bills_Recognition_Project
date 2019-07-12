import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('images/Fifty_thousand_Dram_2.jpg',0)          # queryImage
#img2 = cv2.imread('images/Ten_thousand_Dram_3.jpg',0) # trainImage

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
bf = cv2.BFMatcher()



cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    kp2, des2 = sift.detectAndCompute(grayframe,None)

    # BFMatcher with default params
   
    matches = bf.knnMatch(des1,des2, k=2)
    
    # Apply ratio test.
    
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    
    # cv2.drawMatchesKnn expects list of lists as matches.
    #img3 = cv2.drawMatchesKnn(img1,kp1,grayframe,kp2,good,None,flags=2)
    #Homograpy
    
    if len(good) > 10:
        query_pts = np.float32([img1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        train_pts = np.float32([grayframe[m.trainIdx] for m in good]).reshape(-1,1,2)
    
        matrix, mask = cv2.findHomography(query_pts, train_pts, cv.2RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()
        
        
        h, w = imgshape
        pts = np.float32([0,0],[0,h],[w,h],[w,0]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pst, matrix)
        homography = cv2.polylines(frame,[np,int32(dst)], True, (255,0,0),3)
        cv2.imshow('Homo',homography)
     else:   
        cv2.imshow('Homo',grayframe) 
        
    #cv2.imshow('Frame', img3)
    
    #cv2.imshow('Frame', grayframe)
#    
    
    key = cv2.waitKey(1)
    if key == 27:
        break
    
    
cap.release()
cv2.destroyAllWindows()