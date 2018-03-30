import numpy as np
import cv2
import math

bgSubThreshold = 30 
blurValue = 41
threshold = 40 
rec_h1 = 0
rec_h2 = 500
rec_w1 = 0
rec_w2 = 700
result = [[0,0]]

def drawing(frame):
#draw all the existing points
    if result is not None:
        for i in range(len(result)):
            cv2.circle(frame, tuple(result[i]),2,(255,255,255),3)

def plot_point(frame, point):
    cv2.circle(frame, point,5,(255,0,0),3)

def farthest_point1(defects, contour, centroid):
    s = defects[:,0][:,0]
    cx, cy = centroid
    
    x = np.array(contour[s][:,0][:,0], dtype=np.float)
    y = np.array(contour[s][:,0][:,1], dtype=np.float)
                
    xp = cv2.pow(cv2.subtract(x, cx), 2)
    yp = cv2.pow(cv2.subtract(y, cy), 2)
    dist = cv2.sqrt(cv2.add(xp, yp))

    dist_max_i = np.argmax(dist)

    if dist_max_i < len(s):
        farthest_defect = s[dist_max_i]
        farthest_point = tuple(contour[farthest_defect][0])
        return farthest_point
    else:
        return None

def calculate_fingers(contours,hull,defects):
    cnt = 0
    for i in range(defects.shape[0]):  # calculate the angle
        s, e, f, d = defects[i][0]
        start = tuple(contours[s][0])
        end = tuple(contours[e][0])
        far = tuple(contours[f][0])
        a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
        c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
        angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
        if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
            cnt += 1
    return cnt

def hand_gesture(contours,frame,result):
    cv2.drawContours(frame, contours, -1, (0,255,0), 3)


    maxArea = 0
    contours_i = 0
    contours_len = len(contours)
    if contours_len != 0:
        #max contours
        for i in range(contours_len):
            area = cv2.contourArea(contours[i])
            if area > maxArea:
                maxArea = area
                contours_i = i
        max_contours = contours[contours_i]
        #hull
        hull = cv2.convexHull(max_contours)
        #centroid
        centroid = None
        moments = cv2.moments(max_contours)
        if moments['m00'] != 0:
            cx = int(moments['m10']/moments['m00'])
            cy = int(moments['m01']/moments['m00'])
            centroid = (cx,cy)
        #defects
        defects = None
        hull1 = cv2.convexHull(max_contours, returnPoints=False)
        if hull1 is not None and len(hull > 3) and len(max_contours) > 3:
            defects = cv2.convexityDefects(max_contours, hull1)   

        if centroid is not None and defects is not None and len(defects) > 0:   
            farthest_point = farthest_point1(defects, max_contours, centroid)

            if farthest_point is not None:
                #stop sign?
                finger_numbers = calculate_fingers(max_contours,hull,defects)
                #print(finger_numbers)
                if(finger_numbers is not None and finger_numbers == 0):
                    plot_point(frame, farthest_point)
                    cv2.drawContours(frame,max_contours,0,(0,0,255),3)  
                    temp = np.asarray(farthest_point)
                    result = np.append(result, [temp], axis = 0)
    return result

cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)

while(1):
    ret, frame = cap.read()
    frame = frame[rec_h1:rec_h2, rec_w1:rec_w2]

    #mask
    fgmask = fgbg.apply(frame)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    

    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
    ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
    
    #drawing area
    #cv2.rectangle(frame,(rec_w1, rec_h1),(rec_w2, rec_h2),(0,255,0),3)

    #contours
    im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    result = hand_gesture(contours,frame,result)
    drawing(frame)

    cv2.imshow('frame',frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()