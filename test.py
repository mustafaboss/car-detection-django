import cv2 as cv
import numpy as np

min_width_react = 80 #min width reactangle
min_hieght_react = 80 #min width reactangle
count_line_position = 550
#web camera
cap = cv.VideoCapture("video.mp4")

#initilize subtrutor
algo = cv.createBackgroundSubtractorMOG2()

def center_handle(x, y, w, h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x+x1
    cy = y+y1
    return cx,cy

detect = []
offset = 6 # allowble error between pixel
counter = 0
while True:
    ret, frame1 = cap.read()
    if not ret:
      break
    grey = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(grey,(3,3),5)
    #APPLYING ON EACH FRAME
    img_sub = algo.apply(blur)
    dilat = cv.dilate(img_sub, np.ones((5,5),np.uint8))
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
    dilatada = cv.morphologyEx(dilat,cv.MORPH_CLOSE, kernel)
    dilatada = cv.morphologyEx(dilatada, cv.MORPH_CLOSE, kernel)
    counters, __ = cv.findContours(dilatada, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    cv.line(frame1,(25,count_line_position),(1200,count_line_position),(255, 127, 0),3)


    for (i, c) in enumerate(counters):
        (x,y,w,h) = cv.boundingRect(c)
        validate_counter = (w>= min_width_react) and  (h>= min_hieght_react) 
        if not validate_counter:
            continue

        cv.rectangle(frame1,(x,y),(x+w, y+h),(0,255,0),2)
        center = center_handle(x,y,w,h)
        detect.append(center)
        cv.circle(frame1,center,4,(0,0,255),-1)

        for(x,y) in detect:
            if y<(count_line_position-offset) and y>(count_line_position+offset):
                counter+=1
                cv.line(frame1,(25,count_line_position),(1200, count_line_position),(0, 127, 255),3)
                detect.remove(x,y)
                print("vehicle counter:"+str(counter))
    cv.putText(frame1, "vehicle counter:" +str(counter),(450,70),cv.FONT_HERSHEY_SIMPLEX,2,(0,0,255),5)
    #cv.imshow('DETECTOR', dilatada)
    cv.imshow('video.mp4', frame1)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()
cap.release()