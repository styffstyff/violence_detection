import os
import yolov5_master.detector as DETECT
import cv2
import numpy as np


cap=cv2.VideoCapture("/Users/gabinvrillault/Documents/MP/TIPE/Pedestrian.mp4") # 7 FPS
idimg = 0
while True:

    ret, frame = cap.read()

    cv2.imwrite(f'./images/test1{idimg}.jpg', frame)

    boxes = DETECT.run(source=f'./images/test1{idimg}.jpg', classes=0, view_img=False, nosave=True , device="cpu")
    
    boxes = np.array(boxes)
    print(boxes)

    for box in boxes:
        x, y, w, h = box.astype(int)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    
    cv2.imshow("frame", frame)


    idimg += 1 
    key=cv2.waitKey(70)&0xFF
    if key==ord('r'):
        rectangle = not rectangle
    if key==ord('t'):
        trace = not trace
    if key==ord('q'):
        quit()

