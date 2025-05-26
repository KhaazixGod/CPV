from astra_camera import Camera
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from canbus import post
model = YOLO('best.pt')
cam = Camera()
'''
image 480x640 (y, x)
'''
#model.to('cuda')
center_point = (0, 320)
def get_info():
    try:
        while  True:
            depth, rgb = cam.get_depth_and_color()
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            output = model.predict(rgb,verbose = False) 
            boxes = output[0].boxes
            detection = []
            cls = []
            clearly = float('inf')
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy()) 
                cls_id = int(box.cls[0].cpu().item())
                label = output[0].names[cls_id]
                conf = float(box.conf.cpu().item())
                if conf < 0.7: continue
                if label == 'person':
                    x_mid, y_mid = (x1+x2)/2, (y1 +y2)/2 
                    line = ((x_mid - center_point[1])**2 + (y_mid - center_point[0])**2)**0.5
                    if line < clearly:
                        clearly = line
                        detection = [(x1, y1, x2, y2)]
                    cls.append((x1, y1, x2, y2))
                #cv2.rectangle(rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                #cv2.putText(rgb, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
            x_mid = 0
            y_mid = 0
            if detection: 
                x1, y1, x2, y2 = detection[0]
                # for x, y, h, w in cls:
                #     if x != x1 and y != y1 and h != x2 and w != y2: 
                #         cv2.rectangle(rgb, (x, y), (h, w), (0, 255, 0), 2)
                #         cv2.putText(rgb, "person", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
                # cv2.rectangle(rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
                # cv2.putText(rgb, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv2.LINE_AA)
                x_mid, y_mid = (x1+x2)/2, (y1 +y2)/2 
                # cv2.circle(rgb, (int(x_mid), int(y_mid)), radius = 3, color = (0, 0, 255), thickness= 2)
                distance = (x_mid - center_point[0]) - 320
                #angle = np.degrees(np.arctan((x_mid - center_point[1])/(y_mid - center_point[0]))) + 90
            else:
                distance = 0
            # print(distance, angle)
            # cv2.circle(rgb, (center_point[1], 480), radius = 3, color = (0, 0, 255), thickness = 2)
            # cv2.imshow('rgb', rgb)
            # cv2.imshow('depth', depth)
            key = cv2.waitKey(1)
            if key == 27:
                cv2.destroyAllWindows()
                break
        # aaa
            tmp = +1*(distance<-50) - 1*(distance>50)
            angle = 90 + tmp*30
            post(angle,0,depth[int(y_mid),int(x_mid)]<=12000)
                
    except Exception as e:
        print(e)
    finally:
        cam.unload()

get_info()