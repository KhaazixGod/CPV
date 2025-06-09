import onnxruntime as ort
import numpy as np
import cv2
from astra_camera import Camera
from reid_embedder import ReIDEmbedder
from deep_sort_realtime.deepsort_tracker import DeepSort
def compute_iou(boxA, boxB):
    x1_A, y1_A, x2_A, y2_A = boxA
    x1_B, y1_B, x2_B, y2_B = boxB

    x1_int = max(x1_A, x1_B)
    y1_int = max(y1_A, y1_B)
    x2_int = min(x2_A, x2_B)
    y2_int = min(y2_A, y2_B)

    inter_width = max(0, x2_int - x1_int)
    inter_height = max(0, y2_int - y1_int)
    inter_area = inter_width * inter_height

    area_A = (x2_A - x1_A) * (y2_A - y1_A)
    area_B = (x2_B - x1_B) * (y2_B - y1_B)
    union_area = area_A + area_B - inter_area

    iou = inter_area / union_area if union_area > 0 else 0
    return iou

def first_human(cam,session):
    while  True:
            depth,rgb = cam.get_depth_and_color() 
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if depth is None or depth.size == 0 or rgb is None or rgb.size == 0:
                continue 
            img_height, img_width = rgb.shape[:2]
            rgb = cv2.resize(rgb, (320, 320))
            depth = cv2.resize(depth,(320,320))
            img_input = rgb.transpose(2, 0, 1) / 255.0  # HWC → CHW
            img_input = np.expand_dims(img_input, axis=0).astype(np.float32)
            
            output = session.run(None, {input_name: img_input})[0]  # shape: (1, 84, 2100) or (1, 6, N)
            output = np.squeeze(output).T

            conf_threshold = 0.5
            angle = 90
            for pred in output:
                x, y, w, h, conf = pred
                if conf < conf_threshold:
                    continue
                x1 = int(x-w/2)
                x2 = int(x+w/2)
                y1 = int(y-h/2)
                y2 = int(y+h/2)
                # Bỏ qua vùng crop không hợp lệ
                if x1 < 0 or y1 < 0 or x2 > rgb.shape[1] or y2 > rgb.shape[0]:
                    continue
                if x2 <= x1 or y2 <= y1:
                    continue
                return x1,y1,x2,y2
            cv2.imshow('rgb',  rgb)   
            key = cv2.waitKey(1)
            if key == 27:
                cv2.destroyAllWindows()
def get_info(cam,session):
    try:
        x1_own, y1_own, x2_own, y2_own  = first_human(cam,session)
        dem = 0
        while  True:
            depth,rgb = cam.get_depth_and_color() 
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if depth is None or depth.size == 0 or rgb is None or rgb.size == 0:
                continue 
            img_height, img_width = rgb.shape[:2]
            rgb = cv2.resize(rgb, (320, 320))
            depth = cv2.resize(depth,(320,320))
            img_input = rgb.transpose(2, 0, 1) / 255.0  # HWC → CHW
            img_input = np.expand_dims(img_input, axis=0).astype(np.float32)
            
            output = session.run(None, {input_name: img_input})[0]  # shape: (1, 84, 2100) or (1, 6, N)
            output = np.squeeze(output).T

            conf_threshold = 0.3
            angle = 90
            max_iou = -1
            predict_box = (0,0,0,0)
            for pred in output:
                x, y, w, h, conf = pred
                if conf < conf_threshold:
                    continue
                x1 = int(x-w/2)
                x2 = int(x+w/2)
                y1 = int(y-h/2)
                y2 = int(y+h/2)
                # Bỏ qua vùng crop không hợp lệ
                if x1 < 0 or y1 < 0 or x2 > rgb.shape[1] or y2 > rgb.shape[0]:
                    continue
                if x2 <= x1 or y2 <= y1:
                    continue
                iou = compute_iou((x1,y1,x2,y2),(x1_own,y1_own,x2_own,y2_own))
                if iou > max_iou:
                    max_iou = iou
                    predict_box = (x1,y1,x2,y2)
                if iou >= 0.75:
                    break # Nếu đã ổn thì không sét nữa
            if predict_box == (0,0,0,0):
                print(dem)
                dem+=1 # Check xem co chay khong
                continue
            x1_own,y1_own,x2_own,y2_own = predict_box
            x_mid = x1_own+(x2_own-x1_own)/2
            angle = (x_mid-160) / 160 * 90 + 90
            try :
                dpt = depth[int(y1):int(y2),int(x1):int(x2)]
                dpt[dpt == 0] = 50000
                dpt = np.min(dpt)
            except:
                dpt = 0
            cv2.rectangle(rgb, (x1_own, y1_own), (x2_own, y2_own), (0, 255, 0), 2)
            cv2.imshow('rgb',  rgb)   
            key = cv2.waitKey(1)
            if key == 27:
                cv2.destroyAllWindows()
                break
                
    except Exception as e:
        print(e)
cam = Camera()
try:
    session = ort.InferenceSession("best_yolov8n_2.onnx", providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    get_info(cam,session)
except Exception as e:
    print(e)
finally:
    cam.unload()
    
