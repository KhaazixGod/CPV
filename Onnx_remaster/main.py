import cv2
import numpy as np
import onnxruntime as ort

# Load ONNX model
session = ort.InferenceSession("best_yolov8n_2.onnx", providers=["CPUExecutionProvider"])

input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape  # e.g., [1, 3, 640, 640]
input_height, input_width = input_shape[2], input_shape[3]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize và chuẩn hóa ảnh theo YOLOv8 format
    img = cv2.resize(frame, (input_width, input_height))
    img_input = img.astype(np.float32) / 255.0
    img_input = np.transpose(img_input, (2, 0, 1))  # HWC -> CHW
    img_input = np.expand_dims(img_input, axis=0)   # Add batch dim

    # Run inference
    outputs = session.run(None, {input_name: img_input})

    # Output: shape (1, 5, 2100)
    output = outputs[0][0]  # (5, 2100)
    x = output[0]
    y = output[1]
    w = output[2]
    h = output[3]
    conf = output[4]

    for i in range(conf.shape[0]):
        score = conf[i]
        if score > 0.75:
            cx, cy, bw, bh = x[i], y[i], w[i], h[i]
            x1 = int(cx - bw / 2)
            y1 = int(cy - bh / 2)
            x2 = int(cx + bw / 2)
            y2 = int(cy + bh / 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{score:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            break

    cv2.imshow("YOLOv8 ONNX Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
