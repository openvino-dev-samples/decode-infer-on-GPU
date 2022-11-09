import cv2
import numpy as np
import time
from openvino.runtime import Core

capture = cv2.VideoCapture('/home/ethan/Downloads/sample_2560x1440.h265')
ie = Core()
model = ie.read_model(model="/home/ethan/oneVPL/examples/interop/advanced-decvpp-infer/intel/vehicle-detection-0200/FP16/vehicle-detection-0200.xml")
compiled_model = ie.compile_model(model=model, device_name="GPU")
input_layer = compiled_model.input(0)
N, C, H, W = input_layer.shape
request = compiled_model.create_infer_request()
start = time.time()
while True:
    isTrue, frame = capture.read()
    if isTrue:
        resized_image = cv2.resize(src=frame, dsize=(W,H))
        input_data = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0).astype(np.float32)
        request.infer(inputs={input_layer.any_name: input_data})
    else:
        break
end = time.time()
time_spend = end - start
print(time_spend)
capture.release()

