import tensorflow as tf
import cv2
import numpy as np
import pycoral
import pycoral.utils.edgetpu as edgetpu
from pycoral.adapters import common

model_file = r"C:\VIP\LSTM\model\visual_edgetpu.tflite"
 
movenet_file = r"C:\VIP\LSTM\model\movenet_single_pose_lightning_ptq_edgetpu.tflite"
movenet_interpreter = edgetpu.make_interpreter(movenet_file)
movenet_interpreter.allocate_tensors()
input_size = common.input_size(movenet_interpreter)

LSTM_interpreter = edgetpu.make_interpreter(model_file)
LSTM_interpreter.allocate_tensors()
lstm_input_details = LSTM_interpreter.get_input_details()[0]
print(lstm_input_details)

actions = ['Squatting', 'Standing']
sequence = []
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, input_size)

    common.set_input(movenet_interpreter, img)
    movenet_interpreter.invoke()
    keypoints = common.output_tensor(movenet_interpreter, 0).flatten().copy()

    sequence.append(keypoints)
    sequence = sequence[-45:]

    text_1 = "Waiting for movement..."
    if len(sequence) == 45:
        input_data = np.expand_dims(np.array(sequence, dtype=np.float32), axis=0)
        common.set_input(LSTM_interpreter, input_data)
        LSTM_interpreter.invoke()
        res = common.output_tensor(LSTM_interpreter, 0)[0].copy()

        text_1 = f"{actions[0]} - {res[0]*100:.1f}% - {actions[1]} - {res[1]*100:.1f}%"
        
    cv2.putText(frame, text_1, (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('LSTM Live Feed', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()