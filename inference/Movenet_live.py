import tensorflow as tf
import cv2
import numpy as np
import tensorflow_hub as hub
from Movenet import get_keypoints, movenet_model, movenet_fn



def normalize_keypoints(keypoints):
    """
    Takes raw (51,) keypoints and centers them on the hips.
    """
    kp = keypoints.reshape(17, 3) 
    
    hip_center_y = (kp[11, 0] + kp[12, 0]) / 2
    hip_center_x = (kp[11, 1] + kp[12, 1]) / 2
    
    kp[:, 0] = kp[:, 0] - hip_center_y
    kp[:, 1] = kp[:, 1] - hip_center_x
    
    return kp.flatten()

model = tf.keras.models.load_model('movenet_lstm_2classes.h5')
actions = ['Squatting', 'Standing']
sequence = []
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    
    img = tf.image.resize_with_pad(tf.expand_dims(frame, axis=0), 192, 192)
    input_img = tf.cast(img, dtype=tf.int32)
    keypoints = movenet_fn(input_img)['output_0'].numpy().flatten()


    # norm_kp = normalize_keypoints(keypoints)
    sequence.append(keypoints)
    sequence = sequence[-45:]

    text_1 = "Waiting for movement..."
    if len(sequence) == 45:
        res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]

        text_1 = f"{actions[0]} - {res[0]*100:.1f}% - {actions[1]} - {res[1]*100:.1f}%"
        
    cv2.putText(frame, text_1, (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('LSTM Live Feed', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()