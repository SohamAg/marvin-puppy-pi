import rclpy
from rclpy.node import Node
from std_msgs.msg import Int8

import cv2
import numpy as np
import pycoral.utils.edgetpu as edgetpu
from pycoral.adapters import common

class AIPuppyPublisher(Node):
    def __init__(self):
        super().__init__('ai_puppy_publisher')
        
        # 1. ROS 2 Publisher (sending Int8 to 'robot_state')
        self.publisher_ = self.create_publisher(Int8, 'robot_state', 10)

        # 2. Setup AI Models (Coral TPU)
        model_file = r"models/visual_edgetpu.tflite"
        movenet_file = r"models/movenet_single_pose_lightning_ptq_edgetpu.tflite"
        
        self.movenet_interpreter = edgetpu.make_interpreter(movenet_file)
        self.movenet_interpreter.allocate_tensors()
        self.input_size = common.input_size(self.movenet_interpreter)

        self.lstm_interpreter = edgetpu.make_interpreter(model_file)
        self.lstm_interpreter.allocate_tensors()
        
        self.actions = ['Squatting', 'Standing']
        self.sequence = []
        
        # 3. Setup Camera
        self.cap = cv2.VideoCapture(0)
        
        # 4. Timer to run inference (20 times per second)
        self.timer = self.create_timer(0.05, self.run_inference)
        self.get_logger().info("AI Puppy Publisher started. Watching for poses...")

    def run_inference(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # Prepare image for MoveNet
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.input_size)

        # Run MoveNet
        common.set_input(self.movenet_interpreter, img)
        self.movenet_interpreter.invoke()
        keypoints = common.output_tensor(self.movenet_interpreter, 0).flatten().copy()

        self.sequence.append(keypoints)
        self.sequence = self.sequence[-45:]

        # Create the message
        msg = Int8()

        if len(self.sequence) == 45:
            # Run LSTM
            input_data = np.expand_dims(np.array(self.sequence, dtype=np.float32), axis=0)
            common.set_input(self.lstm_interpreter, input_data)
            self.lstm_interpreter.invoke()
            res = common.output_tensor(self.lstm_interpreter, 0)[0].copy()
            
            # Get the label with highest probability
            label_index = np.argmax(res)
            label = self.actions[label_index]
            
            # --- MAPPING LOGIC ---
            if label == 'Standing':
                msg.data = 4  # Stand ID
            elif label == 'Squatting':
                msg.data = 0  # Stop ID
            
            self.publisher_.publish(msg)
            # Log for debugging
            self.get_logger().info(f"Detected: {label} -> Sending ID: {msg.data}")
            
        # Optional: Show camera feed for debugging locally
        cv2.imshow('AI Publisher Feed', frame)
        cv2.waitKey(1)

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = AIPuppyPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
  
