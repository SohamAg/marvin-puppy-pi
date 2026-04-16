import rclpy
from rclpy.node import Node
from std_msgs.msg import Int8

import pyaudio
import numpy as np
import threading
import tensorflow as tf
from collections import deque
from scipy.signal import resample_poly
from math import gcd

from pycoral.utils import edgetpu
from pycoral.adapters import common
from pycoral.adapters import classify

# Configuration (Matching your hardware setup)
TPU = 1
FORMAT = pyaudio.paInt16
CHANNEL = 1
SAMPLE_RATE = 44100 
WINDOW_LEN = 1 
HOP_LEN = 0.1 
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_LEN)
HOP_SAMPLES = int(HOP_LEN * SAMPLE_RATE)
DEV_INDEX = 0 

# Model Configuration
model_file = "models/wake_edgetpu.tflite"
labels_list = ['backward','bed','bird','cat','dog','down','eight','five','follow','forward','four',
                'go','happy','house','learn','left','marvin','nine','no','off','on','one','right','seven',
                'sheila','six','stop','three','tree','two','up','visual','wow','yes','zero']

# Mel Spectrogram Layer
mel_layer = tf.keras.layers.MelSpectrogram(
    fft_length=512, sequence_stride=160, sampling_rate=16000, num_mel_bins=128, power_to_db=True
)

class VoiceCommandPublisher(Node):
    def __init__(self):
        super().__init__('voice_command_publisher')
        
        # 1. ROS 2 Publisher
        self.publisher_ = self.create_publisher(Int8, 'robot_state', 10)

        # 2. Audio Buffers & Locks
        self.pya = pyaudio.PyAudio()
        self.window_buffer = deque(np.zeros(WINDOW_SAMPLES, dtype=np.int16), maxlen=WINDOW_SAMPLES)
        self.window_lock = threading.Lock()
        self.window_update = threading.Event()
        self.audio_clips = deque(maxlen=10)
        self.clips_lock = threading.Lock()
        self.clips_update = threading.Event()

        # 3. Load TPU Model
        self.get_logger().info("Loading TPU Voice Model...")
        self.tpu_inference = edgetpu.make_interpreter(model_file)
        self.tpu_inference.allocate_tensors()

        # 4. Start Threads
        self.running = True
        self.stream = self.pya.open(
            format=FORMAT, rate=SAMPLE_RATE, channels=CHANNEL,
            input_device_index=DEV_INDEX, input=True,
            frames_per_buffer=HOP_SAMPLES, stream_callback=self.audio_callback
        )
        
        self.processor_thread = threading.Thread(target=self.processor_loop, daemon=True)
        self.snapshot_thread = threading.Thread(target=self.snapshot_loop, daemon=True)
        self.processor_thread.start()
        self.snapshot_thread.start()
        
        self.get_logger().info("Voice Node Online. Listening for commands...")

    def audio_callback(self, in_data, frame_count, time_info, status):
        formatted = np.frombuffer(in_data, dtype=np.int16)
        with self.window_lock:
            self.window_buffer.extend(formatted)
        self.window_update.set()
        return (None, pyaudio.paContinue)

    def snapshot_loop(self):
        while self.running and rclpy.ok():
            self.window_update.wait(timeout=1.0)
            self.window_update.clear()
            with self.window_lock:
                window_copy = np.array(self.window_buffer)
            with self.clips_lock:
                self.audio_clips.append(window_copy)
            self.clips_update.set()

    def processor_loop(self):
        while self.running and rclpy.ok():
            self.clips_update.wait(timeout=1.0)
            self.clips_update.clear()
            
            current_clip = None
            with self.clips_lock:
                if self.audio_clips:
                    current_clip = self.audio_clips.pop()

            if current_clip is not None:
                mel_spec = self.process_audio(current_clip)
                
                # TPU Inference
                common.set_input(self.tpu_inference, mel_spec)
                self.tpu_inference.invoke()
                classes = classify.get_classes(self.tpu_inference)
                if labels_list[classes[0].id] == "marvin":
                    self.handle_voice_command()

    def handle_voice_command(self):
        msg = Int8()
        
        msg.data = 1

        self.publisher_.publish(msg)
        self.get_logger().info(f"Wakeword Detected -> Sending ID: {msg.data}")

    def process_audio(self, audio_data):
        # Resample 44100 -> 16000
        g = gcd(SAMPLE_RATE, 16000)
        resampled = resample_poly(audio_data.astype(np.float64), 16000//g, SAMPLE_RATE//g).astype(np.int16)
        
        # Pad/Clip to exactly 16000 samples
        target = 16000
        if len(resampled) < target:
            resampled = np.pad(resampled, (0, target - len(resampled)))
        else:
            resampled = resampled[:target]

        # Convert to Mel Spectrogram
        tensor_audio = tf.convert_to_tensor(resampled, dtype=tf.float32)
        tensor_audio = tf.expand_dims(tensor_audio, 0)
        mel_spec = mel_layer(tensor_audio)
        return tf.expand_dims(mel_spec, -1)

    def shutdown_node(self):
        self.running = False
        self.stream.stop_stream()
        self.stream.close()
        self.pya.terminate()

def main(args=None):
    rclpy.init(args=args)
    node = VoiceCommandPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown_node()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
