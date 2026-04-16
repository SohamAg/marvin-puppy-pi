import pyaudio
import wave
from collections import deque
import numpy as np
import time
import tensorflow as tf
import threading
from math import gcd

from scipy.signal import resample_poly

import pycoral
from pycoral.utils import edgetpu
from pycoral.adapters import common
from pycoral.adapters import classify

#pya = pyaudio.PyAudio()

#for i in range(pya.get_device_count()):
#    info = pya.get_device_info_by_index(i)
#    print(f"Index {i}: {info['name']} | inputs: {info['maxInputChannels']} | rate: {info['defaultSampleRate']}")

#pya.terminate()

# Configuration
TPU = 1
FORMAT = pyaudio.paInt16 # 16-bit resolution
CHANNEL = 1 # 1 channel
SAMPLE_RATE = 44100 # 16kHz sampling rate
BUFFER_SIZE = 1024 # 2^12 samples for buffer
DEV_INDEX = 0 # device index found by p.get_device_info_by_index(ii)

WINDOW_LEN = 1 #  1s audio clips
HOP_LEN = 0.1 # 0.1s window slides
WINDOW_SAMPLES = int(SAMPLE_RATE*WINDOW_LEN)  # Frames per window
HOP_SAMPLES = int(HOP_LEN*SAMPLE_RATE)  # Frames per hop
wav_output_filename = 'test1.wav' # name of .wav file

# Debugging variables
count = 0
start = 0
length = 0

# Mel spectrogram layer configuration
fft_length = 512
sequence_stride = 160
sample_rate = 16000
num_mel_bins = 128

mel_layer = tf.keras.layers.MelSpectrogram(
        fft_length=fft_length,
        sequence_stride=sequence_stride,
        sampling_rate=sample_rate,
        num_mel_bins=num_mel_bins,
        power_to_db=True
    )

# Paths to access inference library and model
model_file = "models/wake_edgetpu.tflite"
# lib_path = "/usr/local/lib/libedgetpu.1.dylib"

labels_list = ['backward','bed','bird','cat','dog','down','eight','five','follow','forward','four',
                'go','happy','house','learn','left','marvin','nine','no','off','on','one','right','seven',
                'sheila','six','stop','three','tree','two','up','visual','wow','yes','zero']

class AudioProcessor:
  def __init__(self):
    self.stream = None  # stores pyaudio stream
    self.pya = pyaudio.PyAudio()  # Pyaudio object

    # stores current 1s audio data

    self.window_buffer = deque(np.zeros(WINDOW_SAMPLES, dtype=np.int16), maxlen=WINDOW_SAMPLES)   # buffer containing current 1s window
    self.window_lock = threading.Lock()
    self.window_update = threading.Event()

    # stores non processed 1s audio clips
    self.audio_clips = deque(maxlen=10)   # deque containing windows to be processed
    self.clips_lock = threading.Lock()
    self.clips_update = threading.Event()

    if TPU:
      # Set up TPU
        self.tpu_inference = edgetpu.make_interpreter(model_file)
        self.tpu_inference.allocate_tensors()
    else:
        # model used for inference
        self.model = tf.lite.Interpreter(model_path='models/wakeWord.tflite')
        self.model.allocate_tensors()

    # Threads used for processing and data transfer
    self.running = False

    self.processor_worker = None
    self.snapshot_worker = None

  """ Called by pyaudio to capture data from audio input
  via in_data. Stores audio data in self.window_buffer """
  def callback(self, in_data, frame_count, time_info, status):
    # might need to add case where frame_count != HOP Size
    formatted = np.frombuffer(in_data, dtype=np.int16)
    with self.window_lock:
      self.window_buffer.extend(formatted)
    self.window_update.set()

    return (None, pyaudio.paContinue)


  """ Starts pyaudio stream and initializes threads """
  def start_audio_capture(self):
    global start
    # create pyaudio stream
    self.stream = self.pya.open(format = FORMAT,
                        rate = SAMPLE_RATE,
                        channels = CHANNEL,
                        input_device_index = DEV_INDEX,
                        input = True,
                        frames_per_buffer=HOP_SAMPLES,
                        stream_callback=self.callback
                        )

    self.running = True
    self.processor_worker = threading.Thread(target=self.processor_loop, daemon=True)
    self.snapshot_worker = threading.Thread(target=self.snapshot_loop, daemon=True)
    self.processor_worker.start()
    self.snapshot_worker.start()

    self.stream.start_stream()
    print("Starting recording...")
    start = time.time()

  """ Stops audio stream and threads """
  def stop_audio_capture(self):
    global length
    self.running = False
    self.window_update.set() # Unblock threads
    self.clips_update.set()
    # stop the stream, close it, and terminate the pyaudio instantiation
    self.stream.stop_stream()
    self.stream.close()
    self.processor_worker.join()
    self.snapshot_worker.join()
    self.pya.terminate()
    print("Stopped recording")
    length = time.time() - start

  """ Controls flow of program. Calls start_audio_capture to begin program,
  waits for Command-C to stop program. """
  def process_control(self):
    self.start_audio_capture()

    try:
      while True:
        wait = threading.Event()
        wait.wait()
    except KeyboardInterrupt:
      pass
    finally:
      self.stop_audio_capture()

      print("Time of recording: ", length)
      print("Target number of frames: ", length/0.1)
      print("Number of frames recorded: ", count)

      # save the audio frames as .wav file
      wavefile = wave.open(wav_output_filename,'wb')
      wavefile.setnchannels(CHANNEL)
      wavefile.setsampwidth(self.pya.get_sample_size(FORMAT))
      wavefile.setframerate(SAMPLE_RATE)
      wavefile.writeframes(b''.join(self.window_buffer))
      wavefile.close()

  #----------------------------- Processing thread -----------------------------#
  """ Execution for snapshot_worker thread. Transfers 1s
  audio data of window_buffer to audio_clips """
  def snapshot_loop(self):
    # Need to make this multithreading so it does it automatically
    # Processing and inference
    while self.running:
      self.window_update.wait()
      self.window_update.clear()
      with self.window_lock:
        window_copy = np.array(self.window_buffer)
      with self.clips_lock:
        self.audio_clips.append(window_copy)
      self.clips_update.set()

  """ Execution for processor_worker thread. Pops latest 1s audio clip from
  audio_clips, processes and runs inference on it """
  def processor_loop(self):
    global count
    # can implement batch processing
    while self.running:
      self.clips_update.wait()
      self.clips_update.clear()
      with self.clips_lock:
        if self.audio_clips:
          current_clip = self.audio_clips.pop()

      count += 1
      mel_spec = self.process_audio(current_clip)

      if TPU:
        # Run inference
        common.set_input(self.tpu_inference,mel_spec)
        self.tpu_inference.invoke()
        classes = classify.get_classes(self.tpu_inference)
        if labels_list[classes[0].id] == "marvin":

          print("Voice detected. Start listening...")
      else:
         input_details = self.model.get_input_details()
         output_details = self.model.get_output_details()

         # Setting input onto model
         self.model.set_tensor(input_details[0]['index'], mel_spec)
         self.model.invoke()

         output_data = self.model.get_tensor(output_details[0]['index'])
         pred_ind = np.argmax(output_data[0])

         print("TensorFlow Model Prediction: ", labels_list[pred_ind])

         if labels_list[pred_ind] == "marvin":
            print("Activate!")

  """ Helper function to convert audio data into mel spectrogram """
  def process_audio(self, audio_data):
    # Need to resample audio to 16000 since model was trained on that
    g = gcd(44100, 16000)
    up = 16000 // g
    down = 44100 // g
    resampled_data = resample_poly(audio_data.astype(np.float64), up, down).astype(np.int16)

    target_samples = 16000
    front_padding = (target_samples - len(resampled_data)) // 2
    back_padding = (target_samples - len(resampled_data)) - front_padding
    padding = tf.constant([[front_padding,back_padding]])
    padded_waveform = tf.pad(resampled_data, padding)
    padded_waveform = tf.expand_dims(padded_waveform, 0)

    mel_spec = mel_layer(padded_waveform)
    mel_spec = tf.expand_dims(mel_spec, -1) # Add channel dimension for CNN

    return mel_spec



if __name__ == "__main__":
  processor = AudioProcessor()
  processor.process_control()
