import wave
import sys
import pyaudio
import tensorflow as tf

# pip install pyaudio, librosa, matplotlib, tensorflow-datasets, pydub, ffmpeg
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

def mel_spectrogram(audio, label):
    mel_spec = mel_layer(audio)
    mel_spec = tf.expand_dims(mel_spec, -1) # Add channel dimension for CNN
    return mel_spec, label


def record(index):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1 if sys.platform == 'darwin' else 2
    RATE = 16000
    RECORD_SECONDS = 1


    with wave.open('output.wav', 'wb') as wf:
        p = pyaudio.PyAudio()
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)

        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, input_device_index=index)

        print('Recording...')
        for _ in range(0, RATE // CHUNK * RECORD_SECONDS):
            wf.writeframes(stream.read(CHUNK))
        print('Done')

        stream.close()
        p.terminate()


