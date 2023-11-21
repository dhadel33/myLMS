!pip install soundfile
import numpy as np
import soundfile as sf
from scipy.signal import spectrogram, welch
import time
import streamlit as st
import matplotlib.pyplot as plt

# Load audio file
def load_audio_file():
    file_path = st.text_input("Enter the path of the audio file:")
    audio_signal, fs = sf.read(file_path)
    return audio_signal, fs
# Load audio file
audio_signal, fs = load_audio_file()    

# Convert stereo to mono if needed
if audio_signal.ndim == 2:
    audio_signal = np.mean(audio_signal, axis=1)

# Add Gaussian noise to the audio signal
noise_level = 0.1
noise = noise_level * np.random.randn(len(audio_signal))
noisy_audio_signal = audio_signal + noise

# LMS filter parameters
mu = 0.01  # Step size
order = 32  # Filter order

# LMS filter function
def lms_filter(input_signal, desired_signal, order, mu):
    start_time = time.time()  # Record the start time
    num_samples = len(input_signal)
    weights = np.zeros(order)
    output_signal = np.zeros(num_samples)
    mse_values = np.zeros(num_samples - order)

    for i in range(order, num_samples):
        x = input_signal[i-order:i]
        y_hat = np.dot(weights, x)
        error = desired_signal[i] - y_hat
        weights = weights + mu * error * x
        output_signal[i] = y_hat
        mse_values[i - order] = np.mean((desired_signal[i-order+1:i+1] - y_hat) ** 2)

    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate the elapsed time
    print(f"Time taken for LMS process: {elapsed_time:.4f} seconds")

    return output_signal, mse_values

# Apply LMS filter to the noisy audio signal
desired_signal = audio_signal  # Using the original audio as the desired signal for denoising
filtered_audio, mse_values = lms_filter(noisy_audio_signal, desired_signal, order, mu)

# Calculate the error between the input audio and the filtered signal
error = audio_signal - filtered_audio

# Welch method for PSD estimation
frequencies_input, psd_input = welch(audio_signal, fs, nperseg=1024)
frequencies_filtered, psd_filtered = welch(filtered_audio, fs, nperseg=1024)
# MUSIC algorithm for PSD estimation
frequencies, times, psd_original = spectrogram(audio_signal, fs)
_, _, psd_filtered_spect = spectrogram(filtered_audio, fs)

# Plotting
st.title('Audio Denoising with LMS Filter')

# Plot original audio
st.subheader('Original Audio Signal')
st.line_chart(audio_signal)

# Plot noisy audio
st.subheader('Noisy Audio Signal')
st.line_chart(noisy_audio_signal)

# Plot filtered audio
st.subheader('Filtered Audio Signal (LMS)')
st.line_chart(filtered_audio)

# Plot error
error = audio_signal - filtered_audio
st.subheader('Error Between Original and Filtered Signals')
st.line_chart(error)

# Plot time domain MSE
st.subheader('Time Domain MSE')
st.line_chart(mse_values)
# Show the Streamlit app
st.write("This is your Streamlit app.")
