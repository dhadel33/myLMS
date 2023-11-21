import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import spectrogram, welch
import time

# Load audio file
file_path = "C:/Users/Dhadel/ADSPRECORDING.wav"
audio_signal, fs = sf.read(file_path)

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
plt.figure(figsize=(10, 10))

plt.subplot(8, 1, 1)
plt.plot(audio_signal, label='Original Audio Signal')
plt.title('Original Audio Signal')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.legend()
plt.subplots_adjust(hspace=3)

plt.subplot(8, 1, 2)
plt.plot(noisy_audio_signal, label='Noisy Audio Signal', alpha=0.7)
plt.title('Noisy Audio Signal')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.legend()
plt.subplots_adjust(hspace=3)

plt.subplot(8, 1, 3)
plt.plot(filtered_audio, label='Filtered Audio Signal (LMS)', linestyle='dashed')
plt.title('Filtered Audio Signal (LMS)')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.legend()
plt.subplots_adjust(hspace=3)

plt.subplot(8, 1, 4)
plt.plot(error, label='Error', color='red')
plt.title('Error Between Original and Filtered Signals')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.legend()
plt.subplots_adjust(hspace=3)

plt.subplot(8, 1, 5)
plt.plot(mse_values, label='Time Domain MSE', color='green')
plt.title('Time Domain MSE')
plt.xlabel('Sample')
plt.ylabel('MSE')
plt.legend()
plt.subplots_adjust(hspace=3)

plt.subplot(8, 1, 6)
plt.pcolormesh(times, frequencies, 10 * np.log10(psd_original), shading='auto', cmap='viridis')
plt.title('Algorithm PSD - Original Signal')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.subplots_adjust(hspace=3)

plt.subplot(8, 1, 7)
plt.pcolormesh(times, frequencies, 10 * np.log10(psd_filtered_spect), shading='auto', cmap='viridis')
plt.title('Algorithm PSD - Filtered Signal (LMS)')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.subplots_adjust(hspace=3)

plt.subplot(8, 1, 8)
plt.semilogy(frequencies_input, psd_input, label='Input Signal')
plt.semilogy(frequencies_filtered, psd_filtered, label='Filtered Signal (LMS)')
plt.title('Power Spectral Density (PSD)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (dB/Hz)')
plt.legend()
plt.subplots_adjust(hspace=3)

plt.tight_layout()
plt.show()
