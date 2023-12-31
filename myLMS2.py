import numpy as np
import streamlit as st


# Function to generate sinusoidal signal based on user input
def generate_sinusoidal_signal():
    try:
        frequency = float(st.text_input("Enter the frequency of the sinusoidal signal (Hz):"))
        amplitude = float(st.text_input("Enter the amplitude of the sinusoidal signal:"))
        duration = float(st.text_input("Enter the duration of the signal (seconds):"))
        fs = float(st.text_input("Enter the sampling frequency (e.g., 44100):"))

        t = np.arange(0, duration, 1/fs)
        sinusoidal_signal = amplitude * np.sin(2 * np.pi * frequency * t)

        return sinusoidal_signal, fs
    except ValueError as e:
        st.error(f"Error: {e}. Please enter valid numeric values.")
        return None, None

# LMS filter function
def lms_filter(input_signal, desired_signal, order, mu):
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
        mse_values[i - order] = np.mean((desired_signal[i - order:i + 1] - y_hat) ** 2)
    return output_signal, mse_values

# Streamlit app
    st.title('Audio Denoising with LMS Filter - Sinusoidal Signal')
    
    # Explanation or instructions
    st.write("This app allows you to generate a sinusoidal signal, add noise, apply an LMS filter for denoising, and visualize the results.")
    
    # Generate sinusoidal signal
    sinusoidal_signal, fs = generate_sinusoidal_signal()
    
    # Check if sinusoidal signal is generated
    if sinusoidal_signal is not None:
        # Add Gaussian noise to the sinusoidal signal
        noise_level = 0.1
        noise = noise_level * np.random.randn(len(sinusoidal_signal))
        noisy_sinusoidal_signal = sinusoidal_signal + noise
    
        # LMS filter parameters
        mu = st.slider('Step Size (mu)', 0.01, 1.0, 0.01)
        order = st.slider('Filter Order', 5, 100, 32)
    
        # Apply LMS filter to the noisy sinusoidal signal
        desired_signal = sinusoidal_signal
        filtered_signal, mse_values = lms_filter(noisy_sinusoidal_signal, desired_signal, order, mu)
    
        # Plotting
        st.subheader('Original Sinusoidal Signal')
        st.line_chart(sinusoidal_signal)
    
        st.subheader('Noisy Sinusoidal Signal')
        st.line_chart(noisy_sinusoidal_signal)
    
        st.subheader('Filtered Sinusoidal Signal (LMS)')
        st.line_chart(filtered_signal)
    
        st.subheader('Time Domain MSE')
        st.line_chart(mse_values)
    
        # Show the Streamlit app
        st.write("This is your Streamlit app.")
