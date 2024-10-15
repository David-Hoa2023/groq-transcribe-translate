import streamlit as st
import os
import tempfile
import sounddevice as sd
import soundfile as sf
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy import signal
from groq_translator import initialize_client, translate, text_to_speech, LANGUAGE_MAP

st.set_page_config(page_title="Groq Translator", page_icon="🌐", layout="wide")

st.title("Groq Translator with Text-to-Speech and Voice Input")

# Sidebar for API key input
api_key = st.sidebar.text_input("Enter your Groq API key:", type="password")
client = None

if api_key:
    client = initialize_client(api_key)
    if client:
        st.sidebar.success("API key is valid!")
    else:
        st.sidebar.error("Invalid API key. Please try again.")

def safe_remove_file(file_path, max_retries=5, delay=0.1):
    for _ in range(max_retries):
        try:
            os.unlink(file_path)
            return True
        except PermissionError:
            time.sleep(delay)
    return False

def plot_audio_wave(audio_data):
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.plot(audio_data)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Amplitude')
    ax.set_title('Audio Waveform')
    return fig

def reduce_noise(audio_data, sample_rate):
    # Simple noise reduction using a high-pass filter
    b, a = signal.butter(5, 100 / (sample_rate / 2), btype='highpass')
    return signal.filtfilt(b, a, audio_data)

def trim_silence(audio_data, threshold=0.01, chunk_size=1000):
    trim_start = 0
    trim_end = len(audio_data)
    
    for i in range(0, len(audio_data), chunk_size):
        if np.max(np.abs(audio_data[i:i+chunk_size])) > threshold:
            trim_start = i
            break
    
    for i in range(len(audio_data) - chunk_size, 0, -chunk_size):
        if np.max(np.abs(audio_data[i:i+chunk_size])) > threshold:
            trim_end = i + chunk_size
            break
    
    return audio_data[trim_start:trim_end]

def transcribe_audio_with_whisper(filename, language):
    try:
        with open(filename, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(filename, file.read()),
                model="whisper-large-v3-turbo",
                prompt="Specify context or spelling",  # Optional
                response_format="json",  # Optional
                language=language,  # Optional
                temperature=0.0  # Optional
            )
        st.write(f"Debug: Raw transcription result: {transcription}")
        if hasattr(transcription, 'text'):
            return transcription.text
        elif isinstance(transcription, dict) and 'text' in transcription:
            return transcription['text']
        else:
            st.error(f"Unexpected transcription format: {type(transcription)}")
            return "Transcription failed: Unexpected format"
    except Exception as e:
        st.error(f"An error occurred during transcription: {str(e)}")
        return f"Transcription failed: {str(e)}"

# Main app
if client:
    col1, col2 = st.columns(2)

    with col1:
        source_lang = st.selectbox("Select source language:", list(LANGUAGE_MAP.keys()), key="source_lang")
        input_method = st.radio("Choose input method:", ["Text", "Microphone", "System Sound", "Upload Audio"])

        if input_method == "Text":
            text_to_translate = st.text_area("Enter text to translate:", height=150)
        elif input_method in ["Microphone", "System Sound"]:
            show_wave = st.checkbox("Show audio wave during recording")
            wave_placeholder = st.empty()
            
            # Add volume control slider
            volume_gain = st.slider("Volume Gain", min_value=1.0, max_value=5.0, value=2.0, step=0.1)

            # Add recording duration input
            duration = st.number_input("Recording duration (seconds)", min_value=1, max_value=300, value=10)

            if st.button(f"Record Audio ({duration} seconds)"):
                st.write("Recording started...")
                sample_rate = 44100  # Sample rate
                
                try:
                    if input_method == "Microphone":
                        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
                        sd.wait()
                    else:  # System Sound
                        with sd.InputStream(channels=1, samplerate=sample_rate, dtype='float32') as stream:
                            audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
                            sd.wait()
                    
                    audio_data = audio_data.flatten()

                    st.write("Recording finished.")
                    st.write(f"Debug: Raw audio data size: {len(audio_data)}")
                    st.write(f"Debug: Audio data min: {np.min(audio_data)}, max: {np.max(audio_data)}")
                
                    if len(audio_data) > 0:
                        # Apply noise reduction
                        audio_data = reduce_noise(audio_data, sample_rate)
                        
                        # Trim silence
                        audio_data = trim_silence(audio_data)
                        
                        # Apply volume gain
                        audio_data = audio_data * volume_gain
                        
                        st.write(f"Debug: Processed audio data size: {len(audio_data)}")
                        st.write(f"Debug: Processed audio data min: {np.min(audio_data)}, max: {np.max(audio_data)}")
                        
                        # Save as PCM WAV
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                            sf.write(temp_audio.name, audio_data, sample_rate, subtype='PCM_16')
                            st.session_state.audio_file = temp_audio.name
                        st.write(f"Debug: Audio file saved: {st.session_state.audio_file}")
                        st.write(f"Debug: Audio file size: {os.path.getsize(st.session_state.audio_file)} bytes")
                    
                        if show_wave:
                            fig = plot_audio_wave(audio_data)
                            wave_placeholder.pyplot(fig)
                            plt.close(fig)
                    else:
                        st.warning("No audio was detected. Please try the following:")
                        st.write("1. Check if your microphone or system audio is properly connected and enabled.")
                        st.write("2. Increase the volume gain using the slider above.")
                        st.write("3. Speak louder or bring the microphone closer.")
                        st.write("4. If using system sound, make sure audio is playing during recording.")
                        st.write("If the issue persists, try using the 'Upload Audio' option instead.")
                
                except Exception as e:
                    st.error(f"An error occurred during recording: {str(e)}")
                    st.write("Please make sure your audio devices are properly configured and try again.")

            if 'audio_file' in st.session_state and st.session_state.audio_file:
                st.audio(st.session_state.audio_file, format="audio/wav")
                if st.button("Transcribe Audio"):
                    st.write("Transcribing...")
                    if os.path.exists(st.session_state.audio_file):
                        text_to_translate = transcribe_audio_with_whisper(st.session_state.audio_file, LANGUAGE_MAP[source_lang])
                        if text_to_translate.startswith("Transcription failed"):
                            st.error(text_to_translate)
                            text_to_translate = ""
                        else:
                            st.text_area("Transcribed text:", text_to_translate, height=150)
                    else:
                        st.error("Audio file not found. Please record audio again.")
                        st.session_state.audio_file = None

        elif input_method == "Upload Audio":
            uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"])
            if uploaded_file is not None:
                st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")
                if st.button("Transcribe Uploaded Audio"):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{time.time()}.{uploaded_file.name.split('.')[-1]}") as temp_audio:
                        temp_audio.write(uploaded_file.getvalue())
                        text_to_translate = transcribe_audio_with_whisper(temp_audio.name, LANGUAGE_MAP[source_lang])
                        if text_to_translate.startswith("Transcription failed"):
                            st.error(text_to_translate)
                            text_to_translate = ""
                        else:
                            st.text_area("Transcribed text:", text_to_translate, height=150)
                    safe_remove_file(temp_audio.name)

    with col2:
        target_lang = st.selectbox("Select target language:", list(LANGUAGE_MAP.keys()), key="target_lang")
        
        if 'text_to_translate' in locals() and text_to_translate:
            with st.spinner("Translating..."):
                translated_text = translate(client, text_to_translate, source_lang, target_lang)
            if translated_text:
                st.text_area("Translated text:", translated_text, height=150)
                
                with st.spinner("Converting to speech..."):
                    audio_file = text_to_speech(translated_text, LANGUAGE_MAP[target_lang])
                if audio_file:
                    st.audio(audio_file, format="audio/mp3")
                    if not safe_remove_file(audio_file):
                        st.warning(f"Could not remove temporary file: {audio_file}")
                else:
                    st.error("Failed to convert text to speech.")
            else:
                st.error("Translation failed. Please try again.")
        else:
            st.warning("Please enter or record text to translate.")

else:
    st.warning("Please enter a valid Groq API key in the sidebar to use the translator.")

st.markdown("---")
st.markdown("Created with Streamlit and Groq API")
