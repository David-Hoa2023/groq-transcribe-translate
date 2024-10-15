import os
import sys
from groq import Groq
from gtts import gTTS
from gtts.lang import tts_langs
import tempfile
import speech_recognition as sr

# Language mapping
LANGUAGE_MAP = {
    "English": "en",
    "Vietnamese": "vi",
    "French": "fr",
    "Chinese": "zh-CN",
    "Spanish": "es",
    "Korean": "ko",
    "Japanese": "ja"
}

def get_api_key():
    if len(sys.argv) > 1:
        return sys.argv[1]
    return os.environ.get("GROQ_API_KEY")

def validate_api_key(api_key):
    if not api_key or len(api_key) < 20:  # Less strict validation
        return False
    return True

def initialize_client(api_key):
    if not validate_api_key(api_key):
        print("Error: The provided API key seems too short or empty.")
        return None
    return Groq(api_key=api_key)

def translate(client, text, source_lang, target_lang):
    if not client:
        print("Error: Groq client not initialized.")
        return None

    prompt = f"""Translate the following text from {source_lang} to {target_lang}:
    
    {text}
    
    Translation:"""
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="Llama-3.2-90b-Text-Preview",
            temperature=0.5,
            max_tokens=1000,
        )
        
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"An error occurred during translation: {str(e)}")
        return None

def text_to_speech(text, lang):
    try:
        lang_code = LANGUAGE_MAP.get(lang, lang)
        supported_langs = tts_langs()
        
        print(f"Attempting TTS with language code: {lang_code}")
        
        if lang_code not in supported_langs:
            print(f"Language code {lang_code} not supported. Available languages: {', '.join(supported_langs.keys())}")
            fallback_lang = 'en'
            print(f"Falling back to English (en)")
            lang_code = fallback_lang
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            tts = gTTS(text=text, lang=lang_code, slow=False)
            tts.save(temp_file.name)
            return temp_file.name
    except Exception as e:
        print(f"An error occurred during text-to-speech conversion: {str(e)}")
        return None

def transcribe_audio(audio_file_path, language='en-US'):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file_path) as source:
            audio = recognizer.record(source)
        
        # Attempt to recognize speech with increased sensitivity
        text = recognizer.recognize_google(audio, language=language, show_all=True)
        
        if not text:
            return "Speech recognition could not understand the audio. Please try speaking more clearly or in a quieter environment."
        
        # If we get multiple results, return the most confident one
        if isinstance(text, list):
            best_result = max(text, key=lambda alt: alt.get('confidence', 0))
            return best_result.get('transcript', "Could not determine the best transcription.")
        
        return text
    except sr.UnknownValueError:
        return "Speech recognition could not understand the audio. Please try speaking more clearly or in a quieter environment."
    except sr.RequestError as e:
        return f"Could not request results from Google Speech Recognition service. Error: {e}"
    except Exception as e:
        return f"An unexpected error occurred during transcription: {str(e)}"

if __name__ == "__main__":
    print("This script is intended to be imported, not run directly.")
    print("Please use groq_translator_streamlit.py to run the Streamlit app.")
