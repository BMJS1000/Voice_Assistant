
# ==========================================
# VOICE AI ASSISTANT
# ==========================================

import os
import sounddevice as sd
from scipy.io.wavfile import write

from groq import Groq
from dotenv import load_dotenv
from gtts import gTTS
from playsound3 import playsound


# ==========================================
# INITIAL SETUP
# ==========================================

# Get the folder location of main.py

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

AUDIO_FILE = os.path.join(BASE_DIR, "input.wav")

RESPONSE_FILE = os.path.join(BASE_DIR, "response.mp3")


# Load Groq API key

load_dotenv(os.path.join(BASE_DIR, ".env"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("Groq API key not found! Check your .env file.")

client = Groq(api_key=GROQ_API_KEY)

print("Groq connected successfully!")


# ==========================================
# PART 1: USER VOICE INPUT
# ==========================================

def record_audio():

    duration = 8
    sample_rate = 16000

    print("\nListening... Speak now!")

    recording = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="int16"
    )

    sd.wait()

    print("Recording completed!")

    # Save audio recording

    write(
        AUDIO_FILE,
        sample_rate,
        recording
    )

    print("Voice saved successfully as input.wav")


# ==========================================
# PART 2: SPEECH TO TEXT (STT)
# ==========================================

def speech_to_text():

    print("\nConverting voice into text...")

    with open(AUDIO_FILE, "rb") as audio_file:

        transcription = client.audio.transcriptions.create(

            file=audio_file,

            model="whisper-large-v3-turbo",

            language="en",

            response_format="json",

            temperature=0.0

        )

    user_text = transcription.text.strip()

    print("\nTranscribed Text:")
    print(user_text)

    return user_text


# ==========================================
# PART 3: GROQ LLM RESPONSE
# ==========================================

def get_llm_response(user_text):

    print("\nAI is thinking...")

    response = client.chat.completions.create(

        model="openai/gpt-oss-20b",

        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful AI assistant. "
                    "Give simple, clear and concise answers."
                )
            },
            {
                "role": "user",
                "content": user_text
            }
        ]

    )

    answer = response.choices[0].message.content

    if not answer:
        raise ValueError("No response received from the AI.")

    print("\nAI Response:")
    print(answer)

    return answer


# ==========================================
# PART 4: TEXT TO SPEECH (TTS)
# ==========================================

def text_to_speech(answer):

    print("\nConverting AI response into speech...")

    # Convert text into speech

    tts = gTTS(
        text=answer,
        lang="en"
    )

    # Save speech as MP3

    tts.save(RESPONSE_FILE)

    print("\nPlaying AI voice response...")

    # Play generated audio

    playsound(
        RESPONSE_FILE,
        block=True
    )

    print("Audio playback completed!")


# ==========================================
# PART 5: MAIN FUNCTION
# ==========================================

def main():

    print("\n====================================")
    print("       WELCOME TO VOICE AI")
    print("====================================")

    try:

        # STEP 1: USER VOICE INPUT

        record_audio()


        # STEP 2: SPEECH TO TEXT

        user_text = speech_to_text()

        if not user_text:

            print("No speech detected. Please try again.")

            return


        # STEP 3: LLM RESPONSE

        answer = get_llm_response(user_text)


        # STEP 4: TEXT TO SPEECH

        text_to_speech(answer)


        print("\n====================================")
        print(" AI VOICE ASSISTANT COMPLETED!")
        print("====================================")


    except Exception as error:

        print("\nAn error occurred:", error)


# ==========================================
# RUN PROGRAM
# ==========================================

if __name__ == "__main__":

    main()
