
# ==================================================
# VOICE AI ASSISTANT - STREAMLIT UI
# ==================================================

import os
import io
from pathlib import Path

import streamlit as st
from groq import Groq
from dotenv import load_dotenv
from gtts import gTTS


# ==================================================
# 1. INITIAL SETUP
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(
    page_title="Voice AI Studio",
    page_icon="🎙️",
    layout="centered"
)

if not GROQ_API_KEY:
    st.error("Groq API key not found. Please check your .env file.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

LLM_MODEL = "openai/gpt-oss-20b"
STT_MODEL = "whisper-large-v3-turbo"


# ==================================================
# 2. SESSION STATE
# ==================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "latest_audio" not in st.session_state:
    st.session_state.latest_audio = None

if "recording_version" not in st.session_state:
    st.session_state.recording_version = 0


# ==================================================
# 3. CUSTOM CSS - UI DESIGN
# ==================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

.stApp {
    background: linear-gradient(
        135deg,
        #f8f4ff 0%,
        #fff4f8 50%,
        #f0f7ff 100%
    );
    font-family: 'DM Sans', sans-serif;
}

.block-container {
    max-width: 900px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stHeader"] {
    background: transparent;
}

/* HERO SECTION */

.hero {
    background: linear-gradient(
        120deg,
        #8155d9,
        #d768aa,
        #f49b91
    );

    padding: 38px;
    border-radius: 26px;
    color: white;
    margin-bottom: 25px;

    box-shadow: 0 12px 30px rgba(135, 90, 190, 0.18);
}

.hero h1 {
    color: white;
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 8px;
}

.hero p {
    color: #fff3fa;
    font-size: 16px;
}

.hero-tag {
    background: rgba(255,255,255,0.2);
    display: inline-block;
    padding: 7px 13px;
    border-radius: 30px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
}

/* SECTION HEADINGS */

.section-heading {
    color: #46305f;
    font-size: 23px;
    font-weight: 800;
    margin-top: 20px;
    margin-bottom: 10px;
}

/* CARDS */

.info-card {
    background: rgba(255,255,255,0.85);
    padding: 20px;
    border-radius: 20px;
    border: 1px solid #eee5fa;
    margin-bottom: 12px;
}

/* BUTTONS */

.stButton > button {
    border: none;
    border-radius: 14px;
    min-height: 48px;

    background: linear-gradient(
        90deg,
        #8155d9,
        #d768aa
    );

    color: white;
    font-weight: 700;
    font-size: 15px;

    box-shadow: 0 5px 14px rgba(129,85,217,0.2);
}

.stButton > button:hover {
    background: linear-gradient(
        90deg,
        #7045c5,
        #c95298
    );

    color: white;
    border: none;
}

/* CHAT */

[data-testid="stChatMessage"] {
    border-radius: 18px;
    padding: 12px;
    border: 1px solid #eee5fa;
}

/* AUDIO */

[data-testid="stAudioInput"] {
    background: #fff4fa;
    padding: 15px;
    border-radius: 16px;
}

/* SMALL TEXT */

.footer {
    text-align: center;
    color: #927da8;
    font-size: 12px;
    padding: 25px;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# 4. HERO / HEADER
# ==================================================

st.markdown("""
<div class="hero">

    <div class="hero-tag">
        ✦ YOUR PERSONAL AI ASSISTANT
    </div>

    <h1>Hello there! 👋</h1>

    <h2 style="color:white; font-weight:600;">
        Let's talk, not type.
    </h2>

    <p>
        Ask anything using your voice.
        I'll listen, think and talk back to you!
    </p>

    <p style="font-size:14px;">
        🎙️ Voice Input &nbsp; ✨ AI Intelligence
        &nbsp; 🔊 Voice Output
    </p>

</div>
""", unsafe_allow_html=True)


# ==================================================
# 5. SIDEBAR SETTINGS
# ==================================================

with st.sidebar:

    st.markdown("## ⚙️ Settings")

    st.write("Customize your voice assistant.")

    language_choice = st.selectbox(
        "Speech Input Language",
        [
            "Auto Detect",
            "English",
            "Hindi",
            "Gujarati"
        ]
    )

    language_codes = {
        "Auto Detect": None,
        "English": "en",
        "Hindi": "hi",
        "Gujarati": "gu"
    }

    language = language_codes[language_choice]

    st.divider()

    st.markdown("### 🤖 AI Models")

    st.caption("Speech-to-Text: Whisper")
    st.caption("AI Model: GPT-OSS 20B")
    st.caption("Voice Output: Google TTS")

    st.divider()

    if st.button("🗑️ Clear Conversation",
                 use_container_width=True):

        st.session_state.chat_history = []
        st.session_state.latest_audio = None
        st.session_state.recording_version += 1

        st.rerun()


# ==================================================
# 6. VOICE INPUT
# ==================================================

st.markdown(
    '<div class="section-heading">🎙️ Talk to your AI</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="info-card">
    <b>Ready when you are!</b><br>
    <span style="color:#88769b;">
    Click the microphone, record your question,
    and send it to your AI assistant.
    </span>
</div>
""", unsafe_allow_html=True)


recording = st.audio_input(
    "🎤 Record your voice message",
    sample_rate=16000,
    key=f"voice_recording_{st.session_state.recording_version}"
)


if recording is not None:

    st.audio(recording)

    st.caption("✅ Voice recording received!")


# ==================================================
# 7. PROCESS VOICE MESSAGE
# ==================================================

process_button = st.button(
    "✨ Send Message to AI",
    type="primary",
    use_container_width=True,
    disabled=recording is None
)


if process_button and recording is not None:

    try:

        # ------------------------------------------
        # STEP 1: SPEECH TO TEXT
        # ------------------------------------------

        with st.spinner("🎧 Listening and converting your voice into text..."):

            transcription_args = {
                "file": (
                    "voice_input.wav",
                    recording.getvalue()
                ),

                "model": STT_MODEL,

                "response_format": "json",

                "temperature": 0.0
            }

            if language:
                transcription_args["language"] = language

            transcription = client.audio.transcriptions.create(
                **transcription_args
            )

            user_text = transcription.text.strip()


        if not user_text:

            st.warning("No speech detected. Please try recording again.")

            st.stop()


        # ------------------------------------------
        # STEP 2: LLM RESPONSE
        # ------------------------------------------

        with st.spinner("🧠 Your AI assistant is thinking..."):

            messages = [

                {
                    "role": "system",

                    "content": (
                        "You are a friendly and helpful AI assistant. "
                        "Always answer in simple English. "
                        "Give clear, concise and easy-to-understand answers. "
                        "Avoid unnecessary formatting because your "
                        "answers will also be spoken aloud."
                    )
                }

            ]

            messages.extend(
                st.session_state.chat_history
            )

            messages.append({

                "role": "user",

                "content": user_text

            })


            response = client.chat.completions.create(

                model=LLM_MODEL,

                messages=messages

            )

            answer = response.choices[0].message.content

            if not answer:
                raise ValueError("The AI returned an empty response.")


        # Save conversation

        st.session_state.chat_history.append({

            "role": "user",

            "content": user_text

        })

        st.session_state.chat_history.append({

            "role": "assistant",

            "content": answer

        })


        # ------------------------------------------
        # STEP 3: TEXT TO SPEECH
        # ------------------------------------------

        st.session_state.latest_audio = None

        try:

            with st.spinner("🔊 Creating your AI voice response..."):

                audio_buffer = io.BytesIO()

                tts = gTTS(
                    text=answer,
                    lang="en"
                )

                tts.write_to_fp(audio_buffer)

                st.session_state.latest_audio = (
                    audio_buffer.getvalue()
                )

        except Exception as tts_error:

            st.warning(
                f"Text response generated, but voice generation failed: {tts_error}"
            )


        # Reset microphone for the next recording

        st.session_state.recording_version += 1

        st.rerun()


    except Exception as error:

        st.error(f"Something went wrong: {error}")


# ==================================================
# 8. DISPLAY CHAT HISTORY
# ==================================================

st.divider()

st.markdown(
    '<div class="section-heading">💬 Your Conversation</div>',
    unsafe_allow_html=True
)


if len(st.session_state.chat_history) == 0:

    st.markdown("""
    <div class="info-card">

        <h3 style="color:#8155d9;">
            Your conversation starts here! 🌸
        </h3>

        <p style="color:#88769b;">
            Record a voice message above and your AI
            assistant's response will appear here.
        </p>

    </div>
    """, unsafe_allow_html=True)


else:

    for message in st.session_state.chat_history:

        if message["role"] == "user":

            with st.chat_message("user", avatar="🧑"):
                st.markdown(message["content"])

        else:

            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(message["content"])


# ==================================================
# 9. AUDIO RESPONSE
# ==================================================

if st.session_state.latest_audio is not None:

    st.divider()

    st.markdown(
        '<div class="section-heading">🔊 Listen to your AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="info-card">'
        '<b>✨ Your spoken AI response is ready!</b>'
        '</div>',
        unsafe_allow_html=True
    )

    st.audio(
        st.session_state.latest_audio,
        format="audio/mp3"
    )


# ==================================================
# 10. FOOTER
# ==================================================

st.markdown("""
<div class="footer">
    Made with 💜 using Python, Streamlit and Groq AI
    <br>
    Your little voice-powered AI companion ✨
</div>
""", unsafe_allow_html=True)
