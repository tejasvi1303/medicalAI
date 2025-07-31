# app.py

# === patch gradio_client to safely handle boolean schemas ===
import gradio_client.utils as _gu
from gradio_client.utils import APIInfoParseError

_orig_json_schema_to_python_type = getattr(_gu, "_json_schema_to_python_type", None)
_orig_get_type = getattr(_gu, "get_type", None)

def _safe_json_schema_to_python_type(schema, defs=None):
    try:
        if _orig_json_schema_to_python_type:
            return _orig_json_schema_to_python_type(schema, defs)
        return "Any"
    except APIInfoParseError:
        return "Any"
    except Exception:
        return "Any"

def _safe_get_type(schema):
    if isinstance(schema, bool):
        return "Any"
    if _orig_get_type:
        try:
            return _orig_get_type(schema)
        except Exception:
            return "Any"
    return "Any"

if _orig_json_schema_to_python_type:
    _gu._json_schema_to_python_type = _safe_json_schema_to_python_type
if _orig_get_type:
    _gu.get_type = _safe_get_type

# === imports ===
import gradio as gr
import speech_recognition as sr
from gtts import gTTS
from backend import predict_disease
from pydub import AudioSegment

# === ffmpeg path ===
AudioSegment.converter = "C:\\ffmpeg\\bin\\ffmpeg.exe"

# === speech-to-text ===
def speech_to_text(audio_path):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
        return text, None
    except sr.UnknownValueError:
        return "", "Sorry, I could not understand the audio."
    except sr.RequestError:
        return "", "Error connecting to the speech recognition service."
    except Exception as e:
        return "", f"Speech-to-text failure: {e}"

# === text-to-speech ===
def text_to_speech(text):
    try:
        tts = gTTS(text)
        output_path = "output.mp3"
        tts.save(output_path)
        return output_path
    except Exception:
        return None

# === core function ===
def chatbot_interface(audio, typed_input=""):
    if typed_input.strip():  # Use typed input if available
        transcript = typed_input.strip()
        error = None
    else:
        transcript, error = speech_to_text(audio)
        if error:
            return "", "", f"Error: {error}"

    try:
        disease, advice = predict_disease(transcript)
    except Exception as e:
        return transcript, "", f"Prediction error: {e}"

    combined_text = f"Disease: {disease}\nAdvice: {advice}"
    audio_output_path = text_to_speech(f"{disease}. {advice}")
    return transcript, combined_text, audio_output_path

# === Gradio UI ===
with gr.Blocks(title="AI Medical Voice Chatbot") as interface:
    gr.Markdown("## AI Medical Voice Chatbot")
    gr.Markdown("Speak or type your symptoms and get disease prediction with voice advice.")

    audio_input = gr.Audio(type="filepath", label="Upload or record your symptoms (wav/mp3)")
    typed_input = gr.Textbox(label="Or type your symptoms here (overrides audio if filled)")
    transcript_box = gr.Textbox(label="Used Symptoms (transcript or typed)", interactive=False)
    output_box = gr.Textbox(label="Predicted Disease and Advice", interactive=False)
    voice_output = gr.Audio(label="Voice Response", type="filepath")
    submit = gr.Button("Submit")

    submit.click(
        fn=chatbot_interface,
        inputs=[audio_input, typed_input],
        outputs=[transcript_box, output_box, voice_output],
    )

interface.launch(share=False, server_name="127.0.0.1", server_port=7860)
