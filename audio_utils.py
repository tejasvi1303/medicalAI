import speech_recognition as sr
import pyttsx3

recognizer = sr.Recognizer()
engine = pyttsx3.init()

def speech_to_text():
    with sr.Microphone() as source:
        print("Listening for symptoms...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Could not understand audio"

def text_to_speech(text):
    engine.say(text)
    engine.runAndWait()
