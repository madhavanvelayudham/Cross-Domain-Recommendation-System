import speech_recognition as sr

def transcribe_audio(file_path: str) -> str:
    try:
        print("Using Google Speech Recognition...")

        recognizer = sr.Recognizer()

        with sr.AudioFile(file_path) as source:
            audio = recognizer.record(source)

        text = recognizer.recognize_google(audio)

        print("Transcribed:", text)

        return text

    except Exception as e:
        print("❌ ERROR:", e)
        return ""