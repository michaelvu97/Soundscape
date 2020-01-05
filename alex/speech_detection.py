#module to detect speech

import speech_recognition as sr;

r = sr.Recognizer()

def has_speech(audio):
    try:
        r.recognize_sphinx(audio);
        return True
    except sr.UnknownValueError:
        print("unknown")
    except sr.RequestError as e:
        print(e)
    
    return False


if __name__ == "__main__":
    print("speech recog version: " + sr.__version__)
    testwav = 'test.wav'  
    with sr.AudioFile(testwav) as source: 
        audio = r.record(source)
        print(type(audio))
        print(has_speech(audio))
