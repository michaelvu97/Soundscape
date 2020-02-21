#module to detect speech

import speech_recognition as sr;
import time
import threading
import sys

r = sr.Recognizer()
#r.energy_threshold = 1000;
print(r.energy_threshold)


def listen_in_thread(mic, callback):
    stop_listening = r.listen_in_background(mic, callback)
    time.sleep(0.4)
    stop_listening(wait_for_stop=False)

class SpeechDetector:   
    def __init__ (self, index=None):
        if index is None:
            index=7    

        self.mic = sr.Microphone(device_index=index)
    
    def listen(self, callback):
        #listen_thread = threading.Thread(target=listen_in_thread, args=(self.mic, callback))
        #listen_thread.start()
        listen_in_thread(self.mic, callback)
        

    def is_speech(self, audios):
        if(type(audios) == type([])):
            outputs = []
            for a in audios:
                outputs.append(self.signal_is_speech(a))
            return outputs
        else:
            return self.signal_is_speech(audios)
    
    def signal_is_speech(self, audio):
        #return True if recognize(audio) is not None else False
        words = self.recognize(audio)
        if(words is None):
            return False
        return True
    
    
    def recognize(self, audio):
        #if (type(audio) != type(AudioData)):
        #audio = r.listen(audio)
        
        try:
            return r.recognize_google(audio);
        except sr.UnknownValueError:
            #print("unknown value error")
            return None
        except sr.RequestError as e:
            print("request error")
            print(e)
            return None
    
def callback(recognizer, audio):
    try:
        output = recognizer.recognize_google(audio)
        print("Google Speech Recognition thinks you said " + output)
    except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))


if __name__ == "__main__":
    vad =  SpeechDetector();

    print(sr.Microphone.list_microphone_names())
   
    try:

        m = sr.Microphone()
        stop_listening = r.listen_in_background(m, callback)
        #time.sleep(0.4)
        #stop_listening(wait_for_stop=False)
    except:
        print("failed live test with mic")

    print("speech recog version: " + sr.__version__)
    wordtests = [
        ["test1.wav", "all the way"],
        ["test2.wav", None],
        ["test3.wav", "mert say something ass"]
    ];
    

    i = 1;
    for test in wordtests:
        
        testwav = test[0]
        expectedval = test[1]

        with sr.AudioFile(testwav) as source: 
            audio = r.record(source)
            results = vad.recognize(audio)
            if(expectedval == results):
                print("test " + str(i) + " passed")
            else: 
                print("test " + str(i) + " failed")
                print(testwav)
                print("expected = " + str(expectedval))
                print("results = " + str(results))
        
        i = i + 1;
    binarytests = [
        ["btb1.wav", False],
        ["btb2.wav", True]
    ];

    audios = []
    expected = []
    for test in binarytests:

        testwav = test[0]
        expectedval = test[1]
        
        with sr.AudioFile(testwav) as source: 
            audio = r.record(source)
            audios.append(audio)
            expected.append(expectedval)

    results = vad.is_speech(audios)
    for j in range(0, len(results)):
        expectedval = expected[j]
        result = results[j]
        if(expectedval == result):
            print("test " + str(i) + " passed")
        else: 
            print("test " + str(i) + " failed")
            print(testwav)
            print("expected = " + str(expectedval))
            print("results = " + str(result))
        i = i + 1

