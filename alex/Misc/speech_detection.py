#module to detect speech

import speech_recognition as sr;
import time
import threading
import sys



class SpeechDetector:   
if __name__ == "__main__":
    vad =  SpeechDetector();
'''
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
'''
