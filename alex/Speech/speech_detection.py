#module to detect speech

import speech_recognition as sr;

r = sr.Recognizer()
print(r.energy_threshold)
r.energy_threshold = r.energy_threshold * 10;

def has_speech(audio):
    return True if recognize(audio) is not None else False

def recognize(audio):
    try:
        return r.recognize_sphinx(audio);
    except sr.UnknownValueError:
        return None
    except sr.RequestError as e:
        return None



if __name__ == "__main__":
    print("speech recog version: " + sr.__version__)
    wordtests = [
        ["test1.wav", "all the way"],
        ["test2.wav", None]
    ];
    
    i = 1;
    for test in wordtests:
        
        testwav = test[0]
        expectedval = test[1]

        with sr.AudioFile(testwav) as source: 
            audio = r.record(source)
            results = recognize(audio)
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

    for test in binarytests:
        
        testwav = test[0]
        expectedval = test[1]

        with sr.AudioFile(testwav) as source: 
            audio = r.record(source)
            results = has_speech(audio)
            if(expectedval == results):
                print("test " + str(i) + " passed")
            else: 
                print("test " + str(i) + " failed")
                print(testwav)
                print("expected = " + str(expectedval))
                print("results = " + str(results))
        i = i + 1

