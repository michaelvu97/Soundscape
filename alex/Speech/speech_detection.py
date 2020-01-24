#module to detect speech

import speech_recognition as sr;

r = sr.Recognizer()
#r.energy_threshold = 1000;
print(r.energy_threshold)

def has_speech(audio):
    #return True if recognize(audio) is not None else False
    words = recognize(audio)
    if(words is None):
        return False
    print("words " + words)
    return True

def recognize(audio):
    try:
        return r.recognize_google(audio);
    except sr.UnknownValueError:
        #print("unknown value error")
        return None
    except sr.RequestError as e:
        print("request error")
        print(e)
        return None



if __name__ == "__main__":
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

