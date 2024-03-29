#module to detect speech

import speech_recognition as sr;

r = sr.Recognizer()
#r.energy_threshold = 1000;
print(r.energy_threshold)

class SpeechDetector:   
	
	def is_speech(self, audios):
	    if(type(audios) == type([])):
	        outputs = []
	        for a in audios:
	            outputs.append(self.signal_is_speech(a))
	        return outputs
	    else:
	        return signal_is_speech
	
	def signal_is_speech(self, audio):
	    #return True if recognize(audio) is not None else False
	    words = self.recognize(audio)
	    if(words is None):
	        return False
	    return True
	
	
	def recognize(self, audio):
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
    
    vad =  SpeechDetector();

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

