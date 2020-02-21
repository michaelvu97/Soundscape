import RPi.GPIO as GPIO
import time
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

def LED_Init():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    print("Initializing LEDs")
    GPIO.setup(2,GPIO.OUT)
    GPIO.setup(3,GPIO.OUT)
    GPIO.setup(14,GPIO.OUT)
    GPIO.setup(16,GPIO.OUT)
    GPIO.setup(27,GPIO.OUT)
    GPIO.setup(22,GPIO.OUT)
    #Set all LEDs to off initially
    GPIO.output(2,GPIO.LOW)
    GPIO.output(3,GPIO.LOW)
    GPIO.output(14,GPIO.LOW)
    GPIO.output(16,GPIO.LOW)
    GPIO.output(27,GPIO.LOW)
    GPIO.output(22,GPIO.LOW)
def Display_Direction (direction):
    if direction == None:
        GPIO.output(2,GPIO.LOW)
        GPIO.output(3,GPIO.LOW)
        GPIO.output(14,GPIO.LOW)
        GPIO.output(16,GPIO.LOW)
        GPIO.output(27,GPIO.LOW)
        GPIO.output(22,GPIO.LOW)
    elif direction <= 18:
        GPIO.output(2,GPIO.LOW)
        GPIO.output(3,GPIO.LOW)
        GPIO.output(14,GPIO.LOW)
        GPIO.output(16,GPIO.LOW)
        GPIO.output(27,GPIO.LOW)
        GPIO.output(22,GPIO.LOW)
    elif direction <= 54:
        GPIO.output(2,GPIO.LOW)
        GPIO.output(3,GPIO.LOW)
        GPIO.output(14,GPIO.HIGH)
        GPIO.output(16,GPIO.LOW)
        GPIO.output(27,GPIO.LOW)
        GPIO.output(22,GPIO.LOW)
    elif direction <= 90:
        GPIO.output(2,GPIO.LOW)
        GPIO.output(3,GPIO.HIGH)
        GPIO.output(14,GPIO.HIGH)
        GPIO.output(16,GPIO.LOW)
        GPIO.output(27,GPIO.LOW)
        GPIO.output(22,GPIO.LOW)
    elif direction <= 126:
        GPIO.output(2,GPIO.HIGH)
        GPIO.output(3,GPIO.HIGH)
        GPIO.output(14,GPIO.LOW)
        GPIO.output(16,GPIO.LOW)
        GPIO.output(27,GPIO.LOW)
        GPIO.output(22,GPIO.LOW)
    elif direction <= 162:
        GPIO.output(2,GPIO.HIGH)
        GPIO.output(3,GPIO.LOW)
        GPIO.output(14,GPIO.LOW)
        GPIO.output(16,GPIO.LOW)
        GPIO.output(27,GPIO.LOW)
        GPIO.output(22,GPIO.LOW)
    elif direction <= 198:
        GPIO.output(2,GPIO.HIGH)
        GPIO.output(3,GPIO.LOW)
        GPIO.output(14,GPIO.LOW)
        GPIO.output(16,GPIO.LOW)
        GPIO.output(27,GPIO.LOW)
        GPIO.output(22,GPIO.HIGH)
    elif direction <= 234:
        GPIO.output(2,GPIO.LOW)
        GPIO.output(3,GPIO.LOW)
        GPIO.output(14,GPIO.LOW)
        GPIO.output(16,GPIO.LOW)
        GPIO.output(27,GPIO.LOW)
        GPIO.output(22,GPIO.HIGH)
    elif direction <= 270:
        GPIO.output(2,GPIO.LOW)
        GPIO.output(3,GPIO.LOW)
        GPIO.output(14,GPIO.LOW)
        GPIO.output(16,GPIO.LOW)
        GPIO.output(27,GPIO.HIGH)
        GPIO.output(22,GPIO.HIGH)
    elif direction <= 306:
        GPIO.output(2,GPIO.LOW)
        GPIO.output(3,GPIO.LOW)
        GPIO.output(14,GPIO.LOW)
        GPIO.output(16,GPIO.HIGH)
        GPIO.output(27,GPIO.HIGH)
        GPIO.output(22,GPIO.LOW)
    elif direction <= 342:
        GPIO.output(2,GPIO.LOW)
        GPIO.output(3,GPIO.LOW)
        GPIO.output(14,GPIO.LOW)
        GPIO.output(16,GPIO.HIGH)
        GPIO.output(27,GPIO.LOW)
        GPIO.output(22,GPIO.LOW)
    elif direction <= 360:
        GPIO.output(2,GPIO.LOW)
        GPIO.output(3,GPIO.LOW)
        GPIO.output(14,GPIO.LOW)
        GPIO.output(16,GPIO.LOW)
        GPIO.output(27,GPIO.LOW)
        GPIO.output(22,GPIO.LOW)