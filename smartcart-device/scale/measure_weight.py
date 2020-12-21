import logging
import time
import sys
from threading import Thread
from secrets import Secrets

EMULATE_HX711 = False

#reference_unit = -105
reference_unit = Secrets.get_reference_unit()

if not EMULATE_HX711:
    import RPi.GPIO as GPIO
    from hx711 import HX711
    print('1. if')
else:
    from emulated_hx711 import HX711
    print('2. if')


print('now accessing hx')


hx = HX711(5, 6)
print(hx)
hx.set_reading_format("MSB", "MSB")
hx.set_reference_unit(reference_unit)
hx.reset()
hx.tare()

tolerance = 50


def call_weight():
    val = hx.get_weight(5)
    hx.power_down()
    hx.power_up()
    return val


print(call_weight())


def weight_has_increased():
    value1 = call_weight()
    time.sleep(0.2)
    value2 = call_weight()
    diff_increase = value2 - value1
    if value2 > (value1 + tolerance):
        return True, diff_increase
    else:
        return False, diff_increase


def weight_has_decreased():
    value1 = call_weight()
    time.sleep(0.2)
    value2 = call_weight()
    diff_decrease = value1 - value2
    if value1 > (value2 + tolerance):
        return True, diff_decrease
    else:
        return False, diff_decrease


def clean_and_exit():
    if not EMULATE_HX711:
        GPIO.cleanup()
    sys.exit()


'''
### SCALE TEST
while True:
    try:
        val1 = call_weight()
        time.sleep(0.5)
        val2 = call_weight()
        if (val1 > (val2 + 50)):
            diff1 = str(val1 - val2)
            print("Weight decreased by "+diff1+"!")
        elif (val2 > (val1 + 50)):
            diff2 = str(val2 - val1)
            print("Weight increased by "+diff2+"!")
        #else:
            #print("No change")
    except (KeyboardInterrupt, SystemExit):
        cleanAndExit()
'''
