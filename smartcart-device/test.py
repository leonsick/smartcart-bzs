from camera.detection_tflite import Detection
import time

detector = Detection()

time.sleep(2)

while True:
    detector.run()


detector.destroy()