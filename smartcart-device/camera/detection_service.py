from camera.detection_tflite import Detection
from threading import Thread
import time


class DetectionService:
    item_detected = False
    latest_item = None
    detection_thread = None
    detection = Detection()

    @classmethod
    def start(cls):
        cls.detection_thread = Thread(target=cls.detects_item)
        cls.detection_thread.start()

    @classmethod
    def detects_item(cls):
        while True:
            cls.item_detected, cls.latest_item = cls.detection.run()
            if cls.item_detected == True:
                time.sleep(5)
                #cls.item_detected = False
                #cls.latest_item = None
        #print("Detection successful!")
        

    @classmethod
    def get_latest_item(cls):
        return cls.latest_item

    @classmethod
    def get_item_detected(cls):
        return cls.item_detected

    @classmethod
    def reset(cls):
        cls.detection_thread = None
        cls.latest_item = None
        cls.item_detected = False
        #cls.detection_thread = Thread(target=cls.detects_item)
        #cls.detection_thread.start()

    @classmethod
    def detection_test(cls):
        cls.detection.run()
