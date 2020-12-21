import cv2
from threading import Thread
from picamera.array import PiRGBArray
from picamera import PiCamera
import time

class VideoStream:
    """Camera object that controls video streaming from the Picamera"""

    def __init__(self, resolution=(1280, 720), framerate=30):
        # Initialize the PiCamera and the camera image stream
        #self.stream = cv2.VideoCapture(0)
        #ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        #ret = self.stream.set(3, resolution[0])
        #ret = self.stream.set(4, resolution[1])

        # Setup PiCamera
        self.camera = PiCamera()
        self.camera.resolution = (1280, 720)
        self.camera.framerate = 30
        self.rawCapture = PiRGBArray(self.camera, size=(1280, 720))
        self.rawCapture.truncate(0)
        self.stream = self.camera.capture_continuous(self.rawCapture, format="bgr", use_video_port=True)

        # Read first frame from the stream
        #(self.grabbed, self.frame) = self.stream.read()
        self.frame = None

        # Variable to control when the camera is stopped
        self.stopped = False
        
        time.sleep(5)
        print("Camera initialized")
        

    def start(self):
        # Start the thread that reads frames from the video stream
        Thread(target=self.update, args=()).start()
        time.sleep(5)
        print("Stream started")
        return self

    def update(self):
        '''
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()
        '''

        for f in self.stream:
            self.frame = f.array
            self.rawCapture.truncate(0)

            if self.stopped:
                self.stream.close()
                self.rawCapture.close()
                self.camera.close()
                return


    def read(self):
        # Return the most recent frame
        return self.frame

    def stop(self):
        # Indicate that the camera and thread should be stopped
        self.stopped = True
        self.camera.close()

'''stream = VideoStream()
stream.start()

while True:
    frame = stream.read()
    print(type(frame))
    print(frame.dtype)
    print(frame)
    cv2.imshow('Object detector', frame)
    print("Sleeping")
    
    if cv2.waitKey(1) == ord('x'):
        break
    
time.sleep(5)

cv2.destroyAllWindows()'''

