import os
import cv2
import numpy as np
import sys
import time
import random
import colorsys
import requests
import json
from threading import Thread
from PIL import Image
import tensorflow as tf
import importlib.util
from .video_stream import VideoStream


# Parts taken from https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/TFLite_detection_webcam.py

class Detection:
    def __init__(self):
        self.MODEL_NAME = "detect"
        self.GRAPH_NAME = "detect.tflite"
        self.LABELMAP_NAME = "label_map.txt"
        self.min_conf_threshold = 0.70
        self.resW, self.resH = (1280, 720)
        self.imW, self.imH = int(self.resW), int(self.resH)
        # self.use_TPU = (True if 'projects' in str(os.getcwd()) else False)
        self.use_TPU = False
        self.frame_rate_calc = None
        self.item_detected = False
        self.latest_item = None

        self.detection_counter = [
            {
                "name": "apple",
                "counter": 0
            },
            {
                "name": "aubergine",
                "counter": 0
            },
            {
                "name": "banana",
                "counter": 0
            },
            {
                "name": "broccoli",
                "counter": 0
            },
            {
                "name": "cucumber",
                "counter": 0
            },
            {
                "name": "orange",
                "counter": 0
            },
            {
                "name": "paprika",
                "counter": 0
            },
            {
                "name": "pear",
                "counter": 0
            }
        ]

        # Import TFLite requirements
        self.pkg = importlib.util.find_spec('tflite_runtime')
        if self.pkg:
            from tflite_runtime.interpreter import Interpreter
            if self.use_TPU:
                from tflite_runtime.interpreter import load_delegate
        else:
            from tensorflow.lite.python.interpreter import Interpreter
            if self.use_TPU:
                from tensorflow.lite.python.interpreter import load_delegate

        # If using Edge TPU, assign filename for Edge TPU model
        if self.use_TPU:
            # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
            if (self.GRAPH_NAME == 'detect.tflite'):
                self.GRAPH_NAME = 'edgetpu.tflite'

        # Get path to current working directory
        CWD_PATH = os.getcwd()

        PATH_TO_CKPT = "/home/pi/projects/smartcart-device/dojo/tflite/{}".format(self.GRAPH_NAME)

        PATH_TO_LABELS = "/home/pi/projects/smartcart-device/dojo/tflite/{}".format(
            self.LABELMAP_NAME)

        PATH_TO_OBJ_NAMES = "/home/pi/projects/smartcart-device/dojo/yolo/yolov4_smartcart/tflite/coco.names"
        # Load the label map
        with open(PATH_TO_LABELS, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]

        # Fix for potential label map issue
        if self.labels[0] == '???':
            del (self.labels[0])

        if self.use_TPU:
            self.interpreter = Interpreter(model_path=PATH_TO_CKPT,
                                           experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
            print(PATH_TO_CKPT)
        else:
            self.interpreter = Interpreter(model_path=PATH_TO_CKPT)

        self.interpreter.allocate_tensors()
        
        print("Model loaded and tensors allocated")

        # Get model details
        self.input_details = self.interpreter.get_input_details()
        #print("Input details: {}".format(self.input_details))
        self.output_details = self.interpreter.get_output_details()
        #print("Output detais: {}".format(self.output_details))
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]

        self.floating_model = (self.input_details[0]['dtype'] == np.float32)

        self.input_mean = 127.5
        self.input_std = 127.5

        # Initialize frame rate calculation
        self.frame_rate_calc = 1
        self.freq = cv2.getTickFrequency()

        # Initialize video stream
        self.videostream = VideoStream(resolution=(self.imW, self.imH))
        self.videostream = self.videostream.start()

    def filter_boxes(self, box_xywh, scores, score_threshold=0.4, input_shape=tf.constant([416, 416])):
        scores_max = tf.math.reduce_max(scores, axis=-1)

        mask = scores_max >= score_threshold
        class_boxes = tf.boolean_mask(box_xywh, mask)
        pred_conf = tf.boolean_mask(scores, mask)
        class_boxes = tf.reshape(class_boxes, [tf.shape(scores)[0], -1, tf.shape(class_boxes)[-1]])
        pred_conf = tf.reshape(pred_conf, [tf.shape(scores)[0], -1, tf.shape(pred_conf)[-1]])

        box_xy, box_wh = tf.split(class_boxes, (2, 2), axis=-1)

        input_shape = tf.cast(input_shape, dtype=tf.float32)

        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]

        box_mins = (box_yx - (box_hw / 2.)) / input_shape
        box_maxes = (box_yx + (box_hw / 2.)) / input_shape
        boxes = tf.concat([
            box_mins[..., 0:1],  # y_min
            box_mins[..., 1:2],  # x_min
            box_maxes[..., 0:1],  # y_max
            box_maxes[..., 1:2]  # x_max
        ], axis=-1)
        # return tf.concat([boxes, pred_conf], axis=-1)
        return (boxes, pred_conf)

    def read_class_names(self, class_file_name):
        names = {}
        with open(class_file_name, 'r') as data:
            for ID, name in enumerate(data):
                names[ID] = name.strip('\n')
        return names

    # TODO: Definde cfg.YOLO.CLASSES
    def draw_bbox(self, image, bboxes, classes, show_label=True):
        num_classes = len(classes)
        image_h, image_w, _ = image.shape
        hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

        random.seed(0)
        random.shuffle(colors)
        random.seed(None)

        out_boxes, out_scores, out_classes, num_boxes = bboxes
        for i in range(num_boxes[0]):
            if int(out_classes[0][i]) < 0 or int(out_classes[0][i]) > num_classes: continue
            coor = out_boxes[0][i]
            coor[0] = int(coor[0] * image_h)
            coor[2] = int(coor[2] * image_h)
            coor[1] = int(coor[1] * image_w)
            coor[3] = int(coor[3] * image_w)

            fontScale = 0.5
            score = out_scores[0][i]
            class_ind = int(out_classes[0][i])
            bbox_color = colors[class_ind]
            bbox_thick = int(0.6 * (image_h + image_w) / 600)
            c1, c2 = (coor[1], coor[0]), (coor[3], coor[2])
            cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

            if show_label:
                bbox_mess = '%s: %.2f' % (classes[class_ind], score)
                t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
                c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
                cv2.rectangle(image, c1, (np.float32(c3[0]), np.float32(c3[1])), bbox_color, -1)  # filled

                cv2.putText(image, bbox_mess, (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
        return image

    def perform(self):
        while True:
            t1 = cv2.getTickCount()

            frame1 = self.videostream.read()
            
            print("Frame read from stream")

            frame = frame1.copy()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # frame_resized = cv2.resize(frame_rgb, (self.width, self.height))
            # input_data = np.expand_dims(frame_resized, axis=0)

            image_data = cv2.resize(frame, (608, 608))
            image_data = image_data / 255.

            images_data = []
            for i in range(1):
                images_data.append(image_data)
            images_data = np.asarray(images_data).astype(np.float32)

            # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
            # if self.floating_model:
            # input_data = (np.float32(input_data) - self.input_mean) / self.input_std

            # Perform the actual detection by running the model with the image as input
            self.interpreter.set_tensor(self.input_details[0]['index'], images_data)
            print("Performing detection")
            self.interpreter.invoke()
            print("Detection performed")
            pred = [self.interpreter.get_tensor(self.output_details[i]['index']) for i in
                    range(len(self.output_details))]
            boxes, pred_conf = self.filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                 input_shape=tf.constant([608, 608]))

            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=0.3,  # TODO: Make var
                score_threshold=0.3  # TODO: Make var
            )
            pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
            class_names = self.read_class_names(
                "/home/pi/projects/smartcart-device/dojo/yolo/yolov4_smartcart/tflite/coco.names")
            print("Drawing bounding boxes")
            frame = self.draw_bbox(frame, pred_bbox, class_names)
            #frame = Image.fromarray(frame.astype(np.uint8))

           # cv2.imshow('Object detector',frame.astype(np.uint8))
            time.sleep(5)
            image = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)

            if cv2.waitKey(1) == ord('x'):
                break

            if self.item_detected:
                break

        return self.item_detected, self.latest_item

    def run(self, cloud=False):
        #while True:
            # for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):

            # Start timer (for calculating frame rate)
            t1 = cv2.getTickCount()

            # Grab frame from video stream
            frame1 = self.videostream.read()

            # Acquire frame and resize to expected shape [1xHxWx3]
            frame = frame1.copy()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (self.width, self.height))

            if cloud:
                # TODO: Send image to cloud and get data back
                content_type = 'image/jpeg'
                headers = {'content-type': content_type}

                _, img_encoded = cv2.imencode('.jpg', frame_rgb)
                request_address = "http://a24dcb00998c.ngrok.io/api/detect"
                # send http request with image and receive response
                print("Sending image to cloud api and awaiting response")
                response = requests.post(request_address, data=img_encoded.tostring(), headers=headers)
                print("Response received:")
                print(json.loads(response.text))

            else:
                input_data = np.expand_dims(frame_resized, axis=0)

                # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
                if self.floating_model:
                    input_data = (np.float32(input_data) - self.input_mean) / self.input_std

                # Perform the actual detection by running the model with the image as input
                self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                #print("Detection started")
                self.interpreter.invoke()
                #print("Detection complete")

                # Retrieve detection results
                #print(self.output_details)
                boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]  # Bounding coordinates of objects
                classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]  # Class index of detected objects
                scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]  # Confidence of detected objects
                num = self.interpreter.get_tensor(self.output_details[3]['index'])[0]
                # Total number of detected objects (inaccurate and not needed)

            max_score = 0

            # Loop over all detections and draw detection box if confidence is above minimum threshold
            for i in range(len(scores)):
                if ((scores[i] > self.min_conf_threshold) and (scores[i] <= 1.0)):
                    # Specify that item has been detected
                    #self.item_detected = True
                    #if scores[i] > max_score:
                        #max_score = scores[i]
                        #self.latest_item = self.labels[int(classes[i])]


                    # Get bounding box coordinates and draw box
                    # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                    ymin = int(max(1, (boxes[i][0] * self.imH)))
                    xmin = int(max(1, (boxes[i][1] * self.imW)))
                    ymax = int(min(self.imH, (boxes[i][2] * self.imH)))
                    xmax = int(min(self.imW, (boxes[i][3] * self.imW)))

                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

                    # Draw label
                    object_name = self.labels[int(classes[i])]  # Look up object name from "labels" array using class index
                    self.increase_detection_counter(object_name, scores[i])
                    label = '%s: %d%%' % (object_name, int(scores[i] * 100))  # Example: 'person: 72%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)  # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10)  # Make sure not to draw label too close to top of window
                    cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10),
                                  (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255),
                                  cv2.FILLED)  # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0),
                                2)  # Draw label text

            # Draw framerate in corner of frame
            cv2.putText(frame, 'FPS: {0:.2f}'.format(self.frame_rate_calc), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 0), 2,
                        cv2.LINE_AA)

            # All the results have been drawn on the frame, so it's time to display it.
            cv2.imshow('Object detector', frame)
        
            if cv2.waitKey(1) == ord('x'):
                    cv2.destroyAllWindows()
                    #break
        
            # Calculate framerate
            t2 = cv2.getTickCount()
            time1 = (t2 - t1) / self.freq
            self.frame_rate_calc = 1 / time1

            self.item_detected, self.latest_item = self.get_object_with_score_five()

            if self.item_detected:
                self.reset_detection_counter()

            return self.item_detected, self.latest_item

    def increase_detection_counter(self, detected_item, score):
        for object in self.detection_counter:
            if object["name"] == detected_item:
                object["counter"]+=score


    def get_object_with_score_five(self):
        max_score = 0
        latest_object = "None"
        detected_object = False
        for object in self.detection_counter:
            if object["counter"] >= 5 and object["counter"] > max_score:
                latest_object = object["name"]
                detected_object = True
                max_score = object["counter"]
        return detected_object, latest_object

    def reset_detection_counter(self):
        self.detection_counter = [
            {
                "name": "apple",
                "counter": 0
            },
            {
                "name": "aubergine",
                "counter": 0
            },
            {
                "name": "banana",
                "counter": 0
            },
            {
                "name": "broccoli",
                "counter": 0
            },
            {
                "name": "cucumber",
                "counter": 0
            },
            {
                "name": "orange",
                "counter": 0
            },
            {
                "name": "paprika",
                "counter": 0
            },
            {
                "name": "pear",
                "counter": 0
            }
        ]

    def destroy(self):
        # Clean up
        cv2.destroyAllWindows()
        self.videostream.stop()