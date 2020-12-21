# Import packages
import os
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import tensorflow as tf
import argparse

# Import utilites
from utils import visualization_utils as vis_util
from utils import label_map_util

class detection_tf(object):
    def __init__(self, path_to_model, path_to_labels, mode_name, model_type, edge_tpu):
        # Set up camera constants
        self.IM_WIDTH = 1280
        self.IM_HEIGHT = 720

        self.image_tensor = None
        self.detection_boxes = None
        self.detection_scores = None
        self.detection_classes = None
        self.num_detections = None
        self.detection_graph = None
        self.sess = None
        self.frame_rate_calc = None
        self.freq = None
        self.font = None
        self.TL_inside = None
        self.BR_inside = None
        self.TL_outside = None
        self.BR_outside = None




        # Select camera type (if user enters --usbcam when calling this script,
        # a USB webcam will be used)
        self.camera_type = 'picamera'
        parser = argparse.ArgumentParser()
        parser.add_argument('--usbcam', help='Use a USB webcam instead of picamera',
                            action='store_true')
        args = parser.parse_args()
        if args.usbcam:
            camera_type = 'usb'

        #### Initialize TensorFlow model ####

        # This is needed since the working directory is the object_detection folder.
        # sys.path.append('..')

        # Grab path to current working directory
        CWD_PATH = os.getcwd()

        # Path to frozen detection graph .pb file, which contains the model that is used
        # for object detection.
        self.PATH_TO_CKPT = os.path.join(CWD_PATH, 'dojo', 'yolo', 'yolov4_smartcart', 'saved_model.pb')
        print("Path to model: {}".format(self.PATH_TO_CKPT))

        # Path to label map file
        self.PATH_TO_LABELS = os.path.join(CWD_PATH, 'dojo', 'yolo', 'yolov4_smartcart')
        print("Path to labels: {}".format(self.PATH_TO_LABELS))

        # Number of classes the object detector can identify
        self.NUM_CLASSES = 90

        ## Load the label map.
        # Label maps map indices to category names, so that when the convolution
        # network predicts `5`, we know that this corresponds to `airplane`.
        # Here we use internal utility functions, but anything that returns a
        # dictionary mapping integers to appropriate string labels would be fine
        self.label_map = label_map_util.load_labelmap(
            '/home/pi/projects/smartcart-device/dojo/yolo/yolov4_smartcart/testing/mscoco_complete_label_map.pbtxt')
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=self.NUM_CLASSES,
                                                                    use_display_name=True)
        print(self.categories)
        self.category_index = label_map_util.create_category_index(self.categories)

    def load_model(self):
        self.detection_graph = tf.Graph()
        self.sess = tf.Session(graph=self.detection_graph)
        tf.saved_model.loader.load(self.sess, [tf.saved_model.tag_constants.SERVING],
                                   '/home/pi/projects/smartcart-device/dojo/yolo/yolov4_smartcart/testing')

        # Define input and output tensors (i.e. data) for the object detection classifier

        # Input tensor is the image
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Output tensors are the detection boxes, scores, and classes
        # Each box represents a part of the image where a particular object was detected
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represents level of confidence for each of the objects.
        # The score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')

        # Number of objects detected
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')


    def setup_cv(self):
        # Initialize frame rate calculation
        self.frame_rate_calc = 1
        self.freq = cv2.getTickFrequency()
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # Define inside box coordinates (top left and bottom right)
        self.TL_inside = (int(self.IM_WIDTH * 0.1), int(self.IM_HEIGHT * 0.35))
        self.BR_inside = (int(self.IM_WIDTH * 0.45), int(self.IM_HEIGHT - 5))

        # Define outside box coordinates (top left and bottom right)
        self.TL_outside = (int(self.IM_WIDTH * 0.46), int(self.IM_HEIGHT * 0.25))
        self.BR_outside = (int(self.IM_WIDTH * 0.8), int(self.IM_HEIGHT * .85))


    def detector(self, frame):
        # Use globals for the control variables so they retain their value after function exits
        global detected
        frame_expanded = np.expand_dims(frame, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: frame_expanded})

        print("Detected {} objects in frame")
        print("Detected class {} with detection score {}".format(classes[0][0], scores[0][0]))

        # Draw the results of the detection (aka 'visulaize the results')
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.40)

        # Draw boxes defining "outside" and "inside" locations.
        # cv2.rectangle(frame,TL_outside,BR_outside,(255,20,20),3)
        # cv2.putText(frame,"Outside box",(TL_outside[0]+10,TL_outside[1]-10),font,1,(255,20,255),3,cv2.LINE_AA)
        # cv2.rectangle(frame,TL_inside,BR_inside,(20,20,255),3)
        # cv2.putText(frame,"Inside box",(TL_inside[0]+10,TL_inside[1]-10),font,1,(20,255,255),3,cv2.LINE_AA)

        # Check the class of the top detected object by looking at classes[0][0].
        # If the top detected object is a cat (17) or a dog (18) (or a teddy bear (88) for test purposes),
        # find its center coordinates by looking at the boxes[0][0] variable.
        # boxes[0][0] variable holds coordinates of detected objects as (ymin, xmin, ymax, xmax)
        if (((int(classes[0][0]) == 1))):
            x = int(((boxes[0][0][1] + boxes[0][0][3]) / 2) * self.IM_WIDTH)
            y = int(((boxes[0][0][0] + boxes[0][0][2]) / 2) * self.IM_HEIGHT)

            # Draw a circle at center of object
            cv2.circle(frame, (x, y), 5, (75, 13, 180), -1)


        return frame
