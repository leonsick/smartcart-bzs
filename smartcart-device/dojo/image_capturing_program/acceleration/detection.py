import os
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
from .utils import visualization_utils as vis_util
from .utils import label_map_util


class Detection:
    def __init__(self):

        # Set up camera constants
        self.IM_WIDTH = 1280
        self.IM_HEIGHT = 720


        #### Initialize TensorFlow model ####

        # Grab path to current working directory
        CWD_PATH = os.getcwd()

        # Path to frozen detection graph .pb file, which contains the model that is used
        # for object detection.
        self.PATH_TO_CKPT = os.path.join(CWD_PATH, 'model', 'saved_model.pb')
        print("Path to model: {}".format(self.PATH_TO_CKPT))

        # Path to label map file
        self.PATH_TO_LABELS = os.path.join(CWD_PATH, 'model')
        print("Path to labels: {}".format(self.PATH_TO_LABELS))

        # Number of classes the object detector can identify
        self.NUM_CLASSES = 90

        ## Load the label map.
        # Label maps map indices to category names, so that when the convolution
        # network predicts `5`, we know that this corresponds to `airplane`.
        # Here we use internal utility functions, but anything that returns a
        # dictionary mapping integers to appropriate string labels would be fine
        self.label_map = label_map_util.load_labelmap(
            '/Users/leonsick/Desktop/Developer/Dojo/Image-Capturing-Program/acceleration/model/label_map.pbtxt')
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=self.NUM_CLASSES,
                                                                    use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)
        print(self.categories)
        # Load the Tensorflow model into memory.

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile('/Users/leonsick/Desktop/Developer/Dojo/Image-Capturing-Program/acceleration/model/frozen_inference_graph.pb', 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.Session(graph=self.detection_graph)

        #self.detection_graph = tf.Graph()
        #self.sess = tf.Session(graph=self.detection_graph)
        #tf.saved_model.loader.load(self.sess, [tf.saved_model.tag_constants.SERVING],
                                   #'/Users/leonsick/Desktop/Developer/Dojo/Image-Capturing-Program/acceleration/model')

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

        #### Initialize other parameters ####

        # Initialize frame rate calculation
        self.frame_rate_calc = 1
        self.freq = cv2.getTickFrequency()
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def detect(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_expanded = np.expand_dims(frame_rgb, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: frame_expanded})

        #print("Detected {} objects in frame".format(num[0]))
        #print("Detected first class {} with detection score {}".format(classes[0][0], scores[0][0]))
        #print("Detected second class {} with detection score {}".format(classes[0][1], scores[0][1]))
        #print("Detected third class {} with detection score {}".format(classes[0][2], scores[0][3]))
        # Draw the results of the detection (aka 'visulaize the results')
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.95)





        # boxes[0][0] variable holds coordinates of detected objects as (ymin, xmin, ymax, xmax)
        if (((int(classes[0][0]) == 1))):
            x = int(((boxes[0][0][1] + boxes[0][0][3]) / 2) * self.IM_WIDTH)
            y = int(((boxes[0][0][0] + boxes[0][0][2]) / 2) * self.IM_HEIGHT)

            # Draw a circle at center of object
            cv2.circle(frame, (x, y), 5, (75, 13, 180), -1)

        # TODO: Additionally return coords_dict, objects_dict
        return frame, classes[0], boxes[0], scores[0]
