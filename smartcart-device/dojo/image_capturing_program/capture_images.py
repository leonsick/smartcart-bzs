import os
import argparse
import cv2
import uuid
import random
from acceleration.detection import Detection
from acceleration.annotation import PascalVocWriter

parser = argparse.ArgumentParser(description='Take a specific number of images with a pause of 3 seconds.')
parser.add_argument('-n', '--number', type=int, help='`Specifiy the number of images you want to take.')
parser.add_argument('-ic', '--iclass', help='`Specifiy the class of images you want to take.',
                    choices=['apple', 'banana', 'broccoli', 'paprika', 'aubergine', 'cucumber', 'orange', 'pear'],
                    required=True)
parser.add_argument('-ds', '--dataset', help='`Specifiy the class of images you want to take.',
                    choices=['train', 'test'], required=True)
parser.add_argument('-s', '--size', default="large",
                    help='Specifiy the class of images you want to take. If no param is chosen, it will be random between small, medium and large',
                    choices=['large'])
parser.add_argument('-ac', '--acceleration', default="1",
                    help='Specifiy acceleration',
                    choices=["0","1"])

args = parser.parse_args()
number_of_images = args.number
image_class = str(args.iclass)
dataset = str(args.dataset)
size = str(args.size)
acceleration = int(args.acceleration)

print("Setting up camera...")
camera = cv2.VideoCapture(0)
print("Camera is ready!")

print("Now setting up detection...")
detector = Detection()
print("Detection is set up!")

img_counter = 0
img_shape = [0, 0, 0]
# image_detections, classes, boxes, scores = []
print("~~Image capturing device is started~~")
print("~~Press ESC to exit and SPACE to save an image~~")

while img_counter < number_of_images:
    status, image = camera.read()
    if not status:
        print("Image capture failed")
        break
    if size == "random":
        rand = random.randint(1, 3)
        if rand == 1:
            size = "small"
        if rand == 2:
            size = "medium"
        if rand == 3:
            size = "large"
    if size == "small":
        img_shape = [300, 300, 3]
        image_resized = cv2.resize(image, (300, 300))
        original = image_resized.copy()
    elif size == "medium":
        img_shape = [608, 608, 3]
        image_resized = cv2.resize(image, (608, 608))
        original = image_resized.copy()
    else:
        img_shape = [720, 1080, 3]
        image_resized = cv2.resize(image, (1080, 720))
        original = image_resized.copy()

    if not acceleration:
        cv2.imshow("Press SPACE to save image", image_resized)
    else:
        image_detections, classes, boxes, scores = detector.detect(image_resized)
        # print("Classes")
        # print(classes)
        # print("Boxes for best object")
        # print("ymin {}, xmin{}, ymax{}, xmax{}".format(boxes[0][0], boxes[0][1], boxes[0][2], boxes[0][3]))
        cv2.imshow("Press 'A' to save image with all annotations", image_detections)
    k = cv2.waitKey(10)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        camera.release()
        cv2.destroyAllWindows()
        break
    elif k % 256 == 32:
        # SPACE pressed
        img_uuid = uuid.uuid4()
        img_name = "{}_{}.png".format(image_class, img_uuid)
        path = "{}/dataset/{}/{}/".format(os.getcwd(), dataset, image_class)
        print(path + img_name)
        cv2.imwrite(path + img_name, image_resized)
        print("{} written!".format(img_name))
        img_counter += 1
        if args.size == "random":
            size = "random"

    elif k == ord('a'):
        # d pressed
        img_uuid = uuid.uuid4()
        img_folder_name = image_class
        file_name = "{}_{}.xml".format(image_class, img_uuid)
        img_path = os.getcwd() + "/dataset/{}/{}".format(dataset, image_class)

        img_name = "{}_{}.png".format(image_class, img_uuid)
        path = "{}/dataset/{}/{}/".format(os.getcwd(), dataset, image_class)
        print(path + img_name)
        cv2.imwrite(path + img_name, original)
        print("{} written!".format(img_name))

        writer = PascalVocWriter(img_folder_name, file_name, img_shape, localImgPath=img_path)
        num_of_objects = len(classes)
        i = 0
        while i < num_of_objects:
            if scores[i] > 0.95:
                print("Scores: {}".format(scores[i]))
                detected_class = int(classes[i])
                print("Detected class: {}".format(detected_class))
                detected_object = (
                    detector.categories[detected_class - 1]["name"])  ##Attention: -1 because label map starts at 0
                print("Detected object: {}".format(detected_object))
                if detected_object.lower() == image_class:
                    writer.addBndBox(ymin=int(boxes[i][0] * img_shape[0]), xmin=int(boxes[i][1] * img_shape[1]),
                                     ymax=int(boxes[i][2] * img_shape[0]), xmax=int(boxes[i][3] * img_shape[1]),
                                     name=detected_object.lower(),
                                     difficult=0)
                    print("Added BndBox for object {}".format(detected_object))
            i += 1
        writer.save(targetFile=file_name)
        print("Annotations XML written!".format(img_name))
        os.system('mv {} dataset/{}/{}'.format(file_name, dataset, img_folder_name))
        img_counter += 1
        if args.size == "random":
            size = "random"
    """
    elif k == ord('1'):
        # d pressed
        img_uuid = uuid.uuid4()
        # TODO: Sort both dicts for best detection
        # TODO: Save only best coords and classes in XML File

        img_name = "{}_{}.png".format(image_class, img_uuid)
        path = "{}/dataset/{}/{}/".format(os.getcwd(), dataset, image_class)
        print(path + img_name)
        cv2.imwrite(path + img_name, image_resized)
        print("{} written!".format(img_name))
        img_counter += 1
        if args.size == "random":
            size = "random"

    elif k == ord('2'):
        # d pressed
        img_uuid = uuid.uuid4()
        # TODO: Sort both dicts for best detection
        # TODO: Save only the 2 beste coords and classes in XML File

        img_name = "{}_{}.png".format(image_class, img_uuid)
        path = "{}/dataset/{}/{}/".format(os.getcwd(), dataset, image_class)
        print(path + img_name)
        cv2.imwrite(path + img_name, image_resized)
        print("{} written!".format(img_name))
        img_counter += 1
        if args.size == "random":
            size = "random"

    elif k == ord('3'):
        # d pressed
        img_uuid = uuid.uuid4()
        # TODO: Sort both dicts for best detection
        # TODO: Save only the 3 best coords and classes in XML File

        img_name = "{}_{}.png".format(image_class, img_uuid)
        path = "{}/dataset/{}/{}/".format(os.getcwd(), dataset, image_class)
        print(path + img_name)
        cv2.imwrite(path + img_name, image_resized)
        print("{} written!".format(img_name))
        img_counter += 1
        if args.size == "random":
            size = "random" 
    """
