import time

from camera.detection_service import DetectionService as Camera
#import measure_weight as Scale
from scale.scale_service import ScaleService as Scale
from db.db_helpers import db_helpers as DB

# Setup Detection and Scale
print("-> SETTING UP CAMERA")
Camera()
print("-> SETTING UP SCALE")
Scale()
print("-> SETTING UP DB")
db = DB()


# To detect addition of object, trigger comes from camera, confirmation from scale
# To detect removal of object, trigger comes from scale, confirmation from camera
'''
Camera.detects_item(), bool: Stream that check if item with acceptable confidence has been detected. Saves detected item.
Scale.weight_has_increased(), bool: Stream that checks if the weight on the scale has increased. Saves added weight.
Scale.get_weight_change(), float: Get saved weight change from Scale object.
Camera.get_latest_item(), String: Get latest item from Camera object.
DB.push(added_item, added_weight), void: Creates DB entry for new object
Camera.reset(), void: Deletes value for latest detection and bool value for item_detected.
Scale.erase_weight_change(), void: Deletes value for latest weight change.
'''

print("-> STARTING SCALE THREADS")
Scale.start()

print('PRESS THE SCALE FOR ACTIVATION!')
while True:
    refresh_decrease = Scale.get_weight_has_decreased()
    refresh_increase = Scale.get_weight_has_increased()
    if refresh_increase == True:
        print('+++ REFRESH LAST SCALE ACTIVITY +++')
        DB.push_last_activity(db)
        break

print("-> STARTING CAMERA THREADS")
Camera.start()

ready = True

print("-> SET UP")
time.sleep(5)
print("-> START")
while True:
    if ready == True:
        print('+++ READY FOR DETECTION +++')
        ready = False

    if Camera.get_item_detected() == True:
        #print('+++ Camera detects item +++')
        #print(Camera.get_latest_item())
        weight_change = False
        timer = 0
        added_item = None
        added_weight = 0
        print('+++ Camera detects item & WAITING for weight change +++')
        while (not weight_change) and (timer < 100):
            print(timer)
            increase = Scale.get_weight_has_increased()  # Returns two values (boolean and increase)
            if increase == True:
                print('+++ SUCCESS: Item and weight increase have been detected +++')
                added_weight = Scale.get_weight_increase()
                added_item = Camera.get_latest_item()
                weight_change = True
            time.sleep(0.1)
            timer += 1

        if weight_change == True:
            # DB.push(added_item, added_weight)
            print(added_item)
            product_no = 0
            if added_item == 'apple':
                product_no = 1
            elif added_item == 'banana':
                product_no = 2
            elif added_item == 'orange':
                product_no = 3
            elif added_item == 'pear':
                product_no = 4
            elif added_item == 'broccoli':
                product_no = 5
            elif added_item == 'aubergine':
                product_no = 6
            elif added_item == 'paprika':
                product_no = 7
            elif added_item == 'cucumber':
                product_no = 8
            print('+++ Setting product number and pushing product no +++')
            print(product_no)
            print("added weight {}".format(added_weight))
            db.push(product_no, added_weight)
            # Scale.erase_weight_change()  # not needed. because method is called fresh
        Camera.reset()
        time.sleep(3)
        ready = True
    '''
    Scale.weight_has_decreased(), bool: Stream that checks if the weight on the scale has decreased. Saves removed weight.
    Camera.detects_item(), bool: Stream that check if item with acceptable confidence has been detected. Saves detected item.
    Camera.get_latest_item(), String: Get latest item from Camera object.
    Scale.get_weight_change(), float: Get saved weight change from Scale object.
    DB.remove(added_item, added_weight), void: Removes DB entry for removed object
    Camera.reset(), void: Deletes value for latest detection and bool value for item_detected.
    Scale.erase_weight_change(), void: Deletes value for latest weight change.
    '''
    if Scale.get_weight_has_decreased() == True:
        decrease = Scale.get_weight_decrease()  # Returns two values (boolean and decrease)
        print('+++ Scale has decreased +++')
        item_detected = False
        timer = 0
        removed_weight = 0
        removed_item = None
        while (not item_detected) and (timer < 100):
            print('+++ Scale has decreased & WAITING for detection of item +++')
            print(timer)
            if Camera.get_item_detected():
                print('+++ SUCCESS: Item and weight decrease have been detected +++')
                removed_item = Camera.get_latest_item()
                #removed_weight = Scale
                item_detected = True
            timer+=1
            time.sleep(0.1)
        if item_detected == True:
            product_no = 0
            if removed_item == 'apple':
                product_no = 1
            elif removed_item == 'banana':
                product_no = 2
            elif removed_item == 'orange':
                product_no = 3
            elif removed_item == 'pear':
                product_no = 4
            elif removed_item == 'broccoli':
                product_no = 5
            elif removed_item == 'aubergine':
                product_no = 6
            elif removed_item == 'paprika':
                product_no = 7
            elif removed_item == 'cucumber':
                product_no = 8
            print('+++ Setting product number removing product +++')
            print(product_no)
            print(decrease)
            db.remove(product_no, decrease)
            # Scale.erase_weight_change()  # not needed. because method is called fresh
        Camera.reset()
        time.sleep(3)
        ready = True



