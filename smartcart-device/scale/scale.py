import measure_weight
from threading import Thread
import time

class Scale(object):
    def __init__(self):
        self.weight_list = [0,0,0,0,0]
        self.weight_has_decreased = False
        self.started = False
        self.stopped = False
        self.t_measure_weight = None

    def start(self):
        self.started = True
        self.t_measure_weight = Thread(target=self.update)
        self.t_measure_weight.start()
        

    def read(self):
        return measure_weight.call_weight()

    def update(self):
        #print("update(self)")
        while self.started:
            self.weight_list[1:5] = self.weight_list[0:4]
            self.weight_list[0] = self.read()
            time.sleep(0.5)

    def detect_decrease(self):
        counter = 0
        decreased = False
        weight_decrease = 0
        #print("Weight list {}".format(self.weight_list))
        weights = self.weight_list
        
        while counter < (len(self.weight_list)-1):
            if (weights[counter + 1] - weights [counter]) > 50:
                decreased = True
                weight_decrease = weights[counter + 1] - weights[counter]
                print(weights)
                break
            counter+=1
        time.sleep(1)
        return decreased, weight_decrease
    
    def detect_increase(self):
        counter = 0
        increased = False
        weight_increase = 0
        #print("Weight list {}".format(self.weight_list))
        weights = self.weight_list
        
        while counter < (len(self.weight_list)-1):
            if (weights[counter] - weights[counter + 1]) > 50:
                increased = True
                weight_increase = weights[counter] - weights[counter + 1]
                print(weights)
                break
            counter+=1
        time.sleep(1)
        return increased, weight_increase