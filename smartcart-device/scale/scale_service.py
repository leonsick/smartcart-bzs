from .scale import Scale
from threading import Thread

class ScaleService:
    weight_has_decreased = False
    weight_decrease = 0
    weight_has_increased = False
    weight_increased = 0
    
    scale = Scale()

    @classmethod
    def start(cls):
        cls.scale.start()
        Thread(target=cls.weight_has_decreased).start()
        print("weight has decreased thread started")
        Thread(target=cls.weight_has_increased).start()
        print("weight has increased thread started")

    @classmethod
    def weight_has_decreased(cls):
        #print("Called Scale.get_weight_has_decreased()")
        while True:
            #print("scale.detect_decrease()")
            cls.weight_has_decreased, cls.weight_decrease = cls.scale.detect_decrease()
    
    @classmethod
    def weight_has_increased(cls):
        #print("Called Scale.get_weight_has_increased()")
        while True:
            #print("scale.detect_increase()")
            cls.weight_has_increased, cls.weight_increase = cls.scale.detect_increase()


    @classmethod
    def get_weight_decrease(cls):
        return cls.weight_decrease
    
    @classmethod
    def get_weight_increase(cls):
        return cls.weight_increase

    @classmethod
    def get_weight_has_decreased(cls):
        return cls.weight_has_decreased
    
    @classmethod
    def get_weight_has_increased(cls):
        return cls.weight_has_increased

    @classmethod
    def reset(cls):
        cls.weight_has_decreased = False
        cls.weight_decrease = 0
        cls.weight_has_increased = False
        cls.weight_increase = 0