from threading import Thread
import time

def print1():
    while True:
        print("THREAD1")
        time.sleep(1)

def print2():
    while True:
        print("THREAD2")
        time.sleep(1)

Thread(target=print1).start()
Thread(target=print2).start()