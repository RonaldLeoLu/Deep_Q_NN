# some useful things
# some useful functions and tools
from collections import deque
import os
import pickle
import numpy as np
import cv2
import torch

class MyDequeSeq:
    OBSERVATION = 10000

    def __init__(self, filename=None):
        self.filename = filename
        if os.path.exists(filename):
            self.preload = True
        else:
            self.preload = False
        self.replay_mem = deque(maxlen=MyDequeSeq.OBSERVATION)

    def store_transtition(self, states):
        self.replay_mem.append(states)

    def save_cache(self):
        pickle.dump(self.replay_mem, open(self.filename, 'wb'))

    def load_cache(self):
        return pickle.load(open(self.filename, 'rb'))

    def set_init_state(self):
        self.replay_mem.clear()
        if self.preload:
            self.replay_mem = self.load_cache()

    def load_mode(self):
        self.preload = True

class CreatePics:
    def __init__(self):
        self.current_state = None

    def preprocess(self, frame):
        img = cv2.cvtColor(cv2.resize(frame, (80, 80)), cv2.COLOR_BGR2GRAY)
        _, img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
        o = np.reshape(img, (80, 80))
        return o

    def generate_next_input(self, frame=None):
        o = self.preprocess(frame) # 1*M*M
        #print('o size:',o.shape)

        if self.current_state is None:
            self.current_state = np.stack((o, o, o, o))
            #print('current state shape:',self.current_state.shape)
        else:
            self.current_state = np.concatenate(
                (self.current_state[1:,:,:], o.reshape((1,)+o.shape)),axis=0)

        return self.current_state
