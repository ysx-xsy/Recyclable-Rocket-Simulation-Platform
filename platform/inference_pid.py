from rocket_pid import Rocket
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import random

class PIDInference:
    def __init__(self):
        self.env = Rocket(task='landing', max_steps=2000)

    def run_inference(self):
        done = self.env.step()
        frame_0, frame_1 = self.env.render()
        state = self.env.state
        return frame_0, frame_1, state, done
