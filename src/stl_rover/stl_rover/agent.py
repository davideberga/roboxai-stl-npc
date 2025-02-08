from collections import deque
import numpy as np
import tensorflow as tf
from ament_index_python.packages import get_package_share_directory


class Agent:
    def __init__(self, verbose):

        package_dir = get_package_share_directory("stl_rover")
        # load weights of pretrained model
        self.model = tf.keras.models.load_model(package_dir + "/model_trained/DDQN_id940_ep2666.h5")
        if verbose:
            print('==================================================')
            print('  We are using this model for the testing phase:  ')
            print('==================================================')

    def select_action(self, state):
        # we directly take the argmax for that particular state
        action = np.argmax(self.model(state.reshape((1, -1))))
        return action

    def normalize_state(self, state):
        return np.around(state,3)
