import warnings

warnings.filterwarnings("ignore")
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys

sys.path.append("./")
import tensorflow as tf
import numpy as np
import time
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rover_navigation_test import RoverNavigationTest

seed = 42

np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

# Check if a GPU is available
physical_devices = tf.config.list_physical_devices("GPU")

if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
else:
    print("Nessuna GPU trovata. Uso CPU.")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disabilita GPU


FLAG = True


def get_action(state, policy):
    if state[-2] < 0.01:
        state[-2] = 1
    softmax_out = policy(state.reshape((1, -1))).numpy()
    selected_action = np.argmax(softmax_out)
    return selected_action


def main(env, policy_network, iterations=100):
    goal, crash = 0, 0

    # First reset
    state, state_complete = env.reset()

    saved_episodes = []
    
    episode = []

    for ep in range(iterations):
        # An episode is complete when 3 goals are reached or collision or battery finished occur
        goal = 0
        collision = 0
        battery = 0
        state, state_complete = env.reset(battery_reset=True)

        while True:
            action = get_action(state, policy_network)
            v = 0.05
            t = 0
            if action == 1:
                v = 0
                t = np.pi / 3
            if action == 2:
                v = 0
                t = -np.pi / 3
            state, reward, done, info, state_complete = env.step(action)

            if info["target_reached"]:
                goal = 1
            if info["collision"]:
                collision = 1
            if info["battery_ended"]:
                battery = 1
                
            episode.append(np.concatenate((state_complete, np.array([v, t, goal, collision, battery]))))

            if done:
                break

        saved_episodes.append(np.array(episode))
        print(f" Episode {ep}")

    return np.array(saved_episodes)


if __name__ == "__main__":
    policy_network = tf.keras.models.load_model("models/DDQN_paper_id940_ep4883_success82.h5")

    ENV_TYPE = "test"

    try:
        env = RoverNavigationTest(env_type=ENV_TYPE, seed=seed, worker_id=0, is_env_for_paper=False)
        saved_episodes = main(env, policy_network)
        np.savez("test-result/dqn.result.npz", episodes=saved_episodes)
        env.close()
    finally:
        print("Test DDQN finished!")
