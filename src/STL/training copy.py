import traceback
from rover_navigation import RoverNavigation
from alg.RoverSTL import RoverSTL
import time
import tensorflow as tf
import config


physical_devices = tf.config.list_physical_devices("GPU")
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)


def train(env, args):
    # Execution of the training loop
    try:
        algo = RoverSTL(env, args)
        algo.training_loop(args)

    # Listener for errors and print of the eventual error message
    except Exception as e:
        raise e

    # In any case, close the Unity3D environment
    finally:
        env.close()


def generate_environment(editor_build, env_type):
    
    return 


# Call the main function
if __name__ == "__main__":
    # Default parameters
    args = config.parse_args()

    # seed = None implies random seed
    editor_build = True
    env_type = "training"

    print("STL Rover training with Davide and Martina!\n")
    
    worker_id = int(round(time.time() % 1, 4) * 10000)
    env = RoverNavigation(worker_id=worker_id, is_training=True)
    
    train(env, args)
