from env.rover_navigation_DQN import RoverNavigation
from alg.DDQN import DDQN
import time, sys, argparse
import tensorflow as tf
import config
import os

# Check if a GPU is available
physical_devices = tf.config.list_physical_devices('GPU')

if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
else:
    print("Nessuna GPU trovata. Uso CPU.")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disabilita GPU


def train(env, args):

	# Execution of the training loop
	try: 
		algo = DDQN(env, args)
		algo.loop(args)
	
	# Listener for errors and print of the eventual error message
	except Exception as e: 
		print(e)

	# In any case, close the Unity3D environment
	finally:
		env.close()

def generate_environment():
	worker_id = int(round(time.time() % 1, 4)*10000)
	return RoverNavigation( worker_id=worker_id )


# Call the main function
if __name__ == "__main__":

	# Default parameters
	args = config.parse_args()
	# seed = None implies random seed
	editor_build = True
	env_type = "training"

	print( "Mobile Robotics Lecture on ML-agents and DDQN! \n")
	env = generate_environment()
	train(env, args)


