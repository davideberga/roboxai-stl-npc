import warnings; warnings.filterwarnings("ignore")
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys; sys.path.append("./")
import tensorflow as tf
import numpy as np
import time
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rover_navigation import RoverNavigation

seed = 42

np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

# Check if a GPU is available
physical_devices = tf.config.list_physical_devices('GPU')

if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
else:
    print("Nessuna GPU trovata. Uso CPU.")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disabilita GPU


FLAG = True

def get_action( state, policy ):

	if state[-2] < 0.01: state[-2] = 1
 
	#print(state)

	softmax_out = policy(state.reshape((1, -1))).numpy()
	selected_action = np.argmax( softmax_out )
	return selected_action


def main( env, policy_network, iterations=100 ):

	goal, crash = 0, 0

	for ep in range(iterations):

		state = env.reset()

		while True:
			action = get_action( state, policy_network )
			state, reward, done, info = env.step(action)
			if done:
				print(f'Episode: {ep}')	
				break		
			# if done: 
			# 	print(f'Ep: {ep}, Reward: {reward}, Batteria: {info["battery"]}, Goal_reached: {info["target_reached"]}, Collision: {info["collision"]}, Charger_reached: {info["n_charged"]}, d_n_target: {info["d_n_target"]}, d_n_charger: {info["d_n_charger"]}')	
			#   break

		if info["target_reached"]: 
			#print( f"{ep:3}: Goal!" )
			goal += 1

		elif info["collision"]:  
			#print( f"{np.round(state, 4)} => {action}")
			#print( f"{ep:3}: Crash!" )
			crash += 1

		#else:
			#print( f"{ep:3}: Time Out!" )

	return goal, crash, iterations


if __name__ == "__main__":

	policy_network = tf.keras.models.load_model("models/DDQN_id940_ep3_success100.h5")

	try:
		env = RoverNavigation(env_type= "test", seed=seed, worker_id=0) 
		success = main( env, policy_network )
		#print('\n======================================')
		#print(f'\nSuccess: {success[0]}/{success[2]}\nCrash: {success[1]}/{success[2]}\n')
		#print('======================================\n')

	finally:
		env.close()
