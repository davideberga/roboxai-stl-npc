from collections import deque
from importlib.resources import path
from .log_utils import *
import numpy as np
import tensorflow as tf
import gym, sys
import time
import warnings; warnings.filterwarnings("ignore")
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

class DDQN():

	"""
	Double DQN algorithm implementation, the original paper can be found here:
	https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf [1] and https://ojs.aaai.org/index.php/AAAI/article/view/10295 [2]

	[1] Playing atari with deep reinforcement learning, 
		Mnih et al., 
		arXiv preprint arXiv:1312.5602, 2013

	[1] Playing atari with deep reinforcement learning, 
		Van Hasselt et al., 
		Proceedings of the AAAI conference on artificial intelligence, 2016
	"""


	# Constructor of the class
	def __init__( self, env, args):
		
		self.seed = args.seed
		if self.seed is None: 
			self.seed = np.random.randint(0, 1000)

		tf.random.set_seed( self.seed )
		np.random.seed( self.seed )

		self.env = env
		self.run_name = f"{args.alg}__{args.tag if args.tag != '' else ''}__{args.seed}__{int(time.time())}"
		self.input_shape = self.env.observation_space.shape
		self.action_space = self.env.action_space
		
		# Training hyperparameters
		self.memory_size = args.memory_size
		self.gamma = args.gamma
		self.epoch = args.n_epochs
		self.batch_size = args.batch_size
		self.eps_decay = args.eps_decay
		self.tau = args.tau
		self.layers = args.n_layers
		self.nodes = args.layer_size

		
		# initialize memory buffer to store all the trajectories
		self.memory_buffer = deque( maxlen=self.memory_size )
		self.eps_greedy = 1

		# create the DNN actor and target
		self.actor = self.create_model(self.input_shape, self.action_space.n, layers=self.layers, nodes=self.nodes)
		self.actor_target = self.create_model(self.input_shape, self.action_space.n, layers=self.layers, nodes=self.nodes)
		self.actor_target.set_weights(self.actor.get_weights())

		# optimizer for the DNN
		self.optimizer = tf.keras.optimizers.Adam()



	# Class that generate a basic neural netowrk from the given parameters.
	# Can be overrided in the inheriting class for a specific architecture (e.g., dueling)
	def create_model(self, input_shape, output_size=1, layers=2, nodes=32, last_activation='linear', output_bounds=None):

		# Fix if the output shape is received as a gym spaces,conversion to integer
		if isinstance(output_size, gym.spaces.discrete.Discrete): output_size = output_size.n
		if isinstance(output_size, gym.spaces.box.Box): output_size = output_size.shape[0]		

		# Iterate over the provided parametrs to create the network, the input shape must be defined as a multidimensional tuple (e.g, (4,))
		# While the output size as an integer
		hiddens_layers = [tf.keras.layers.Input(shape=input_shape)]

		for _ in range(layers):	
			hiddens_layers.append(tf.keras.layers.Dense( nodes, activation='relu')( hiddens_layers[-1]))

		hiddens_layers.append(tf.keras.layers.Dense( output_size, activation=last_activation)( hiddens_layers[-1]))	

		# Normalize the output layer between a range if given, usually used with continuous control for a sigmoid final activation
		if output_bounds is not None: 
			hiddens_layers[-1] = hiddens_layers[-1] * (output_bounds[1] - output_bounds[0]) + output_bounds[0]

		# Create the model with the keras format and return
		return tf.keras.Model( hiddens_layers[0], hiddens_layers[-1] )

	
	def network_update_rule(self, terminal):

		# Update of the networks for DDQN!
		# Update of the eps greedy strategy after each episode of the training
		if terminal: 
			self.update_networks(np.array(self.memory_buffer, dtype=object))
			self.eps_greedy *= self.eps_decay
			self.eps_greedy = max(0.05, self.eps_greedy)

		# Update toward the target network
		self.update_target(self.actor.variables, self.actor_target.variables, tau=self.tau)


	# Application of the gradient with TensorFlow and based on the objective function
	def update_networks(self, memory_buffer):

		for _ in range(self.epoch):

			# Computing a random sample of elements from the batch for the training, randomized at each iteration
			idx = np.random.randint(memory_buffer.shape[0], size=self.batch_size)
			training_batch = memory_buffer[idx]

			with tf.GradientTape() as tg:
				# Compute the objective function, compute the gradient information and apply the gradient with the optimizer
				objective_function = self.objective_function(training_batch)
				gradient = tg.gradient(objective_function, self.actor.trainable_variables)
				self.optimizer.apply_gradients(zip(gradient, self.actor.trainable_variables))


	# Soft update (polyak avg) of the network toward the target network
	def update_target(self, weights, target_weights, tau):
		for (a, b) in zip(target_weights, weights):
			a.assign(b * tau + a * (1 - tau))

	# we select thea action based on the state, for eps greedy based policy we perform the exploration selecting random action with a decreasing frequency
	def get_action(self, state):

		if np.random.random() < self.eps_greedy:
			action = np.random.choice(self.action_space.n)
		else:
			action = np.argmax(self.actor(state.reshape((1, -1))))

		return action, 0


	# Computing the objective function of the DDQN for the gradient descent procedure, here it applies the Bellman equation
	def objective_function(self, memory_buffer):

		# Extract values from buffer
		state = np.vstack(memory_buffer[:, 0])
		action = memory_buffer[:, 1]
		reward  = np.vstack(memory_buffer[:, 3])
		new_state = np.vstack(memory_buffer[:, 4])
		done = np.vstack(memory_buffer[:, 5])

		# compute Q'(s',a')
		next_state_action = np.argmax(self.actor(new_state), axis=1)
		target_mask = self.actor_target(new_state) * tf.one_hot(next_state_action, self.action_space.n)
		target_mask = tf.reduce_sum(target_mask, axis=1, keepdims=True)
		
		
		target_value = reward + (1 - done.astype(int)) * self.gamma * target_mask
		mask = self.actor(state) * tf.one_hot(action, self.action_space.n)
		prediction_value = tf.reduce_sum(mask, axis=1, keepdims=True)

		
		mse = tf.math.square(prediction_value - target_value)
		
		
		return tf.math.reduce_mean(mse)

	def loop(self, args):		

		# Initialize the logger
		logger = create_logger(self.run_name, args)
		if args.wandb_log: init_wandb(self.run_name, args)

		logger_dict = { "reward": [], "success": [], "step": [], "cost": []}
		
		

		# Iterate the training loop over multiple episodes
		for episode in range(args.n_episode):

			# Reset the environment at each new episode
			state = self.env.reset()

			# Initialize the values for the logger
			logger_dict['reward'].append(0)
			logger_dict['success'].append(0)
			logger_dict['step'].append(0)
			logger_dict['cost'].append(0)
            
			# Main loop of the current episode
			while True:

				# Select the action, perform the action and save the returns in the memory buffer
				action, action_prob = self.get_action(state)
				new_state, reward, done, info = self.env.step(action)
				self.memory_buffer.append([state, action, action_prob, reward, new_state, done])

				# Update the dictionaries for the logger and the trajectory
				logger_dict['reward'][-1] += reward	
				logger_dict['step'][-1] += 1	
				logger_dict['cost'][-1] += info['cost']
				logger_dict['success'][-1] = 1 if info['goal_reached'] else 0

				# Call the update rule of the algorithm
				self.network_update_rule(done)

				# Exit if terminal state and eventually update the state
				if done: break
				state = new_state


			# after each episode log and print information
			last_n =  min(len(logger_dict['reward']), 100)
			reward_last_100 = logger_dict['reward'][-last_n:]
			cost_last_100 = logger_dict['cost'][-last_n:]
			step_last_100 = logger_dict['step'][-last_n:]
			success_last_100 = logger_dict['success'][-last_n:]

			record = {
					'Episode': episode,
					'Step': int(np.mean(step_last_100)),
					'Avg_Cost': int(np.mean(cost_last_100)*100),
					'Avg_Success': int(np.mean(success_last_100)*100),
					'Avg_Reward': np.mean(reward_last_100),	
				}
			logger.write(record)

			print( f"(DDQN) Ep: {episode:5}", end=" " )
			print( f"reward: {logger_dict['reward'][-1]:5.2f} (last_100: {np.mean(reward_last_100):5.2f})", end=" " )
			print( f"cost_last_100: {int(np.mean(cost_last_100))}", end=" " )
			print( f"step_last_100 {int(np.mean(step_last_100)):3d}", end=" " )
			if 'eps_greedy' in self.__dict__.keys(): print( f"eps: {self.eps_greedy:3.2f}", end=" " )
			if 'sigma' in self.__dict__.keys(): print( f"sigma: {self.sigma:3.2f}", end=" " )
			print( f"success_last_100 {int(np.mean(success_last_100)*100):4d}%" )

			if args.wandb_log:
				wandb.log(record) 

			# save model if we reach avg_success greater than 78%
			if int(np.mean(success_last_100)*100) >= 79:
					self.actor.save(f"models/DDQN_id{self.seed}_ep{episode}_success{int(np.mean(success_last_100)*100)}.h5")

		
	
		
				



	

	