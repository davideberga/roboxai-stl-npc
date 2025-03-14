import gym
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
import numpy as np
import os
import random
import logging

gym.logger.set_level(40)


class RoverNavigation(gym.Env):
    """
    A class that implements a wrapper between the Unity Engine environment of and a custom Gym environment.
    """

    def __init__(
        self,
        env_type,
        seed,
        step_limit=300,
        worker_id=0,
        battery_time=5,
        charger_hold_time=1,
    ):
        # Load the scan number given in input
        self.scan_number = 7

        if env_type == "train":
            env_path = "env/training_rover_DQN/TrainingRoverDDQN"
        elif env_type == "test":
            env_path = "env/testing_rover_DQN/TestingRoverDDQN"
        else:
            env_path = None
            worker_id = 0
        

        unity_env = UnityEnvironment(env_path, worker_id=worker_id, seed=seed)
        self.env = UnityToGymWrapper(unity_env, flatten_branched=True)

        self.action_space = self.env.action_space

        # Override the state_size
        state_size = self.env.observation_space.shape[0] - self.scan_number

        #print(f"State size: {state_size}")
        #print(f"Scan size: {self.scan_number}")

        # Sanity check for the scan number
        assert_messages = "Mismatching between the given scan number and the observations (check the cost value)"
        assert state_size == self.scan_number + 4, assert_messages

        # Initialize the counter for the maximum step counter
        # NOT NEEDED ANYMORE, NOT episodic task
        self.step_counter = 0

        # Initialize battery time and charger hold time
        if env_type == "train":
            #self.FULL_BATTERY_TIME = battery_time if battery_time is not None else random.uniform(1, 5)
            self.FULL_BATTERY_TIME = battery_time
        else:
            self.FULL_BATTERY_TIME = battery_time

        self.FULL_CHARGER_HOLD_TIME = charger_hold_time

        self.battery_time = self.FULL_BATTERY_TIME
        self.charger_hold_time = self.FULL_CHARGER_HOLD_TIME

        # Acoording to the previous line, we override the observation space
        # lidar_scan + 6 elements,
        # - normalized in [0, 1] ==> heading (first) + distance (second) of the TARGET
        # - normalized in [0, 1] ==> heading (first) + distance (second) of the NEAREST CHARGER
        # - Battery time (number of steps) [0, 10000], Charger hold time (number of steps) [0,5]
        self.observation_space = gym.spaces.Box(
            np.array([0 for _ in range(self.scan_number)] + [0, 0, 0, 0, 0, 0]),
            np.array([1 for _ in range(self.scan_number)] + [1, 1, 1, 1, 10000, 5]),
            dtype=np.float64,
        )

    def reset(self):
        # Reset the counter for the maximum step counter and battery status
        self.battery_time = self.FULL_BATTERY_TIME
        self.charger_hold_time = self.FULL_CHARGER_HOLD_TIME
        self.step_counter = 0

        # Override the state to return with the fixed state, as described in the constructor
        state = self.env.reset()
        state = self.fix_state(state)

        self.target_distance = state[-3]

        env_var = self.extractValues(state)
        env_var['n_charged'] = 0
        return state

    def step(self, action):
        info = {}
        done = False
        state, reward, a, b = self.env.step(action)
        #print(f'Batteria prima dello step: {self.battery_time}')
        #print('Reward prima dello step: ', reward)

        state = self.fix_state(state)
        env_var = self.extractValues(state)

        info["target_reached"] = False
        info["collision"] = False

        if reward == -1:
            done = True
            info["collision"] = True
            print("\n Collisione rilevata! Episodio terminato.\n")
        elif reward == 1:
            done = True
            info["target_reached"] = True
            print("\n Target raggiunto! Episodio terminato.\n")
        
        # ------ HANDLE BATTERY -----
        # If near charger
        if env_var["d_n_charger"] < 0.3 and self.battery_time > 0:  # close to charger
            if self.charger_hold_time > 0:
                #print(f'before charger_hold_time: {self.charger_hold_time}')
                self.battery_time = min(self.FULL_BATTERY_TIME, self.battery_time + 0.1)  # Recharge
                self.charger_hold_time -= 1
                #print(f'before charger_hold_time: {self.charger_hold_time}')
                #print(f'Il robot si sta ricaricando. Batteria: {self.battery_time}')
                env_var['n_charged'] += 1
            elif self.charger_hold_time == 0:
                self.charger_hold_time = self.FULL_CHARGER_HOLD_TIME
                #print(f'Il robot ha finito di caricarsi. Batteria: {self.battery_time} e Charger_hold_time: {self.charger_hold_time}')
        else:
            if self.battery_time > 0:
                self.battery_time -= 0.01  # Discharge when not near charger                
                #print(f'Batteria dopo lo step: {self.battery_time} perchè non vicino al charger')
            else:
                done = True
                print("\n Batteria esaurita! Episodio terminato.\n")
        # ------ / HANDLE BATTERY -----
        
        #print(f"env_var: {env_var}")
        #print(f"Step terminato: {done}\n")

        if not done: # if done=False
            new_distance = env_var['d_n_target'] # update target distance
            reward = self.override_reward2(state, reward, action, done)
            self.target_distance = new_distance
        else:
            reward = self.override_reward2(state, reward, action, done)

        #print(f'Reward dopo lo step: {reward}')

        if done:
            info["battery"] = self.battery_time
            info["n_charged"] = env_var["n_charged"]
            info["d_n_target"] = env_var["d_n_target"]
            info["d_n_charger"] = env_var["d_n_charger"]
            print(f'Reward: {reward}, Batteria: {info["battery"]}, Goal_reached: {info["target_reached"]}, Collision: {info["collision"]}, Charger_reached: {info["n_charged"]}, d_n_target: {info["d_n_target"]}, d_n_charger: {info["d_n_charger"]}')	
        
        return state, reward, done, info


    def override_reward3(self, state, reward, action, done):
        env_var = self.extractValues(state)

        collision_penalty = -100
        #out_of_map_penalty = -100
        out_of_battery_penalty = -100
        reach_station_reward = 10
        reach_dest_reward = 10
        stay_station_reward = 10
        reward = 0
        security_threshold = 0.1

        lidar1 = state[0]
        lidar2 = state[1]
        lidar3 = state[2]
        lidar4 = state[3]
        lidar5 = state[4]
        lidar6 = state[5]
        lidar7 = state[6]
        
        lidar = [lidar1, lidar2, lidar3, lidar4, lidar5, lidar6, lidar7]

        min_lidar = min(lidar)

        # Se il robot ha avuto una collisione, penalizzare fortemente
        if (done and reward == -1) or min_lidar < security_threshold:  # Se reward è -1, c'è stata una collisione
            reward += collision_penalty

        if done and reward == 1:  # Se reward è 1; il target è stato raggiunto
            reward += reach_dest_reward

        if self.battery_time < 0:
            reward += out_of_battery_penalty

        if env_var["h_n_charger"] < 1: # il robot si sta caricando
            reward += stay_station_reward 

        if env_var["d_n_charger"] < 0.3: # il robot è vicino al target
            reward  += reach_station_reward

        return reward


    def override_reward2(self, state, reward, action, done):
        env_var = self.extractValues(state)
        step_penalty = 0.0001
        collision_penalty = -50
        out_of_battery_penalty = -50
        security_threshold = 0.1
        reach_dest_reward = 10

        lidar1 = state[0]
        lidar2 = state[1]
        lidar3 = state[2]
        lidar4 = state[3]
        lidar5 = state[4]
        lidar6 = state[5]
        lidar7 = state[6]
        
        lidar = [lidar1, lidar2, lidar3, lidar4, lidar5, lidar6, lidar7]

        min_lidar = min(lidar)

        if (done and reward == -1) or min_lidar < security_threshold:
            reward += collision_penalty

        if done and reward == 1: 
            reward += reach_dest_reward

        if self.battery_time > 2.5:
            reward += self.battery_time -(abs(env_var['d_n_target'])) + step_penalty
        elif self.battery_time < 0:
            reward += out_of_battery_penalty
        else:
            reward += self.battery_time -(abs(env_var['d_n_charger'])) + step_penalty
        
        if done and reward == -1:
            reward += collision_penalty
        
        return reward 

    def override_reward1(self, state, reward, action, done):
        # Dati estratti dallo stato
        env_var = self.extractValues(state)

        lidar1 = state[0]
        lidar2 = state[1]
        lidar3 = state[2]
        lidar4 = state[3]
        lidar5 = state[4]
        lidar6 = state[5]
        lidar7 = state[6]
        
        lidar = [lidar1, lidar2, lidar3, lidar4, lidar5, lidar6, lidar7]

        min_lidar = min(lidar)

        collision_penalty = -30
        out_of_battery_penalty = -30
        reach_station_reward = 10
        reach_dest_reward = 10
        stay_station_reward = 10
        security_threshold = 0.1

        reward = 0

        # Se il robot ha avuto una collisione, penalizzare fortemente
        if (done and reward == -1) or min_lidar < security_threshold:  # Se reward è -1, c'è stata una collisione
            reward += collision_penalty
        elif done and reward == 1: 
            reward += reach_dest_reward
        elif min_lidar > security_threshold:  
            reward += min_lidar

        if self.battery_time < 0:
            reward += out_of_battery_penalty
        elif self.battery_time > 2.5: # Se la batteria è carica abbastanza, vai al target
            reward -= env_var["d_n_target"]
        elif self.battery_time <= 2.5: # Se la batteria è scarica, vai alla stazione di ricarica
            reward += reach_station_reward - env_var["d_n_charger"] 


        return reward

    def extractValues(self, state):
        non_lidar_state = {}
        non_lidar_state["bat_hold"] = state[-1]
        non_lidar_state["bat_time"] = state[-2]
        non_lidar_state["d_n_charger"] = state[-3]
        non_lidar_state["h_n_charger"] = state[-4]
        non_lidar_state["d_n_target"] = state[-5]
        non_lidar_state["h_n_target"] = state[-6]
        non_lidar_state["n_charged"] = 0
        return non_lidar_state

    def fix_state(self, state):
        # Compute the size of the observation array that correspond to the lidar sensor,
        # the other portion is maintened
        # [LIDAR, LIDAR, LIDAR, LIDAR, LIDAR, LIDAR, LIDAR, HEAD_TARGET, DIST_TARGET, HEAD_N_CHARGER, DIST_N_CHARGER, B_TIME, C_TIME]
        scan_limit = 2 * (self.scan_number)
        state_lidar = [s for id, s in enumerate(state[:scan_limit]) if id % 2 == 1]

        # Change the order of the lidar scan to the order of the wrapper (see the class declaration for details)
        lidar_ordered_1 = [
            s for id, s in enumerate(reversed(state_lidar)) if id % 2 == 0
        ]
        lidar_ordered_2 = [s for id, s in enumerate(state_lidar) if id % 2 == 1]
        lidar_ordered = lidar_ordered_1 + lidar_ordered_2

        # Concatenate the ordered lidar state with the other values of the state
        state_fixed = (
            lidar_ordered
            + list(state[scan_limit:])
            + [self.battery_time, self.charger_hold_time]
        )

        state_fixed = np.array(state_fixed)
        
        #print('len(state_fixed): ', len(state_fixed))
        return state_fixed

    # Override the "close" function
    def close(self):
        self.env.close()

    # Override the "render" function
    def render(self):
        pass
