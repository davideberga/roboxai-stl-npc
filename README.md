## XAI - Robotics project

This repo contains the code for the project of Robotics in AI and XAI courses.
- [XAI - Robotics project](#xai---robotics-project)
- [:cyclone: Overview and purpose](#cyclone-overview-and-purpose)
- [:whale: Porting of navigation (rover) task to ROS2-Unity](#whale-porting-of-navigation-rover-task-to-ros2-unity)
- [:book: Project structure](#book-project-structure)
- [:syringe: Installation](#syringe-installation)
  - [System requirements](#system-requirements)
- [:trumpet: Running](#trumpet-running)
  - [Train / Test](#train--test)
  - [DQN](#dqn)
  - [STL](#stl)
  - [ROS](#ros)
- [:checkered\_flag: Test result](#checkered_flag-test-result)
  - [Test in graphical env](#test-in-graphical-env)
  - [Test in unity](#test-in-unity)


**Team members:**
- Davide Bergamasco
- Martina Toffoli

## :cyclone: Overview and purpose 

This project aims to use the **Neural Predictive Control** (npc) method to predict a sequence of actions, given an initial agent state, for a fixed time horizon. The sequence predicted should follow a set of Signal Temporal Logic formula with which safety and task goal contraints are expressed. We take the idea and a small portion of the code from this [paper](https://arxiv.org/abs/2309.05131) and its [code](https://github.com/mengyuest/stl_npc). Specifically, in this project we have ported the Rover task, where an agent must reach a destination without collision or ending its battery. 
We have rethinked the state in a more general way a rewite the STL rules for the collision event. At the end we performed a comparison with a DDQN model with the same state, trained in a Unity envrinoment with ML-agent.


| STL Repository | Our Model | ROS Simulation |
|----------------|-----------|------------------|
| ![Alt Text](./asset/paper.gif)   | ![Alt Text](./asset/our.gif) | |


## :whale: Porting of navigation (rover) task to ROS2-Unity

|                | STL Repository                                                                                                 | Unity Simulation                                                                                      |
|----------------|----------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| **Need for**   | Exact rover, obstacles, chargers positions, destinations                                                       | Position of rover, target and  nearest charger, lidar values                                          |
| **State**      | Rover's (x, y) position, destination's (x, y) position, charger's (x, y) position, battery time and hold time. | List of lidar values, (dist, heading) for destination and nearest charger, battery time and hold time |
| **Enviroment** | Dynamic destination, always static chargers, static obstacles, battery value, charger handling                 | Dynamic destination inter/infra episodes, dynamic charger pos inter episodes                          |
| **Actions**    | Bounded continuos values : velocity [0, 1] and theta [-pi, pi]                                                 | Bounded continuos values as before                                                                    |


## :book: Project structure

 - `UnityEnvs/`: contains all the unity environment we used during the project, in particular there are three envs with ML-Agents integrated, used for training and testing the models and one for ros demostration.
 - `src/`: main code
   - `DQN/`: code for training the DDQN network (tensorflow)
   - `STL/`: code for training the STL model (Pytorch)
   - `stl_rover/`: ROS2 nodes for ros demostration

## :syringe: Installation

> **This project has been developed and tested on a native installation of Ubuntu 20.04. This linux distro or a derivative of it is required.**

To be able to run this project you will need the following dependecies for a **complete setup**:

### System requirements

1. ROS2 Foxy `ros-foxy-desktop`: [complete instructions here](https://docs.ros.org/en/foxy/Installation/Ubuntu-Install-Debians.html)
2. Turtlebot3 and tf ros packages: `ros-foxy-turtlebot3-msgs ros-foxy-turtlebot3  ros-foxy-tf-transformations`
3. The colcon build system: *on ubuntu* `apt install python3-colcon-common-extensions -y`
4. Anaconda: [miniconda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/linux.html)
5. For ROS nodes you'll need the following packages installed system wide or anyway available ***for the same python interpreter ROS uses*** (`pyhton3.8 -m pip install ...` to be sure):
    - numpy (*mandatory*)
    - tensorflow==2.8.0 (*mandatory*)
    - transforms3d (*mandatory*)
    - wandb (*mandatory*)
    - scipy (*mandatory*)
    - torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 (*mandatory*)
    - gym==0.20.0 (*optional*)
    - gym_unity==0.28.0 (*optional*)
    - mlagents_envs==0.28.0 (*optional*)
6. For training testing ouside of ROS a conda env is provided with all the necessary dependencies `conda env create -f environment.yml`

The dependencies marked as *optional* are required only for testing/training outside of ROS context.
    


## :trumpet: Running

### Train / Test

### DQN 
 - `conda activate unity-ros`
 - `cd src/DQN`
 - **Train**: `python training_DQN.py`
 - **Test**: `python test_unity.py`

### STL
 - `conda activate unity-ros`
 - `cd src/STL`
 - **Train**: `python training.py --lr=0.0001`
 - **Test in perfect graphic enviroment**: `python test_graphic.py`
 - **Test**: `python test_unity.py`


### ROS

1. `cd src/stl_rover`
2. `colcon build && source install/setup.(bash | zsh)`
3. To start the node with the model of the paper: `colcon build && ros2 run stl_rover paper`
4. To start the node with our model: `colcon build && ros2 run stl_rover STL`


## :checkered_flag: Test result

## METRICS

#### N° of goals reached: 
The average percentage of the sum of the goals achieved in each episode.
#### Mean Battery: 
The average percentage of the battery average value for each episode.
#### Mean Velocity:
The average of the velocity average value for each episode.
#### Delta Velocity:
Given an array containing lists of velocities at each step for each episode, the velocity delta is represents the average of the difference between each velocity value and the previous one for each step.
#### Safety:
Set a threshold = 0.15 representing the maximum distance to an obstacle, safety is the average percentage of times the lidar, with minimum distance at each step, does not exceed this threshold.
#### Low Battery:
The number of times, as a percentage, that the battery discharged to the total number of episodes.
#### Accuracy:
The percentage of how well the robot complies with STL rules.
#### Battery correlation:
-
#### Collision:
The number of times, as a percentage, that the robot collides to the total number of episodes.
#### Avoid:
- 
#### Totale distance:
The average of the total distance traveled for each episode.


<!-- START TABLES -->




### Test in graphical env

|Method|N_Goals_Reached|Mean Battery %|Mean Velocity|Mean Abs Delta Velocity|Safety %|Low Battery %|Accuracy %|Battery correlation|Collision %|Avoid %|Total distance|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|Paper|76.4|90.32 ± 0.81|0.21 ± 0.17|0.21|691|0.1|20.14|0.86|23.5|16.66|1.15 ± 0.94|
|OUR|69.3|72.91 ± 1.5|0.13 ± 0.13|0.16|686|4.6|31.3|-0.35|26.1|17.25|0.78 ± 0.91|
|No avoid rule|75.2|65.05 ± 1.37|0.21 ± 0.14|0.22|695|0.4|2.33|-0.54|24.5|16.07|0.77 ± 0.66|






### Test in unity

|Method|N_Goals_Reached|Mean Battery %|Mean Velocity|Mean Abs Delta Velocity|Safety %|Low Battery %|Accuracy %|Battery correlation|Collision %|Avoid %|Total distance|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|Paper|34.0|42.66 ± 1.59|0.32 ± 0.42|0.28|45|37.0|82.28|0.36|29.0|24.03|4.5 ± 4.06|
|DQN|53.0|45.51 ± 1.29|0.04 ± 0.02|0.01|99|19.0|13.45|-0.04|2.0|68.27|nan ± nan|
|OUR|4.0|48.05 ± 1.52|0.34 ± 0.28|0.4|27|13.0|11.03|-0.13|83.0|12.48|3.29 ± 2.71|
|No avoid rule|1.0|48.15 ± 1.47|0.67 ± 0.27|0.67|20|0.0|5.88|-0.19|99.0|9.6|3.81 ± 2.46|

<!-- END TABLES -->