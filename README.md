# XAI - Robotics project

This repo contains the code for the project of Robotics in AI and XAI courses.

**Team members:**
- Davide Bergamasco
- Martina Toffoli

## :cyclone: Purpose 

## :book: Project structure

 - `UnityEnvs`: contains all the unity environment we used during the porject, in particular there are three envs with ML-Agents integrated, used for training and testing the models and one for ros demostration.
 - `src`: main code
   - `DQN`: code for training the DDQN network (tensorflow)
   - `STL`: code for training the STL model (Pytorch)
   - `stl_rover`: ROS2 nodes for ros demostration

## :syringe: Installation

> **This project has been developed and tested on a native installation of Ubuntu 20.04. This linux distro or a derivative of it is required.**

To be able to run this project you will need the following dependecies for a **complete setup**:

### System requirements

1. ROS2 Foxy `ros-foxy-desktop`: [complete instructions here](https://docs.ros.org/en/foxy/Installation/Ubuntu-Install-Debians.html)
2. The colcon build system: *on ubuntu* `apt install python3-colcon-common-extensions -y`
3. Anaconda: [miniconda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/linux.html)

### 


## :trumpet: Running

### Training / Testing

### ROS

1. `cd src/stl_rover`
2. `colcon build && source install/setup.(bash | zsh)`
3. To start the node with the model of the paper: `colcon build && ros2 run stl_rover paper`
4. To start the node with our model: `colcon build && ros2 run stl_rover STL`



## :whale: Porting of navigation (rover) task to  to ROS2-Unity

|                | STL Repository                                                                                                 | Unity Simulation                                                                                      |
| -------------- | -------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| **Need for**   | Exact rover, obstacles, chargers positions, destinations                                                       | Position of rover, target and  nearest charger, lidar values                                          |
| **State**      | Rover's (x, y) position, destination's (x, y) position, charger's (x, y) position, battery time and hold time. | List of lidar values, (dist, heading) for destination and nearest charger, battery time and hold time |
| **Enviroment** | Dynamic destination, always static chargers, static obstacles, battery value, charger handling                 | Dynamic destination inter/infra episodes, dynamic charger pos inter episodes                          |
| **Actions**    | Bounded continuos values : velocity [0, 1] and theta [-pi, pi]                                                 | Bounded continuos values as before                                                                    |




