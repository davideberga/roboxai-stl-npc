# XAI - Robotics project

This repo contains the code for the project of Robotics in AI and XAI courses.

**Team members:**
- Martina Toffoli
- Davide Bergamasco


## :whale: Porting of navigation (rover) task to  to ROS2-Unity

|                | STL Repository                                                                                                 | Unity Simulation                                                                                      |
| -------------- | -------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| **Need for**   | Exact rover, obstacles, chargers positions, destinations                                                       | Position of rover, target and  nearest charger, lidar values                                          |
| **State**      | Rover's (x, y) position, destination's (x, y) position, charger's (x, y) position, battery time and hold time. | List of lidar values, (dist, heading) for destination and nearest charger, battery time and hold time |
| **Enviroment** | Dynamic destination, always static chargers, static obstacles, battery value, charger handling                 | Dynamic destination inter/infra episodes, dynamic charger pos inter episodes                          |
| **Actions**    | Bounded continuos values : velocity [0, 1] and theta [-pi, pi]                                                 | Bounded continuos values as before                                                                    |


