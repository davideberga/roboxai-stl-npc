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

## STL Paper Dynamics

> Example State
> `tensor([[6.6312, 3.9914, 6.9342, 0.9568, 6.0255, 3.9109, 5.0000, 0.4000]],
       device='cuda:0')`

> Example Rules score
> ``` 
>    in_map: tensor([[33.6980, 29.4170, 27.8107, 27.8107, 27.8107, 26.0355, 25.5567, 25.5567,
>         23.5417, 22.3649, 22.3649]], device='cuda:0', grad_fn=<SelectBackward0>)
>    Avoid Poly: tensor([[10.8836, 12.1679, 12.6498, 12.6498, 12.6498, 13.1823, 13.3260, 13.3260,
>             13.9305, 14.2835, 14.2835]], device='cuda:0', grad_fn=<NegBackward0>)
>    Avoid Seg Int: tensor([[0.9972, 0.9972, 0.9972, 0.9972, 0.9972, 0.9972, 0.9972, 0.9972, 0.9972,
>             0.9972, 1.0000]], device='cuda:0', grad_fn=<MulBackward0>)
>    Battery > Limit: tensor([[3.0000, 3.0000, 2.8000, 2.6000, 2.4000, 2.2000, 2.0000, 1.8000, 1.6000,
>             1.4000, 1.2000]], device='cuda:0', grad_fn=<SubBackward0>)
>    at_dest: tensor([[-2.2497, -2.0504, -2.0620, -2.0620, -2.0620, -2.0110, -2.0153, -2.0153,
>             -2.0650, -2.0928, -2.0928]], device='cuda:0', grad_fn=<RsubBackward1>)
>    at_charger: tensor([[ 0.1890, -0.2392, -0.3991, -0.3991, -0.3991, -0.5839, -0.6319, -0.6319,
>             -0.8312, -0.9486, -0.9486]], device='cuda:0', grad_fn=<RsubBackward1>)
>    Standby (norm): tensor([[ 0.1000, -0.3671, -0.5177, -0.5177, -0.5177, -0.7098, -0.7565, -0.7565,
>             -0.9483, -1.0635, -1.0635]], device='cuda:0', grad_fn=<RsubBackward1>)
>    Enough Stay: tensor([[-0.4000, -0.2000, -0.2000, -0.2000, -0.2000, -0.2000, -0.2000, -0.2000,
>             -0.2000, -0.2000, -0.2000]], device='cuda:0', grad_fn=<NegBackward0>)
>    Battery Always: tensor([[5.0000, 5.0000, 4.8000, 4.6000, 4.4000, 4.2000, 4.0000, 3.8000, 3.6000,
>             3.4000, 3.2000]], device='cuda:0', grad_fn=<SelectBackward0>)
>    Battery < Limit: tensor([[-3.0000, -3.0000, -2.8000, -2.6000, -2.4000, -2.2000, -2.0000, -1.8000,
>             -1.6000, -1.4000, -1.2000]], device='cuda:0', grad_fn=<RsubBackward1>)
>    at_charger: tensor([[ 0.1890, -0.2392, -0.3991, -0.3991, -0.3991, -0.5839, -0.6319, -0.6319,
>             -0.8312, -0.9486, -0.9486]], device='cuda:0', grad_fn=<RsubBackward1>)
> ```

## STL Rover 4 Unity and ROS

> Example state: `tensor([[0.3069, 0.1040, 0.0960, 0.1164, 0.1544, 0.1531, 1.0000, 0.0578, 0.3586,
         2.0751, 0.3886, 1.0000, 1.0000]], device='cuda:0')`

> Example estimated next states: 
> ```
> tensor([[[0.3069, 0.1040, 0.0960, 0.1164, 0.1544, 0.1531, 1.0000, 0.0578,
>           0.3586, 2.0751, 0.3886, 1.0000, 1.0000],
>          [0.4331, 0.4913, 0.5459, 0.6292, 0.4926, 0.4807, 0.5405, 0.2483,
>           0.3132, 2.3539, 0.4125, 0.9000, 1.0000],
>          [0.4246, 0.5225, 0.5736, 0.6482, 0.7347, 0.5684, 0.3791, 0.4048,
>           0.2663, 2.5424, 0.4482, 0.8000, 1.0000],
>          [0.4271, 0.6065, 0.5532, 0.6926, 0.6658, 0.5926, 0.4367, 0.5858,
>           0.2224, 2.6934, 0.4891, 0.7000, 1.0000],
>          [0.5416, 0.6067, 0.6281, 0.7404, 0.7149, 0.7490, 0.5456, 0.5892,
>           0.1819, 2.5819, 0.5358, 0.6000, 1.0000],
>          [0.7269, 0.8170, 0.9468, 0.7832, 0.7019, 0.7485, 0.5365, 0.5340,
>           0.1436, 2.3801, 0.5781, 0.5000, 1.0000],
>          [0.9305, 0.8967, 0.8422, 0.8802, 0.7862, 0.8899, 0.9376, 1.0418,
>           0.1030, 2.6896, 0.6161, 0.4000, 1.0000],
>          [0.9887, 0.9352, 0.8896, 0.8963, 0.9626, 0.9821, 0.9749, 1.4186,
>           0.0889, 2.6072, 0.6602, 0.3000, 1.0000],
>          [0.9960, 0.9356, 0.9394, 0.8922, 1.0000, 0.9998, 0.9504, 1.6973,
>           0.0945, 2.3922, 0.7021, 0.2000, 1.0000],
>          [0.9998, 0.9936, 0.9659, 0.9546, 0.9886, 1.0000, 0.9981, 2.3661,
>           0.1109, 2.6686, 0.7374, 0.1000, 1.0000],
>          [1.0000, 0.9998, 0.9985, 0.9897, 0.9952, 0.9941, 1.0000, 2.8496,
>           0.1554, 2.9317, 0.7872, 0.0000, 1.0000]]], device='cuda:0',
>        grad_fn=<StackBackward0>)
> ```

> Example rules: 
>
> ``` 
>     Lidar safety: tensor([[-0.1422,  0.2067,  0.1974,  0.2273,  0.3209,  0.3881,  0.5729,  0.6450,
>               0.6570,  0.6897,  0.7021]], device='cuda:0', grad_fn=<SubBackward0>)
>     Battery level: tensor([[1.0000, 0.9000, 0.8000, 0.7000, 0.6000, 0.5000, 0.4000, 0.3000, 0.2000,
>              0.1000, 0.0000]], device='cuda:0', grad_fn=<SelectBackward0>)
>     Distance to charger: tensor([[-2.8864, -3.1253, -3.4819, -3.8913, -4.3578, -4.7812, -5.1613, -5.6020,
>              -6.0214, -6.3740, -6.8718]], device='cuda:0', grad_fn=<DivBackward0>)
>     Stand by (distance from charger): tensor([[0.2886, 0.3125, 0.3482, 0.3891, 0.4358, 0.4781, 0.5161, 0.5602, 0.6021,
>              0.6374, 0.6872]], device='cuda:0', grad_fn=<SubBackward0>)
>     Stay > 1 steps: tensor([[-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.]],
>            device='cuda:0', grad_fn=<NegBackward0>)
>     Battery level < limit: tensor([[-9.0000e-01, -8.0000e-01, -7.0000e-01, -6.0000e-01, -5.0000e-01,
>              -4.0000e-01, -3.0000e-01, -2.0000e-01, -1.0000e-01,  7.4506e-08,
>               1.0000e-01]], device='cuda:0', grad_fn=<RsubBackward1>)
>     Distance to charger: tensor([[-2.8864, -3.1253, -3.4819, -3.8913, -4.3578, -4.7812, -5.1613, -5.6020,
>              -6.0214, -6.3740, -6.8718]], device='cuda:0', grad_fn=<DivBackward0>)
>     Battery level > limit: tensor([[ 9.0000e-01,  8.0000e-01,  7.0000e-01,  6.0000e-01,  5.0000e-01,
>               4.0000e-01,  3.0000e-01,  2.0000e-01,  1.0000e-01, -7.4506e-08,
>              -1.0000e-01]], device='cuda:0', grad_fn=<SubBackward0>)
>     Distance to destination: tensor([[-2.5858, -2.1320, -1.6630, -1.2241, -0.8193, -0.4360, -0.0295,  0.1106,
>               0.0547, -0.1086, -0.5540]], device='cuda:0', grad_fn=<DivBackward0>)
> ```

### Config

- self.safe_distance = 0.12 # Inferred from ros experiments in unity
- self.enough_close_to = 0.05
- max_range = 5.0 # Simulate a similar lidar wrt unity env
