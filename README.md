# LM-RL

:dizzy: **Large Model and Reinforcement Learning Integration for Autonomous Robotic Navigation**

:wrench: Realized in ROS Gazebo simulator with Ubuntu 20.04, ROS noetic, and Pytorch. 

# Basic Dependency Installation
:one: [ROS Noetic](http://wiki.ros.org/noetic/Installation)

:two: [Gazebo](https://classic.gazebosim.org/tutorials?tut=install_ubuntu)

:three: [Pytorch](https://pytorch.org/get-started/locally/)

# Performance
| Algorithm | Target Point | Avg. Dist | Var. Dist | Avg. Time | Var. Time | Success Rate |
|-----------|--------------|-----------|-----------|-----------|-----------|--------------|
|           | 1st          | -         | -         | -         | -         | -            |
| DDPG      | 2nd          | -         | -         | -         | -         | -            |
|           | 3rd          | -         | -         | -         | -         | -            |
|           | 1st          | -         | -         | -         | -         | -            |
| DQN       | 2nd          | -         | -         | -         | -         | -            |
|           | 3rd          | -         | -         | -         | -         | -            |
|           | 1st          | -         | -         | -         | -         | -            |
| Multimodal| 2nd          | -         | -         | -         | -         | -            |
|           | 3rd          | -         | -         | -         | -         | -            |
|           | 1st          | -         | -         | -         | -         | -            |
| GLI       | 2nd          | -         | -         | -         | -         | -            |
|           | 3rd          | -         | -         | -         | -         | -            |
|           | 1st          | -         | -         | -         | -         | -            |
| PPO       | 2nd          | -         | -         | -         | -         | -            |
|           | 3rd          | -         | -         | -         | -         | -            |
|           | 1st          | -         | -         | -         | -         | -            |
| SAC       | 2nd          | -         | -         | -         | -         | -            |
|           | 3rd          | -         | -         | -         | -         | -            |
|           | 1st          | -         | -         | -         | -         | -            |
| TD3       | 2nd          | -         | -         | -         | -         | -            |
|           | 3rd          | -         | -         | -         | -         | -            |
|           | 1st          | -         | -         | -         | -         | -            |
| TD3-GRU   | 2nd          | -         | -         | -         | -         | -            |
|           | 3rd          | -         | -         | -         | -         | -            |

# User Guidance
## Create a new Virtual environment (conda is suggested).
Specify your own name for the virtual environment, e.g., gtrl:
```
conda create -n gtrl python=3.7
```
## Activate virtual environment.
```
conda activate gtrl
```
## Install Dependencies.
```
pip install numpy tqdm natsort cpprb matplotlib einops squaternion opencv-python rospkg rosnumpy yaml
sudo apt install python3-catkin-tools python3-osrf-pycommon
sudo apt-get install ros-noetic-cv-bridge
```
### Optional step for visualizing real-time plotting (reward curve) with Spyder. 
```
conda install spyder==5.2.2
```
## Clone the repository.
cd to your workspace and clone the repo.
```
git clone https://github.com/OscarHuangWind/DRL-Transformer-SimtoReal-Navigation.git
```

## Source the workspace.
```
source devel/setup.bash
```

# Framework

<p align="center">
<img src="https://github.com/OscarHuangWind/DRL-Transformer-SimtoReal-Navigation/blob/master/Materials/framework_final.png" width="70%">
</p>

# Goal-guided Transformer (GoT)
<p align="center">
<img src="https://github.com/OscarHuangWind/DRL-Transformer-SimtoReal-Navigation/blob/master/Materials/GoalTransformer_final.png" width="80%">
</p>

# Noise-augmented RGB images from fisheye camera
<p align="center">
<img src="https://github.com/OscarHuangWind/DRL-Transformer-SimtoReal-Navigation/blob/master/Materials/fisheye_final.png" width="60%">
</p>

# AGV and lab environment model in simulation and real world.
<p align="center">
  <img src="https://github.com/OscarHuangWind/DRL-Transformer-SimtoReal-Navigation/blob/master/Materials/gazebo_scout.png" height= "150" />
  <img src="https://github.com/OscarHuangWind/DRL-Transformer-SimtoReal-Navigation/blob/master/Materials/gazebo_world.png" height= "150" />
  <img src="https://github.com/OscarHuangWind/DRL-Transformer-SimtoReal-Navigation/blob/master/Materials/AGV.png" height= "150" />
  <img src="https://github.com/OscarHuangWind/DRL-Transformer-SimtoReal-Navigation/blob/master/Materials/Robotics_Research_Centre.png" height= "150" />
</p>

# Sim-to-Real navigaiton experiment in office environment.
<p align="center">
<img src="https://github.com/OscarHuangWind/DRL-Transformer-SimtoReal-Navigation/blob/master/Materials/office_environment.png" width="60%">
</p>
