# ⭐ *Large Model and Reinforcement Learning Integration for Autonomous Robotic Navigation*



## 📆✅Update
| Date       | Updates                                                                                                                         | Address |
|------------|---------------------------------------------------------------------------------------------------------------------------------|---------|
| 2023-12-01 | Implemented DQN, DDPG, SAC, TD3.                                                                                                | LMRL-RL |
| 2024-01-12 | Implemented Multidomol-GIL, PPO, and TD3-GRU algorithms.                                                                        | LMRL-RL |
| 2024-01-14 | Added multiple test scenarios, including one with complex pedestrian movements and multiple scenarios with complex obstacles.   | Env     |
| 2024-01-20 | Completed testing for some algorithms in GAZEBO, added to README.                                                               | README  |
| 2024-01-24 | Added ChatGPT interface in ROS, implemented demo controlling a turtle with ChatGPT.                                             | LLM     |
| 2024-01-27 | Added interfaces for basic vision models in ROS, such as SAM, FastSAM, CaptionAnything, and YOLO.                               | VFM     |


## 📆✅Future Plan




## ✨Datasets and Models
| Datasets and Models                            | Links                                              |
|-----------------------------------------------|----------------------------------------------------|
| **Datasets in dark conditions**               | [dark](http://host.robots.ox.ac.uk/pascal/VOC/)   |
|                                               |                                                    |
| **Datasets under dense fog conditions**       | [fog](http://host.robots.ox.ac.uk/pascal/VOC/)    |
|                                               |                                                    |
| **DIOR remote sensing dataset**               | [DIOR](http://host.robots.ox.ac.uk/pascal/VOC/)   |
|                                               |                                                    |
| **DIOR remote sensing dataset with fog**      | [DIOR-FOG](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset/tree/master/Dataset) |


## 👀Problems and Visualization

|![space-1.jpg](https://github.com/Nichaojun/Feature-Adaptive-YOLO/blob/master/picture/0.4.png) | ![space-1.jpg](https://github.com/Nichaojun/Feature-Adaptive-YOLO/blob/master/picture/6.4.png)                                                                                                                                                                                                |
|:----------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| *Failure detection examples from the YOLOV5 model. (a) Clean images. (b) The same image under adverse weather conditions. (c) Activations of feature maps from YOLOV5. (d) Detection results, where green boxes indicate correct detections, red boxes indicate false detections, orange boxes indicate missed detections.*                                                           | *Visualization of detection results. (a) Clean images. (b) The same image under adverse weather conditions. (c) Results of IA-YOLO. (d) Results of FA-YOLO, where green boxes indicate correct detections, red boxes indicate false detections, and orange boxes indicate missed detections.* 

## 🛸Framework

|    ![space-1.jpg](https://github.com/Nichaojun/Feature-Adaptive-YOLO/blob/master/picture/1.1.png)      |
|:-----|
|   *Fig. 1. Algorithm framework diagram of FA-YOLO.*    |


## ⏳AFM and DG-Head
| ![space-1.jpg](https://github.com/Nichaojun/Feature-Adaptive-YOLO/blob/master/picture/12.png) |    ![space-1.jpg](https://github.com/Nichaojun/Feature-Adaptive-YOLO/blob/master/picture/3.png)     |
|:----------------------------------------------------------------------------------------------|----------------------------------- |
| *The structure of Adaptive Filters*                                                           |            *The structure of DG-Head.*                   |

## 🔥Result
| ![space-1.jpg](https://github.com/Nichaojun/Feature-Adaptive-YOLO/blob/master/picture/t3.png)       |
|:----------------------------------------------------------------------------------------------------|
| *Performance comparisons with state-of-the-art methods on the Dior\_Foggy and Dior\_Severe\_Foggy.* | *Performance comparisons with state-of-the-art methods on the RTTS*     |
