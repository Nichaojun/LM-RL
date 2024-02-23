# ⭐ *Large Model and Reinforcement Learning Integration for Autonomous Robotic Navigation*
# ⭐ *Realized in ROS Gazebo simulator with Ubuntu 20.04, ROS noetic, and Pytorch.*
##  🎉 Accepted by VCIP 2023 [[IEEE]](https://ieeexplore.ieee.org/document/10402716) 
[Chaojun Ni](https://github.com/Nichaojun), [**Wenhui Jiang**](http://sim.jxufe.edu.cn/down/show-31909.aspx?id=98), Chao Cai, Qishou Zhu, [**Yuming Fang**](http://sim.jxufe.edu.cn/down/show-1226.aspx?id=98)

|                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             ![space-1.jpg](https://github.com/Nichaojun/Feature-Adaptive-YOLO/blob/master/picture/1.1.png)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | 
|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:| 
|                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   *Fig. 1. Algorithm framework diagram of FA-YOLO.*                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| **Abstract:** *Target detection in remote sensing has been one of the most challenging tasks in the past few decades. However, the detection performance in adverse weather conditions still needs to be satisfactory, mainly caused by the low-quality image features and the fuzzy boundary information. This work proposes a novel framework called Feature Adaptive YOLO (FA-YOLO). Specifically, we present a Hierarchical Feature Enhancement Module (HFEM), which adaptively performs feature-level enhancement to tackle the adverse impacts of different weather conditions. Then, we propose an Adaptive receptive Field enhancement Module (AFM) that dynamically adjusts the receptive field of the features and thus can enrich the context information for feature augmentation. In addition, we introduce Deformable Gated Head (DG-Head) which reduces the clutter caused by adverse weather. Experimental results on RTTS and two synthetic datasets demonstrate that our proposed FA-YOLO significantly outperforms other state-of-the-art target detection models.* |

## 📆✅Update

| Date       | Updates                                 | Bug Fixes                                         |
|------------|-----------------------------------------|---------------------------------------------------|
| 2023-03-13 | Reproduced IA-YOLO algorithm, achieving results close to the original paper on RTTS dataset. | Fixed the issue of incorrect image size reading in FA-YOLO, and added a new data augmentation module. |
| 2023-03-21 | Reproduced GDIP-YOLO algorithm.         | Fixed the "HFDIP" parameter, allowing it to control whether to enable the filtering module.         |
| 2023-04-12 | Reproduced Togethernet algorithm.      |                                                   |


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
