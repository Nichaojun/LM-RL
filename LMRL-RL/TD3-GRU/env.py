import rospy
import subprocess
from os import path
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from numpy import inf
import numpy as np
import random
import math
from gazebo_msgs.msg import ModelState
from squaternion import Quaternion
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.random.seed(0)

#* 检查生成的目标点是否在障碍物上, 随机生成目标点
def check_pos(x, y):
    goalOK = True
    if -3.8 > x > -6.2 and 6.2 > y > 3.8: goalOK = False
    if -1.3 > x > -2.7 and 4.7 > y > -0.2: goalOK = False
    if -0.3 > x > -4.2 and 2.7 > y > 1.3: goalOK = False
    if -0.8 > x > -4.2 and -2.3 > y > -4.2: goalOK = False
    if -1.3 > x > -3.7 and -0.8 > y > -2.7: goalOK = False
    if 4.2 > x > 0.8 and -1.8 > y > -3.2: goalOK = False
    if 4 > x > 2.5 and 0.7 > y > -3.2: goalOK = False
    if 6.2 > x > 3.8 and -3.3 > y > -4.2: goalOK = False
    if 4.2 > x > 1.3 and 3.7 > y > 1.5: goalOK = False
    if -3.0 > x > -7.2 and 0.5 > y > -1.5: goalOK = False
    if x > 3.5 or x < -3.5 or y > 3.5 or y < -3.5: goalOK = False
    
    return goalOK

#* 处理传入的激光雷达数据
def binning(lower_bound, data, quantity):
    width = round(len(data) / quantity)
    quantity -= 1
    bins = []
    for low in range(lower_bound, lower_bound + quantity * width + 1, width):
        bins.append(min(data[low:low + width]))
    return np.array([bins])

def launchRVIZ(launchfile):
    # 启动launch文件
    port = '11311'
    subprocess.Popen(["roscore", "-p", port])
    print("Roscore launched!")
    rospy.init_node('gym', anonymous=True)
    if launchfile.startswith("/"): fullpath = launchfile
    else: fullpath = os.path.join(os.path.dirname(__file__), "assets", launchfile)
    if not path.exists(fullpath): raise IOError("File " + fullpath + " does not exist")
    subprocess.Popen(["roslaunch", "-p", port, fullpath])
    print("Gazebo launched!")

#! 仿真环境定义
class GazeboEnv:

    
    def __init__(self, launchfile):
        
        self.odomX = 0                                          # 里程计, x, y方向上的位移
        self.odomY = 0
        self.goalX = 1                                          # 目标点坐标
        self.goalY = 0.0
        self.upper = 5.0
        self.lower = -5.0
        self.velodyne_data = np.ones(20) * 10                   # 激光雷达数据 [10]*20的array
        self.set_self_state = ModelState()                      # self.set_self_state实例化可以通过.pose.position/.pose.orientation访问当前状态
        self.set_self_state.model_name = 'r1'                   # 给机器人命名
        self.set_self_state.pose.position.x = 0.                # 位置
        self.set_self_state.pose.position.y = 0.
        self.set_self_state.pose.position.z = 0.
        self.set_self_state.pose.orientation.x = 0.0            # 四元数
        self.set_self_state.pose.orientation.y = 0.0
        self.set_self_state.pose.orientation.z = 0.0
        self.set_self_state.pose.orientation.w = 1.0

        self.last_laser = None                                  # laser data
        self.last_odom = None                                   # odom data
        
        # 当前和目标点的距离
        self.distOld = math.sqrt(math.pow(self.odomX - 
                        self.goalX, 2) + math.pow(self.odomY - self.goalY, 2))
        self.gaps = [[-1.6, -1.57 + 3.14 / 20]]
        for m in range(19): self.gaps.append([self.gaps[m][1], self.gaps[m][1] + 3.14 / 20])
        self.gaps[-1][-1] += 0.03                               # 每一束激光的角度，总共180度范围，20条射线

        launchRVIZ(launchfile)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)
    
        self.publisher = rospy.Publisher('vis_mark_array', MarkerArray, queue_size=10)
        self.vel_pub = rospy.Publisher('/r1/cmd_vel', Twist, queue_size=10)
        self.set_state = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size=10)

        self.velodyne = rospy.Subscriber('/velodyne_points', PointCloud2, self.velodyne_callback, queue_size=10)
        self.laser = rospy.Subscriber('/r1/front_laser/scan', LaserScan, self.laser_callback, queue_size=10)
        self.odom = rospy.Subscriber('/r1/odom', Odometry, self.odom_callback, queue_size=10)

    def laser_callback(self, scan):
        self.last_laser = scan

    def odom_callback(self, od_data):
        self.last_odom = od_data

    def velodyne_callback(self, v):
        data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
        self.velodyne_data = np.ones(20) * 10                                                   # velodyne激光雷达的最大测距为10
        for i in range(len(data)):
            if data[i][2] > -0.2:
                dot = data[i][0] * 1 + data[i][1] * 0                                           # x
                mag1 = math.sqrt(math.pow(data[i][0], 2) + math.pow(data[i][1], 2))             # sqrt(x^2 + y^2)
                mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))                               # 1.0
                beta = math.acos(dot / (mag1 * mag2)) * np.sign(data[i][1])                     # 激光与x轴的夹角(机器人是坐标原点)
                dist = math.sqrt(data[i][0] ** 2 + data[i][1] ** 2 + data[i][2] ** 2)           # sqrt(x^2 + y^2 + z^2)

                for j in range(len(self.gaps)):
                    if self.gaps[j][0] <= beta < self.gaps[j][1]:                               
                        self.velodyne_data[j] = min(self.velodyne_data[j], dist)                # 更新激光雷达距离的list, 现在list内所有的值都是遇到障碍物的值
                        break

    def calculate_observation(self, data):
        min_range = 0.3
        min_laser = 2
        done = False
        col = False

        for i, item in enumerate(data.ranges):          # 遍历所有的激光束
            if min_laser > data.ranges[i]:
                min_laser = data.ranges[i]              # min_laser记录所有激光束中最小值
            if (min_range > data.ranges[i] > 0):        # 如果激光距离小于0.3，表明撞到了
                done = True
                col = True
        return done, col, min_laser
        # return col, min_laser
    #! step
    def step(self, act, timestep):
        
        target = False
        vel_cmd = Twist()
        vel_cmd.linear.x = act[0]
        vel_cmd.angular.z = act[1]
        self.vel_pub.publish(vel_cmd)

        rospy.wait_for_service('/gazebo/unpause_physics')
        try: self.unpause()
        except (rospy.ServiceException) as e: print("/gazebo/unpause_physics service call failed")

        time.sleep(0.1)
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            pass
            self.pause()
        except (rospy.ServiceException) as e: print("/gazebo/pause_physics service call failed")

        data = self.last_laser
        dataOdom = self.last_odom
        laser_state = np.array(data.ranges[:])
        v_state = []
        v_state[:] = self.velodyne_data[:]
        laser_state = [v_state]
        done, col, min_laser = self.calculate_observation(data)

        #* 从里程计数据计算机器人朝向(欧拉角)
        self.odomX = dataOdom.pose.pose.position.x
        self.odomY = dataOdom.pose.pose.position.y
        quaternion = Quaternion(dataOdom.pose.pose.orientation.w, dataOdom.pose.pose.orientation.x, dataOdom.pose.pose.orientation.y, dataOdom.pose.pose.orientation.z)
        euler = quaternion.to_euler(degrees=False)
        angle = round(euler[2], 4)

        #* 机器人和目标点距离
        Dist = math.sqrt(math.pow(self.odomX - self.goalX, 2) + math.pow(self.odomY - self.goalY, 2))
        #* 计算机器人的朝向和到目标点朝向的角度差
        skewX = self.goalX - self.odomX
        skewY = self.goalY - self.odomY
        dot = skewX * 1 + skewY * 0
        mag1 = math.sqrt(math.pow(skewX, 2) + math.pow(skewY, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))
        if skewY < 0:
            if skewX < 0: beta = -beta
            else: beta = 0 - beta
        beta2 = (beta - angle)
        if beta2 > np.pi:
            beta2 = np.pi - beta2
            beta2 = -np.pi - beta2
        if beta2 < -np.pi:
            beta2 = -np.pi - beta2
            beta2 = np.pi - beta2


        #* 奖励计算
        r3 = lambda x: 1 - x if x < 1 else 0.0
        reward = act[0] / 2 - abs(act[1]) / 2 - r3(min(laser_state[0])) / 2
        self.distOld = Dist
        # 如果机器人和目标点很近, 给大奖励
        if Dist < 0.3:
            target = True
            done = True
            self.distOld = math.sqrt(math.pow(self.odomX - self.goalX, 2) + math.pow(self.odomY - self.goalY, 2))
            reward = 80
        # 如果撞墙了
        if col:
            reward = -100
        if timestep == 499:
            reward -= 100
        

        toGoal = [Dist, beta2, act[0], act[1]]

        state = np.append(laser_state, toGoal)

        return state, reward, done, target

    #! reset
    def reset(self):
        
        rospy.wait_for_service('/gazebo/reset_world')
        try: self.reset_proxy() # 初始化仿真地图
        except rospy.ServiceException as e: print("/gazebo/reset_simulation service call failed")
        
        angle = np.random.uniform(-np.pi, np.pi)                                # 角度
        # angle = 0
        quaternion = Quaternion.from_euler(0., 0., angle)                       # 四元数
        object_state = self.set_self_state

        x = 0
        y = 0
        chk = False
        
        while not chk:                                                          # 随机初始化机器人的位置并检查初始化的位置是否合理
            x = np.random.uniform(-4.5, 4.5)
            y = np.random.uniform(-4.5, 4.5)
            chk = check_pos(x, y)

        object_state.pose.position.x = x
        object_state.pose.position.y = y
        # object_state.pose.position.z = 0.
        object_state.pose.orientation.x = quaternion.x
        object_state.pose.orientation.y = quaternion.y
        object_state.pose.orientation.z = quaternion.z
        object_state.pose.orientation.w = quaternion.w
        self.set_state.publish(object_state)                                    # 将初始化的机器人状态(位置, 四元数)进行发布

        self.odomX = object_state.pose.position.x                               # 更新里程计数据
        self.odomY = object_state.pose.position.y

        self.change_goal()                                                      # 随机初始化目标点的位置并检查初始化的位置是否合理
        self.random_box()                                                       # 随机初始化障碍物的位置并检查初始化的位置是否合理

        # Publish visual data in Rviz
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.goalX
        marker.pose.position.y = self.goalY
        marker.pose.position.z = 0
        markerArray.markers.append(marker)
        self.publisher.publish(markerArray)

        self.distOld = math.sqrt(math.pow(self.odomX - self.goalX, 2) + math.pow(self.odomY - self.goalY, 2))       # 与目标点距离

        rospy.wait_for_service('/gazebo/unpause_physics')
        try: self.unpause()
        except (rospy.ServiceException) as e: print("/gazebo/unpause_physics service call failed")

        data = None
        while data is None:
            try: data = rospy.wait_for_message('/r1/front_laser/scan', LaserScan, timeout=0.5)
            except: pass
        laser_state = np.array(data.ranges[:])                                  # 360个激光束
        laser_state[laser_state == inf] = 10
        laser_state = binning(0, laser_state, 20)

        rospy.wait_for_service('/gazebo/pause_physics')
        try: self.pause()
        except (rospy.ServiceException) as e: print("/gazebo/pause_physics service call failed")


        Dist = math.sqrt(math.pow(self.odomX - self.goalX, 2) + math.pow(self.odomY - self.goalY, 2))
        skewX = self.goalX - self.odomX
        skewY = self.goalY - self.odomY
        dot = skewX * 1 + skewY * 0
        mag1 = math.sqrt(math.pow(skewX, 2) + math.pow(skewY, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))

        if skewY < 0:
            if skewX < 0: beta = -beta
            else:beta = 0 - beta
        beta2 = (beta - angle)
        if beta2 > np.pi:
            beta2 = np.pi - beta2
            beta2 = -np.pi - beta2
        if beta2 < -np.pi:
            beta2 = -np.pi - beta2
            beta2 = np.pi - beta2
        toGoal = [Dist, beta2, 0.0, 0.0]

        state = np.append(laser_state, toGoal)
        return state

    # Place a new goal and check if its lov\cation is not on one of the obstacles
    def change_goal(self):
        if self.upper < 10:
            self.upper += 0.004
        if self.lower > -10:
            self.lower -= 0.004
        
        gOK = False

        while not gOK:
            self.goalX = self.odomX + np.random.uniform(self.upper, self.lower)
            self.goalY = self.odomY + np.random.uniform(self.upper, self.lower)
            gOK = check_pos(self.goalX, self.goalY)

    #* 随机carboard_box位置
    def random_box(self):
        for i in range(4):
            name = 'cardboard_box_' + str(i)

            x = 0
            y = 0
            chk = False
            while not chk:
                x = np.random.uniform(-6, 6)
                y = np.random.uniform(-6, 6)
                chk = check_pos(x, y)
                d1 = math.sqrt((x - self.odomX) ** 2 + (y - self.odomY) ** 2)
                d2 = math.sqrt((x - self.goalX) ** 2 + (y - self.goalY) ** 2)
                if d1 < 1.5 or d2 < 1.5:
                    chk = False
            box_state = ModelState()
            box_state.model_name = name
            box_state.pose.position.x = x
            box_state.pose.position.y = y
            box_state.pose.position.z = 0.
            box_state.pose.orientation.x = 0.0
            box_state.pose.orientation.y = 0.0
            box_state.pose.orientation.z = 0.0
            box_state.pose.orientation.w = 1.0
            self.set_state.publish(box_state)
