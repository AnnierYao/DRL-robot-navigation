import math
import time
from os import path


import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
# import sensor_msgs.msg as sensor_msg
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

GOAL_REACHED_DIST = 0.3
COLLISION_DIST = 0.35
TIME_DELTA = 0.1


# Check if the random goal position is located on an obstacle and do not accept it if it is
def check_pos(x, y):
    goal_ok = True

    if -3.8 > x > -6.2 and 6.2 > y > 3.8:
        goal_ok = False

    if -1.3 > x > -2.7 and 4.7 > y > -0.2:
        goal_ok = False

    if -0.3 > x > -4.2 and 2.7 > y > 1.3:
        goal_ok = False

    if -0.8 > x > -4.2 and -2.3 > y > -4.2:
        goal_ok = False

    if -1.3 > x > -3.7 and -0.8 > y > -2.7:
        goal_ok = False

    if 4.2 > x > 0.8 and -1.8 > y > -3.2:
        goal_ok = False

    if 4 > x > 2.5 and 0.7 > y > -3.2:
        goal_ok = False

    if 6.2 > x > 3.8 and -3.3 > y > -4.2:
        goal_ok = False

    if 4.2 > x > 1.3 and 3.7 > y > 1.5:
        goal_ok = False

    if -3.0 > x > -7.2 and 0.5 > y > -1.5:
        goal_ok = False

    if x > 4.5 or x < -4.5 or y > 4.5 or y < -4.5:
        goal_ok = False

    return goal_ok


class GazeboEnv:
    """Superclass for all Gazebo environments."""

    def __init__(self, launchfile, environment_dim):
        self.environment_dim = environment_dim
        self.odom_x = 0
        self.odom_y = 0

        self.goal_x = 1
        self.goal_y = 0.0

        self.upper = 9.0
        self.lower = -9.0
        self.velodyne_data = np.ones(self.environment_dim) * 10
        self.odom_state = []

        self.last_odom = None
        self.angle = 0

        self.set_self_state = ModelState()
        self.set_self_state.model_name = "p3dx"
        self.set_self_state.pose.position.x = 0.0
        self.set_self_state.pose.position.y = 0.0
        self.set_self_state.pose.position.z = 0.0
        self.set_self_state.pose.orientation.x = 0.0
        self.set_self_state.pose.orientation.y = 0.0
        self.set_self_state.pose.orientation.z = 0.0
        self.set_self_state.pose.orientation.w = 1.0

        self.gaps = [[-np.pi / 2 - 0.03, -np.pi / 2 + np.pi / self.environment_dim]]
        for m in range(self.environment_dim - 1):
            self.gaps.append(
                [self.gaps[m][1], self.gaps[m][1] + np.pi / self.environment_dim]
            )
        self.gaps[-1][-1] += 0.03

        # start_angle = -np.pi/2,
        # end_angle  =np.pi/2
        # angle_resolution = (end_angle - start_angle) / self.environment_dim
        # gaps = [[start_angle, angle_resolution]]
        # for m in range(self.environment_dim - 1):
        #     gaps.append([gaps[m][1], gaps[m][1] + angle_resolution])
        # gaps[-1][-1] = end_angle

        # port = "11311"
        # subprocess.Popen(["roscore", "-p", port])


        # Launch the simulation with the given launchfile name
        rospy.init_node("gym", anonymous=True)
        # if launchfile.startswith("/"):
        #     fullpath = launchfile
        # else:
        #     fullpath = os.path.join(os.path.dirname(__file__), "assets", launchfile)
        # if not path.exists(fullpath):
        #     raise IOError("File " + fullpath + " does not exist")

        # subprocess.Popen(["roslaunch", "-p", port, fullpath])
        # print("Gazebo launched!")

        # Set up the ROS publishers and subscribers
        self.vel_pub = rospy.Publisher("/p3dx/cmd_vel", Twist, queue_size=1)
        self.set_state = rospy.Publisher(
            "gazebo/set_model_state", ModelState, queue_size=10
        )
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        self.publisher = rospy.Publisher("goal_point", MarkerArray, queue_size=3)
        self.publisher1 = rospy.Publisher("last_new_goal_point", MarkerArray, queue_size=3)
        self.publisher12 = rospy.Publisher("last_old_goal_point", MarkerArray, queue_size=3)
        self.publisher2 = rospy.Publisher("linear_velocity", MarkerArray, queue_size=1)
        self.publisher3 = rospy.Publisher("angular_velocity", MarkerArray, queue_size=1)
        self.traj_publisher = rospy.Publisher("odom_traj", MarkerArray, queue_size=10)
        self.velodyne = rospy.Subscriber(
            "/velodyne_points", PointCloud2, self.velodyne_callback, queue_size=1
        )
        self.odom = rospy.Subscriber(
            "/p3dx/odom", Odometry, self.odom_callback, queue_size=1
        )

    # Read velodyne pointcloud and turn it into distance data, then select the minimum value for each angle
    # range as state representation
    def velodyne_callback(self, v):
        data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
        self.velodyne_data = np.ones(self.environment_dim) * 10
        for i in range(len(data)):
            if data[i][2] > -0.2:
                dot = data[i][0] * 1 + data[i][1] * 0
                mag1 = math.sqrt(math.pow(data[i][0], 2) + math.pow(data[i][1], 2))
                mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
                beta = math.acos(dot / (mag1 * mag2)) * np.sign(data[i][1])
                dist = math.sqrt(data[i][0] ** 2 + data[i][1] ** 2 + data[i][2] ** 2)

                for j in range(len(self.gaps)):
                    if self.gaps[j][0] <= beta < self.gaps[j][1]:
                        self.velodyne_data[j] = min(self.velodyne_data[j], dist)
                        break

    def odom_callback(self, od_data):
        self.last_odom = od_data

    # Perform an action and read a new state
    def step(self, action):
        target = False
        collision = False
        info = {'odom_x': self.odom_x, 'odom_y': self.odom_y, 'angle': self.angle, 'goal_x': self.goal_x, 'goal_y': self.goal_y, 'collision': collision, ' collision_point': [], 'odom_state': self.odom_state, 'min_laser': 10}

        # Publish the robot action
        vel_cmd = Twist()
        vel_cmd.linear.x = action[0]
        vel_cmd.angular.z = action[1]
        self.vel_pub.publish(vel_cmd)
        self.publish_markers(action)

        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        # propagate state for TIME_DELTA seconds
        time.sleep(TIME_DELTA)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            pass
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        # read velodyne laser state
        done, collision, min_laser = self.observe_collision(self.velodyne_data)
        v_state = []
        v_state[:] = self.velodyne_data[:]
        laser_state = [v_state]

        # Calculate robot heading from odometry data
        self.odom_x = self.last_odom.pose.pose.position.x
        self.odom_y = self.last_odom.pose.pose.position.y
        quaternion = Quaternion(
            self.last_odom.pose.pose.orientation.w,
            self.last_odom.pose.pose.orientation.x,
            self.last_odom.pose.pose.orientation.y,
            self.last_odom.pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        self.angle = round(euler[2], 4)
        self.odom_traj.append([self.odom_x, self.odom_y])

        info['odom_x'] = self.odom_x
        info['odom_y'] = self.odom_y
        info['goal_x'] = self.goal_x
        info['goal_y'] = self.goal_y
        info['collision'] = collision
        info['min_laser'] = min_laser
        info["angle"] = self.angle
        if collision:
            info['collision_point'] = [self.odom_x, self.odom_y]

        # Calculate distance to the goal from the robot
        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )

        # Calculate the relative angle between the robots heading and heading toward the goal
        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y
        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))
        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - self.angle
        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        # Detect if the goal has been reached and give a large positive reward
        if distance < GOAL_REACHED_DIST:
            target = True
            done = True

        robot_state = [distance, theta, action[0], action[1]]
        odom_state = [self.odom_x, self.odom_y, min_laser, distance, theta, action[0], action[1]]
        info['odom_state'] = odom_state
        state = np.append(laser_state, robot_state)
        # print(state)
        reward = self.get_reward(target, collision, action, min_laser)
        return state, reward, done, target, info

    def reset(self):

        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_proxy()

        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")

        info = {'odom_x': self.odom_x, 'odom_y': self.odom_y, 'angle':self.angle, 'goal_x': self.goal_x, 'goal_y': self.goal_y, 'collision': False, ' collision_point': [], 'odom_state': self.odom_state, 'min_laser': 10}

        self.angle = np.random.uniform(-np.pi, np.pi)
        quaternion = Quaternion.from_euler(0.0, 0.0, self.angle)
        object_state = self.set_self_state

        x = 0
        y = 0
        # position_ok = False
        # while not position_ok:
        x = np.random.uniform(-9, 9)
        y = np.random.uniform(-9, 9)
        #     position_ok = check_pos(x, y)
        object_state.pose.position.x = x
        object_state.pose.position.y = y
        # object_state.pose.position.z = 0.
        object_state.pose.orientation.x = quaternion.x
        object_state.pose.orientation.y = quaternion.y
        object_state.pose.orientation.z = quaternion.z
        object_state.pose.orientation.w = quaternion.w
        self.set_state.publish(object_state)

        self.odom_x = object_state.pose.position.x
        self.odom_y = object_state.pose.position.y

        # set a random goal in empty space in environment
        self.change_goal()
        # randomly scatter boxes in the environment
        # self.random_box()
        self.publish_markers([0.0, 0.0])

        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        time.sleep(TIME_DELTA)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")
        v_state = []
        v_state[:] = self.velodyne_data[:]
        laser_state = [v_state]

        done, collision, min_laser = self.observe_collision(self.velodyne_data)

        self.odom_traj = []

        info['odom_x'] = self.odom_x
        info['odom_y'] = self.odom_y
        info['goal_x'] = self.goal_x
        info['goal_y'] = self.goal_y
        info['collision'] = collision
        info['min_laser'] = min_laser
        info["angle"] = self.angle

        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )

        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y

        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))

        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - self.angle

        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        robot_state = [distance, theta, 0.0, 0.0]
        odom_state = [self.odom_x, self.odom_y, min_laser, distance, theta, 0.0, 0.0]
        info['odom_state'] = odom_state
        state = np.append(laser_state, robot_state)
        return state, info

    def change_goal(self):
        # Place a new goal and check if its location is not on one of the obstacles
        if self.upper < 10:
            self.upper += 0.004
        if self.lower > -10:
            self.lower -= 0.004

        # goal_ok = False

       #  while not goal_ok:
        self.goal_x = self.odom_x + np.random.uniform(self.upper, self.lower)
        self.goal_y = self.odom_y + np.random.uniform(self.upper, self.lower)
            #goal_ok = check_pos(self.goal_x, self.goal_y)

    # def random_box(self):
    #     # Randomly change the location of the boxes in the environment on each reset to randomize the training
    #     # environment
    #     for i in range(4):
    #         name = "cardboard_box_" + str(i)

    #         x = 0
    #         y = 0
    #         box_ok = False
    #         while not box_ok:
    #             x = np.random.uniform(-6, 6)
    #             y = np.random.uniform(-6, 6)
    #             box_ok = check_pos(x, y)
    #             distance_to_robot = np.linalg.norm([x - self.odom_x, y - self.odom_y])
    #             distance_to_goal = np.linalg.norm([x - self.goal_x, y - self.goal_y])
    #             if distance_to_robot < 1.5 or distance_to_goal < 1.5:
    #                 box_ok = False
    #         box_state = ModelState()
    #         box_state.model_name = name
    #         box_state.pose.position.x = x
    #         box_state.pose.position.y = y
    #         box_state.pose.position.z = 0.0
    #         box_state.pose.orientation.x = 0.0
    #         box_state.pose.orientation.y = 0.0
    #         box_state.pose.orientation.z = 0.0
    #         box_state.pose.orientation.w = 1.0
    #         self.set_state.publish(box_state)

    def publish_markers(self, action):
        # Publish visual data in Rviz
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "odom"
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
        marker.pose.position.x = self.goal_x
        marker.pose.position.y = self.goal_y
        marker.pose.position.z = 0

        markerArray.markers.append(marker)

        self.publisher.publish(markerArray)

        markerArray2 = MarkerArray()
        marker2 = Marker()
        marker2.header.frame_id = "odom"
        marker2.type = marker.CUBE
        marker2.action = marker.ADD
        marker2.scale.x = abs(action[0])
        marker2.scale.y = 0.1
        marker2.scale.z = 0.01
        marker2.color.a = 1.0
        marker2.color.r = 1.0
        marker2.color.g = 0.0
        marker2.color.b = 0.0
        marker2.pose.orientation.w = 1.0
        marker2.pose.position.x = 5
        marker2.pose.position.y = 0
        marker2.pose.position.z = 0

        markerArray2.markers.append(marker2)
        self.publisher2.publish(markerArray2)

        markerArray3 = MarkerArray()
        marker3 = Marker()
        marker3.header.frame_id = "odom"
        marker3.type = marker.CUBE
        marker3.action = marker.ADD
        marker3.scale.x = abs(action[1])
        marker3.scale.y = 0.1
        marker3.scale.z = 0.01
        marker3.color.a = 1.0
        marker3.color.r = 1.0
        marker3.color.g = 0.0
        marker3.color.b = 0.0
        marker3.pose.orientation.w = 1.0
        marker3.pose.position.x = 5
        marker3.pose.position.y = 0.2
        marker3.pose.position.z = 0

        markerArray3.markers.append(marker3)
        self.publisher3.publish(markerArray3)

    @staticmethod
    def observe_collision(laser_data):
        # Detect a collision from laser data
        min_laser = min(laser_data)
        # print('min_laser:', min_laser)
        # print('laser_data:', laser_data)
        if min_laser < COLLISION_DIST:
            return True, True, min_laser
        return False, False, min_laser

    @staticmethod
    def get_reward(target, collision, action, min_laser):
        if target:
            return 100.0
        elif collision:
            return -100.0
        else:
            r3 = lambda x: 1 - x if x < 1 else 0.0
            return action[0] / 2 - abs(action[1]) / 2 - r3(min_laser) / 2
        
    def calculate_reward_and_state(self, state, odom_x, odom_y, angle, new_goal_x, new_goal_y):
        # 获取机器人当前位置
        robot_x = odom_x
        robot_y = odom_y
        robot_theta = angle

        # 计算新目标的距离
        distance = np.linalg.norm([robot_x - new_goal_x, robot_y - new_goal_y])

        # 计算机器人朝向与新目标的相对角度
        skew_x = new_goal_x - robot_x
        skew_y = new_goal_y - robot_y

        # 判断是否已经到达目标位置
        if skew_x == 0 and skew_y == 0:
            # 机器人已经到达目标，角度为0
            theta = 0
        else:
            dot = skew_x * 1 + skew_y * 0
            mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
            mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
            beta = math.acos(dot / (mag1 * mag2))

            if skew_y < 0:
                if skew_x < 0:
                    beta = -beta
                else:
                    beta = 0 - beta

            theta = beta - robot_theta
            if theta > np.pi:
                theta = np.pi - theta
                theta = -np.pi - theta
            if theta < -np.pi:
                theta = -np.pi - theta
                theta = np.pi - theta

        # 计算奖励
        target = False
        collision = min(state[:-4]) < COLLISION_DIST
        min_laser = min(state[:-4])

        if collision:
            reward = -100.0
        elif distance < GOAL_REACHED_DIST:
            target = True
            reward = 100.0
        else:
            r3 = lambda x: 1 - x if x < 1 else 0.0
            reward = state[-2] / 2 - abs(state[-1]) / 2 - r3(min_laser) / 2

        # 更新状态
        robot_state = [distance, theta, state[-2], state[-1]]
        new_state = np.append(state[:-4], robot_state)

        return new_state, reward, target


    def perturb_goal(self, goal_x, goal_y, laser_data, safe_distance=0.5, lidar_range=(-np.pi/2, np.pi/2)):
        """
        Perturb the goal slightly away from obstacles if it’s too close, based on LiDAR readings.
        :param goal: The original goal (x, y) coordinates.
        :param lidar_data: LiDAR readings (list of 20 points).
        :param safe_distance: Minimum safe distance from obstacles.
        :param lidar_range: The angular range of the LiDAR sensor (tuple, e.g., (-π/2, π/2)).
        :return: Perturbed goal coordinates (x, y).
        """
        # Calculate the angle step for each LiDAR point
        num_points = self.environment_dim
        angle_step = (lidar_range[1] - lidar_range[0]) / num_points

        # Find the direction of the closest obstacle
        min_lidar_distance = min(laser_data)
        if min_lidar_distance < safe_distance:
            closest_index = np.argmin(laser_data)
            closest_angle = lidar_range[0] + closest_index * angle_step

            # Calculate the shift vector to move the goal away from the obstacle direction
            shift_vector = np.array([np.cos(closest_angle), np.sin(closest_angle)]) * (safe_distance - min_lidar_distance)
            goal_x = goal_x + shift_vector[0]
            goal_y = goal_y + shift_vector[1]

        return goal_x, goal_y


    def publish_last_new_markers(self, goal_x, goal_y):
        # Publish visual data in Rviz
        markerArray1 = MarkerArray()
        marker1 = Marker()
        marker1.header.frame_id = "odom"
        marker1.type = marker1.CYLINDER
        marker1.action = marker1.ADD
        marker1.scale.x = 0.1
        marker1.scale.y = 0.1
        marker1.scale.z = 0.01
        marker1.color.a = 1.0
        marker1.color.r = 1
        marker1.color.g = 0
        marker1.color.b = 1
        marker1.pose.orientation.w = 1.0
        marker1.pose.position.x = goal_x
        marker1.pose.position.y = goal_y
        marker1.pose.position.z = 0

        markerArray1.markers.append(marker1)

        self.publisher1.publish(markerArray1)

    def publish_last_old_markers(self, goal_x, goal_y):
        # Publish visual data in Rviz
        markerArray12 = MarkerArray()
        marker12 = Marker()
        marker12.header.frame_id = "odom"
        marker12.type = marker12.CYLINDER
        marker12.action = marker12.ADD
        marker12.scale.x = 0.1
        marker12.scale.y = 0.1
        marker12.scale.z = 0.01
        marker12.color.a = 1.0
        marker12.color.r = 0.5
        marker12.color.g = 0.5
        marker12.color.b = 0.5
        marker12.pose.orientation.w = 1.0
        marker12.pose.position.x = goal_x
        marker12.pose.position.y = goal_y
        marker12.pose.position.z = 0

        markerArray12.markers.append(marker12)

        self.publisher12.publish(markerArray12)

    def publish_traj(self, odom_sequence):
        # Publish visual data in Rviz
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.type = marker.LINE_STRIP
        marker.action = marker.ADD
        marker.scale.x = 0.05
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        for i in range(len(odom_sequence)):
            point = Point()
            point.x = odom_sequence[i][0]
            point.y = odom_sequence[i][1]
            point.z = 0
            marker.points.append(point)

        markerArray.markers.append(marker)

        self.traj_publisher.publish(markerArray)
