#!/usr/bin/env python

# IMPORT
import gym
import rospy
import numpy as np
import time
import random
import sys
import yaml
import math
import datetime
import rospkg
from gym import utils, spaces
from gym.utils import seeding
from gym.envs.registration import register
#from transformations import quaternion_from_euler

# OTHER FILES
import util_env as U
import math_util as UMath
#from environments.gazebo_connection import GazeboConnection
#from environments.controllers_connection import ControllersConnection
#from joint_publisher import JointPub
from joint_array_publisher import JointArrayPub
import logger

# MESSAGES/SERVICES
from std_msgs.msg import String
from std_msgs.msg import Bool
from sensor_msgs.msg import JointState
from gazebo_msgs.msg import ContactsState
from sensor_msgs.msg import Image
#from gazebo_msgs.srv import GetModelState
#from gazebo_msgs.srv import SetModelState
#from gazebo_msgs.srv import GetLinkState
from geometry_msgs.msg import Point, Quaternion, Vector3
#from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point, PoseStamped
import geometry_msgs
from openai_ros.msg import RLExperimentInfo
from moveit_msgs.msg import MoveGroupActionFeedback

#from simulation.msg import VacuumGripperState
#from simulation.srv import VacuumGripperControl
register(
    id='PickbotReach-v0',
    entry_point='aubo_env:PickbotEnv',
    max_episode_steps=110,
)
# DEFINE ENVIRONMENT CLASS
class PickbotEnv(gym.Env):

    def __init__(self, joint_increment=None, sim_time_factor=0.005, random_object=False, random_position=False,
                 use_object_type=False, populate_object=False, env_object_type='free_shapes'):
        """
        initializing all the relevant variables and connections
        :param joint_increment: increment of the joints
        :param running_step: gazebo simulation time factor
        :param random_object: spawn random object in the simulation
        :param random_position: change object position in each reset
        :param use_object_type: assign IDs to objects and used them in the observation space
        :param populate_object: to populate object(s) in the simulation using sdf file
        :param env_object_type: object type for environment, free_shapes for boxes while others are related to use_case
            'door_handle', 'combox', ...
        """
        rospy.init_node('env_node', anonymous=True)

        # Assign Parameters
        self._joint_increment = joint_increment  # joint_increment in rad
        self._random_object = random_object
        self._random_position = random_position
        self._use_object_type = use_object_type
        self._populate_object = populate_object
        #self.final_object_pos = [1.23, 0.81, -1.7, -1.63, -1.61, 0.99]
        #self.final_object_pos = [-0.055, -0.703, 1.576, 2.279, 1.548, 0.0001]
        self.final_object_pos = [1.84, 0.71, -1.01, -1.07, -1.40, -1.66]
        self.distance_threshold = 0.1
        # Assign MsgTypes
        self.joints_state = JointState()
        self.curr_pose = PoseStamped()
        #self.contact_1_state = ContactsState()
        #self.contact_2_state = ContactsState()
        #self.collisions = Bool()
        #self.camera_rgb_state = Image()
        #self.camera_depth_state = Image()
        #self.contact_1_force = Vector3()
        #self.contact_2_force = Vector3()
        #self.gripper_state = VacuumGripperState()
        self.movement_complete = Bool()
        self.movement_complete.data = False
        self.moveit_action_feedback = MoveGroupActionFeedback()
        self.feedback_list = []

        self._list_of_observations = ["distance_gripper_to_object",
                                      "elbow_joint_state",
                                      "shoulder_lift_joint_state",
                                      "shoulder_pan_joint_state",
                                      "wrist_1_joint_state",
                                      "wrist_2_joint_state",
                                      "wrist_3_joint_state",
                                      "object_pos_x",
                                      "object_pos_y",
                                      "object_pos_z",
                                      "min_distance_gripper_to_object"]

        if self._use_object_type:
            self._list_of_observations.append("object_type")

        # Establishes connection with simulator
        """
        1) Gazebo Connection 
        2) Controller Connection
        3) Joint Publisher 
        """
        #self.gazebo = GazeboConnection(sim_time_factor=sim_time_factor)
        #self.controllers_object = ControllersConnection()
        #self.pickbot_joint_pubisher_object = JointPub()
        self.publisher_to_moveit_object = JointArrayPub()
        
        # Define Subscribers as Sensordata
        """
        1) /joint_states
        2) /gripper_contactsensor_1_state
        3) /gripper_contactsensor_2_state
        4) /gz_collisions
        not used so far but available in the environment 
        5) /pickbot/gripper/state
        6) /camera_rgb/image_raw   
        7) /camera_depth/depth/image_raw
        """
        rospy.Subscriber("/joint_states", JointState, self.joints_state_callback)
        rospy.Subscriber("/curr_pose", geometry_msgs.msg.PoseStamped,  self.curr_robot_pose_callback)
  
        #rospy.Subscriber("/gripper_contactsensor_1_state", ContactsState, self.contact_1_callback)
        #rospy.Subscriber("/gripper_contactsensor_2_state", ContactsState, self.contact_2_callback)
        #rospy.Subscriber("/gz_collisions", Bool, self.collision_callback)
        rospy.Subscriber("/pickbot/movement_complete", Bool, self.movement_complete_callback)
        rospy.Subscriber("/move_group/feedback", MoveGroupActionFeedback, self.move_group_action_feedback_callback, queue_size=4)
        # rospy.Subscriber("/pickbot/gripper/state", VacuumGripperState, self.gripper_state_callback)
        # rospy.Subscriber("/camera_rgb/image_raw", Image, self.camera_rgb_callback)
        # rospy.Subscriber("/camera_depth/depth/image_raw", Image, self.camera_depth_callback)

        # Define Action and state Space and Reward Range
        """
        Action Space: Box Space with 6 values.
        
        State Space: Box Space with 12 values. It is a numpy array with shape (12,)
        Reward Range: -infitity to infinity 
        """
        
        if self._joint_increment is None:
            low_action = np.array([
                -(1.0),
                -(1.0),
                -(1.0),
                -(1.0),
                -(1.0),
                -(1.0)])

            high_action = np.array([
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0])
        else: # Use joint_increments as action
            low_action = np.array([
                -self._joint_increment,
                -self._joint_increment,
                -self._joint_increment,
                -self._joint_increment,
                -self._joint_increment,
                -self._joint_increment])

            high_action = np.array([
                self._joint_increment,
                self._joint_increment,
                self._joint_increment,
                self._joint_increment,
                self._joint_increment,
                self._joint_increment])
        self.action_space = spaces.Discrete(12)
        print('increment+++++++++++++++++++++++++++++++')
        print(math.pi - 0.05)
        high = np.array([
            999,
            math.pi,
            math.pi,
            math.pi,
            math.pi,
            math.pi,
            math.pi,
            1,
            1.4,
            1.5,
            999])

        low = np.array([
            0,
            -math.pi,
            -math.pi,
            -math.pi,
            -math.pi,
            -math.pi,
            -math.pi,
            -1,
            0,
            0,
            0])

        if self._use_object_type:
            high = np.append(high, 9)
            low = np.append(low, 0)
            
        self.observation_space = spaces.Box(low, high)
        self.reward_range = (-np.inf, np.inf)

        self._seed()
        self.done_reward = 0

        # set up everything to publish the Episode Number and Episode Reward on a rostopic
        self.episode_num = 0
        self.accumulated_episode_reward = 0
        self.episode_steps = 0
        self.reward_pub = rospy.Publisher('/openai/reward', RLExperimentInfo, queue_size=1)
        self.reward_list = []
        self.episode_list = []
        self.step_list = []

        self.success_2_contacts = 0
        self.success_1_contact = 0


        self.object_name = ''
        self.object_type_str = ''
        self.object_type = 0

        if self._populate_object:
            # populate objects from object list
            self.populate_objects()


        init_joint_pos = [0, 0, 0, 0, 0, 0]
        self.publisher_to_moveit_object.set_joints(init_joint_pos)
        print(self.joints_state.position)
        self.max_distance, _ = U.get_distance_gripper_to_object(self.joints_state.position)
        self.min_distace = 999

    # Callback Functions for Subscribers to make topic values available each time the class is initialized 
    def joints_state_callback(self, msg):
        self.joints_state = msg

    def movement_complete_callback(self, msg):
        self.movement_complete = msg

    def move_group_action_feedback_callback(self, msg):
        self.moveit_action_feedback = msg

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def reset(self):
        """
        Reset The Robot to its initial Position and restart the Controllers 
        1) Publish the initial joint_positions to MoveIt
        2) Busy waiting until the movement is completed by MoveIt
        3) set target_object to random position
        4) Check all Systems work
        5) Create YAML Files for contact forces in order to get the average over 2 contacts
        6) Create YAML Files for collision to make shure to see a collision due to high noise in topic
        7) Get Observations and return current State
        8) Publish Episode Reward and set accumulated reward back to 0 and iterate the Episode Number
        9) Return State
        """
        init_joint_pos = [0, 0, 0, 0, 0, 0]
        self.publisher_to_moveit_object.set_joints(init_joint_pos)

        
        while not self.movement_complete.data:
            pass
        # print(">>>>>>>>>>>>>>>>>>> RESET: Waiting complete")

        start_ros_time = rospy.Time.now()
        while True:
            elapsed_time = rospy.Time.now() - start_ros_time
            if np.isclose(init_joint_pos, self.joints_state.position, rtol=0.0, atol=0.01).all():
                break
            elif elapsed_time > rospy.Duration(2): # time out
                break

        #self.set_target_object(random_object=self._random_object, random_position=self._random_position)
        self._check_all_systems_ready()
       
        observation = self.get_obs()
        #self.object_position = observation[9:12]
        self.object_position = observation[7:10]
        
        self.max_distance, _ = U.get_distance_gripper_to_object(self.joints_state.position)
        self.min_distace = self.max_distance
        state = U.get_state(observation)
        self._update_episode()
        return state

    def step(self, action):
        """
        Given the action selected by the learning algorithm,
        we perform the corresponding movement of the robot
        return: the state of the robot, the corresponding reward for the step and if its done(terminal State)
        1) Read last joint positions by getting the observation before acting
        2) Get the new joint positions according to chosen action (actions here are the joint increments)
        3) Publish the joint_positions to MoveIt, meanwhile busy waiting, until the movement is complete
        4) Get new observation after performing the action
        5) Convert Observations into States
        6) Check if the task is done or crashing happens, calculate done_reward and pause Simulation again
        7) Calculate reward based on Observatin and done_reward
        8) Return State, Reward, Done
        """
        print("############################")
        print("action: {}".format(action))
        
        self._check_all_systems_ready()
        self.movement_complete.data = False

        # 1) Read last joint positions by getting the observation before acting
        old_observation = self.get_obs()

        print("===========================Old observation=====================================")
        print(old_observation)
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        # 2) Get the new joint positions according to chosen action (actions here are the joint increments)
        '''
        if self._joint_increment is None:
            next_action_position = action
        else:
        '''
        next_action_position = self.get_action_to_position(action, old_observation[1:7])

        print(next_action_position)
        # 3) Move to position and wait for moveit to complete the execution
        self.publisher_to_moveit_object.pub_joints_to_moveit(next_action_position)
        # rospy.wait_for_message("/pickbot/movement_complete", Bool)
        while not self.movement_complete.data:
            pass

        start_ros_time = rospy.Time.now()
        while True:
            elapsed_time = rospy.Time.now() - start_ros_time
            if np.isclose(next_action_position, self.joints_state.position, rtol=0.0, atol=0.01).all():
                break
            elif elapsed_time > rospy.Duration(2): # time out
                break
        # time.sleep(s

       
        # 4) Get new observation and update min_distance after performing the action
        new_observation = self.get_obs()

        print("=============================new observation===============================")
        print(new_observation)
        print("=-==============++++++++++++++++++++=============++++++++++++++============")
        if new_observation[0] < self.min_distace:
            self.min_distace = new_observation[0]
        # print("observ: {}".format( np.around(new_observation[1:7], decimals=3)))

        # 5) Convert Observations into state
        state = U.get_state(new_observation)

        # 6) Check if its done, calculate done_reward
        done, done_reward = self.is_done(new_observation)

        # 7) Calculate reward based on Observatin and done_reward and update the accumulated Episode Reward
        #reward = UMath.compute_reward(new_observation, done_reward)
        
        info = {
            "is_success": self._is_success(new_observation[1:7], self.final_object_pos),
        }
        reward = self.compute_reward(new_observation[1:7], self.final_object_pos, info)

        ### TEST ###
        if done:
            joint_pos = self.joints_state.position
            print("Joint in step (done): {}".format(np.around(joint_pos, decimals=3)))
        ### END of TEST ###

        self.accumulated_episode_reward += reward

        self.episode_steps += 1

        return state, reward, done, {}

    def _is_success(self, achieved_goal, desired_goal):
        d = self.goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = self.goal_distance(achieved_goal, goal)
        print("reward_distance: {}".format(-d))
        #for sparse rewards
        #if self.reward_type == "sparse":
         #   return -(d > self.distance_threshold).astype(np.float32)
        #else:
        #for dense rewards
        return -d

    def goal_distance(self, goal_a, goal_b):
        goal_a_array = np.asarray(goal_a) 
        goal_b_array = np.asarray(goal_b)
        assert goal_a_array.shape == goal_b_array.shape
        return np.linalg.norm(goal_a_array - goal_b_array, axis=-1)

    def _check_all_systems_ready(self):
        """
        Checks that all subscribers for sensortopics are working
        1) /joint_states
        2) /gripper_contactsensor_1_state
        3) /gripper_contactsensor_2_state
        7) Collisions
        not used so far
        4) /camera_rgb/image_raw   
        5) /camera_depth/depth/image_raw
        """
        self.check_joint_states()
        rospy.logdebug("ALL SYSTEMS READY")

    def check_joint_states(self):
        joint_states_msg = None
        while joint_states_msg is None and not rospy.is_shutdown():
            try:
                joint_states_msg = rospy.wait_for_message("/joint_states", JointState, timeout=0.1)
                self.joints_state = joint_states_msg
                rospy.logdebug("Current joint_states READY")
            except Exception as e:
                rospy.logdebug("Current joint_states not ready yet, retrying==>" + str(e))
                print("EXCEPTION: Joint States not ready yet, retrying.")
    
  
    def get_action_to_position(self, action, last_position):
        """
        Take the last published joint and increment/decrement one joint acording to action chosen
        :param action: Integer that goes from 0 to 11, because we have 12 actions.
        :return: list with all joint positions acording to chosen action
        """

        distance = U.get_distance_gripper_to_object(self.joints_state.position)
        self._joint_increment_value = 0.18 * distance[0] + 0.01

        joint_states_position = last_position
        action_position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        rospy.logdebug("get_action_to_position>>>" + str(joint_states_position))
        if action == 0:  # Increment joint3_position_controller (elbow joint)
            action_position[0] = joint_states_position[0] + self._joint_increment_value / 2
            action_position[1] = joint_states_position[1]
            action_position[2] = joint_states_position[2]
            action_position[3] = joint_states_position[3]
            action_position[4] = joint_states_position[4]
            action_position[5] = joint_states_position[5]
        elif action == 1:  # Decrement joint3_position_controller (elbow joint)
            action_position[0] = joint_states_position[0] - self._joint_increment_value / 2
            action_position[1] = joint_states_position[1]
            action_position[2] = joint_states_position[2]
            action_position[3] = joint_states_position[3]
            action_position[4] = joint_states_position[4]
            action_position[5] = joint_states_position[5]

        elif action == 2:  # Increment joint2_position_controller (shoulder_lift_joint)
            action_position[0] = joint_states_position[0]
            action_position[1] = joint_states_position[1] + self._joint_increment_value / 2
            action_position[2] = joint_states_position[2]
            action_position[3] = joint_states_position[3]
            action_position[4] = joint_states_position[4]
            action_position[5] = joint_states_position[5]
        elif action == 3:  # Decrement joint2_position_controller (shoulder_lift_joint)
            action_position[0] = joint_states_position[0]
            action_position[1] = joint_states_position[1] - self._joint_increment_value / 2
            action_position[2] = joint_states_position[2]
            action_position[3] = joint_states_position[3]
            action_position[4] = joint_states_position[4]
            action_position[5] = joint_states_position[5]

        elif action == 4:  # Increment joint1_position_controller (shoulder_pan_joint)
            action_position[0] = joint_states_position[0]
            action_position[1] = joint_states_position[1]
            action_position[2] = joint_states_position[2] + self._joint_increment_value / 2
            action_position[3] = joint_states_position[3]
            action_position[4] = joint_states_position[4]
            action_position[5] = joint_states_position[5]
        elif action == 5:  # Decrement joint1_position_controller (shoulder_pan_joint)
            action_position[0] = joint_states_position[0]
            action_position[1] = joint_states_position[1]
            action_position[2] = joint_states_position[2] - self._joint_increment_value / 2
            action_position[3] = joint_states_position[3]
            action_position[4] = joint_states_position[4]
            action_position[5] = joint_states_position[5]

        elif action == 6:  # Increment joint4_position_controller (wrist_1_joint)
            action_position[0] = joint_states_position[0]
            action_position[1] = joint_states_position[1]
            action_position[2] = joint_states_position[2]
            action_position[3] = joint_states_position[3] + self._joint_increment_value
            action_position[4] = joint_states_position[4]
            action_position[5] = joint_states_position[5]
        elif action == 7:  # Decrement joint4_position_controller (wrist_1_joint)
            action_position[0] = joint_states_position[0]
            action_position[1] = joint_states_position[1]
            action_position[2] = joint_states_position[2]
            action_position[3] = joint_states_position[3] - self._joint_increment_value
            action_position[4] = joint_states_position[4]
            action_position[5] = joint_states_position[5]

        elif action == 8:  # Increment joint5_position_controller (wrist_2_joint)
            action_position[0] = joint_states_position[0]
            action_position[1] = joint_states_position[1]
            action_position[2] = joint_states_position[2]
            action_position[3] = joint_states_position[3]
            action_position[4] = joint_states_position[4] + self._joint_increment_value
            action_position[5] = joint_states_position[5]
        elif action == 9:  # Decrement joint5_position_controller (wrist_2_joint)
            action_position[0] = joint_states_position[0]
            action_position[1] = joint_states_position[1]
            action_position[2] = joint_states_position[2]
            action_position[3] = joint_states_position[3]
            action_position[4] = joint_states_position[4] - self._joint_increment_value
            action_position[5] = joint_states_position[5]

        elif action == 10:  # Increment joint6_position_controller (wrist_3_joint)
            action_position[0] = joint_states_position[0]
            action_position[1] = joint_states_position[1]
            action_position[2] = joint_states_position[2]
            action_position[3] = joint_states_position[3]
            action_position[4] = joint_states_position[4]
            action_position[5] = joint_states_position[5] + self._joint_increment_value
        elif action == 11:  # Decrement joint6_position_controller (wrist_3_joint)
            action_position[0] = joint_states_position[0]
            action_position[1] = joint_states_position[1]
            action_position[2] = joint_states_position[2]
            action_position[3] = joint_states_position[3]
            action_position[4] = joint_states_position[4]
            action_position[5] = joint_states_position[5] - self._joint_increment_value
       
        joint_exceeds_limits = False
        for joint_pos in action_position:
            joint_correction = []
            if joint_pos < -(math.pi - 0.1) or joint_pos > (math.pi - 0.1):
                joint_exceeds_limits = True
                print('>>>>>>>>>>>>>>>>>>>> joint exceeds limit <<<<<<<<<<<<<<<<<<<<<<<')
                joint_correction.append(-joint_pos)
            else:
                joint_correction.append(0.0)

        if joint_exceeds_limits:
            print("is_done: Joints: {}".format(np.round(self.joints_state.position, decimals=3)))
            self.publisher_to_moveit_object.pub_joints_to_moveit([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            while not self.movement_complete.data:
                pass
            self.publisher_to_moveit_object.pub_relative_joints_to_moveit(joint_correction)
            while not self.movement_complete.data:
                pass
            print('>>>>>>>>>>>>>>>> joint corrected <<<<<<<<<<<<<<<<<')

        return action_position

    def get_obs(self):
        """
        Returns the state of the robot needed for Algorithm to learn
        The state will be defined by a List (later converted to numpy array) of the:
        1)          Distance from desired point in meters
        2-7)        States of the 6 joints in radiants
        8,9)        Force in contact sensor in Newtons
        10,11,12)   x, y, z Position of object?
        MISSING
        10)     RGBD image 
        
        
        self._list_of_observations = ["distance_gripper_to_object",
                                    "elbow_joint_state",
                                    "shoulder_lift_joint_state",
                                    "shoulder_pan_joint_state",
                                    "wrist_1_joint_state",
                                    "wrist_2_joint_state",
                                    "wrist_3_joint_state",
                                    "contact_1_force",
                                    "contact_2_force",
                                    "object_pos_x",
                                    "object_pos_y",
                                    "object_pos_z",
                                    "object_type", -- if use_object_type set to True
                                    "min_distance_gripper_to_object"]
        :return: observation
        """

        # Get Distance Object to Gripper and Objectposition from Service Call. Needs to be done a second time cause we need the distance and position after the Step execution
        distance_gripper_to_object, position_xyz_object = U.get_distance_gripper_to_object(self.joints_state.position)
        object_pos_x = position_xyz_object[0]
        object_pos_y = position_xyz_object[1]
        object_pos_z = position_xyz_object[2]

        # Get Joints Data out of Subscriber
        joint_states = self.joints_state
        elbow_joint_state = joint_states.position[0]
        shoulder_lift_joint_state = joint_states.position[1]
        shoulder_pan_joint_state = joint_states.position[2]
        wrist_1_joint_state = joint_states.position[3]
        wrist_2_joint_state = joint_states.position[4]
        wrist_3_joint_state = joint_states.position[5]

        for joint in joint_states.position:
            if joint > 2 * math.pi or joint < -2 * math.pi:
                print(joint_states.name)
                print(np.around(joint_states.position, decimals=3))
                sys.exit("Joint exceeds limit")

        # Get Contact Forces out of get_contact_force Functions to be able to take an average over some iterations otherwise chances are high that not both sensors are showing contact the same time
        #contact_1_force = self.get_contact_force_1()
        #contact_2_force = self.get_contact_force_2()

        # Stack all information into Observations List
        observation = []
        for obs_name in self._list_of_observations:
            if obs_name == "distance_gripper_to_object":
                observation.append(distance_gripper_to_object)
            elif obs_name == "elbow_joint_state":
                observation.append(elbow_joint_state)
            elif obs_name == "shoulder_lift_joint_state":
                observation.append(shoulder_lift_joint_state)
            elif obs_name == "shoulder_pan_joint_state":
                observation.append(shoulder_pan_joint_state)
            elif obs_name == "wrist_1_joint_state":
                observation.append(wrist_1_joint_state)
            elif obs_name == "wrist_2_joint_state":
                observation.append(wrist_2_joint_state)
            elif obs_name == "wrist_3_joint_state":
                observation.append(wrist_3_joint_state)
            elif obs_name == "object_pos_x":
                observation.append(object_pos_x)
            elif obs_name == "object_pos_y":
                observation.append(object_pos_y)
            elif obs_name == "object_pos_z":
                observation.append(object_pos_z)
            elif obs_name == "object_type":
                observation.append(self.object_type)
            elif obs_name == "min_distance_gripper_to_object":
                observation.append(self.min_distace)
            else:
                raise NameError('Observation Asked does not exist==' + str(obs_name))

        return observation
  

    def is_done(self, observations):
        """Checks if episode is done based on observations given.
        
        Done when:
        -Successfully reached goal: Contact with both contact sensors and contact is a valid one(Wrist3 or/and Vavuum Gripper with unit_box)
        -Crashing with itself, shelf, base
        -Joints are going into limits set
        """
        ####################################################################
        #                        Plan0: init                               #
        ####################################################################
        # done = False
        # done_reward = 0
        # reward_reached_goal = 2000
        # reward_crashing = -200
        # reward_no_motion_plan = -50
        # reward_joint_range = -150

        ####################################################################################
        # Plan1: Reach a point in 3D space (usually right above the target object)         #
        # Reward only dependent on distance. Nu punishment for crashing or joint_limits    #
        ####################################################################################
        done = False
        done_reward = 0
        reward_reached_goal = 100
        reward_crashing = 0
        reward_no_motion_plan = 0
        reward_joint_range = 0


        # Check if there are invalid collisions
        #invalid_collision = self.get_collisions()

        # print("##################{}: {}".format(self.moveit_action_feedback.header.seq, self.moveit_action_feedback.status.text))
        if self.moveit_action_feedback.status.text == "No motion plan found. No execution attempted." or  \
                self.moveit_action_feedback.status.text == "Solution found but controller failed during execution" or \
                self.moveit_action_feedback.status.text == "Motion plan was found but it seems to be invalid (possibly due to postprocessing).Not executing.":

            print(">>>>>>>>>>>> NO MOTION PLAN!!! <<<<<<<<<<<<<<<")
            done = True
            done_reward = reward_no_motion_plan

        if np.isclose(self.final_object_pos, self.joints_state.position, rtol=0.0, atol=0.01).all():
            done = True
            done_reward = reward_reached_goal
            print("++++++++++++++++++++++++++++++++reached position++++++++++++++++++++++++++++++++++++")

       
        joint_exceeds_limits = False
        for joint_pos in self.joints_state.position:
            joint_correction = []
            if joint_pos < -math.pi or joint_pos > math.pi:
                joint_exceeds_limits = True
                done = True
                done_reward = reward_joint_range
                print('>>>>>>>>>>>>>>>>>>>> joint exceeds limit <<<<<<<<<<<<<<<<<<<<<<<')
                joint_correction.append(-joint_pos)
            else:
                joint_correction.append(0.0)

        if joint_exceeds_limits:
            print("is_done: Joints: {}".format(np.round(self.joints_state.position, decimals=3)))
            self.publisher_to_moveit_object.pub_joints_to_moveit([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            while not self.movement_complete.data:
                pass
            self.publisher_to_moveit_object.pub_relative_joints_to_moveit(joint_correction)
            while not self.movement_complete.data:
                pass
            print('>>>>>>>>>>>>>>>> joint corrected <<<<<<<<<<<<<<<<<')

        return done, done_reward

    def _update_episode(self):
        """
        Publishes the accumulated reward of the episode and 
        increases the episode number by one.
        :return:
        """
        if self.episode_num > 0:
            self._publish_reward_topic(
                self.accumulated_episode_reward,
                self.episode_steps,
                self.episode_num
            )

        self.episode_num += 1
        self.accumulated_episode_reward = 0
        self.episode_steps = 0

    def _publish_reward_topic(self, reward, steps, episode_number=1):
        """
        This function publishes the given reward in the reward topic for
        easy access from ROS infrastructure.
        :param reward:
        :param episode_number:
        :return:
        """
        reward_msg = RLExperimentInfo()
        reward_msg.episode_number = episode_number
        reward_msg.episode_reward = reward
        self.reward_pub.publish(reward_msg)
        self.reward_list.append(reward)
        self.episode_list.append(episode_number)
        self.step_list.append(steps)
        list = str(reward) + ";" + str(episode_number) + ";" + str(steps) + "\n"

        #with open(self.csv_name + '.csv', 'a') as csv:
        #    csv.write(str(list))
