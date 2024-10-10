import numpy as np
import math

from pynput.keyboard import Listener, Key, KeyCode
from collections import defaultdict
from enum import Enum
import subprocess

import threading
import time

import time

import pickle
import cv2

import airsim
from airsim import Vector3r
import sys

import usb

from base_method import *
from mss import mss

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers
#from skimage.measure import compare_ssim
#import exceptions
driving_agent = "RL"
version = 17
home_dir = "/home/xpsucla/Rahul_FPGA_Project/Minato_drone_project/Drone_Experiments/Eightbyeight_exp/RL/Wind/01212023/Experiment_data/"
# parser = argparse.ArgumentParser()
# parser.add_argument("-f","--filename",type=str,default='Drone_Navigation_airsim.avi',help="Enter destination filename for video file")
# args = parser.parse_args()
# mdir_vidsav = "/home/xpsucla/Rahul_FPGA_Project/Minato_drone_project/External Recording Files/"
fname_external = "External_Video_" + driving_agent + "_v" + str(version) + ".avi"
fname_onsite = "Drone_navigation_obstacle_" + driving_agent + "_v" + str(version) + ".avi"
fname_data = driving_agent + "_test_v" + str(version) + ".pkl"

HIDDEN_UNITS_1 = 64
HIDDEN_UNITS_2 = 64
ACTION_DIM = 9
LEARNING_RATE = 5e-5
GAMMA = 0.95
NUM_EPS = 5000
num_inp = 8
EXP_END = 0
PENALTY = 1000

mdir = '/home/xpsucla/Rahul_FPGA_Project/Minato_drone_project/Drone_Experiments/Eightbyeight_exp/RL/Wind/02152023/Inference/v1_RL_lr_' + str(LEARNING_RATE) + '_Gamma_' + str(GAMMA) + '_itr_'
mdir_dat = '/home/xpsucla/Rahul_FPGA_Project/Minato_drone_project/Drone_Experiments/Eightbyeight_exp/RL/Wind/02152023/Inf Data/v1_RL_lr_' + str(LEARNING_RATE) + '_Gamma_' + str(GAMMA) + '_itr_'
# Connect to the Vehicle (in this case a UDP endpoint)

class Ctrl(Enum):
    (
        QUIT,
        TAKEOFF,
        LANDING,
        MOVE_LEFT,
        MOVE_RIGHT,
        MOVE_FORWARD,
        MOVE_BACKWARD,
        MOVE_UP,
        MOVE_DOWN,
        TURN_LEFT,
        TURN_RIGHT,
        START_EXP,
        END_EXP,
        TAKE_REF,
        FAN1_ON,
        FAN2_ON,
        FAN1_OFF,
        FAN2_OFF
    ) = range(18)


QWERTY_CTRL_KEYS = {
    Ctrl.QUIT: Key.esc,
    Ctrl.TAKEOFF: "t",
    Ctrl.LANDING: "l",
    Ctrl.MOVE_LEFT: "a",
    Ctrl.MOVE_RIGHT: "d",
    Ctrl.MOVE_FORWARD: "w",
    Ctrl.MOVE_BACKWARD: "s",
    Ctrl.MOVE_UP: Key.up,
    Ctrl.MOVE_DOWN: Key.down,
    Ctrl.TURN_LEFT: Key.left,
    Ctrl.TURN_RIGHT: Key.right,
    Ctrl.START_EXP: "o",
    Ctrl.END_EXP: "p",
    Ctrl.TAKE_REF: "r",
    Ctrl.FAN1_OFF: Key.space,
    Ctrl.FAN2_ON: Key.enter,
    Ctrl.FAN1_ON: "g",
    Ctrl.FAN2_OFF: "k"
}

AZERTY_CTRL_KEYS = QWERTY_CTRL_KEYS.copy()
AZERTY_CTRL_KEYS.update(
    {
        Ctrl.MOVE_LEFT: "q",
        Ctrl.MOVE_RIGHT: "d",
        Ctrl.MOVE_FORWARD: "z",
        Ctrl.MOVE_BACKWARD: "s",
    }
)

## KEYBOARD CLASS
class KeyboardCtrl(Listener):
    def __init__(self, ctrl_keys=None):
        self._ctrl_keys = self._get_ctrl_keys(ctrl_keys)
        self._key_pressed = defaultdict(lambda: False)
        self._last_action_ts = defaultdict(lambda: 0.0)
        super().__init__(on_press=self._on_press, on_release=self._on_release)
        self.start()

    def _on_press(self, key):
        if isinstance(key, KeyCode):
            self._key_pressed[key.char] = True
        elif isinstance(key, Key):
            self._key_pressed[key] = True
        if self._key_pressed[self._ctrl_keys[Ctrl.QUIT]]:
            return False
        else:
            return True

    def _on_release(self, key):
        if isinstance(key, KeyCode):
            self._key_pressed[key.char] = False
        elif isinstance(key, Key):
            self._key_pressed[key] = False
        return True

    def quit(self):
        return not self.running or self._key_pressed[self._ctrl_keys[Ctrl.QUIT]]

    def _axis(self, left_key, right_key):
        diff = int(self._key_pressed[right_key]) - int(self._key_pressed[left_key])
        if (diff>0):
            return '01'
        elif (diff<0):
            return '10'
        else:
            return '00'

    def roll(self):
        return self._axis(
            self._ctrl_keys[Ctrl.MOVE_LEFT],
            self._ctrl_keys[Ctrl.MOVE_RIGHT]
        )

    def pitch(self):
        return self._axis(
            self._ctrl_keys[Ctrl.MOVE_BACKWARD],
            self._ctrl_keys[Ctrl.MOVE_FORWARD]
        )

    def yaw(self):
        return self._axis(
            self._ctrl_keys[Ctrl.TURN_LEFT],
            self._ctrl_keys[Ctrl.TURN_RIGHT]
        )

    def throttle(self):
        return self._axis(
            self._ctrl_keys[Ctrl.MOVE_DOWN],
            self._ctrl_keys[Ctrl.MOVE_UP]
        )


    def has_piloting_cmd(self):
        return (
            bool(self.roll())
            or bool(self.pitch())
            or bool(self.yaw())
            or bool(self.throttle())
        )

    def _rate_limit_cmd(self, ctrl, delay):
        now = time.time()
        if self._last_action_ts[ctrl] > (now - delay):
            return str(1)
        elif self._key_pressed[self._ctrl_keys[ctrl]]:
            self._last_action_ts[ctrl] = now
            return str(1)
        else:
            return str(0)

    def takeoff(self):
        return self._rate_limit_cmd(Ctrl.TAKEOFF, 2.0)

    def landing(self):
        return self._rate_limit_cmd(Ctrl.LANDING, 2.0)

    def take_reference(self):
        return self._rate_limit_cmd(Ctrl.TAKE_REF, 2.0)

    def start_fan1(self):
        return self._rate_limit_cmd(Ctrl.FAN1_ON, 1.0)
    
    def stop_fan1(self):
        return self._rate_limit_cmd(Ctrl.FAN1_OFF, 1.0)

    def start_fan2(self):
        return self._rate_limit_cmd(Ctrl.FAN2_ON, 1.0)

    def stop_fan2(self):
        return self._rate_limit_cmd(Ctrl.FAN2_OFF, 1.0)

    def start_experiment(self):
        return self._rate_limit_cmd(Ctrl.START_EXP, 2.0)

    def end_experiment(self):
        return self._rate_limit_cmd(Ctrl.END_EXP, 2.0)

    def _get_ctrl_keys(self, ctrl_keys):
        # Get the default ctrl keys based on the current keyboard layout:
        if ctrl_keys is None:
            ctrl_keys = QWERTY_CTRL_KEYS
            try:
                # Olympe currently only support Linux
                # and the following only works on *nix/X11...
                keyboard_variant = (
                    subprocess.check_output(
                        "setxkbmap -query | grep 'variant:'|"
                        "cut -d ':' -f2 | tr -d ' '",
                        shell=True,
                    )
                    .decode()
                    .strip()
                )
            except subprocess.CalledProcessError:
                pass
            else:
                if keyboard_variant == "azerty":
                    ctrl_keys = AZERTY_CTRL_KEYS
        return ctrl_keys 

#RL Policy Class

class PolicyNet(keras.Model):
    def __init__(self, action_dim= ACTION_DIM):
        super(PolicyNet, self).__init__()
        # self.fc1 = layers.Dense(HIDDEN_UNITS_1, activation="relu", input_shape=(1,num_inp),\
        #     kernel_initializer=initializers.RandomUniform(minval=-0.01, maxval=0.01),bias_initializer=initializers.Zeros())
        self.fc1 = layers.Dense(action_dim, activation="softmax", input_shape=(1,num_inp),\
            kernel_initializer=initializers.RandomUniform(minval=-0.01, maxval=0.01),bias_initializer=initializers.Zeros())
        # self.bn1 = layers.BatchNormalization()
        # self.fc2 = layers.Dense(HIDDEN_UNITS_2, activation="relu",kernel_initializer=initializers.RandomUniform(minval=-0.01, maxval=0.01)\
        #     ,bias_initializer=initializers.Zeros())
        # self.fc3 = layers.Dense(action_dim,activation="softmax"\
        #     ,kernel_initializer=initializers.RandomUniform(minval=-0.01, maxval=0.01),bias_initializer=initializers.Zeros())
    
    def call(self, x):
        x = self.fc1(x)
        # x = self.bn1(x)
        # x = self.fc2(x)
        # x = self.fc3(x)
        return x

    def process(self, observations):
        # Process batch observations using `call(x)`
        # behind-the-scenes
        action_probabilities = self.predict_on_batch(observations)
        return action_probabilities#np.clip(action_probabilities,1e-7,1-1e-7)

class Agent(object):
    def __init__(self, action_dim=ACTION_DIM):
        """Agent with a neural-network brain powered
        policy
        Args:
        action_dim (int): Action dimension
        """
        self.policy_net = PolicyNet(action_dim=action_dim)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, epsilon=1e-8)
        self.gamma = GAMMA

    def policy(self, observation):
        observation = observation.reshape(1, num_inp)
        observation = tf.convert_to_tensor(observation,dtype=tf.float32)
        # print(observation)
        action_logits = self.policy_net(observation)
        # print('Action: ',action_logits)
        action = tf.random.categorical(tf.math.log(action_logits), num_samples=1)
        # print('Action: ',action_logits)
        return action

    def get_action(self, observation):
        action = self.policy(observation).numpy()
        # print(action)
        return action.squeeze()

    def learn(self, states, rewards, actions):
        discounted_reward = 0
        discounted_rewards = []
        # print(rewards)
        rewards.reverse()
        #print(self.policy_net.trainable_variables)
        for r in rewards:
            discounted_reward = r + self.gamma * discounted_reward
            discounted_rewards.append(discounted_reward)
        discounted_rewards.reverse()
        # print(discounted_rewards)
        discounted_rewards = list(np.array(discounted_rewards) - np.mean(np.array(discounted_rewards)))
        # print(discounted_rewards)
        for state, reward, action in zip(states,discounted_rewards, actions):
            with tf.GradientTape() as tape:
                action_probabilities = tf.clip_by_value(self.policy_net(np.array([state]),training=True), clip_value_min=1e-2, clip_value_max=1-1e-2)
                # action_probabilities = self.policy_net(np.array([state]),training=True)
                # print(action_probabilities)
                loss = self.loss(action_probabilities,action, reward)
                grads = tape.gradient(loss,self.policy_net.trainable_variables)
                self.optimizer.apply_gradients(zip(grads,self.policy_net.trainable_variables))
        # print(self.policy_net.trainable_variables)

    def loss(self, action_probabilities, action, reward):
        # log_prob = tf.math.log(action_probabilities(action))
        dist = tfp.distributions.Categorical(probs=action_probabilities, dtype=tf.float32)
        log_prob = dist.log_prob(action)
        # print(tf.math.log(action_probabilities))
        # print(action)
        # print(log_prob)
        loss = -log_prob * reward
        # print(reward)
        # print(loss)
        return loss

def reward_scheme(states):
    # print(np.any(np.abs(states) > 100))
    if (np.sum(states) < 200):
        s = 230/np.sqrt(np.sum(states)+1)
    else:
        s =  0#-1 * np.sum(states)
    # s =  np.sum(states)
    # if(np.any(np.abs(states) > 100) == False):
    #     s =  np.sum(states)
    # else:
    #     s =  np.array(PENALTY)
    # print('Rewards: ',s)
    return s

class DroneTracking(threading.Thread):
    def __init__(self):
        self.w = 1280
        self.h = 720
        self.bounding_box = {'top': 100, 'left': 100, 'width': 1600, 'height': 900}
        # self.fps_screen = 20.0
        self.save_file_ext = home_dir + fname_external
        self.save_file_onsite = home_dir + fname_onsite
        self.Width = 640 #852
        self.Height = 360 #480
        self.CameraFOV = 90
        self.Fx = self.Fy = self.Width / (2 * math.tan(self.CameraFOV * math.pi / 360))
        self.Cx = self.Width / 2
        self.Cy = self.Height / 2
        self.filterSizeX = 20
        self.filterSizeY = 20
        self.strideX = 20
        self.strideY = 20
        self.zper = 50
        self.client = airsim.MultirotorClient()
        self.client2 = airsim.MultirotorClient()
        self.fontFace = cv2.FONT_HERSHEY_SIMPLEX
        self.fontScale = 1.0
        self.thickness = 2
        self.textSize, self.baseline = cv2.getTextSize("FPS", self.fontFace, self.fontScale, self.thickness)
        self.textOrg = (10, 10 + self.textSize[1])
        self.frameCount = 0
        self.fps = 0
        self.stop_processing = False
        self.state = np.zeros(8)
        self.current_pos = np.zeros(4)
        self.state_gains = [10.0,10.0,10.0,350.0]
        self.xtarget = -205.0
        self.ytarget = -95.0
        self.ztarget = 1370.0 #432.0
        self.yawtarget = 0.0
        self.z_l = 0#pos.z_val
        self.local_targets = np.linspace(1250,1370,25)
        print(self.local_targets)
        # self.local_targets = np.zeros(self.z_num)#np.sign(self.ztarget) * np.linspace(self.z_l,np.abs(int(self.ztarget)),self.z_num, endpoint=True)
        # print('Local Checkpoints: ',self.local_targets)
        # self.xtarget_temp = -np.array([180,185,190,193,197,197,197,197,197,199,201,203,203,203,199,199,201,202,203,205,205,205,205,205,205]) #self.xtarget
        # self.ytarget_temp = -np.array([94,95,96,98,100,101,102,101,101,99,98,96,96,95,95,95,95,97,97,98,96,95,95,95,95])#self.ytarget
        self.xtarget_temp = self.xtarget
        self.ytarget_temp = self.ytarget
        self.ztarget_temp = self.local_targets[0]
        self.target_pos = np.array([self.xtarget, self.ytarget, self.local_targets[-1],0])
        self.target_dev = self.target_pos - self.current_pos
        self.z_num = self.local_targets.shape[0]#(np.abs(int(self.ztarget)))//self.z_l
        self.zthres_low = 20.0
        self.zthres_min = 0.5#float('inf')
        self.frame = None
        self.frame_rate_video = 10.0
        self.stop_video = False
        self.start_video = False
        self.start_time = None
        self.start_local_time = None
        self.chp_num = 0
        self.ylim = -100
        self.reached_checkpoint = False
        self.ztol = 1.9#1.5
        self.dist_target = float('inf')
        self.img_target_c = [200,200]
        self.z_obs_dist = float('inf')
        self.zper_obs = 10
        self.reached_target = False

        self.state4_tol = 15
        self.final_checkpoint = False
        self.current_count = 0
        self.total_count = 10
        self.time_thres = np.random.choice(np.arange(50,70,0.5))
        self.process_image = False

        # Set Wind
        self.v_list = [0, 5, 10, 12, 14, 14, 10]#[0, 5, 7.5, 12, 14, 15, 15] #[0, 0, 0, 0, 0, 0, 0]
        # self.v_len = len(self.v_list)

        
        vxw = 0
        vyw = 0
        vzw = 0
        # print('Setting Wind Velocity(vx,vy,vz): ',vxw,'m/s ',vyw,'m/s ',vzw,'m/s')
        self.client2.simSetWind(airsim.Vector3r(vxw,vyw,vzw))
        self.vxyz = np.array([vyw,vzw,vxw])
        
        super().__init__()
        super().start()

    def start(self):
        pass

    def generatepointcloud(self,depth):
        rows, cols = depth.shape
        c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
        z = depth
        x = z * (c - self.Cx) / self.Fx
        y = z * (r - self.Cy) / self.Fy
        zn = (z*np.cos(self.current_pos[3]))+(x*np.sin(self.current_pos[3]))
        xn = (x*np.cos(self.current_pos[3]))-(z*np.sin(self.current_pos[3]))

        return np.dstack((xn, y, zn))
    
    def next_checkpoint(self):
        # if self.start_local_time is not None:
        #     curr_time = time.time()
        #     if ((curr_time-self.start_local_time)>self.time_thres):
        #         wind_ind = self.chp_num // 4
        #         if ((wind_ind==12) or (wind_ind==13) or (wind_ind==14)):
        #             vxw = (np.random.choice(np.arange(800,900))/1000.0) * self.v_list[wind_ind]
        #             vyw = (np.random.choice(np.arange(-900,-800))/1000.0) * self.v_list[wind_ind]
        #             vzw = 0#(np.random.choice(np.append(np.arange(-1000,-400),np.arange(400,1000)))/1000.0) * self.v_list[wind_ind]
        #         else:
        #             vxw = (np.random.choice(np.append(np.arange(-900,-600),np.arange(600,900)))/1000.0) * self.v_list[wind_ind]
        #             vyw = (np.random.choice(np.append(np.arange(-900,-600),np.arange(600,900)))/1000.0) * self.v_list[wind_ind]
        #             vzw = 0#(np.random.choice(np.append(np.arange(-1000,-400),np.arange(400,1000)))/1000.0) * self.v_list[wind_ind]
        #         print(f'Setting Wind Velocity(vx,vy,vz): {vxw}m/s, {vyw}m/s, {vzw}m/s')
        #         self.client2.simSetWind(airsim.Vector3r(vxw,vyw,vzw))
        #         self.vxyz = np.array([vyw,vzw,vxw])
        #         self.time_thres = np.random.choice(np.arange(50,70,0.5))
        #         self.start_local_time = time.time()
        point1 = self.current_pos[:3]
        point2 = np.array([self.xtarget_temp,self.ytarget_temp,self.ztarget_temp])
        self.dist_target = np.linalg.norm(point1-point2)
        if((self.dist_target < self.ztol) and (np.abs(self.state[6]-self.state[7]) < self.state4_tol) and (not self.reached_checkpoint)):
            # self.client.simFlushPersistentMarkers()
            self.start_local_time = time.time()
            if (self.ztarget_temp <= self.local_targets[-2]):
                if (self.ztarget_temp == self.local_targets[self.chp_num]):
                    self.chp_num += 1
                self.reached_checkpoint = True
                time.sleep(0.1)
                print('Next checkpoint: ',self.chp_num,'\n')
                # if (self.chp_num % 4 == 0):
                #     wind_ind = self.chp_num // 4
                #     if ((wind_ind==12) or (wind_ind==13) or (wind_ind==14)):
                #         vxw = (np.random.choice(np.arange(800,900))/1000.0) * self.v_list[wind_ind]
                #         vyw = (np.random.choice(np.arange(-900,-800))/1000.0) * self.v_list[wind_ind]
                #         vzw = 0#(np.random.choice(np.append(np.arange(-1000,-400),np.arange(400,1000)))/1000.0) * self.v_list[wind_ind]
                #     else:
                #         vxw = (np.random.choice(np.append(np.arange(-900,-600),np.arange(600,900)))/1000.0) * self.v_list[wind_ind]
                #         vyw = (np.random.choice(np.append(np.arange(-900,-600),np.arange(600,900)))/1000.0) * self.v_list[wind_ind]
                #         vzw = 0#(np.random.choice(np.append(np.arange(-1000,-400),np.arange(400,1000)))/1000.0) * self.v_list[wind_ind]
                #     print(f'Setting Wind Velocity(vx,vy,vz): {vxw}m/s, {vyw}m/s, {vzw}m/s')
                #     self.client2.simSetWind(airsim.Vector3r(vxw,vyw,vzw))
                #     self.vxyz = np.array([vyw,vzw,vxw])
            elif not self.final_checkpoint:
                self.reached_checkpoint = True
                time.sleep(0.1)
                self.final_checkpoint = True
            else:
                self.reached_checkpoint = True
                self.reached_target = True
            return True
        else:
            return False
    
    def convolve2d_img(self,img):
        h, w, d = img.shape
        w_out = (w-self.filterSizeX+self.strideX)//self.strideX
        h_out = (h-self.filterSizeY+self.strideY)//self.strideY

        is_occupied = np.zeros((w_out,h_out),dtype=np.int32)
        center_coordinate = np.zeros((w_out,h_out,2))
        target_blocked = False
        target_blocked_coordinate = (None,None)
        #print(w,h,d)
        # target_depth = abs(self.current_pos[2] - self.local_targets[self.chp_num])
        xdev = self.current_pos[0]-self.xtarget
        ydev = self.current_pos[1]-self.ytarget
        xtarget_tmp = self.current_pos[0] - (np.sign(xdev)*min(5,abs(xdev)))
        ytarget_tmp = self.current_pos[1] - (np.sign(ydev)*min(3,abs(ydev)))
        iout = 0
        jout = 0
        i = 0
        min_dist = float('inf')
        img_coordinate = [None,None]
        print(img.shape)
        print(f'Is_occupied Shape: {is_occupied.shape}')
        while ((i+self.filterSizeX) <= w):
            j = 0
            jout = 0
            while ((j+self.filterSizeY) <= h):
                img_crop = img[j:(j+self.filterSizeY),i:(i+self.filterSizeX)]
                img_crop_flatten = img_crop.reshape((-1,3))
                x_c, y_c, z_c = np.percentile(img_crop_flatten,self.zper,axis=0)
                if ((z_c <= self.zthres_low) and (z_c > self.zthres_min)):
                    is_occupied[iout,jout] = 1
                    center_coordinate[iout,jout,:] = (x_c + self.current_pos[0],y_c + self.current_pos[1])
                    min_idx = np.argmin(img_crop_flatten,axis=0)
                    max_idx = np.argmax(img_crop_flatten,axis=0)
                    # xmin, zxmin = img_crop_flatten[min_idx[0]][0], img_crop_flatten[min_idx[0]][2]
                    # xmin = abs(target_depth/zxmin)*xmin
                    # ymin, zymin = img_crop_flatten[min_idx[1]][1], img_crop_flatten[min_idx[1]][2]
                    # ymin = abs(target_depth/zymin)*ymin
                    # xmax, zxmax = img_crop_flatten[max_idx[0]][0], img_crop_flatten[max_idx[0]][2]
                    # xmax = abs(target_depth/zxmax)*xmax
                    # ymax, zymax = img_crop_flatten[max_idx[1]][1], img_crop_flatten[max_idx[1]][2]
                    # ymax = abs(target_depth/zymax)*ymax
                    xmin = img_crop_flatten[min_idx[0]][0]
                    ymin = img_crop_flatten[min_idx[1]][1]
                    xmax = img_crop_flatten[max_idx[0]][0]
                    ymax = img_crop_flatten[max_idx[1]][1]

                    xmin, xmax, ymin, ymax = xmin + self.current_pos[0], xmax + self.current_pos[0], ymin + self.current_pos[1], ymax + self.current_pos[1]

                    if ((xmin <= xtarget_tmp <= xmax) and (ymin <= ytarget_tmp <= ymax)):
                        if target_blocked:
                            dist = ((xtarget_tmp-center_coordinate[iout,jout,0])**2) + ((ytarget_tmp-center_coordinate[iout,jout,1])**2)
                            if dist<min_dist:
                                target_blocked_coordinate = (iout,jout)
                                min_dist = dist
                        else:
                            target_blocked = True
                            target_blocked_coordinate = (iout,jout)
                            min_dist = ((xtarget_tmp-center_coordinate[iout,jout,0])**2) + ((ytarget_tmp-center_coordinate[iout,jout,1])**2)
                        # print('Target Blocked')
   
                j += self.strideY
                jout += 1
            
            i += self.strideX
            iout += 1

        if target_blocked:
            iN,jN = target_blocked_coordinate
            iNW,jNW = target_blocked_coordinate
            iW,jW = target_blocked_coordinate
            iSW,jSW = target_blocked_coordinate
            iS,jS = target_blocked_coordinate
            iSE,jSE = target_blocked_coordinate
            iE,jE = target_blocked_coordinate
            iNE,jNE = target_blocked_coordinate
            displacement_x = 2.0
            displacement_y = 3.0
            nN,nNW,nW,nSW,nS,nSE,nE,nNE = 0,0,0,0,0,0,0,0
            nmax = 2
            nmax_curr = 0

            while ((jN>=0) | ((iNW>=0) & (jNW>=0)) | (iW>=0) | ((iSW>=0) & (jSW<h_out)) | (jS<h_out) | ((iSE<w_out) & (jSE<h_out)) | (iE<w_out) | ((iNE<w_out) & (jNE>=0))):
                
                jN -= 1
                if (jN>=0):
                    if (is_occupied[iN,jN]==0) & (nN==nmax):
                        ytarget_tmp = center_coordinate[iN,(jN+(nmax+1)),1] - displacement_y
                        break
                    elif (is_occupied[iN,jN]==0) & (nN<nmax):
                        if (nN>= nmax_curr):
                            ytarget_tmp = center_coordinate[iN,(jN+(nN+1)),1] - displacement_y
                            nmax_curr += 1 
                        nN += 1
                    else:
                        nN = 0
                
                jS += 1
                if (jS<h_out):
                    if (is_occupied[iS,jS]==0) & (nS==nmax):
                        ytarget_tmp = (center_coordinate[iS,(jS-(nmax+1)),1] + displacement_y)
                        break
                    elif (is_occupied[iS,jS]==0) & (nS<nmax):
                        if (nS>= nmax_curr):
                            ytarget_tmp = (center_coordinate[iS,(jS-(nS+1)),1] + displacement_y)
                            nmax_curr += 1
                        nS += 1
                    else:
                        nS = 0

                iNW -= 1
                jNW -= 1
                if ((iNW>=0) & (jNW>=0)):
                    if (is_occupied[iNW,jNW]==0) & (nNW==nmax):
                        xtarget_tmp, ytarget_tmp = (center_coordinate[(iNW+(nmax+1)),(jNW+(nmax+1)),0] - displacement_x), (center_coordinate[(iNW+(nmax+1)),(jNW+(nmax+1)),1] - displacement_y)
                        break
                    elif (is_occupied[iNW,jNW]==0) & (nNW<nmax):
                        if (nNW>= nmax_curr):
                            xtarget_tmp, ytarget_tmp = (center_coordinate[(iNW+(nNW+1)),(jNW+(nNW+1)),0] - displacement_x), (center_coordinate[(iNW+(nNW+1)),(jNW+(nNW+1)),1] - displacement_y)
                            nmax_curr += 1
                        nNW += 1
                    else:
                        nNW = 0

                iNE += 1
                jNE -= 1
                if ((iNE<w_out) & (jNE>=0)):
                    if (is_occupied[iNE,jNE]==0) & (nNE==nmax):
                        xtarget_tmp, ytarget_tmp = (center_coordinate[(iNE-(nmax+1)),(jNE+(nmax+1)),0] + displacement_x), (center_coordinate[(iNE-(nmax+1)),(jNE+(nmax+1)),1] - displacement_y)
                        break
                    elif (is_occupied[iNE,jNE]==0) & (nNE<nmax):
                        if (nNE>= nmax_curr):
                            xtarget_tmp, ytarget_tmp = (center_coordinate[(iNE-(nNE+1)),(jNE+(nNE+1)),0] + displacement_x), (center_coordinate[(iNE-(nNE+1)),(jNE+(nNE+1)),1] - displacement_y)
                            nmax_curr += 1
                        nNE += 1
                    else:
                        nNE = 0

                iSW -= 1
                jSW += 1
                if ((iSW>=0) & (jSW<h_out)):
                    if (is_occupied[iSW,jSW]==0) & (nSW==nmax):
                        xtarget_tmp, ytarget_tmp = (center_coordinate[(iSW+(nmax+1)),(jSW-(nmax+1)),0] - displacement_x), (center_coordinate[(iSW+(nmax+1)),(jSW-(nmax+1)),1] + displacement_y)
                        break
                    elif (is_occupied[iSW,jSW]==0) & (nSW<nmax):
                        if (nSW>= nmax_curr):
                            xtarget_tmp, ytarget_tmp = (center_coordinate[(iSW+(nSW+1)),(jSW-(nSW+1)),0] - displacement_x), (center_coordinate[(iSW+(nSW+1)),(jSW-(nSW+1)),1] + displacement_y)
                            nmax_curr += 1
                        nSW += 1
                    else:
                        nSW = 0

                iSE += 1
                jSE += 1
                if ((iSE<w_out) & (jSE<h_out)):
                    if (is_occupied[iSE,jSE]==0) & (nSE==nmax):
                        xtarget_tmp, ytarget_tmp = (center_coordinate[(iSE-(nmax+1)),(jSE-(nmax+1)),0] + displacement_x), (center_coordinate[(iSE-(nmax+1)),(jSE-(nmax+1)),1] + displacement_y)
                        break
                    elif (is_occupied[iSE,jSE]==0) & (nSE<nmax):
                        if (nSE>= nmax_curr):
                            xtarget_tmp, ytarget_tmp = (center_coordinate[(iSE-(nSE+1)),(jSE-(nSE+1)),0] + displacement_x), (center_coordinate[(iSE-(nSE+1)),(jSE-(nSE+1)),1] + displacement_y)
                            nmax_curr += 1
                        nSE += 1
                    else:
                        nSE = 0
                
                iW -= 1
                if (iW>=0):
                    if (is_occupied[iW,jW]==0) & (nW==nmax):
                        xtarget_tmp = (center_coordinate[(iW+(nmax+1)),jW,0] - displacement_x)
                        break
                    elif (is_occupied[iW,jW]==0) & (nW<nmax):
                        if (nW>= nmax_curr):
                            xtarget_tmp = (center_coordinate[(iW+(nW+1)),jW,0] - displacement_x)
                            nmax_curr += 1
                        nW += 1
                    else:
                        nW = 0

                iE += 1
                if (iE<w_out):
                    if (is_occupied[iE,jE]==0) & (nE==nmax):
                        xtarget_tmp = (center_coordinate[(iE-(nmax+1)),jE,0] + displacement_x)
                        break
                    elif (is_occupied[iE,jE]==0) & (nE<nmax):
                        if (nE>= nmax_curr):
                            xtarget_tmp = (center_coordinate[(iE-(nE+1)),jE,0] + displacement_x)
                            nmax_curr += 1
                        nE += 1
                    else:
                        nE = 0

        # if entered_if:
        #     print("Entered if loop\n")
        self.xtarget_temp, self.ytarget_temp = xtarget_tmp, ytarget_tmp
        
        
        return img_coordinate
    
    def crash_check(self,img):
        w, h, d = img.shape
        img_center = img[w//2-self.filterSizeY//2:w//2+self.filterSizeY//2,h//2-self.filterSizeX//2:h//2+self.filterSizeX//2]
        img_center_flatten = img_center.reshape((-1,3))
        self.z_obs_dist = np.percentile(img_center_flatten,self.zper_obs,axis=0)[2]

    def get_obs(self):
        drone_state = self.client.getMultirotorState(vehicle_name='Drone1')
        pos = drone_state.kinematics_estimated.position
        orientation_q = drone_state.kinematics_estimated.orientation
        self.current_pos[3] = airsim.utils.to_eularian_angles(orientation_q)[2]

        self.current_pos[0], self.current_pos[1], self.current_pos[2] = pos.y_val, pos.z_val, pos.x_val

        self.state[0] = (np.clip((self.current_pos[0]-self.xtarget_temp)*self.state_gains[0],-100,100)) if self.current_pos[0]-self.xtarget_temp > 0 else 0
        self.state[1] = -(np.clip((self.current_pos[0]-self.xtarget_temp)*self.state_gains[0],-100,100)) if self.current_pos[0]-self.xtarget_temp <= 0 else 0
        self.state[2] = (np.clip((self.current_pos[1]-self.ytarget_temp)*self.state_gains[1],-100,100)) if self.current_pos[1]-self.ytarget_temp > 0 else 0
        self.state[3] = -(np.clip((self.current_pos[1]-self.ytarget_temp)*self.state_gains[1],-100,100)) if self.current_pos[1]-self.ytarget_temp <= 0 else 0
        self.state[4] = (np.clip((self.current_pos[2]-self.ztarget_temp)*self.state_gains[2],-100,100)) if self.current_pos[2]-self.ztarget_temp > 0 else 0
        self.state[5] = -(np.clip((self.current_pos[2]-self.ztarget_temp)*self.state_gains[2],-100,100)) if self.current_pos[2]-self.ztarget_temp <= 0 else 0
        self.state[6] = (np.clip(self.current_pos[3]*self.state_gains[3],-100,100)) if self.current_pos[3] > 0 else 0
        self.state[7] = -(np.clip(self.current_pos[3]*self.state_gains[3],-100,100)) if self.current_pos[3] <= 0 else 0
        self.target_dev = self.target_pos - self.current_pos
        # gps_data = self.client.getGpsData()
        # geo_point = gps_data.gnss.geo_point
        # print(geo_point.latitude, geo_point.longitude, geo_point.altitude)
        # print('Pos(x,y,z): ',self.current_pos[0], self.current_pos[1], self.current_pos[2],'\n')

        '''
        self.state["velocity"] = self.drone_state.kinematics_estimated.linear_velocity

        collision = self.drone.simGetCollisionInfo().has_collided
        self.state["collision"] = collision
        '''

        #print("Pos: ({:.2f},{:.2f},{:.2f},{:.2f})".format(self.current_pos[0],self.current_pos[1],self.current_pos[2],self.current_pos[3]), end="\r")

    def image_processing(self):
        str_time = time.time()
        '''
        rawImages = self.client.simGetImages([airsim.ImageRequest("1", airsim.ImageType.DepthPlanar, False, True)])
        rawImage = rawImages[0]
        '''
        # reach_tar = False
        # reach_chk = False
        # if ((self.reached_checkpoint) and (self.current_count==0)):
        #     self.client.simFlushPersistentMarkers()
        #     self.current_count = 1
        #     # time.sleep(0.5)
        # elif ((self.reached_checkpoint) and (self.current_count<self.total_count)):
        #     self.current_count += 1
        # elif ((self.reached_checkpoint) and (self.current_count==self.total_count)):
        #     reach_chk = True

        image_request = [airsim.ImageRequest("1", airsim.ImageType.DepthPlanar, True, False)]
        rawImages = self.client.simGetImages(image_request,vehicle_name='Drone1')
        rawImage = rawImages[0]
        #Image = rawImages[1]
        #Image = self.client.simGetImage("1", airsim.ImageType.Scene)
        #Image = self.client.simGetImage("1", airsim.ImageType.DepthPlanar)
        if (rawImage is None):
            print("Camera is not returning image, please check airsim for error messages")
            airsim.wait_key("Press any key to exit")
            sys.exit(0)
        else:
            img1d = np.array(rawImage.image_data_float, dtype=np.float)
            img2d = np.reshape(img1d, (rawImage.height, rawImage.width))
            if not self.final_checkpoint:
                img1d = np.array(rawImage.image_data_float, dtype=np.float32)
                #img1d[img1d > 255] = 255
                img2d = np.reshape(img1d, (rawImage.height, rawImage.width))
                # print(f'Average Depth: {np.mean(img2d)}')
                Image3D = self.generatepointcloud(img2d)
                _ = self.convolve2d_img(Image3D)
                zdist = self.local_targets[self.chp_num] - self.current_pos[2]
                if np.abs(zdist)<= 6.0 :
                    self.ztarget_temp = self.local_targets[self.chp_num]
                else:
                    self.ztarget_temp = self.current_pos[2] + (np.sign(zdist)*5.0)
                # self.ztarget_temp = np.random.choice(np.arange(2.0,5.5,0.5)) + self.current_pos[2]
            else:
                self.xtarget_temp, self.ytarget_temp, self.ztarget_temp = self.xtarget, self.ytarget, self.ztarget
                
                # if (self.chp_num < (self.z_num-1)):
                #     self.chp_num += 1
                
                # if (self.chp_num!=(self.z_num-1)):
                # xe, ye, ze = (self.local_targets[self.chp_num]),self.xtarget_temp,(self.ytarget_temp)
                # xs, ys, zs = (self.local_targets[self.chp_num]),self.xtarget_temp,(self.ytarget_temp-0.3)
                # self.client.simPlotArrows(points_start=[Vector3r(xs,ys,zs)], points_end=[Vector3r(xe,ye,ze)], 
                #                             color_rgba=[1.0,0.0,0.0,1.0], arrow_size=100,thickness=7,is_persistent=True)
                
                # self.reached_checkpoint = False
                # self.current_count = 0
            
            # self.client.simPlotStrings(strings=["X"], positions=[Vector3r(self.local_targets[self.chp_num],self.xtarget_temp,self.ytarget_temp)], scale=3, color_rgba=[1.0,0.0,0.0,1.0],duration=0.5)
            #print(Image3D_conv)
            Image3D_2 = self.generatepointcloud(img2d)
            self.crash_check(Image3D_2)
            img_meters = airsim.list_to_2d_float_array(rawImage.image_data_float, rawImage.width, rawImage.height)
            img_meters = img_meters.reshape(rawImage.height, rawImage.width, 1)
            img_pixel = np.interp(img_meters,(0,100),(0,255))
            png = img_pixel.astype('uint8')
            png = cv2.cvtColor(png,cv2.COLOR_GRAY2RGB)
            #png = cv2.adaptiveThreshold(png_numpy, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)

            #img1d[img1d > 255] = 255
            #png = np.reshape(png, (Image.height, Image.width))
            #print(png.shape)
            #png_color = cv2.imdecode(airsim.string_to_uint8_array(Image), cv2.IMREAD_UNCHANGED)[:, :, :3]
            #cv2.putText(png,'FPS ' + str(self.fps),self.textOrg, self.fontFace, self.fontScale,(255,0,255),self.thickness)
            # if img_target_c[0] is not None:
            cx, cy = self.img_target_c[1], self.img_target_c[0]
            text_center = 'State: (' + str(int(self.state[0]-self.state[1])) + ',' + str(int(self.state[2]-self.state[3])) + ',' + str(int(self.state[4]-self.state[5])) + ',' + str(int(self.state[6]-self.state[7])) + ')'
            # text_center = 'State: (' + str(int(self.state[0])) + ',' + str(int(self.state[1])) + ',' + str(int(self.state[2])) + ',' + str(int(self.state[3])) + ')'
            cv2.putText(png,text_center,(10,50), self.fontFace, self.fontScale,(0,255,0),self.thickness)
            text_center = 'Local Target Number: ' + str(self.chp_num+1)
            cv2.putText(png,text_center,(10,100), self.fontFace, self.fontScale,(255,0,0),self.thickness)
            # text_center = 'Local Target: (' + str(int(self.xtarget_temp[self.chp_num])) + ',' + str(int(self.ytarget_temp[self.chp_num])) + ',' + str(int(self.local_targets[self.chp_num])) + ')'
            # cv2.putText(png,text_center,(10,80), self.fontFace, self.fontScale,(255,0,0),self.thickness)
            if self.reached_target:
                text_center = 'Target Reached!'
                cv2.putText(png,text_center,(10,150), self.fontFace, self.fontScale,(0,0,255),self.thickness)

            #png = cv2.circle(png, (cx,cy), 20, (0,0,255), self.thickness)
            #print(f'{cx}, {cy}',end='/r')
            #png = cv2.resize(png, None, fx=4.0, fy=4.0)
            #print(png.shape)
            #print(png.shape)
            #print(png_color.shape)
            self.frame = png #np.concatenate((png,png_color),axis=0)
            cv2.imshow("Depth", self.frame)
            cv2.waitKey(1) 
        
        #print("Convolution Time: {}".format(time.time()-str_time),end='\r')


    def record_video(self):
        print("Recording Started")
        # filename = args.filename

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fourcc2 = cv2.VideoWriter_fourcc(*'XVID')
        bounding_box = {'top': 100, 'left': 100, 'width': 1600, 'height': 900}

        sct = mss()

        frame_width = 1920
        frame_height = 1080
        frame_rate = 10.0
        # PATH_TO_MIDDLE = mdir_vidsav + fname
        out = cv2.VideoWriter(self.save_file_ext, fourcc2, frame_rate,(frame_width, frame_height))

        vout = cv2.VideoWriter(self.save_file_onsite, fourcc, self.frame_rate_video, (self.Width,self.Height))
        # out = cv2.VideoWriter(self.save_dir, fourcc2, self.frame_rate_video,(self.Width,(self.Height)))

        strt_time = time.time()
        while True :
            if self.stop_video:
                break
            if ((self.start_video) and (self.start_time is not None)):
                #print('Entered')
                time_duration = time.time() - strt_time
                if (time_duration >= (1/self.frame_rate_video)):
                    strt_time = time.time()
                    current_time = int((strt_time - self.start_time))
                    #print("Writing Frame")
                    sct_img = sct.grab(bounding_box)
                    img = np.array(sct_img)
                    img = cv2.resize(img,(frame_width,frame_height))
                    frame2 = img
                    frame2 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                    out.write(frame2)
                    # cv2.imshow('screen', img)
                    vout.write(self.frame)
                else:
                    time.sleep(0.0005)
            else:
                time.sleep(0.001)

        # Release everything if job is finished
        vout.release()
        out.release()
        # cv2.destroyAllWindows()

    def run(self):
        # video_thread = threading.Thread(target=self.record_video)
        # video_thread.start()

        #print("Drone Tracking Started")
        startTime = time.time()

        while True:
            self.get_obs()
            if self.process_image:
                self.image_processing()
                self.process_image = False
            # print(self.z_obs_dist)
            self.frameCount = self.frameCount  + 1
            endTime = time.time()
            diff = endTime - startTime
            if (diff > 1):
                self.fps = self.frameCount
                # print(f"Current Position: {self.current_pos[0]}, {self.current_pos[1]}, {self.current_pos[2]}")
                #print("FPS = {}".format(self.fps))
                #print("Frame Size = {}".format(self.frame.shape))
                # print(f'State: ({int(self.state[0]-self.state[1])},{int(self.state[2]-self.state[3])},{int(self.state[4]-self.state[5])},{int(self.state[6]-self.state[7])})')
                self.frameCount = 0
                startTime = endTime

            if self.stop_processing:
                break
        
        # After the loop release the cap object
        # Destroy all the windows
        cv2.destroyAllWindows()
                               
        self.stop_video = True
        time.sleep(3)
        #cv2.destroyAllWindows()

class AirSimDroneEnv(threading.Thread):
    def __init__(self):
        self.drone = airsim.MultirotorClient()
        self.RLagent = Agent(action_dim = ACTION_DIM)
        self.drone.confirmConnection()
        self.pose = self.drone.simGetVehiclePose(vehicle_name='Drone1')
        #print(self.pose)

        # teleport the drone + 10 meters in x-direction
        #pose.position.x_val -= 10
        #pose.position.y_val -= 112

        self.pose.position.x_val = 1245 #np.random.randint(495,high=505)#550#
        self.pose.position.y_val = -176 #np.random.randint(495,high=505)#550#
        self.pose.position.z_val = -100
        self.drone.simSetVehiclePose(pose=self.pose, ignore_collision= True, vehicle_name='Drone1')

        self.pose2 = self.drone.simGetVehiclePose(vehicle_name='Drone2')
        #print(self.pose)

        # teleport the drone + 10 meters in x-direction
        #pose.position.x_val -= 10
        #pose.position.y_val -= 112

        self.pose2.position.x_val = 1364 #np.random.randint(495,high=505)#550#
        self.pose2.position.y_val = -205 #np.random.randint(495,high=505)#550#
        self.pose2.position.z_val = -100
        self.drone.simSetVehiclePose(self.pose2, True,'Drone2')

        # Set PID gains for better stability
        # Position
        kpx = 0.5
        kix = .1
        kdx = 1

        kpy = 0.5
        kiy = .1
        kdy = 1

        kpz = 0.5
        kiz = .1
        kdz = 1

        # Velocity
        # kpx = 0.2
        # kix = 0.2
        # kdx = 0.2

        # kpy = 0.2
        # kiy = 0.2
        # kdy = 0.2

        # kpz = 0.2
        # kiz = 0.2
        # kdz = 0.2
        # print('Position Gains: ',airsim.PositionControllerGains.x_gains)
        # print('Velocity Gains: ',airsim.VelocityControllerGains())
        # self.drone.setPositionControllerGains(position_gains=airsim.PositionControllerGains()) # Reset PID Gains Position
        # self.drone.setVelocityControllerGains(velocity_gains=airsim.VelocityControllerGains())# Reset PID Gains Velocity
        self.drone.setPositionControllerGains(vehicle_name='Drone1', position_gains=airsim.PositionControllerGains(airsim.PIDGains(kpx,kix,kdx)\
            ,airsim.PIDGains(kpy,kiy,kdy),airsim.PIDGains(kpz,kiz,kdz)))
        # self.drone.setVelocityControllerGains(velocity_gains=airsim.VelocityControllerGains(airsim.PIDGains(kpx,kix,kdx)\
        #     ,airsim.PIDGains(kpy,kiy,kdy),airsim.PIDGains(kpz,kiz,kdz)))
        # Set the camera to manual_pose mode
        # self.drone.simSetCameraOrientation("0", airsim.to_quaternion(0, 0, 0), airsim.ManualPoseCameraMsg)

        cam_info = self.drone.simGetCameraInfo("1", 'Drone1')
        print("Camera Pre-Pose: ",airsim.utils.to_eularian_angles(cam_info.pose.orientation))
        cam_info.pose.orientation = airsim.utils.to_quaternion(0.0,0.0,0.0)
        self.drone.simSetCameraPose("1",cam_info.pose,vehicle_name='Drone1')
        time.sleep(5.0)
        new_cam_info = self.drone.simGetCameraInfo("1", 'Drone1')
        print("Camera Post-Pose: ",airsim.utils.to_eularian_angles(new_cam_info.pose.orientation))

        self.drone.enableApiControl(True,'Drone1')
        self.drone.enableApiControl(True, 'Drone2')
        self.control = KeyboardCtrl()
        self.preprocessing = DroneTracking()
        roll_gain = 1
        self.pitch_ang = 0.1
        self.roll_ang = self.pitch_ang*roll_gain
        self.yaw_rate = 5
        self.dalt = 0.5
        self.takeoff = False
        self.exitcode = False
        self.keypress_delay = 0.08
        self.duration = 0.1
        self.thres_slw = 10
        self.thres_sup = 90
        self.exp_time = 30 # in seconds
        self.takeoff_drone_var = 0
        self.landing_drone_var = 0
        self.action = -1
        self.episodes = NUM_EPS
        self.start_exp = 0
        self.states = self.preprocessing.state
        self.stop_fpga = False
        self.action1 = '00'
        self.action2 = '00'
        self.action3 = '00'
        self.action4 = '00'
        self.ch_check = False
        self.dvx = 3
        self.dvy = 3
        self.dvz = 3
        self.yaw = 0.0
        self.dyaw = 2
        self.crashed = False
        self.crash_thres = 0.7
        self.crash_quit = False
        self.reached_target = False
        self.target_quit = False
        self.target_dev = list(self.preprocessing.target_dev)
        self.vxyz = list(self.preprocessing.vxyz)
        self.yaw_range = 10 * (math.pi/180)
        self.episode = 1
        self.v_list1 = [0.0,2.0,4.0,6.0]
        self.v_list2 = [10.0,12.0,14.0]
        self.ground_dist = self.drone.getDistanceSensorData(distance_sensor_name='Distance1', vehicle_name='Drone1').distance

        super().__init__()
        super().start()

    def start(self):
        pass
    
    def takeoff_drone(self):
        if ((self.control.takeoff() == '1') and (self.takeoff is False)):
            #self.drone.armDisarm(True)
            self.drone.takeoffAsync(vehicle_name='Drone1').join()
            time.sleep(0.2)
            #drone_state = self.drone.getMultirotorState(vehicle_name='Drone1')

            self.drone.takeoffAsync(vehicle_name='Drone2').join()
            time.sleep(0.2)
            #drone_state = self.drone.getMultirotorState(vehicle_name='Drone2')
            self.drone.moveToPositionAsync(1364, -205, -75, 3, yaw_mode={'is_rate': False, 'yaw_or_rate': self.yaw}, vehicle_name="Drone2").join()
            #state = drone_state.kinematics_estimated.position
            # self.drone.moveByRollPitchYawrateZAsync(0, 0, 0, state.z_val-5, 3)
            #self.drone.moveToZAsync(z=-np.random.randint(2,high=12), velocity=1).join()
            # self.drone.moveByRollPitchYawrateZAsync(0, 0, 0, state.z_val-np.random.randint(2,high=15), self.duration)
            time.sleep(3)
            self.drone.moveByVelocityAsync(0.0, 0.0, 0.0, 1.0, yaw_mode={'is_rate': False, 'yaw_or_rate': self.yaw},vehicle_name='Drone1')
            time.sleep(2)
            # print('Out of TO func!')
            self.takeoff = True
    
    def takeoff_drone_RL(self):
        if ((self.takeoff_drone_var == 1) and (self.takeoff is False)):
            #self.drone.enableApiControl(True)
            self.drone.armDisarm(True)
            print('\nTaking Off!\n')
            self.drone.takeoffAsync(vehicle_name='Drone1').join()
            time.sleep(0.2)
            #drone_state = self.drone.getMultirotorState(vehicle_name='Drone1')

            self.drone.takeoffAsync(vehicle_name='Drone2').join()
            time.sleep(0.2)
            #drone_state = self.drone.getMultirotorState(vehicle_name='Drone2')
            self.drone.moveToPositionAsync(1364, -205, -75, 3, yaw_mode={'is_rate': False, 'yaw_or_rate': self.yaw}, vehicle_name="Drone2").join()
            #state = drone_state.kinematics_estimated.position
            # self.drone.moveByRollPitchYawrateZAsync(0, 0, 0, state.z_val-5, 3)
            #self.drone.moveToZAsync(z=-np.random.randint(2,high=12), velocity=1).join()
            # self.drone.moveByRollPitchYawrateZAsync(0, 0, 0, state.z_val-np.random.randint(2,high=15), self.duration)
            time.sleep(3)
            self.drone.moveByVelocityAsync(0.0, 0.0, 0.0, 1.0, yaw_mode={'is_rate': False, 'yaw_or_rate': self.yaw},vehicle_name='Drone1')
            time.sleep(5.0)
            # yaw_init = np.random.choice(np.append(np.arange(-492,-170),np.arange(170,492)))/100
            # self.drone.moveByVelocityAsync(0.0, 0.0, 0.0, 1.0, yaw_mode={'is_rate': False, 'yaw_or_rate': yaw_init},vehicle_name='Drone1')
            # time.sleep(5.0)
            # print('Out of TO func!')

    def reset_yaw(self):
        self.drone.moveByVelocityAsync(0.0, 0.0, 0.0, 1.0, yaw_mode={'is_rate': False, 'yaw_or_rate': self.yaw},vehicle_name='Drone1')
        time.sleep(10.0)

    def initialize_yaw(self):
        yaw_init = np.random.choice(np.append(np.arange(-492,-170),np.arange(170,492)))/100
        self.drone.moveByVelocityAsync(0.0, 0.0, 0.0, 1.0, yaw_mode={'is_rate': False, 'yaw_or_rate': yaw_init},vehicle_name='Drone1')
        time.sleep(5.0)

    def reset_camera(self):
        cam_info = self.drone.simGetCameraInfo("1", 'Drone1')
        print("Camera Pre-Pose: ",airsim.utils.to_eularian_angles(cam_info.pose.orientation))
        cam_info.pose.orientation = airsim.utils.to_quaternion(0.0,0.0,0.0)
        camera_pose = airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(0.0, 0.0, 0.0)) 
        self.drone.simSetCameraPose("1",camera_pose,vehicle_name='Drone1')
        time.sleep(5.0)
        new_cam_info = self.drone.simGetCameraInfo("1", 'Drone1')
        print("Camera Post-Pose: ",airsim.utils.to_eularian_angles(new_cam_info.pose.orientation))

    def land_drone(self):   
        if (((self.control.landing() == '1') and (self.takeoff is True)) or (self.control.quit())):
            self.drone.landAsync(vehicle_name='Drone1').join()
            time.sleep(0.2)
            self.takeoff = False
            self.drone.armDisarm(False,vehicle_name='Drone1')
        elif (self.crashed and self.takeoff is True):
            # self.drone.moveByVelocityAsync(0, 0, 0, 0.5, yaw_mode={'is_rate': False, 'yaw_or_rate': self.yaw})
            # time.sleep(0.2)
            self.drone.moveByAngleRatesThrottleAsync(0,0,0,0.1,3,vehicle_name='Drone1')
            time.sleep(3)
            self.drone.landAsync(vehicle_name='Drone1').join()
            time.sleep(0.2)
            self.takeoff = False
            self.drone.armDisarm(False,vehicle_name='Drone1')
            self.preprocessing.start_video = False
            # time.sleep(1)
            self.crash_quit = True
        elif (self.reached_target and self.takeoff):
            time.sleep(10.0)
            self.preprocessing.start_video = False
            # self.drone.landAsync(vehicle_name='Drone1').join()
            # time.sleep(0.2)
            # self.takeoff = False
            self.drone.armDisarm(False,vehicle_name='Drone1')
            # time.sleep(1)
            self.target_quit = True
    
    def land_drone_RL(self):
        print("Landing")
        if (((self.landing_drone_var == 1) and (self.takeoff is True)) or (self.control.quit())):
            self.drone.landAsync(vehicle_name='Drone1').join()
            time.sleep(5.0)
            self.takeoff = False
            self.drone.armDisarm(False,vehicle_name='Drone1')
    
    # def recording_state(self):
    #     # if (self.control.start_experiment() == '1'):
    #     self.crash_check()
    #     self.reached_target = self.preprocessing.reached_target
    #     if ((self.control.start_experiment() == '1') and (self.start_exp == 0)):
    #         # if(np.abs(self.preprocessing.current_pos[2]) < np.abs(self.preprocessing.ztarget)):
    #         #     self.preprocessing.z_l = (self.preprocessing.ztarget - self.preprocessing.current_pos[2])/self.preprocessing.z_num
    #         #     self.preprocessing.local_targets = np.linspace(self.preprocessing.current_pos[2] + self.preprocessing.z_l,\
    #         #             int(self.preprocessing.ztarget), self.preprocessing.z_num, endpoint=True)
    #         #     # self.preprocessing.norms12 = 
    #         # else:
    #         #     self.preprocessing.z_l = (self.preprocessing.current_pos[2] - self.preprocessing.ztarget)/self.preprocessing.z_num
    #         #     self.preprocessing.local_targets = np.flip(np.linspace(int(self.preprocessing.ztarget), \
    #         #         self.preprocessing.current_pos[2] - self.preprocessing.z_l, self.preprocessing.z_num, endpoint=True))
    #         # print('Local Targets: ',self.preprocessing.local_targets)
    #         self.preprocessing.reached_checkpoint = True
    #         time.sleep(0.5)
    #         print('Experiment Started')
    #         self.start_exp = 1
    #         self.preprocessing.start_video = True
    #         self.preprocessing.start_time = time.time()
    #         self.preprocessing.start_local_time = time.time()

    #     elif ((self.control.end_experiment() == '1') and (self.start_exp == 1)):
    #         print("Experiment Ended")
    #         self.start_exp = 0
    #         self.preprocessing.start_video = False
    #     elif (self.reached_target and (self.start_exp == 1)):
    #         print("Experiment Ended")
    #         self.start_exp = 0
    #     elif(self.crashed and self.start_exp == 1):
    #         print('Crashed!')
    #         print("Experiment Ended")
    #         # self.land_drone()
    #         self.start_exp = 0
    #         # self.preprocessing.start_video = False
        
                       

    # def action_update(self,action):
    #     if action==0:
    #         self.action1 = '01'
    #     elif action==1:
    #         self.action1 = '10'
    #     else:
    #         self.action1 = '00'
        
    #     if action==2:
    #         self.action3 = '01'
    #     elif action==3:
    #         self.action3 = '10'
    #     else:
    #         self.action3 = '00'

    #     if action==4:
    #         self.action2 = '01'
    #     elif action==5:
    #         self.action2 = '10'
    #     else:
    #         self.action2 = '00'
        
    #     if action==6:
    #         self.action4 = '01'
    #     elif action==7:.orientation
    #         self.action4 = '10'
    #     else:
    #         self.action4 = '00'
    
    # def crash_check(self):
    #     # print(self.preprocessing.z_obs_dist)
    #     if(self.preprocessing.z_obs_dist <= self.crash_thres and self.takeoff):
    #         self.crashed = True
    #         # print('Crashed!\n')
    
    # def drone_control(self):
    #     # self.crash_check
    #     if(self.crashed is False):
    #         vy = 0.0
    #         if (self.control.roll() == '01'):
    #             vy = self.dvy
    #         elif (self.control.roll() == '10'):
    #             vy = -self.dvy
    #         else:
    #             if self.start_exp==1:
    #                 if (self.action1 == '01'):
    #                     # if self.state[0]<= 90:
    #                     vy = self.dvy
    #                 elif (self.action1 == '10'):
    #                     # if self.state[0] >= -90:
    #                     vy = -self.dvy
                        
    #         vx = 0.0
    #         if (self.control.pitch() == '01'):
    #             vx = self.dvx
    #         elif (self.control.pitch() == '10'):
    #             vx = -self.dvx
    #         else:
    #             if self.start_exp==1:
    #                 if (self.action3 == '01'):
    #                     # if self.state[2]<= 90:
    #                     vx = self.dvx
    #                 elif (self.action3 == '10'):
    #                     # if self.state[2] >= -90:
    #                     vx = -self.dvx

    #         vz = 0.0
    #         if (self.control.throttle() == '01'):
    #             vz = -self.dvz
    #         elif (self.control.throttle() == '10'):
    #             vz = self.dvz
    #         else:
    #             if self.start_exp==1:
    #                 if (self.action2 == '01'):
    #                     # if self.state[1]>= -90:
    #                     vz = -self.dvz
    #                 elif (self.action2 == '10'):
    #                     # if self.state[1] <= 90:
    #                     vz = self.dvz
            
    #         yawrate = 0.0
    #         if (self.control.yaw() == '01'):
    #             # if self.state[3]<= 90:
    #             yawrate = self.yaw_rate
    #         elif (self.control.yaw() == '10'):
    #             # if self.state[3] >= -90:
    #             yawrate = -self.yaw_rate
    #         else:
    #             if self.start_exp==1:
    #                 if (self.action4 == '01'):
    #                     # if self.state[3]<= 90:
    #                     yawrate = self.yaw_rate
    #                 elif (self.action4 == '10'):
    #                     # if self.state[3] >= -90:
    #                     yawrate = -self.yaw_rate

    #         self.drone.moveByVelocityAsync(vx, vy, vz, self.duration, yaw_mode={'is_rate': True, 'yaw_or_rate': yawrate},vehicle_name='Drone1')
    #         self.drone.moveToPositionAsync(1364, -205, -75, 3, yaw_mode={'is_rate': False, 'yaw_or_rate': self.yaw}, vehicle_name="Drone2")
    #         time.sleep(self.keypress_delay)
    
    # def update_state(self):
    #     #self.preprocessing.get_obs()
    #     self.state = self.preprocessing.state
    #     self.target_dev = list(self.preprocessing.target_dev)
    #     self.vxyz = list(self.preprocessing.vxyz)

    
    def reset_drone(self):
        self.drone.armDisarm(False,vehicle_name='Drone1')
        self.drone.armDisarm(False,vehicle_name='Drone2')
        self.drone.reset()
        self.drone.enableApiControl(False,vehicle_name='Drone1')
        self.drone.enableApiControl(False,vehicle_name='Drone2')
        time.sleep(5)
        #self.drone.simSetVehiclePose(self.pose, True)
        #self.drone.enableApiControl(True)
        # cam_info = self.drone.simGetCameraInfo("1")
        # print("Camera Pre-Pose: ",airsim.utils.to_eularian_angles(cam_info.pose.orientation))
        # cam_info.pose.orientation = airsim.utils.to_quaternion(0.0,0.0,0.0)
        # self.drone.simSetCameraPose("0",cam_info.pose)
        # print("Camera Post-Pose: ",airsim.utils.to_eularian_angles(cam_info.pose.orientation))
        #self.drone.armDisarm(False)
        #self.drone.enableApiControl(False)
        time.sleep(2)

    def setup_flight(self):
        self.drone.reset()
        self.drone.confirmConnection()
        self.drone.enableApiControl(True,vehicle_name='Drone1')
        self.drone.enableApiControl(True,vehicle_name='Drone2')

    def setup(self):
        self.setup_flight()

    def check_done(self,states,dt,ch_check,crash):
        # print(np.all(states <= self.thres_s))
        # print("States: ",states,'\n')
        # point1 = self.preprocessing.current_pos[:3]
        # point2 = np.array([crash.y_val, crash.z_val, crash.x_val])
        # crash_dist = np.linalg.norm(point1-point2)
        check = np.all(np.abs(states) <= self.thres_slw)
        check2 = np.any(np.abs(states) > self.thres_sup)
        # print(self.preprocessing.current_pos)
        if(dt >= self.exp_time or ch_check == True or check2 == True):
            if(check2):
                print('Drone Crashed!\n')
            # if(check):
            #     print('Expt Successful!\n')
            # self.drone.simSetWind(airsim.Vector3r(self.vx_list[np.random.randint(0,self.v_len)], self.vy_list[np.random.randint(0,self.v_len)]\
            #     , self.vz_list[np.random.randint(0,self.v_len)]))
            return 1
        else:
            return 0

    def step(self):
        chk_num = self.preprocessing.chp_num
        zcurr_tar = self.preprocessing.local_targets[chk_num]
        vy = 0.0
        if (self.action == 0):
            if (self.preprocessing.current_pos[0]< -150.0):
                vy = self.dvy
        elif (self.action == 1):
            if (self.preprocessing.current_pos[0]> -235.0):
                vy = -self.dvy
                       
        vx = 0.0
        if (self.action == 2):
            if (self.preprocessing.current_pos[2]< (zcurr_tar+15.0)):
                vx = self.dvx
        elif (self.action == 3):
            if (self.preprocessing.current_pos[2]> (zcurr_tar-15.0)):
                vx = -self.dvx

        vz = 0.0
        if (self.action == 4):
            if (self.preprocessing.current_pos[1] > -105.0):
                vz = -self.dvz
        elif (self.action == 5):
            if (np.abs(self.ground_dist)>1.3):
                vz = self.dvz

        yawrate = 0.0
        if (self.action == 6):
            yawrate = self.yaw_rate
        elif (self.action == 7):
            yawrate = -self.yaw_rate
        
        # if (self.control.yaw() == '01'):
        #     self.yaw -= self.dyaw
        # elif (self.control.yaw() == '10'):
        #     self.yaw += self.dyaw

        # self.drone.moveByVelocityAsync(vx, vy, vz, self.duration, yaw_mode={'is_rate': False, 'yaw_or_rate': self.yaw})
        self.drone.moveByVelocityAsync(vx, vy, vz, self.duration, yaw_mode={'is_rate': True, 'yaw_or_rate': yawrate})

        time.sleep(self.keypress_delay)
       
    def reset_drone(self):
        self.drone.armDisarm(False,vehicle_name='Drone1')
        self.drone.armDisarm(False,vehicle_name='Drone2')
        self.drone.reset()
        self.drone.enableApiControl(False,vehicle_name='Drone1')
        self.drone.enableApiControl(False,vehicle_name='Drone2')
        time.sleep(5)
        self.preprocessing.chp_num = 0
        time.sleep(1)
        # self.pose.position.x_val = np.random.choice(np.append(np.arange(49700,49900),np.arange(50100,50300)))/100.0 #np.random.uniform(496,504)#np.random.randint(496,high=504)#550#
        # self.pose.position.y_val = np.random.choice(np.append(np.arange(49700,49900),np.arange(50100,50300)))/100.0 #np.random.uniform(496,504)#np.random.randint(496,high=504)#550#
        # self.drone.simSetVehiclePose(self.pose, True)
        # self.initial_pos_num = np.random.choice(np.arange(0,8))
        # self.preprocessing.xtarget = self.y_val_array[(self.initial_pos_num + 1)]
        # self.preprocessing.ytarget = self.z_val_array[(self.initial_pos_num + 1)]
        # self.preprocessing.ztarget = self.x_val_array[(self.initial_pos_num + 1)]
        # # self.pose.position.x_val = np.random.choice(np.append(np.arange(49700,49900),np.arange(50100,50300)))/100#np.random.uniform(496,504)#np.random.randint(496,high=504)#550#
        # # self.pose.position.y_val = np.random.choice(np.append(np.arange(49700,49900),np.arange(50100,50300)))/100#np.random.uniform(496,504)#np.random.randint(496,high=504)#550#
        # self.pose.position.x_val = self.x_val_array[self.initial_pos_num]
        # self.pose.position.y_val = self.y_val_array[self.initial_pos_num]
        # self.pose.position.z_val = -100.0

        # self.drone.simSetVehiclePose(self.pose, True,vehicle_name='Drone1')
        # time.sleep(2)
        # self.drone.enableApiControl(True,vehicle_name='Drone1')
        # cam_info = self.drone.simGetCameraInfo("1")
        # print("Camera Pre-Pose: ",airsim.utils.to_eularian_angles(cam_info.pose.orientation))
        # cam_info.pose.orientation = airsim.utils.to_quaternion(0.0,0.0,0.0)
        # self.drone.simSetCameraPose("0",cam_info.pose,vehicle_name='Drone1')
        # print("Camera Post-Pose: ",airsim.utils.to_eularian_angles(cam_info.pose.orientation))
        # #self.drone.armDisarm(False)
        # #self.drone.enableApiControl(False)
        # time.sleep(2)

    def train(self):#agent: Agent, env: AirSimDroneEnv, episodes: int):
        """Train `agent` in `env` for `episodes`
        Args:
        agent (Agent): Agent to train
        episodes (int): Number of episodes to train
        """
        ch_check = False
        flag = 0
        fac = 0
        input_arr = tf.random.uniform((1, num_inp))
        model = self.RLagent.policy_net
        outputs = model(input_arr)
        model._set_inputs(input_arr)

        self.takeoff_drone_var = 1
        self.takeoff_drone_RL()
        
        while self.episode <= self.episodes:
            print(self.preprocessing.current_pos)
            if(self.exitcode == True):
                break
            # if(ch_check == True):
            #     self.episode -= 1
            # model = get_model()
            model = self.RLagent.policy_net
            model.optimizer = self.RLagent.optimizer
            model.save(mdir + str(self.episode),save_format='tf',include_optimizer=True)
            done = False
            # if(ch_check == False):
            #     self.reset_drone()
            #     # print(self.takeoff_drone_var)
            #     self.takeoff_drone_var = 1
            #     self.takeoff_drone_RL()
            # if(flag == 0):
            #     if(np.abs(self.preprocessing.current_pos[2]) < np.abs(self.preprocessing.ztarget)):
            #         fac = 1
            #         self.preprocessing.z_l = (self.preprocessing.ztarget - self.preprocessing.current_pos[2])/self.preprocessing.z_num
            #         self.preprocessing.local_targets = np.linspace(self.preprocessing.current_pos[2] + self.preprocessing.z_l,\
            #              int(self.preprocessing.ztarget), self.preprocessing.z_num, endpoint=True)
            #         # self.preprocessing.norms12 = 
            #     else:
            #         fac = 2
            #         self.preprocessing.z_l = (self.preprocessing.current_pos[2] - self.preprocessing.ztarget)/self.preprocessing.z_num
            #         self.preprocessing.local_targets = np.flip(np.linspace(int(self.preprocessing.ztarget), \
            #             self.preprocessing.current_pos[2] - self.preprocessing.z_l, self.preprocessing.z_num, endpoint=True))
            #     print('Local Targets: ',self.preprocessing.local_targets)
            #     flag = 1     
            self.reset_yaw()

            self.reset_camera()
            self.preprocessing.process_image = True
            time.sleep(5.0)
            print(f'New Target: {self.preprocessing.xtarget_temp},{self.preprocessing.ytarget_temp},{self.preprocessing.ztarget_temp}')
            self.initialize_yaw()

            state_c = self.preprocessing.state
            print(f'State: ({int(state_c[0]-state_c[1])},{int(state_c[2]-state_c[3])},{int(state_c[4]-state_c[5])},{int(state_c[6]-state_c[7])})')
            if self.preprocessing.chp_num < 13:
                wind_num = np.random.choice(np.arange(len(self.v_list1)))
                vxw = (np.random.choice(np.append(np.arange(-900,-600),np.arange(600,900)))/1000.0) * self.v_list1[wind_num]
                vyw = (np.random.choice(np.append(np.arange(-900,-600),np.arange(600,900)))/1000.0) * self.v_list1[wind_num]
                vzw = 0#(np.random.choice(np.append(np.arange(-1000,-400),np.arange(400,1000)))/1000.0) * self.v_list[wind_ind]
            else:
                wind_num = np.random.choice(np.arange(len(self.v_list2)))
                vxw = (np.random.choice(np.append(np.arange(-900,-600),np.arange(600,900)))/1000.0) * self.v_list2[wind_num]
                vyw = (np.random.choice(np.append(np.arange(-900,-600),np.arange(600,900)))/1000.0) * self.v_list2[wind_num]
                vzw = 0#(np.random.choice(np.append(np.arange(-1000,-400),np.arange(400,1000)))/1000.0) * self.v_list[wind_ind]
            print(f'Setting Wind Velocity(vx,vy,vz): {vxw}m/s, {vyw}m/s, {vzw}m/s')
            time.sleep(1.0)
            vxyz = np.array([vyw,vzw,vxw])
            self.drone.simSetWind(airsim.Vector3r(vxw,vyw,vzw))
            self.states = self.preprocessing.state#np.array([100,100,100,100])
            total_reward = 0
            rewards = []
            states = []
            actions = []
            time_array = []
            target_devs = []
            # crash = []
            time.sleep(0.2)
            # env.takeoff_drone_RL(1)
            ts = time.time()
            while not done:
                if (self.control.quit()):
                    print("Quitting Code")
                    self.exitcode = True
                    break
                # print(self.states)
                self.ground_dist = self.drone.getDistanceSensorData(distance_sensor_name='Distance1', vehicle_name='Drone1').distance
                self.action = self.RLagent.get_action(self.states)
                self.step()
                next_state = self.preprocessing.state#np.array([100,100,100,100])
                # ch_check = self.preprocessing.next_checkpoint(fac)
                crash_check_var = self.drone.simGetCollisionInfo()
                # print(crash_check_var.has_collided)
                # crash.append(crash_check_var.has_collided)
                crash = crash_check_var.position
                self.ch_check = self.preprocessing.next_checkpoint()
                reward = reward_scheme(np.array(self.states))
                target_dev = list(self.preprocessing.target_dev)
                # print(time.time() - ts,'\n')
                done  = self.check_done(self.states,time.time() - ts,ch_check,crash)
                rewards.append(reward)
                states.append(list(self.states))
                actions.append(self.action)
                target_devs.append(target_dev)
                # print(self.action)
                time_array.append(time.time() - ts)
                self.states = next_state
                total_reward += reward

                if self.preprocessing.reached_checkpoint :
                    done  = True
                    self.preprocessing.reached_checkpoint = False
                
                if (self.preprocessing.reached_target):
                    print("Target Reached!")
                    self.exitcode = True
                    done = True
    
                if done:
                    vxw = 0
                    vyw = 0
                    vzw = 0
                    # print('Setting Wind Velocity(vx,vy,vz): ',vxw,'m/s ',vyw,'m/s ',vzw,'m/s')
                    self.drone.simSetWind(airsim.Vector3r(vxw,vyw,vzw))

                    time.sleep(5.0)
                    state_c = self.preprocessing.state
                    print(f'State: ({int(state_c[0]-state_c[1])},{int(state_c[2]-state_c[3])},{int(state_c[4]-state_c[5])},{int(state_c[6]-state_c[7])})')
                    # state_array = np.array(states,dtype=np.float64)
                    # rewards_array = np.array(rewards,dtype=np.float64)
                    # actions_array = np.array(actions,dtype=np.int64)
                    # # time_array = (time_array-time_array[0])
                    # time_arr = np.array(time_array,dtype=np.float64)
                    # time_arr = time_arr - time_arr[0]
                    # time_arr_new = np.arange(time_arr[0],time_arr[-1],0.0008)
                    # # rewards_array_new = np.interp(time_arr_new,time_arr,rewards_array)
                    # actions_array_new = np.zeros(time_arr_new.shape)
                    # for tma_idx in range(1,len(time_arr)):
                    #     idx_act = ((time_arr_new>=time_arr[tma_idx-1]) & (time_arr_new<time_arr[tma_idx]))
                    #     actions_array_new[idx_act] = actions_array[tma_idx-1]

                    # states_stack = []
                    # for state_idx in range(8):
                    #     tmp_array = np.interp(time_arr_new,time_arr,state_array[:,state_idx])
                    #     states_stack.append(tmp_array)
                    
                    # state_array_new = np.stack(states_stack,axis=-1)

                    # rewards_array_new = np.zeros(time_arr_new.shape)
                    # for ran_idx in range(time_arr_new.shape[0]):
                    #     rewards_array_new[ran_idx] = reward_scheme(state_array_new[ran_idx])
                    
                    # print(f'State Array Shape: {state_array_new.shape}')
                    learn_start = time.time()

                    self.RLagent.learn(states, rewards, actions)

                    learning_time = time.time() - learn_start

                    print(f'Learning Time: {learning_time} s')
                    checkpoint_num = self.preprocessing.chp_num

                    with open(mdir_dat + str(self.episode) + '.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
                        pickle.dump([states, rewards, actions,time_array,vxyz,learning_time,target_devs,checkpoint_num], f)
                    print("\n")
                    print(f"Episode#:{self.episode} ep_reward:{np.mean(np.array(rewards))} CP: {self.preprocessing.chp_num}/{self.preprocessing.z_num}", end="\n")
            self.episode += 1
            # time.sleep(5)
        
        self.landing_drone_var = 1
        self.land_drone_RL()
        EXP_END = 1
        self.exitcode = True

    def run(self):
        print("Drone is ready to fly")

        time.sleep(2)
        while True:
            time.sleep(0.1)
            if not self.exitcode:
                self.train()

            if self.exitcode:
                self.preprocessing.stop_processing = True
                time.sleep(2)
                break

        # print('Done!')
        self.reset_drone()
        time.sleep(2)

# class FPGAComm():

#     def __init__(self):
#         # Create the olympe.Drone object from its IP address
#         self.mainfunc = AirSimDroneEnv()
#         self.state = self.mainfunc.state
#         self.start_exp = self.mainfunc.start_exp
#         self.check_ch = 0
#         self.mainfunc.start()
#         self.fname = home_dir + fname_data #"FPGA_data_test_v35.pkl"
#         self.save_data = True
#         self.policy_net = PolicyNet(action_dim=ACTION_DIM)
#         learning_rate = 5e-7
#         gamma = 0.9995

#         episode = 750
#         chp_num = 0
#         self.model_dir = '/home/xpsucla/Rahul_FPGA_Project/Minato_drone_project/Drone_Experiments/Eightbyeight_exp/RL/Wind/01212023/Inference/v1_RL_lr_' + str(learning_rate) + '_Gamma_' + str(gamma) + '_itr_' + str(episode) + " CP " + str(chp_num)
#         time.sleep(2)

#     def policy(self, observation):
#         observation = observation.reshape(1, INPUT_DIM)
#         observation = tf.convert_to_tensor(observation,dtype=tf.float32)
#         # print(observation)
#         action_logits = self.policy_net(observation)
#         # print('Action: ',action_logits)
#         action = tf.random.categorical(tf.math.log(action_logits), num_samples=1)
#         # print('Action: ',action_logits)
#         return action
    
#     def get_action(self, observation):
#         action = self.policy(observation).numpy()

#         # print(action)
#         return action.squeeze()
        
#     def load_model(self,model):
#         self.policy_net = model#tf.keras.models.load_model(self.load_file)
    
    
#     def run(self):
#         print("Start!")
#         # find device
#         #usb_dev, usb_ep_in, usb_ep_out = self.find_device()
#         #usb_dev.set_configuration()
        
#         # initial reset usb and fpga
#         #usb_dev.reset()

#         #fpga_data_array = []
#         time_array = []
#         target_dev_array = []
#         vxyz_array = []
#         action_array = []

#         model = tf.keras.models.load_model(self.model_dir)
#         self.load_model(model)

#         # num = 64 * 1
#         str_time = time.time()
#         while not self.mainfunc.stop_fpga:
#             self.mainfunc.update_state()
#             self.state = self.mainfunc.state
#             self.start_exp = self.mainfunc.start_exp
#             #self.check_ch = int(self.mainfunc.ch_check)
#             # np_data1 = np.array([(self.state[0]-self.state[1]),(self.state[2]-self.state[3]), (self.state[4]-self.state[5]),0,self.start_exp],dtype=np.uint8)
#             # np_data2 = np.random.randint(0, high=255, size = num-5, dtype=np.uint8)
#             # np_data = np.concatenate((np_data1,np_data2))
#             # wr_data = list(np_data)
#             # length = len(wr_data)
        
#             # # write data to ddr
#             # opu_dma(wr_data, num, 10, 0, usb_dev, usb_ep_out, usb_ep_in)
        
#             # # start calculation
#             # opu_run([], 0, 0, 3, usb_dev, usb_ep_out, usb_ep_in)

#             # # read data from FPGA
#             # rd_data = []
#             # opu_dma(rd_data, num, 11, 2, usb_dev, usb_ep_out, usb_ep_in)

#             # action1 = '{0:02b}'.format(int(rd_data[0]))
#             # action2 = '{0:02b}'.format(int(rd_data[1]))
#             # action3 = '{0:02b}'.format(int(rd_data[2]))
#             # action4 = '{0:02b}'.format(int(rd_data[3]))
            
#             # action = 6
#             action = self.get_action(self.state)

#             self.mainfunc.action_update(action)

#             '''action3 = rd_data[0]
#             action2 = rd_data[1]
#             action1 = rd_data[2]'''

#             if self.start_exp==1:
#                 #fpga_data_array.append(rd_data)
#                 time_array.append((time.time()-str_time))
#                 target_dev_array.append(self.mainfunc.target_dev)
#                 vxyz_array.append(self.mainfunc.vxyz)
#                 action_array.append(action)
        
#         if self.save_data:
#             with open(self.fname, "wb") as fout:
#                 # default protocol is zero
#                 # -1 gives highest prototcol and smallest data file size
#                 pickle.dump((time_array, target_dev_array, vxyz_array, action), fout, protocol=-1)

if __name__ == "__main__":
    airsim_drone = AirSimDroneEnv()
    # drone_tracking = DroneTracking()
    # Start the fpga communication
    airsim_drone.start()
    time.sleep(1)
    