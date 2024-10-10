import numpy as np
import math

from pynput.keyboard import Listener, Key, KeyCode
from collections import defaultdict
from enum import Enum
import subprocess
import os
import queue

import threading
import traceback
import time

import socket
import time

# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
import pickle

# import argparse
import cv2

import airsim
from airsim import Vector3r
import sys

import usb

from base_method import *
from mss import mss
# from PIL import Image
#from skimage.measure import compare_ssim
#import exceptions
driving_agent = 'Human'
version = 3
sub_folder = '12222022'
if driving_agent=='Human':
    driver = 'test'
    home_dir = "/home/xpsucla/Rahul_FPGA_Project/Minato_drone_project/Drone_Experiments/Eightbyeight_exp/" + driving_agent + "/" + driver + "/"
else:
    home_dir = "/home/xpsucla/Rahul_FPGA_Project/Minato_drone_project/Drone_Experiments/Eightbyeight_exp/" + driving_agent + "/" + sub_folder + "/"
# parser = argparse.ArgumentParser()
# parser.add_argument("-f","--filename",type=str,default='Drone_Navigation_airsim.avi',help="Enter destination filename for video file")
# args = parser.parse_args()
# mdir_vidsav = "/home/xpsucla/Rahul_FPGA_Project/Minato_drone_project/External Recording Files/"
fname_external = "External_Video_" + driving_agent + "_v" + str(version) + ".avi"
fname_onsite = "Drone_navigation_obstacle_" + driving_agent + "_v" + str(version) + ".avi"
fname_data = driving_agent + "_test_v" + str(version) + ".pkl"
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
        self.state = np.zeros(4)
        self.current_pos = np.zeros(4)
        self.state_gains = [10.0,10.0,10.0,350.0]
        self.xtarget = -205.0
        self.ytarget = -95.0
        self.ztarget = 1370 #432.0
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
        self.target_pos = np.array([self.xtarget, self.ytarget, self.local_targets[-1],0.0])
        self.target_dev = self.target_pos - self.current_pos
        self.z_num = self.local_targets.shape[0]#(np.abs(int(self.ztarget)))//self.z_l
        self.zthres_low = 20.0
        self.zthres_min = 0.5#float('inf')
        self.frame = None
        self.frame_rate_video = 10.0
        self.stop_video = False
        self.start_video = False
        self.start_time = None
        self.chp_num = -2
        self.ylim = -100
        self.reached_checkpoint = True
        self.ztol = 1.9 #2.5
        self.dist_target = float('inf')
        self.img_target_c = [200,200]
        self.z_obs_dist = float('inf')
        self.zper_obs = 10
        self.reached_target = False
        self.state4_tol = 15
        self.final_checkpoint = False
        self.current_count = 0
        self.total_count = 10

        # Set Wind
        self.v_list = [0, 5, 7.5, 10, 12, 14, 14] #[0, 0, 0, 0, 0, 0, 0]
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
        if (self.current_count==0):
            point1 = self.current_pos[:3]
            point2 = np.array([self.xtarget_temp,self.ytarget_temp,self.local_targets[self.chp_num]])
            self.dist_target = np.linalg.norm(point1-point2)
        if((self.dist_target < self.ztol) and (np.abs(self.state[3]) < self.state4_tol) and (not self.reached_checkpoint) and (self.chp_num>=0)):
            # self.client.simFlushPersistentMarkers()
            if (self.chp_num < (self.z_num-1)):
                if (self.chp_num == (self.z_num-2)):
                    self.final_checkpoint = True
                self.reached_checkpoint = True
                time.sleep(0.1)
                # self.chp_num += 1
                print('Next checkpoint: ',self.chp_num,'\n')
                if (self.chp_num % 4 == 0):
                    wind_ind = self.chp_num // 4
                    vxw = (np.random.choice(np.append(np.arange(-900,-600),np.arange(600,900)))/1000.0) * self.v_list[wind_ind]
                    vyw = (np.random.choice(np.append(np.arange(-900,-600),np.arange(600,900)))/1000.0) * self.v_list[wind_ind]
                    vzw = 0#(np.random.choice(np.append(np.arange(-1000,-400),np.arange(400,1000)))/1000.0) * self.v_list[wind_ind]
                    print(f'Setting Wind Velocity(vx,vy,vz): {vxw}m/s, {vyw}m/s, {vzw}m/s')
                    self.client2.simSetWind(airsim.Vector3r(vxw,vyw,vzw))
                    self.vxyz = np.array([vyw,vzw,vxw])
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
        if (self.chp_num<=13):
            xtarget_tmp = self.current_pos[0] - (np.sign(xdev)*min(5,abs(xdev)))
            ytarget_tmp = self.current_pos[1] - (np.sign(ydev)*min(0.5,abs(ydev)))
        else:
            xtarget_tmp = self.current_pos[0] - (np.sign(xdev)*min(5,abs(xdev)))
            ytarget_tmp = self.current_pos[1] - (np.sign(ydev)*min(1.5,abs(ydev)))
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

        self.state[0] = np.clip((self.current_pos[0]-self.xtarget_temp)*self.state_gains[0],-100,100)
        self.state[1] = np.clip((self.current_pos[1]-self.ytarget_temp)*self.state_gains[1],-100,100)
        self.state[2] = np.clip((self.current_pos[2]-self.local_targets[self.chp_num])*self.state_gains[2],-100,100)
        self.state[3] = np.clip(self.current_pos[3]*self.state_gains[3],-100,100)
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
        reach_chk = False
        if ((self.reached_checkpoint) and (self.current_count==0)):
            self.client.simFlushPersistentMarkers()
            self.current_count = 1
            # time.sleep(0.5)
        elif ((self.reached_checkpoint) and (self.current_count<self.total_count)):
            self.current_count += 1
        elif ((self.reached_checkpoint) and (self.current_count==self.total_count)):
            reach_chk = True

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
            if reach_chk:
                if not self.final_checkpoint:
                    img1d = np.array(rawImage.image_data_float, dtype=np.float32)
                    #img1d[img1d > 255] = 255
                    img2d = np.reshape(img1d, (rawImage.height, rawImage.width))
                    # print(f'Average Depth: {np.mean(img2d)}')
                    Image3D = self.generatepointcloud(img2d)
                    _ = self.convolve2d_img(Image3D)
                else:
                    self.xtarget_temp, self.ytarget_temp = self.xtarget, self.ytarget
                
                if (self.chp_num < (self.z_num-1)):
                    self.chp_num += 1
                
                # if (self.chp_num!=(self.z_num-1)):
                xe, ye, ze = (self.local_targets[self.chp_num]),self.xtarget_temp,(self.ytarget_temp)
                xs, ys, zs = (self.local_targets[self.chp_num]),self.xtarget_temp,(self.ytarget_temp-0.3)
                self.client.simPlotArrows(points_start=[Vector3r(xs,ys,zs)], points_end=[Vector3r(xe,ye,ze)], 
                                            color_rgba=[1.0,0.0,0.0,1.0], arrow_size=100,thickness=7,is_persistent=True)
                
                self.reached_checkpoint = False
                self.current_count = 0
            
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
            text_center = 'State: (' + str(int(self.state[0])) + ',' + str(int(self.state[1])) + ',' + str(int(self.state[2])) + ',' + str(int(self.state[3])) + ')'
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
        video_thread = threading.Thread(target=self.record_video)
        video_thread.start()

        #print("Drone Tracking Started")
        startTime = time.time()
        
        # self.client.simPlotArrows(points_start=[Vector3r(1250,-180,-92)], points_end=[Vector3r(1250,-180,-96)], color_rgba=[1.0,0.0,0.0,1.0], arrow_size=50,thickness=1,is_persistent=True)
        # time.sleep(10.0)
        # str_time = time.time()
        # self.client.simFlushPersistentMarkers()
        # time_str = time.time() - str_time
        # print(f"Time: {time_str} s")
        while True:
            self.get_obs()
            self.image_processing()
            # print(self.z_obs_dist)
            self.frameCount = self.frameCount  + 1
            endTime = time.time()
            diff = endTime - startTime
            if (diff > 1):
                self.fps = self.frameCount
                # print(f"Current Position: {self.current_pos[0]}, {self.current_pos[1]}, {self.current_pos[2]}, {self.current_pos[3]}")
                # print("FPS = {}".format(self.fps))
                #print("Frame Size = {}".format(self.frame.shape))
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

        # cam_info = self.drone.simGetCameraInfo("1", 'Drone1')
        # print("Camera Pre-Pose: ",airsim.utils.to_eularian_angles(cam_info.pose.orientation))
        # cam_info.pose.orientation = airsim.utils.to_quaternion(0.0,0.0,0.0)
        # self.drone.simSetCameraPose("1",cam_info.pose,vehicle_name='Drone1')
        # print("Camera Post-Pose: ",airsim.utils.to_eularian_angles(cam_info.pose.orientation))

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
        self.keypress_delay = 0.01
        self.duration = 0.1
        self.start_exp = 0
        self.state = self.preprocessing.state
        self.stop_fpga = False
        self.action1 = '00'
        self.action2 = '00'
        self.action3 = '00'
        self.action4 = '00'
        self.human_actions = ['00','00','00','00']
        # self.human_action2 = '00'
        # self.human_action3 = '00'
        # self.human_action4 = '00'
        self.ch_check = False
        self.dvx = 3
        self.dvy = 3
        self.dvz = 3
        self.yaw = 0.0
        self.dyaw = 2
        self.crashed = False
        self.crash_thres = 0.5
        self.crash_quit = False
        self.reached_target = False
        self.target_quit = False
        self.target_dev = list(self.preprocessing.target_dev)
        self.vxyz = list(self.preprocessing.vxyz)

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
    
    def recording_state(self):
        # if (self.control.start_experiment() == '1'):
        self.crash_check()
        self.reached_target = self.preprocessing.reached_target
        if ((self.control.start_experiment() == '1') and (self.start_exp == 0)):
            # if(np.abs(self.preprocessing.current_pos[2]) < np.abs(self.preprocessing.ztarget)):
            #     self.preprocessing.z_l = (self.preprocessing.ztarget - self.preprocessing.current_pos[2])/self.preprocessing.z_num
            #     self.preprocessing.local_targets = np.linspace(self.preprocessing.current_pos[2] + self.preprocessing.z_l,\
            #             int(self.preprocessing.ztarget), self.preprocessing.z_num, endpoint=True)
            #     # self.preprocessing.norms12 = 
            # else:
            #     self.preprocessing.z_l = (self.preprocessing.current_pos[2] - self.preprocessing.ztarget)/self.preprocessing.z_num
            #     self.preprocessing.local_targets = np.flip(np.linspace(int(self.preprocessing.ztarget), \
            #         self.preprocessing.current_pos[2] - self.preprocessing.z_l, self.preprocessing.z_num, endpoint=True))
            # print('Local Targets: ',self.preprocessing.local_targets)
            self.preprocessing.reached_checkpoint = True
            time.sleep(1.0)
            print('Experiment Started')
            self.start_exp = 1
            self.preprocessing.start_video = True
            self.preprocessing.start_time = time.time()

        elif ((self.control.end_experiment() == '1') and (self.start_exp == 1)):
            print("Experiment Ended")
            self.start_exp = 0
            self.preprocessing.start_video = False
        elif (self.reached_target and (self.start_exp == 1)):
            print("Experiment Ended")
            self.start_exp = 0
        elif(self.crashed and self.start_exp == 1):
            print('Crashed!')
            print("Experiment Ended")
            # self.land_drone()
            self.start_exp = 0
            # self.preprocessing.start_video = False
        
                       
    def human_action_update(self):
        human_action1 = self.control.roll()
        human_action2 = self.control.throttle()
        human_action3 = self.control.pitch()
        human_action4 = self.control.yaw()
        self.human_actions = [human_action1,human_action2,human_action3,human_action4]

    def action_update(self,act1,act2,act3,act4):
        self.action1 = act1
        self.action2 = act2
        self.action3 = act3
        self.action4 = act4
    
    def crash_check(self):
        # print(self.preprocessing.z_obs_dist)
        if(self.preprocessing.z_obs_dist <= self.crash_thres and self.takeoff):
            if (self.preprocessing.dist_target>= self.preprocessing.ztol):
                self.crashed = True
            # print('Crashed!\n')
    
    def drone_control(self):
        # self.crash_check
        if(self.crashed is False):
            vy = 0.0
            if (self.control.roll() == '01'):
                if self.state[0]<= 90:
                    vy = self.dvy
            elif (self.control.roll() == '10'):
                if self.state[0] >= -90:
                    vy = -self.dvy
            else:
                if self.start_exp==1:
                    if (self.action1 == '01'):
                        if self.state[0]<= 90:
                            vy = self.dvy
                    elif (self.action1 == '10'):
                        if self.state[0] >= -90:
                            vy = -self.dvy
                        
            vx = 0.0
            if (self.control.pitch() == '01'):
                if self.state[2]<= 90:
                    vx = self.dvx
            elif (self.control.pitch() == '10'):
                if self.state[2] >= -90:
                    vx = -self.dvx
            else:
                if self.start_exp==1:
                    if (self.action3 == '01'):
                        if self.state[2]<= 90:
                            vx = self.dvx
                    elif (self.action3 == '10'):
                        if self.state[2] >= -90:
                            vx = -self.dvx

            vz = 0.0
            if (self.control.throttle() == '01'):
                if self.state[1]>= -90:
                    vz = -self.dvz
            elif (self.control.throttle() == '10'):
                if self.state[1] <= 90:
                    vz = self.dvz
            else:
                if self.start_exp==1:
                    if (self.action2 == '01'):
                        if self.state[1]>= -90:
                            vz = -self.dvz
                    elif (self.action2 == '10'):
                        if self.state[1] <= 90:
                            vz = self.dvz
            
            yawrate = 0.0
            if (self.control.yaw() == '01'):
                if self.state[3]<= 90:
                    yawrate = self.yaw_rate
            elif (self.control.yaw() == '10'):
                if self.state[3] >= -90:
                    yawrate = -self.yaw_rate
            else:
                if self.start_exp==1:
                    if (self.action4 == '01'):
                        if self.state[3]<= 90:
                            yawrate = self.yaw_rate
                    elif (self.action4 == '10'):
                        if self.state[3] >= -90:
                            yawrate = -self.yaw_rate

            self.drone.moveByVelocityAsync(vx, vy, vz, self.duration, yaw_mode={'is_rate': True, 'yaw_or_rate': yawrate},vehicle_name='Drone1')
            self.drone.moveToPositionAsync(1364, -205, -75, 3, yaw_mode={'is_rate': False, 'yaw_or_rate': self.yaw}, vehicle_name="Drone2")
            time.sleep(self.keypress_delay)
    
    def update_state(self):
        #self.preprocessing.get_obs()
        self.state = self.preprocessing.state
        self.target_dev = list(self.preprocessing.target_dev)
        self.vxyz = list(self.preprocessing.vxyz)

    
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

    def run(self):
        print("Drone is ready to fly")

        time.sleep(2)
        strt_time = time.time()
        frame_num = 0
        while True:
            self.recording_state()
            if not self.takeoff:
                self.takeoff_drone()
            else:
                self.land_drone()
                self.drone_control()
            
            self.ch_check = self.preprocessing.next_checkpoint()
            time.sleep(0.01)
            
            if(self.control.quit() or self.crash_quit or self.target_quit) :
                print("Quitting Code")
                self.exitcode = True

            if self.exitcode:
                self.preprocessing.stop_processing = True
                time.sleep(2)
                self.stop_fpga = True
                time.sleep(5)
                break
            '''
            if ((time.time()-strt_time)>=1):
                self.get_obs()
                strt_time = time.time()
            '''

        self.reset_drone()

        # that's enough fun for now. let's quit cleanly
        #self.drone.enableApiControl(False)
        #time.sleep(2)

class FPGAComm():

    def __init__(self):
        # Create the olympe.Drone object from its IP address
        self.mainfunc = AirSimDroneEnv()
        self.state = self.mainfunc.state
        self.start_exp = self.mainfunc.start_exp
        self.check_ch = 0
        self.mainfunc.start()
        self.fname = home_dir + fname_data #"FPGA_data_test_v35.pkl"
        self.save_data = True
        time.sleep(2)

    def find_device(self):
        """
        Find FX3 device and the corresponding endpoints (bulk in/out).
        If find device and not find endpoints, this may because no images are programed, we will program image;
        If image is programmed and still not find endpoints, raise error;
        If not find device, raise error.

        :return: usb device, usb endpoint bulk in, usb endpoint bulk out
        """

        # find device
        dev = usb.core.find(idVendor=0x04b4)
        intf = dev.get_active_configuration()[(0, 0)]

        # find endpoint bulk in
        ep_in = usb.util.find_descriptor(intf,
                                        custom_match=lambda e:
                                        usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_IN)

        # find endpoint bulk out
        ep_out = usb.util.find_descriptor(intf,
                                        custom_match=lambda e:
                                        usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_OUT)

        if ep_in is None and ep_out is None:
            print('Error: Cannot find endpoints after programming image.')
            return -1
        else:
            return dev, ep_in, ep_out
    
    def run(self):
        print("Start!")
        # find device
        usb_dev, usb_ep_in, usb_ep_out = self.find_device()
        usb_dev.set_configuration()
        
        # initial reset usb and fpga
        usb_dev.reset()
        if driving_agent=='SNIC':
            is_SNIC = True
        else:
            is_SNIC = False

        fpga_data_array = []
        time_array = []
        target_dev_array = []
        vxyz_array = []
        human_actions_array = []


        num = 64 * 1
        str_time = time.time()
        while not self.mainfunc.stop_fpga:
            self.mainfunc.update_state()
            self.state = self.mainfunc.state
            self.start_exp = self.mainfunc.start_exp
            #self.check_ch = int(self.mainfunc.ch_check)
            np_data1 = np.array([self.state[0],self.state[1],self.state[2], self.state[3],self.start_exp],dtype=np.uint8)
            np_data2 = np.random.randint(0, high=255, size = num-5, dtype=np.uint8)
            np_data = np.concatenate((np_data1,np_data2))
            wr_data = list(np_data)
            length = len(wr_data)
        
            # write data to ddr
            opu_dma(wr_data, num, 10, 0, usb_dev, usb_ep_out, usb_ep_in)
        
            # start calculation
            opu_run([], 0, 0, 3, usb_dev, usb_ep_out, usb_ep_in)

            # read data from FPGA
            rd_data = []
            opu_dma(rd_data, num, 11, 2, usb_dev, usb_ep_out, usb_ep_in)

            if is_SNIC:
                action1 = '{0:02b}'.format(int(rd_data[0]))
                action2 = '{0:02b}'.format(int(rd_data[1]))
                action3 = '{0:02b}'.format(int(rd_data[2]))
                action4 = '{0:02b}'.format(int(rd_data[3]))
                self.mainfunc.action_update(action1,action2,action3,action4)
            else:
                self.mainfunc.human_action_update()

            '''action3 = rd_data[0]
            action2 = rd_data[1]
            action1 = rd_data[2]'''

            if self.start_exp==1:
                fpga_data_array.append(rd_data)
                time_array.append((time.time()-str_time))
                target_dev_array.append(self.mainfunc.target_dev)
                vxyz_array.append(self.mainfunc.vxyz)
                human_actions_array.append(self.mainfunc.human_actions)
        
        if self.save_data:
            with open(self.fname, "wb") as fout:
                # default protocol is zero
                # -1 gives highest prototcol and smallest data file size
                pickle.dump((fpga_data_array, time_array, target_dev_array, vxyz_array,human_actions_array), fout, protocol=-1)

if __name__ == "__main__":
    fpga_comm = FPGAComm()
    # Start the fpga communication
    fpga_comm.run()

    time.sleep(1)
    