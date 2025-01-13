#!/usr/bin/env python3

import os

import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
import time
# ROS2 stuffs
import rclpy
from rclpy.node import Node
from std_msgs.msg import String as StringMsg
from std_msgs.msg import UInt16
from geometry_msgs.msg import Twist
from ament_index_python.packages import get_package_share_directory

model = YOLO(
	os.path.join(
		os.path.expanduser("~"),
		'abu_ws',
		'src',
		'abu2024_area3',
		'model',
		'bestn2.pt',
	)
)
silo_model = YOLO(
	os.path.join(
		os.path.expanduser("~"),
		'abu_ws',
		'src',
		'abu2024_area3',
		'model',
		'cyroV8-size-n.pt',
	)
)
box_color = (255, 255, 255)
	
def predict_cylo(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf,verbose=False)
    else:
        results = chosen_model.predict(img, conf=conf,verbose=False)

    return results
    
def predict_and_detect_cylo(
    chosen_model, img, classes=[], conf=0.5, rectangle_thickness=1, text_thickness=1
):
    results = predict_cylo(chosen_model, img, classes, conf=conf)
    # print(results)
    for result in results:
        for box in result.boxes:
            
            if result.names[int(box.cls[0])] == "Cylo 0":  #  red RedBall
                text_color = (255, 0, 0)
                rectangle_thickness = 1
                center_x = int((box.xyxy[0][0] + box.xyxy[0][2]) / 2)
                center_y = int((box.xyxy[0][1] + box.xyxy[0][3]) / 2)

                cv2.circle(
                    img,
                    (center_x, center_y),
                    5,  
                    (255, 255, 255),   
                    -1 
                )

            elif result.names[int(box.cls[0])] == "Cylo 1":  #BlueBall blue
                text_color = (0, 255, 0)
                rectangle_thickness = 1
                center_x = int((box.xyxy[0][0] + box.xyxy[0][2]) / 2)
                center_y = int((box.xyxy[0][1] + box.xyxy[0][3]) / 2)

                cv2.circle(
                    img,
                    (center_x, center_y),
                    5, 
                    (255, 255, 255),
                    -1 
                )

            elif result.names[int(box.cls[0])] == "Cylo 2":  #PurpleBall  purple
                text_color = (0, 0, 255)
                rectangle_thickness = 1
                center_x = int((box.xyxy[0][0] + box.xyxy[0][2]) / 2)
                center_y = int((box.xyxy[0][1] + box.xyxy[0][3]) / 2)

                cv2.circle(
                    img,
                    (center_x, center_y),
                    5, 
                    (255, 255, 255),
                    -1 
                )

            elif result.names[int(box.cls[0])] == "Cylo 3": 
                text_color = (255, 0, 255)
                rectangle_thickness = 1
                center_x = int((box.xyxy[0][0] + box.xyxy[0][2]) / 2)
                center_y = int((box.xyxy[0][1] + box.xyxy[0][3]) / 2)

                cv2.circle(
                    img,
                    (center_x, center_y),
                    5, 
                    (255, 255, 255),
                    -1 
                )

            cv2.rectangle(
                img,
                (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                (int(box.xyxy[0][2]), int(box.xyxy[0][3])),
                text_color,
                rectangle_thickness,
            )
            
            cv2.putText(
                img,
                f"{result.names[int(box.cls[0])]} ({round(box.conf[0].item()*100, 2)}%)",
                (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                text_color,
                text_thickness,
            )

    return img, results
    
def predict(chosen_model, img, classes=[], conf=0.9):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf,verbose=False)
    else:
        results = chosen_model.predict(img, conf=conf,verbose=False)

    return results


def predict_and_detect(
    chosen_model, img, classes=[], conf=0.9, rectangle_thickness=1, text_thickness=1
):
    results = predict(chosen_model, img, classes, conf=conf)
    # print(results)
    for result in results:
        for box in result.boxes:
            
            if result.names[int(box.cls[0])] == "red":  # red RedBall
                text_color = (0, 0, 255)
                rectangle_thickness = 2

            elif result.names[int(box.cls[0])] == "blue":  # BlueBall blue
                text_color = (255, 0, 0)
                rectangle_thickness = 2
                
            elif result.names[int(box.cls[0])] == "purple":  # PurpleBall  purple
                text_color = (255, 255, 255)
                rectangle_thickness = 1

            cv2.rectangle(
                img,
                (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                (int(box.xyxy[0][2]), int(box.xyxy[0][3])),
                box_color,
                rectangle_thickness,
            )

            cv2.putText(
                img,
                f"{result.names[int(box.cls[0])]} ({round(box.conf[0].item()*100, 2)})",
                (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                text_color,
                text_thickness,
            )

    return img, results

def draw_center_line(image):
    height, width = image.shape[:2]
    center_x = width // 2
    center_y = height // 2

    cv2.line(image, (0, center_y), (width, center_y), (255, 255, 255), 3) # นอน

    cv2.line(image, (center_x, 0), (center_x, height), (255, 255, 255), 3) # ตั้ง

    return image, center_x

def select_ball(img, results, select_ball_togo):
    max_area = 0
    # max_confidence = 0

    if select_ball_togo is None:   # เลือกบอลก่อน 
        for result in results:
            for box in result.boxes:
                area = (box.xyxy[0][2] - box.xyxy[0][0]) * (box.xyxy[0][3] - box.xyxy[0][1])
                if result.names[int(box.cls[0])] in ["red", "blue"]:  # Considering only red and blue balls
                    # confidence = box.conf[0].item() 
                    if area > max_area:
                        max_area = area
                        select_ball_togo = box
                # if confidence > max_confidence:
                # max_confidence = confidence
                # chosen_box = box
    else :    # ตามบอล
        closest_box = None
        min_distance = float('inf')
        for result in results:
            for box in result.boxes:
                if result.names[int(box.cls[0])] in ["red", "blue"]: 
                    # Calculate distance between box and select_ball_togo
                    distance = ((box.xyxy[0][0] - select_ball_togo.xyxy[0][0])**2 + 
                                (box.xyxy[0][1] - select_ball_togo.xyxy[0][1])**2)**0.5
                    if distance < min_distance:
                        min_distance = distance
                        closest_box = box
        select_ball_togo = closest_box

    if select_ball_togo is not None:
        img = cv2.rectangle(img, (int(select_ball_togo.xyxy[0][0]), int(select_ball_togo.xyxy[0][1])), 
                            (int(select_ball_togo.xyxy[0][2]), int(select_ball_togo.xyxy[0][3])), (0, 0, 255), 2)
        
        center_x = int((select_ball_togo.xyxy[0][0] + select_ball_togo.xyxy[0][2]) / 2)
        center_y = int((select_ball_togo.xyxy[0][1] + select_ball_togo.xyxy[0][3]) / 2)

        cv2.circle(img, (center_x, center_y), 2, (255, 255, 255), -1)

        return center_x, center_y, img, select_ball_togo

    return None, None, img, None

def select_silo(img,results, select_silo_togo):

    piority = 99 # 1 for silo 2 , 2 for silo 0, 3 for silo 1
    distance_this = float('inf')
    height, width = img.shape[:2]
    center_screen_x = 295
    center_screen_y = height // 2
    cv2.line(img, (center_screen_x, 0), (center_screen_x, height), (255, 255, 255), 3)
    cv2.line(img, (0, center_screen_y), (width, center_screen_y), (255, 255, 255), 3)
    if select_silo_togo is None:
        for result in results:
            for box in result.boxes:
                center_x = int((box.xyxy[0][0] + box.xyxy[0][2]) / 2)
                center_y = int((box.xyxy[0][1] + box.xyxy[0][3]) / 2)
                if result.names[int(box.cls[0])] in ["Cylo 2"]:
                    piority = min(piority,1)
                    if piority == 1 :
                        #select_silo_togo = box
                        distance = abs(center_screen_x - center_x)
                        if distance < distance_this :
                            distance_this = distance
                            select_silo_togo = box
                elif result.names[int(box.cls[0])] in ["Cylo 0"]:
                    piority = min(piority,2)
                    if piority == 2 :
                        distance = abs(center_screen_x - center_x)
                        if distance < distance_this :
                            distance_this = distance
                            select_silo_togo = box
                elif result.names[int(box.cls[0])] in ["Cylo 1"]:
                    piority = min(piority,3)
                    if piority == 3 :
                        distance = abs(center_screen_x - center_x)
                        if distance < distance_this :
                            distance_this = distance
                            select_silo_togo = box

    else :
        closest_box = None
        min_distance = float('inf')
        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) == int(select_silo_togo.cls[0]): 
                    # Calculate distance between box and select_ball_togo
                    distance = ((box.xyxy[0][0] - select_silo_togo.xyxy[0][0])**2 + 
                                (box.xyxy[0][1] - select_silo_togo.xyxy[0][1])**2)**0.5
                    if distance < min_distance:
                        min_distance = distance
                        closest_box = box
        select_silo_togo = closest_box
    if select_silo_togo is not None:
        img = cv2.rectangle(img, (int(select_silo_togo.xyxy[0][0]), int(select_silo_togo.xyxy[0][1])), 
                            (int(select_silo_togo.xyxy[0][2]), int(select_silo_togo.xyxy[0][3])), (0, 0, 255), 2)
        
        center_x = int((select_silo_togo.xyxy[0][0] + select_silo_togo.xyxy[0][2]) / 2)
        center_y = int((select_silo_togo.xyxy[0][1] + select_silo_togo.xyxy[0][3]) / 2)
        return center_x, center_y, img, select_silo_togo
    return None, None, img, None
# ROS2 node
class abu_area3(Node):

	def __init__(self):
		super().__init__('ABUArea3')
		self.trackball = cv2.VideoCapture('/dev/trackball')
		self.tracksilo = cv2.VideoCapture('/dev/tracksilo')
		
		self.trackball.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
		self.trackball.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
		
		self.tracksilo.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
		self.tracksilo.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

		# System FSMs and counter variable
		self.balltrack_fsm = 0
		self.prev_state = -1
		self.state_delay = 0 # General purpose state delay
		self.state_delay_2 = 0
		# Area 3 start flag
		self.at_area3 = False # True when robot reached area 3
		self.at_stop = False
		# toggle recovery
		
		self.toggle_recovery = False
		# Ball tracking algorithon variables (CV, YOLO, etc..)
		self.select_ball_togo = None
		self.select_silo_togo = None
		
		# Ball Feed system flags
		self.ball_ar = 0 # Ball accept/reject 1 == accept, 2 == reject
		self.ball_out_stat = 0 # Ball out status 1 == out, 0 == ball still at the top
		
		# recovery search attempt
		self.search_count = 0
		
		# time of flight data
		self.front_tof = 0
		self.front2_tof = 0
		# Ball track motion variables, Kp and min max vel.
		self.trackball_kp = 0.0025
		self.trackball_az_min = 0.1
		self.trackball_az_max = 1.0
		# align robot with orientation

		self.angular_min = 0.1
		self.angular_max = 0.4
		self.angular_kp = 0.0020
		#Silo track motion, Kp and min max vel
		self.silo_kp = 0.00125 # 0.00125
		self.silo_y_max = 0.4
		self.silo_y_min = 0.1
		
		self.blind_kp = 0.0025
		self.blind_max = 0.4
		self.blind_min = 0.1
		# Ball track command
		self.trackball_track_cmd = True
		
		# Ball track motion behaviors.
		# Jogging for certian amount of time to make sure that ball gets into the robot.
		self.trackball_jogging_flag = False
		
		# Command interlock
		self.ball_feed_started = False
		self.ball_feed_stopped = True
		self.ball_feed_out  = False
		
		self.number_crash_calibration = 0
		self.number_right_turn = 0
		self.number_left_turn = 0
		self.turn_direction_select = 'left'
		# Command Robot velocity
		self.publisher_vel = self.create_publisher(
			Twist,
			'cmd_vel_a3',
			10)
		
		# Publish ball feed command
		self.pub_cmd = self.create_publisher(
			StringMsg,
			'ball_feed_cmd',
			10)
			
		# Subscribe to ball feed accpet/reject status topic
		self.pub_status = self.create_subscription(
			StringMsg,
			'ball_feed_ar',
			self.ball_ar_callback,
			10)
			
		# Subscibe to abu_nav /abu_nav_stat, wait for Area3 mission trigger
		self.sub_nav_stat = self.create_subscription(
			StringMsg,
			'abu_nav_stat',
			self.nav_stat_callback,
			1)
		self.sub_front_tof = self.create_subscription(
			UInt16,
			'right_tof',
			self.front_tof_callback,
			10)
		self.sub_front2_tof = self.create_subscription(
			UInt16,
			'left_tof',
			self.front2_tof_callback,
			10
			)
		self.sub_teammode = self.create_subscription(
			StringMsg,
			'teammode',
			self.teammode_callback,
			1
			)

		# Create timer callback
		self.timer = self.create_timer(0.03, self.timer_callback)

	def front_tof_callback(self,msg):
		#print(msg.data)
		self.front_tof = int(msg.data)
	def front2_tof_callback(self,msg):
		self.front2_tof = int(msg.data) -60
	# Ball accept/reject callback
	def ball_ar_callback(self, msg):
		cmd_str = msg.data
		
		match cmd_str:
			case 'Accept':
				print('Accept this ball!')
				self.ball_ar = 1
				self.ball_out_stat = 0
			
			case 'Reject':
				print('Reject this ball!')
				self.ball_ar = 2
				self.ball_out_stat = 0
				
			case 'OUT':
				print('Ball out!')
				self.ball_ar = 0
				self.ball_out_stat = 1

			case _:
				print('Unknow feed ar status!')
				self.ball_ar = 0
				self.ball_out_stat = 0

	# abu_nav callback. Check if robot reached Area 3
	def nav_stat_callback(self, msg):
		nav_stat = msg.data
		
		match nav_stat:
			case 'START':
				print("abu_nav start mode, running on area 1 and 2")
				self.at_area3 = False
			
			case 'RETRY':
				print("abu_nav retry mode, running on area 1 and 2")
				self.at_area3 = False
			
			case 'DONE':
				print("abu_nav reached Area 3, Starting Ball/Silo mission")
				self.at_area3 = True
			case _:
				self.at_area3 = False
				 
	def teammode_callback(self, msg):
		teammode_msg = msg.data

		match teammode_msg:
			case 'stop,stop':
				print('Stop mode detected')
				self.at_stop = True
				self.at_area3 = False
			case _:
				self.at_stop = False

	def Publish_msg_Twist(self,x,y,an_z):
		msg = Twist()
		msg.linear.x = x
		msg.linear.y = y
		msg.linear.z = 0.0
		msg.angular.x = 0.0
		msg.angular.y = 0.0
		msg.angular.z = an_z
		self.publisher_vel.publish(msg)
		
	def Command_ball_feed(self, cmd_str):
		msg = StringMsg()
		msg.data = cmd_str
		self.pub_cmd.publish(msg)	
		
	def timer_callback(self):
		success, img = self.trackball.read()
		suc_silo,img_silo = self.tracksilo.read()
		#print(img_silo)
		silo_x = silo_y = silo = result_silo_img = sult_silo = None
		if suc_silo :
				result_silo_img,sult_silo = predict_and_detect_cylo(silo_model, img_silo, classes=[], conf=0.5)
				#print(result_silo_img)
				silo_x, silo_y, img_silo, silo = select_silo(result_silo_img,sult_silo,self.select_silo_togo)
				self.select_silo_togo = silo
		if success :

			result_img, sult = predict_and_detect(model, img, classes=[], conf=0.7)
			result_img, center_screen_x = draw_center_line(result_img)  # มีค่ากลางของจอ center_pic_x


			pos_x, pos_y, result_img, ball = select_ball(result_img, sult, self.select_ball_togo)
			self.select_ball_togo = ball

			#print(center_screen_x)

			#print(self.balltrack_fsm)
			#print(self.front_tof,self.balltrack_fsm)

			if self.at_stop is True:
				self.at_stop = False
				self.at_are3 = False
				self.Publish_msg_Twist(0.0, 0.0, 0.0)
				self.balltrack_fsm = 0

			if self.balltrack_fsm != self.prev_state:
				self.prev_state = self.balltrack_fsm
				match self.balltrack_fsm:
					case 0:
						print('FSM: Idle state')
					case 1:
						print('FSM: Find ball')
					case 2:
						print('FSM: Feed ball in')
					case 10:
						print('FSM: Wait ball out')
					case 20:
						print('FSM: Backup recovery')
					case 30:
						print('FSM: Find silo')
					case 45:
						print('FSM: Silo Center Alignment')
					case 50:
						print('FSM: Approaching Silo')
					case 60:
						print('FSM: back-off until Silo found')
					case 70:
						print('FSM: Feed ball out')

			match self.balltrack_fsm:
				case 0: # Idle state
					if self.at_area3 is True:
						self.at_area3 = False
						self.state_delay = 0
						self.number_crash_calibration = 0
						self.number_right_turn = 0
						self.number_left_turn = 0
						self.turn_direction_select = 'left'
						self.balltrack_fsm = 1

				case 1: # Find ball case
					if pos_x is not None and pos_y is not None:
						self.state_delay = 0
						self.search_count = 0
						print('State: Tracking ball')
						ang_vel = -(pos_x - 280) * self.trackball_kp
						
						if ang_vel > self.trackball_az_max:
							ang_vel = self.trackball_az_max
						elif ang_vel < -self.trackball_az_max:
							ang_vel = -self.trackball_az_max
							
						# Velocity Dead-band
						if abs(ang_vel) < self.trackball_az_min:
							ang_vel = 0.0;
						
						
						self.Publish_msg_Twist(0.45, 0.0, ang_vel)
						
						if(pos_y > 350):# Wait unti ball reach at some point
							if self.ball_feed_started is False:
								self.ball_feed_started = True
								self.ball_feed_stopped = False
								self.Command_ball_feed('start')
								self.balltrack_fsm = 2
						else:
							if self.ball_feed_stopped is False:
								self.ball_feed_started = False
								self.ball_feed_stopped = True
								self.Command_ball_feed('stop')	
						

						#print(pos_x, pos_y)
					elif self.search_count < 3: # search for find a ball
						
						self.state_delay += 1
						if 10 < self.state_delay < 25:
							self.Publish_msg_Twist(0.4,0.0,0.0)
						else :
							orientation_count = self.state_delay - 25
							turn = 1 if self.number_right_turn >= self.number_left_turn else -1

							if orientation_count <= 30 :
								self.Publish_msg_Twist(0.0,0.0,0.3*turn)
							elif orientation_count <= 60 :
								self.Publish_msg_Twist(0.0,0.0,0.3*(-turn))
							else :
								self.state_delay = 0
								if turn == 1 :
									self.number_left_turn += 1
								else :
									self.number_right_turn += 1
					else :
						self.Publish_msg_Twist(0.0, 0.0, 0.0)
						#print('State:Tracking ball no ball')
				case 2: # Feed ball in
					
					if self.ball_ar == 1 or self.ball_ar == 2:
						print('State:Feed ball Ball in')
						self.Publish_msg_Twist(0.0, 0.0, 0.0)
					else:
						print('State:Feed ball Ball feeding')
						self.state_delay += 1
						self.state_delay_2 += 1
						
						if(self.state_delay > 25):
							self.state_delay = 0
							self.trackball_jogging_flag = not self.trackball_jogging_flag
							
						if self.trackball_jogging_flag is False:
							self.Publish_msg_Twist(0.4, 0.0, 0.0) # change here
						else:
							self.Publish_msg_Twist(0.0, 0.0, 0.0)
							
						if self.state_delay_2 > 133: # 0.03 second * 100 == 3 seconds
							self.state_delay = 0
							self.state_delay_2 = 0
							self.ball_feed_started = False
							self.ball_feed_stopped = True
							self.Command_ball_feed('stop')	
							self.Publish_msg_Twist(0.0, 0.0, 0.0)
							self.balltrack_fsm = 20

					# Reject ball
					
					if self.ball_ar == 2:
						print('State:Feed ball reject')
						self.balltrack_fsm = 10
					# accept ball go to silo
					elif self.ball_ar == 1 :
						print('State:Feed ball accept')
						self.balltrack_fsm = 30

				case 10: # Wait for ball out
					print('State:Ball out')
					if self.ball_out_stat == 1:
						self.ball_out_stat = 0
						self.balltrack_fsm = 1

				case 20: # Back-off recovery to detect another ball
					self.state_delay += 1
					self.Publish_msg_Twist(-0.4, 0.0, 0.0)
					if(self.state_delay > 25):
						self.state_delay = 0
						self.Publish_msg_Twist(0.0, 0.0, 0.0)
						self.balltrack_fsm = 1
				
				case 30: # Ball-in find out silo
					#print('HI 2')
					if self.select_silo_togo is not None :
						area = (self.select_silo_togo.xyxy[0][2] - self.select_silo_togo.xyxy[0][0]) * (self.select_silo_togo.xyxy[0][3] - self.select_silo_togo.xyxy[0][1])
						area = int(area)
						if area >= 27000:
							self.Publish_msg_Twist(0.0, 0.0, 0.0)
							self.balltrack_fsm = 45
					#if self.front_tof < 700 :
						#self.Publish_msg_Twist(0.0, 0.0, 0.0)
						#self.balltrack_fsm = 40

					if silo_x is not None and silo_y is not None :
						self.state_delay = 0
						self.search_count = 0
						y_vel = (silo_x - 295) * self.silo_kp
						#print(y_vel,silo_x,silo_y)
						if y_vel > self.silo_y_max:
							y_vel = self.silo_y_max
						elif y_vel < -self.silo_y_max:
							y_vel = -self.silo_y_max
							
						# Velocity Dead-band
						if abs(y_vel) < self.silo_y_min:
							y_vel = 0.0;
							
						#self.Publish_msg_Twist(-0.5,y_vel,0.0)
						self.Publish_msg_Twist(-0.5,0.0,-y_vel)
						
						if(y_vel > 400): # Wait until robot reach silo 
							self.Publish_msg_Twist(0.0, 0.0, 0.0)
							self.balltrack_fsm = 255
						else:
							pass
					elif self.search_count < 3: # search for find a ball
						
						self.state_delay += 1
						if 10 < self.state_delay < 25:
							self.Publish_msg_Twist(-0.4,0.0,0.0)
						else :
							orientation_count = self.state_delay - 25
							turn = 1 if self.number_right_turn >= self.number_left_turn else -1

							if orientation_count <= 30 :
								self.Publish_msg_Twist(0.0,0.0,0.3*turn)
							elif orientation_count <= 60 :
								self.Publish_msg_Twist(0.0,0.0,0.3*(-turn))
							else :
								self.state_delay = 0
								if turn == 1 :
									self.number_left_turn += 1
								else :
									self.number_right_turn += 1
					else :
						self.Publish_msg_Twist(0.0, 0.0, 0.0)
						print('State:Tracking ball no ball')

				# ********* CURRENTLY UNUSED *************
				case 40 : # set directon of robot
					angular = (self.front_tof - self.front2_tof) * self.angular_kp			#for align orientation

					print((self.front_tof - self.front2_tof),self.front_tof,self.front2_tof)

					if abs(angular) < self.angular_min :
						angular = self.angular_min if angular >= 0 else - self.angular_min
					elif abs(angular) > self.angular_max :
						angular = self.angular_max if angular >= 0 else -self.angular_max
					self.Publish_msg_Twist(0.0,0.0,angular)
					if abs(self.front_tof - self.front2_tof) <= 5 :
						self.state_delay += 1
						if self.state_delay > 10: # 0.3s delay for stability
							self.Publish_msg_Twist(0.0,0.0,0.0)
							self.state_delay = 0
							self.balltrack_fsm = 45
				# ********* CURRENTLY UNUSED *************

				case 45 : # Slide robot on Y axis to align with Silo.
					if silo_x is not None and silo_y is not None :
						print(silo_x,silo_y, silo_x - 295)
						if abs(silo_x - 295) <= 5 :
							self.Publish_msg_Twist(0.0,0.0,0.0)
							self.balltrack_fsm = 50
							
						y_vel = (silo_x - 295) * self.silo_kp
												
						if y_vel > self.silo_y_max:
							y_vel = self.silo_y_max
						elif y_vel < -self.silo_y_max:
							y_vel = -self.silo_y_max

						if abs(y_vel) < self.silo_y_min and y_vel > 0.0 :
							y_vel = self.silo_y_min
						elif abs(y_vel) < self.silo_y_min and y_vel <= 0.0 :
							y_vel = -self.silo_y_min
							
						self.Publish_msg_Twist(0.0,y_vel, 0.0)
					else :
						self.Publish_msg_Twist(0.4,0.0,0.0) # Why commanding X ?
				case 47 : # Slide robot on Y axis to align with Silo.
					if silo_x is not None and silo_y is not None :
						print(silo_x,silo_y, silo_x - 325)
						if abs(silo_x - 325) <= 5 :
							self.Publish_msg_Twist(0.0,0.0,0.0)
							self.balltrack_fsm = 50

						y_vel = (silo_x - 325) * self.silo_kp

						if y_vel > self.silo_y_max:
							y_vel = self.silo_y_max
						elif y_vel < -self.silo_y_max:
							y_vel = -self.silo_y_max

						if abs(y_vel) < self.silo_y_min and y_vel > 0.0 :
							y_vel = self.silo_y_min
						elif abs(y_vel) < self.silo_y_min and y_vel <= 0.0 :
							y_vel = -self.silo_y_min

						self.Publish_msg_Twist(0.0,y_vel,0.0)
					else :
 						self.Publish_msg_Twist(0.4,0.0,0.0) # Why commanding X ?


				case 50 : # blind walk to silo with tof
					self.state_delay += 1
					vel_x = self.front_tof * self.blind_kp
					if vel_x > self.blind_max:
						vel_x = self.blind_max
					elif vel_x < -self.blind_max:
						vel_x = -self.blind_max
                                                # Velocity Dead-band
					if abs(vel_x) < self.blind_min:
						vel_x = self.blind_min;
					#if self.state_delay >= 60 :
					#	if (self.state_delay % 8) <= 3 :
					#		self.Publish_msg_Twist(-0.3,0.0,0.5)
					#	elif (self.state_delay % 8) <= 7 :
					#		self.Publish_msg_Twist(-0.3,0.0,-0.5)
					if self.front_tof < 10 :
						self.Publish_msg_Twist(0.0, 0.0, 0.0)
						if self.number_crash_calibration == 1 :
							self.balltrack_fsm = 70
						else :
							self.balltrack_fsm = 60
						self.state_delay = 0
						self.number_crash_calibration += 1
					else :
						self.Publish_msg_Twist(-vel_x,0.0, 0.0)
						
				case 60 : # back from silo after calibration
					self.state_delay += 1
					if self.state_delay < 2 :
						self.Publish_msg_Twist(-0.15,0.0, 0.0)
					elif silo_x is not None and silo_y is not None and self.state_delay >= 25:
						self.state_delay = 0
						self.Publish_msg_Twist(0.0, 0.0, 0.0)
						self.balltrack_fsm = 47

					else :
						self.Publish_msg_Twist(0.4, 0.0, 0.0)
						
				case 70 : # go to shoot ball to silo
					self.state_delay += 1
					if self.state_delay < 2 : # offset slide left
						self.Publish_msg_Twist(-0.15,0.0, 0.0)
					elif  2 <= self.state_delay <= 35 :
						self.Publish_msg_Twist(0.0, 0.0, 0.0)
						self.Command_ball_feed('out')
						self.Command_ball_feed('start')
					elif self.state_delay < 85 :
						self.Command_ball_feed('stop')
						self.Publish_msg_Twist(0.6, 0.0, 0.0)
					elif self.state_delay >= 85 :
						self.Publish_msg_Twist(0.0, 0.0, 0.0)
						self.state_delay = 0
						self.number_crash_calibration = 0
						self.balltrack_fsm = 255
					#print("here")
					#print(self.front_tof)
					#if self.front_tof < 40 :
						#self.Publish_msg_Twist(0.0, 0.0, 0.0)
					#else :
						#print('here')
						#self.Publish_msg_Twist(-0.5,0.0,0.0)
				case 255: # inter-state
					print('FSM: inter-state')
					self.balltrack_fsm = 1
										
			#print(result_silo_img)
			#if result_silo_img is not None :
				#cv2.imshow("Image",result_silo_img)
			if result_silo_img is not None :
				ball_img = cv2.resize(result_img,(640,480))
				silo_img = cv2.resize(result_silo_img,(640,480))
				#numpy_horizontal = np.hstack((silo_img,ball_img))
				numpy_horizontal = np.concatenate((silo_img,ball_img),axis=1)
				cv2.imshow("Image",numpy_horizontal)
		# Break the loop if 'q' is pressed
			if cv2.waitKey(1) & 0xFF == ord('q'):
				self.destroy_node()

def main(args=None):
	rclpy.init(args=args)
	abu_area3_ball_node = abu_area3()
	rclpy.spin(abu_area3_ball_node)
	
	#abu_area3_ball_node.destroy_node()
	rclpy.shutdown()
	# Release the capture
	trackball.release()
	cv2.destroyAllWindows()
	
if __name__ == '__main__':
	main()	 

