#!/usr/bin/env python3

import os

import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np

# ROS2 stuffs
import rclpy
from rclpy.node import Node
from std_msgs.msg import String as StringMsg
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
)# bestn2.pt  Ball-Colors-V8-images640

box_color = (255, 255, 255)
	
# Ball yolo begins here

def predict(chosen_model, img, classes=[], conf=0.9):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)

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

    cv2.line(image, (0, center_y), (width, center_y), (255, 255, 255), 1) # นอน

    cv2.line(image, (center_x, 0), (center_x, height), (255, 255, 255), 1) # ตั้ง

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


# ROS2 node
class abu_area3(Node):

	def __init__(self):
		super().__init__('ABUArea3')
		self.trackball = cv2.VideoCapture('/dev/trackball')
		self.trackball.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
		self.trackball.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

		# System FSMs and counter variable
		self.balltrack_fsm = 0
		self.state_delay = 0 # General purpose state delay
		self.state_delay_2 = 0
		
		# Ball tracking algorithon variables (CV, YOLO, etc..)
		self.select_ball_togo = None
		
		# Area 3 start flag
		self.at_area3 = 0 # True when robot reached area 3
		
		# Ball Feed system flags
		self.ball_ar = 0 # Ball accept/reject 1 == accept, 2 == reject
		self.ball_out_stat = 0 # Ball out status 1 == out, 0 == ball still at the top
		
		# recovery search attempt
		self.search_count = 0
		# Ball track motion variables, Kp and min max vel.
		self.trackball_kp = 0.005
		self.trackball_az_min = 0.1
		self.trackball_az_max = 1.0
		
		# Ball track command
		self.trackball_track_cmd = True
		
		# Ball track motion behaviors.
		# Jogging for certian amount of time to make sure that ball gets into the robot.
		self.trackball_jogging_flag = False
		
		# Command interlock
		self.ball_feed_started = False
		self.ball_feed_stopped = True
		self.ball_feed_out  = False
		
		# Command Robot velocity
		self.publisher_vel = self.create_publisher(
			Twist,
			'cmd_vel', 
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
			10)
			
		# Create timer callback
		self.timer = self.create_timer(0.03, self.timer_callback)

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
				self.at_area3 = 0
			
			case 'RETRY':
				print("abu_nav retry mode, running on area 1 and 2")
				self.at_area3 = 0
			
			case 'DONE':
				print("abu_nav reached Area 3, Starting Ball/Silo mission")
				self.at_area3 = 1
				 

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
		if success:
			
			result_img, sult = predict_and_detect(model, img, classes=[], conf=0.5)
			result_img, center_screen_x = draw_center_line(result_img)  # มีค่ากลางของจอ center_pic_x


			pos_x, pos_y, result_img, ball = select_ball(result_img, sult, self.select_ball_togo)
			self.select_ball_togo = ball

			#print(center_screen_x)

				
			match self.balltrack_fsm:
				case 0: # Find ball case
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
						
						
						self.Publish_msg_Twist(0.4, 0.0, ang_vel)
						
						if(pos_y > 400):# Wait unti ball reach at some point
							if self.ball_feed_started is False:
								self.ball_feed_started = True
								self.ball_feed_stopped = False
								self.Command_ball_feed('start')
								self.balltrack_fsm = 1
						else:
							if self.ball_feed_stopped is False:
								self.ball_feed_started = False
								self.ball_feed_stopped = True
								self.Command_ball_feed('stop')	
						

						print(pos_x, pos_y)
					elif self.search_count < 3: # search for find a ball
						
						self.state_delay += 1
						if self.state_delay < 25:
							self.Publish_msg_Twist(-0.4,0.0,0.0)
						else :
							orientation_count = self.state_delay - 25
							if orientation_count <= 105: # turn right 180 degrees
								self.Publish_msg_Twist(0.0,0.0,0.5)
							elif orientation_count <= 315:
								self.Publish_msg_Twist(0.0,0.0,-0.5) # turn left 360 degrees
							elif orientation_count <= 420:
								self.Publish_msg_Twist(0.0,0.0,0.5) # turn right 180 degrees
							else:
								self.state_delay = 0
								self.search_count += 1
					else :
						self.Publish_msg_Twist(0.0, 0.0, 0.0)
						print('State:Tracking ball no ball')

				case 1: # Feed ball in
					
					if self.ball_ar == 1 or self.ball_ar == 2:
						print('State:Feed ball Ball in')
						self.Publish_msg_Twist(0.0, 0.0, 0.0)
					else:
						print('State:Feed ball Ball feeding')
						self.state_delay += 1
						self.state_delay_2 += 1
						
						if(self.state_delay > 15):
							self.state_delay = 0
							self.trackball_jogging_flag = not self.trackball_jogging_flag
							
						if self.trackball_jogging_flag is False:
							self.Publish_msg_Twist(0.4, 0.0, 0.0)
						else:
							self.Publish_msg_Twist(0.0, 0.0, 0.0)
							
						if self.state_delay_2 > 100: # 0.03 second * 100 == 3 seconds
							self.state_delay = 0
							self.state_delay_2 = 0
							self.ball_feed_started = False
							self.ball_feed_stopped = True
							self.Command_ball_feed('stop')	
							self.Publish_msg_Twist(0.0, 0.0, 0.0)
							self.balltrack_fsm = 255

					# Reject ball
					if self.ball_ar == 2:
						print('State:Feed ball reject')
						self.balltrack_fsm = 10

				case 10:# Wait for ball out
					print('State:Ball out')
					if self.ball_out_stat == 1:
						self.ball_out_stat = 0
						self.balltrack_fsm = 0	

				case 255:# Stop case
					print('State:Stop case')
					self.balltrack_fsm = 0

			cv2.imshow("Image", result_img)

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
