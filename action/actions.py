#!/usr/bin/env python3
# coding=utf8

import math
import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool
from puppy_control_msgs.msg import Velocity, Pose, Gait


class PuppyActions(Node):
    def __init__(self):
        super().__init__('marvin_puppy_controller')

        self.pose_pub = self.create_publisher(Pose, '/puppy_control/pose', 10)
        self.gait_pub = self.create_publisher(Gait, '/puppy_control/gait', 10)
        self.velocity_pub = self.create_publisher(Velocity, '/puppy_control/velocity', 10)
        self.mark_time_client = self.create_client(SetBool, '/puppy_control/set_mark_time')

        self.get_logger().info('PuppyActions initialized.')

    def set_pose(
        self,
        roll=0.0,
        pitch=0.0,
        yaw=0.0,
        height=-10.0,
        x_shift=-0.9,
        stance_x=0.0,
        stance_y=0.0,
        run_time=500
    ):
        msg = Pose(
            stance_x=stance_x,
            stance_y=stance_y,
            x_shift=x_shift,
            height=height,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            run_time=run_time
        )
        self.pose_pub.publish(msg)

    def set_gait(self, gait_type='Amble'):
        if gait_type == 'Trot':
            gait_config = {
                'overlap_time': 0.2,
                'swing_time': 0.3,
                'clearance_time': 0.0,
                'z_clearance': 5.0
            }
        elif gait_type == 'Walk':
            gait_config = {
                'overlap_time': 0.3,
                'swing_time': 0.4,
                'clearance_time': 0.4,
                'z_clearance': 5.0
            }
        else:
            gait_config = {
                'overlap_time': 0.1,
                'swing_time': 0.2,
                'clearance_time': 0.1,
                'z_clearance': 5.0
            }

        msg = Gait(
            overlap_time=gait_config['overlap_time'],
            swing_time=gait_config['swing_time'],
            clearance_time=gait_config['clearance_time'],
            z_clearance=gait_config['z_clearance']
        )
        self.gait_pub.publish(msg)

    def set_velocity(self, x=0.0, y=0.0, yaw_rate=0.0):
        msg = Velocity(x=x, y=y, yaw_rate=yaw_rate)
        self.velocity_pub.publish(msg)

    def set_mark_time(self, enabled=False):
        while not self.mark_time_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /puppy_control/set_mark_time service...')

        request = SetBool.Request()
        request.data = enabled
        future = self.mark_time_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

    def stand(self):
        self.get_logger().info('Action: stand')
        self.set_pose(
            roll=math.radians(0),
            pitch=math.radians(0),
            yaw=0.0,
            height=-10.0,
            x_shift=-0.9,
            stance_x=0.0,
            stance_y=0.0,
            run_time=500
        )

    def walk_forward(self, speed=6.0, gait_type='Amble'):
        self.get_logger().info('Action: walk_forward')
        self.set_gait(gait_type)
        self.set_mark_time(False)
        self.set_velocity(x=speed, y=0.0, yaw_rate=0.0)

    def turn_left(self, yaw_rate=0.3):
        self.get_logger().info('Action: turn_left')
        self.set_mark_time(False)
        self.set_velocity(x=0.0, y=0.0, yaw_rate=yaw_rate)

    def turn_right(self, yaw_rate=-0.3):
        self.get_logger().info('Action: turn_right')
        self.set_mark_time(False)
        self.set_velocity(x=0.0, y=0.0, yaw_rate=yaw_rate)

    def stop(self):
        self.get_logger().info('Action: stop')
        self.set_velocity(x=0.0, y=0.0, yaw_rate=0.0)