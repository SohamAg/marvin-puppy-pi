
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int8
from actions.actions import PuppyActions

class PuppyControlSubscriber(Node):
    def __init__(self):
        super().__init__('puppy_control_subscriber')
        
        # Initialize the hardware controller
        self.controller = PuppyActions()
        
        # Startup sequence
        self.get_logger().info("Puppy Subscriber Starting...")
        self.controller.stand()

        # Create subscription to 'robot_state' topic
        # It expects an Int8: 0=Stop, 1=Walk, 2=Left, 3=Right, 4=Stand
        self.subscription = self.create_subscription(
            Int8,
            'robot_state',
            self.command_callback,
            10)

    def command_callback(self, msg):
        command_id = msg.data
        
        if command_id == 4: # Stand
            self.get_logger().info("Action: Standing")
            self.controller.stand()

        elif command_id == 1: # Walk
            self.get_logger().info("Action: Walking")
            self.controller.walk_forward(speed=6.0, gait_type='Amble')

        elif command_id == 2: # Left
            self.get_logger().info("Action: Turning Left")
            self.controller.turn_left()

        elif command_id == 3: # Right
            self.get_logger().info("Action: Turning Right")
            self.controller.turn_right()

        elif command_id == 0: # Stop
            self.get_logger().info("Action: Stopping")
            self.controller.stop()

        else:
            self.get_logger().warning(f"Unknown command ID: {command_id}")

    def stop_and_cleanup(self):
        self.controller.stop()
        self.controller.destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = PuppyControlSubscriber()

    try:
        # spin() keeps the node alive and listening for messages
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt, shutting down...")
    finally:
        node.stop_and_cleanup()
        rclpy.shutdown()

if __name__ == '__main__':
    main()