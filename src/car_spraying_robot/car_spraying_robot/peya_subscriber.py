#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Float64
from sensor_msgs.msg import JointState

class PeyaSubscriber(Node): # Node creation

    def __init__(self):
        super().__init__('peya_subscriber') # Node name

        # 1. Subscriber: Listen to the incoming angles from /peya
        self.subscription = self.create_subscription(
            Float64MultiArray,
            '/peya',      # Topic name    
            self.listener_callback,
            10                
        )

        # 2. Publishers: Create 6 individual publishers for each joint
        # Change this in peya_subscriber.py
# 2. Publishers: Create 6 individual publishers for each joint
        self.joint_pubs = [
            self.create_publisher(
                Float64, 
                f'/joint_{i}/position_cmd', 
                10)  # Store up to 10 msgs if subscriber is slow
            for i in range(6)
        ]
        # 3. Optional: Subscriber to Joint States (for syncing/monitoring)
        self.create_subscription(JointState, '/joint_states', self.state_callback, 10) # Recieves messages
        self.get_logger().info('✅ Peya Subscriber Node started with 6 joint publishers.')

    def state_callback(self, msg):
        # This keeps track of where the robot actually is in Gazebo
        # Useful for debugging or initial sync
        pass

    def listener_callback(self, msg): # Extracts the data from the message and sends it to Gazebo
        data = list(msg.data)
        
        if len(data) >= 6:
            actual_angles = data[:6]
            self.send_to_gazebo(actual_angles)
        else:
            self.get_logger().error(f"❌ Data mismatch! Received {len(data)} elements.")
    
    def send_to_gazebo(self, angles):
        # Loop through each angle and publish it to its corresponding joint topic
        for i, angle in enumerate(angles):
            command = Float64()
            command.data = float(angle)
            self.joint_pubs[i].publish(command) # Publishes Angles
        
        # Log to confirm publishing is happening
        rounded = [round(a, 3) for a in angles]
        self.get_logger().info(f"🚀 Sent to Joint Commands: {rounded}")

def main(args=None):
    rclpy.init(args=args)
    node = PeyaSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()