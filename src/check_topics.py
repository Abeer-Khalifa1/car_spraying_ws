#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import time

class TopicChecker(Node):
    def __init__(self):
        super().__init__('topic_checker')
        
    def check_topics(self):
        # Get all topics
        topics = self.get_topic_names_and_types()
        
        print("\n" + "="*70)
        print("AVAILABLE ROS2 TOPICS")
        print("="*70)
        
        image_topics = []
        for topic_name, types in topics:
            if 'image' in topic_name.lower() or 'camera' in topic_name.lower():
                image_topics.append((topic_name, types))
                print(f"  {topic_name}")
                print(f"    Type: {types}")
        
        print("\n" + "="*70)
        if not image_topics:
            print("⚠️  NO IMAGE/CAMERA TOPICS FOUND!")
            print("\nAll topics available:")
            for topic_name, types in topics:
                print(f"  {topic_name}: {types}")
        else:
            print(f"Found {len(image_topics)} image-related topic(s)")
        
        print("="*70 + "\n")

def main():
    rclpy.init()
    node = TopicChecker()
    node.check_topics()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
