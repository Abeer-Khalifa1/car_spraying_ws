#!/usr/bin/env python3
"""Display the camera stream for the vision PASS 2 workflow."""

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CompressedImage


class VisionStreamWindow(Node):
    def __init__(self):
        super().__init__('vision_stream_window')
        self.declare_parameter('camera_topic', '/color_image/compressed')
        self.declare_parameter('window_name', 'vision_camera_stream')

        self.camera_topic = self.get_parameter('camera_topic').value
        self.window_name = self.get_parameter('window_name').value
        self._latest_frame = None

        qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(CompressedImage, self.camera_topic, self._image_cb, qos)
        self.create_timer(0.05, self._draw_frame)

        self._display_available = True
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        except cv2.error as exc:
            self._display_available = False
            self.get_logger().warn(
                f'No display available for window "{self.window_name}": {exc}. '
                f'The node will still listen to {self.camera_topic} and log frames.'
            )
        except Exception as exc:
            self._display_available = False
            self.get_logger().warn(f'Could not create display window: {exc}')

        self.get_logger().info(
            f'Vision stream window ready on topic {self.camera_topic} '
            f'using window "{self.window_name}"'
        )

    def _image_cb(self, msg: CompressedImage) -> None:
        try:
            np_arr = np.frombuffer(msg.data, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is not None:
                self._latest_frame = frame
        except Exception as exc:
            self.get_logger().warn(f'Failed to decode compressed image: {exc}')

    def _draw_frame(self) -> None:
        if self._latest_frame is None or not self._display_available:
            return
        try:
            cv2.imshow(self.window_name, self._latest_frame)
            cv2.waitKey(1)
        except cv2.error as exc:
            self.get_logger().warn(
                f'Cannot display window: {exc}. Stream frames are still being received.'
            )
        except Exception as exc:
            self.get_logger().warn(f'Failed to display frame: {exc}')

    def destroy_node(self):
        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VisionStreamWindow()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
