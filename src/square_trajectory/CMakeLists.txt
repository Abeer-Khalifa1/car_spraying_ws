cmake_minimum_required(VERSION 3.8)
project(square_trajectory)

find_package(ament_cmake REQUIRED)

find_package(rclcpp REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(nav_msgs REQUIRED) 
find_package(geometry_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)

add_executable(square_xz src/square_xz.cpp)

ament_target_dependencies(square_xz
  rclcpp
  moveit_core
  moveit_ros_planning_interface
  moveit_ros_planning
  nav_msgs
  geometry_msgs
  moveit_msgs
  tf2_eigen
)

install(TARGETS square_xz
  DESTINATION lib/${PROJECT_NAME}
)

install(
  PROGRAMS
    src/rl_agent_node.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
