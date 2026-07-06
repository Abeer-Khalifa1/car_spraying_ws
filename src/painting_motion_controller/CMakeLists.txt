cmake_minimum_required(VERSION 3.8)
project(painting_motion_controller)

find_package(ament_cmake REQUIRED)

find_package(rclcpp REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(builtin_interfaces REQUIRED)

# ============================================================
# Shared library containing common spraying functionality
# ============================================================

add_library(spray_common
    src/spray_common.cpp
    src/correction_pass.cpp
)

ament_target_dependencies(spray_common
    rclcpp
    moveit_core
    moveit_ros_planning_interface
    moveit_ros_planning
    moveit_msgs
    geometry_msgs
    nav_msgs
    sensor_msgs
    std_msgs
    tf2
    tf2_eigen
    Eigen3
    builtin_interfaces
)

target_include_directories(spray_common PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
)

# ============================================================
# Cartesian trajectory controller
# ============================================================

add_executable(cartesian_trajectory_controller
    src/cartesian_trajectory_controller.cpp
)

ament_target_dependencies(cartesian_trajectory_controller
    rclcpp
    moveit_core
    moveit_ros_planning_interface
    moveit_ros_planning
    nav_msgs
    geometry_msgs
    moveit_msgs
    sensor_msgs
    std_msgs
    tf2
    tf2_eigen
    Eigen3
    builtin_interfaces
)

target_link_libraries(cartesian_trajectory_controller
    spray_common
)

# ============================================================
# Vision pass executor
# ============================================================

add_executable(vision_pass_executor
    src/vision_pass_executor.cpp
)

ament_target_dependencies(vision_pass_executor
    rclcpp
    moveit_core
    moveit_ros_planning_interface
    moveit_ros_planning
    geometry_msgs
    nav_msgs
    std_msgs
)

target_link_libraries(vision_pass_executor
    spray_common
)

# ============================================================
# Installation
# ============================================================

install(TARGETS
    spray_common
    cartesian_trajectory_controller
    vision_pass_executor
    DESTINATION lib/${PROJECT_NAME}
)

install(
    PROGRAMS
        src/rl_agent_node.py
    DESTINATION lib/${PROJECT_NAME}
)

install(
    PROGRAMS
        src/rl_agent_node.py
    RENAME rl_agent_node
    DESTINATION lib/${PROJECT_NAME}
)

install(
    PROGRAMS
        src/vision_rl_agent.py
    RENAME vision_rl_agent_node
    DESTINATION lib/${PROJECT_NAME}
)

ament_package()
