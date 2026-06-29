cmake_minimum_required(VERSION 3.8)
project(car_spraying_robot)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED) #Check later

install(
	DIRECTORY launch urdf config meshes models worlds car_spraying_robot
	DESTINATION share/${PROJECT_NAME}
	)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_python_install_package(${PROJECT_NAME})
# ADD THIS BLOCK: Installs the actual executable nodes
install(PROGRAMS
  car_spraying_robot/peya_subscriber.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME peya_subscriber
)

# (Optional: Add this if you also need ik_square to run)
#  install(PROGRAMS
#    car_spraying_robot/ik_square.py
#    DESTINATION lib/${PROJECT_NAME}
#    RENAME ik_square
#  )

ament_package()
