# ---------------------------------------------------------------------------------------
# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ---------------------------------------------------------------------------------------

import launch
import launch_ros
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
import os 
def generate_launch_description():
    pkg_share = launch_ros.substitutions.FindPackageShare(package='ros2_trt_pose').find('ros2_trt_pose')
    print(pkg_share)
    default_rviz_config_path = 'src/ros2_trt_pose/ros2_trt_pose/launch/pose_estimation.rviz'

    trt_pose_node = Node(
            package="ros2_trt_pose",
            node_executable="pose-estimation",
            node_name="pose_estimation",
            output="screen",
            parameters = [{
                'base_dir':'/home/ak-nv/trt_pose/tasks/human_pose',
                'model': 'resnet18',
                'point_range' : 10,
                'show_image' : False,
                }],
            )
    cam2image_node = Node(
            package="image_tools",
            node_executable="cam2image",
            node_name="cam",
            )

    rviz_node = Node(
            package="rviz2",
            node_executable="rviz2",
            node_name="rviz2",
            arguments=['-d', LaunchConfiguration('rvizconfig')],
            )
    
    return LaunchDescription([
        launch.actions.DeclareLaunchArgument(name='rvizconfig', default_value=default_rviz_config_path),
        trt_pose_node,
        cam2image_node,
        rviz_node
        ])





