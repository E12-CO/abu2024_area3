# Test package launch file : 
import os

import launch
import launch_ros.actions
import launch.actions
import launch_ros.descriptions
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    #ABU Nodes and executables

    # Area 3 mission launch
    abu_area3_instant = launch_ros.actions.Node(
        package='abu_area3',
	name='abu_area3_node',
        executable='abu_area3_node.py',
	output='screen'
    )

    return launch.LaunchDescription([
        abu_area3_instant,
    ])

