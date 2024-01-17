
####################################################################################
# play omniverse
from omni.isaac.kit import SimulationApp
CONFIG = {"renderer": "Defaults", "headless": False}
simulation_app = SimulationApp(CONFIG)

####################################################################################
# enable ROS2 bridge extension
from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.isaac.ros2_bridge")

####################################################################################
# import packages
import os
import random
import pickle
import numpy as np
import omni
from omni.isaac.core import World
from omni.isaac.core import SimulationContext
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import define_prim, get_prim_at_path
from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
from omni.isaac.wheeled_robots.robots import WheeledRobot

world = World(stage_units_in_meters=1)

##############################################################################
# background scene
base_path = os.getcwd()
directory = f"{base_path}/standalone_examples/warehouse_test"
bg_scene_path = f"{directory}/data/scene/random/0/generated_0.usd"

add_reference_to_stage(usd_path=bg_scene_path, prim_path="/World")

##############################################################################
# object scene
tile_path = f"{directory}/data/scene/random/0/tile.pickle"
with open(tile_path, 'rb') as fr:
    tile_loaded = pickle.load(fr)

search_string_1 = 'single_shelf'
search_string_2 = 'double_shelf_a'
search_string_3 = 'empty_3_0'       # obstacle or forklit or people or empty
search_string_4 = 'empty_7_0'       # obstacle or forklit or people or empty
search_string_5 = '/World/empty_cross_7_1'
search_string_6 = '/World/empty_cross_7_3'
search_string_7 = '/World/empty_cross_3_3'
search_string_8 = '/World/empty_cross_5_3'

# random object 선정 필요
for tile_name in tile_loaded['tile_name']:

    # single shelf
    if search_string_1 in tile_name:
        numb_shelf = random.randint(1, 2)
        extras_path = f"{directory}/data/object/single_shelf_{numb_shelf}.usd"
        add_reference_to_stage(extras_path, f"{tile_name}/single_shelf_{numb_shelf}")

    # double shelf
    if search_string_2 in tile_name:
        numb_shelf = random.randint(1, 3)
        extras_path = f"{directory}/data/object/double_shelf_{numb_shelf}.usd"
        add_reference_to_stage(extras_path, f"{tile_name}/double_shelf_{numb_shelf}")

    # obstacle
    if search_string_3 in tile_name:
        extras_path = f"{directory}/data/object/obstacle_1.usd"
        add_reference_to_stage(extras_path, f"{tile_name}/obstacle_1")

    if search_string_4 in tile_name:
        extras_path = f"{directory}/data/robot/forklift/forklift_2.usd"
        add_reference_to_stage(extras_path, f"{tile_name}/forklift_2")

    # robot #1
    if search_string_5 == tile_name:
        rb_asset_path = f"{directory}/data/robot/transporter/transporter.usd"
        my_transporter_1 = world.scene.add(
            WheeledRobot(
                prim_path=f"{tile_name}/transporter",
                name="my_transporter_1",
                wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
                create_robot=True,
                usd_path=rb_asset_path,
            )

        )

    # robot #2
    if search_string_6 == tile_name:
        rb_asset_path = f"{directory}/data/robot/transporter/transporter.usd"
        my_transporter_2 = world.scene.add(
            WheeledRobot(
                prim_path=f"{tile_name}/transporter",
                name="my_transporter_2",
                wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
                create_robot=True,
                usd_path=rb_asset_path,
            )

        )

    # robot #3
    if search_string_7 == tile_name:
        rb_asset_path = f"{directory}/data/robot/carter/carter_v1.usd"
        my_carter_v1= world.scene.add(
            WheeledRobot(
                prim_path=f"{tile_name}/carter_v1",
                name="my_carter_v1",
                wheel_dof_names=["left_wheel", "right_wheel"],
                create_robot=True,
                usd_path=rb_asset_path,
            )

        )

    # robot #4
    if search_string_8 == tile_name:
        rb_asset_path = f"{directory}/data/robot/carter/nova_carter_sensors.usd"
        my_nova_carter_sensors= world.scene.add(
            WheeledRobot(
                prim_path=f"{tile_name}/nova_carter_sensors",
                name="my_nova_carter_sensors",
                wheel_dof_names=["joint_wheel_left", "joint_wheel_right"],
                create_robot=True,
                usd_path=rb_asset_path,
            )

        )

################################
# controller
my_controller = DifferentialController(name="simple_control", wheel_radius=0.03, wheel_base=0.1125)

simulation_app.update()
world.reset()

iter_numb = 0
while simulation_app.is_running():
    world.step(render=True)
    if world.is_playing():
        if world.current_time_step_index == 0:
            world.reset()
            my_controller.reset()
        if iter_numb >=0 and iter_numb < 1000:
            # forward
            my_transporter_1.apply_wheel_actions(my_controller.forward(command=[0.05, 0]))
            my_transporter_2.apply_wheel_actions(my_controller.forward(command=[0.03, 0]))
            my_carter_v1.apply_wheel_actions(my_controller.forward(command=[0.06, 0]))
            # my_nova_carter_sensors.apply_wheel_actions(my_controller.forward(command=[0.04, 0]))
        elif iter_numb >= 1000 and iter_numb < 1300:
            # rotate
            my_transporter_1.apply_wheel_actions(my_controller.forward(command=[0, np.pi/10]))
            my_transporter_2.apply_wheel_actions(my_controller.forward(command=[0, np.pi/30]))
            my_carter_v1.apply_wheel_actions(my_controller.forward(command=[0, np.pi/10]))
            my_nova_carter_sensors.apply_wheel_actions(my_controller.forward(command=[0, np.pi/20]))
        elif iter_numb >= 1300 and iter_numb < 2000:
            my_transporter_1.apply_wheel_actions(my_controller.forward(command=[0.05, 0]))
            my_transporter_2.apply_wheel_actions(my_controller.forward(command=[0.04, 0]))
            my_carter_v1.apply_wheel_actions(my_controller.forward(command=[0.05, 0]))
            # my_nova_carter_sensors.apply_wheel_actions(my_controller.forward(command=[0.03, 0]))
        elif iter_numb == 2000:
            break
        iter_numb += 1

simulation_app.close()

