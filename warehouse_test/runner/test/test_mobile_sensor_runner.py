
####################################################################################
# play omniverse

from omni.isaac.kit import SimulationApp
CONFIG = {"renderer": "RayTracedLighting", "headless": False}
simulation_app = SimulationApp(CONFIG)

####################################################################################
# import packages for path planning
import os
import numpy as np

####################################################################################
# import packages for robot navigation
import sys
import carb
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import define_prim, get_prim_at_path
from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.isaac.sensor import LidarRtx

####################################################################################
# enable ROS2 bridge extension
ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("omni.isaac.ros2_bridge", True)

####################################################################################
# ogm mapping
class basic_runner(object):

    def __init__(self, params, simulation_app, directory):
        self._params = params
        self._simulation_app = simulation_app
        self._directory = directory
        self._world = World(stage_units_in_meters=1.0)

    def _get_the_scene(self):
        prim = define_prim(f"/World/{self._params['scene_prim_name']}", "Xform")
        asset_path = self._directory + self._params['scene_path']
        prim.GetReferences().AddReference(asset_path)

    def _get_the_robot(self):
        rb_asset_path = self._directory + self._params['robot_path']
        self._robot = self._world.scene.add(
            WheeledRobot(
                prim_path=f"/World/{self._params['robot_prim_name']}",
                name=self._params['robot_name'],
                wheel_dof_names=["joint_wheel_left", "joint_wheel_right"],
                create_robot=True,
                usd_path=rb_asset_path,
                position=np.array([0, 0, 0]),
            )
        )
    
    def _get_the_lidar(self):
        self._lidar = self._world.scene.add(
            LidarRtx(prim_path=f"/World/{self._params['robot_prim_name']}/{self._params['lidar_sensor_position']}", 
                     name=f"{self._params['lidar_sensor_name']}")
        )

    def run(self):
        self._world.reset()

        ##############################################
        # initial setting
        self._get_the_scene()
        self._get_the_robot()
        # self._get_the_lidar()
        self._simulation_app.update()
        
        # self._lidar.add_range_data_to_frame()
        # self._lidar.add_point_cloud_data_to_frame()
        # self._lidar.enable_visualization()

        count_iter = 10000

        i = 0
        while self._simulation_app.is_running():
            self._world.step(render=True)
            if self._world.is_playing():
                if self._world.current_time_step_index == 0:
                    self._world.reset()
                    self._controller.reset()

            if i >= count_iter:
                simulation_app.close()
            i += 1

def main():

    base_path = os.getcwd()
    directory = f"{base_path}/standalone_examples/warehouse_test"

    # parameters
    import yaml
    with open(f"{directory}/config/mobile_sensor_params.yaml", 'r') as f:
        params = yaml.full_load(f)

    # running
    basic_runner(params, simulation_app, directory).run()


if __name__ == "__main__":
    main()



