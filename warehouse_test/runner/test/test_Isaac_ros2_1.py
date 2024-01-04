
####################################################################################
# play omniverse

from omni.isaac.kit import SimulationApp
CONFIG = {"renderer": "RayTracedLighting", "headless": False}
simulation_app = SimulationApp(CONFIG)

####################################################################################
# import packages for path planning
import os
import cv2
import math
import time
import random
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import collections
import itertools
import largestinteriorrectangle as lir
import planners.astar_planner as astar_planner
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from parameter import Parameter
from position import Position
from exceptions import SHGGeometryError
from matplotlib import pyplot as plt
from typing import Dict, List, Sequence, Tuple
from environment import Environment
from shapely.ops import nearest_points, unary_union, polygonize_full
from shapely import MultiLineString, Polygon, LineString, MultiPolygon, GeometryCollection, Point
from webcolors import rgb_to_hex
from path_planning.search_planning import AStar

####################################################################################
# import packages for robot navigation
import sys
import carb
import omni
from typing import Optional
from omni.isaac.sensor import IMUSensor
from omni.isaac.core.controllers.articulation_controller import ArticulationController
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core import World
from omni.isaac.occupancy_map import _occupancy_map
from omni.isaac.occupancy_map.scripts.utils import update_location, compute_coordinates, generate_image
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import define_prim, get_prim_at_path
from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.isaac.sensor import LidarRtx, RotatingLidarPhysX
from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
from omni.isaac.wheeled_robots.controllers.stanley_control import State, calc_target_index, pid_control, stanley_control, normalize_angle
from omni.isaac.wheeled_robots.controllers.holonomic_controller import HolonomicController

####################################################################################
# enable ROS2 bridge extension
ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("omni.isaac.ros2_bridge", True)

####################################################################################
# ogm mapping
class ogm_runner(object):
    def __init__(self, args, directory, simulation_app, world) -> None:
        self._args = args
        self._directory = directory
        self._simulation_app = simulation_app
        self._world = world
        self._om = _occupancy_map.acquire_occupancy_map_interface()
        self._timeline = omni.timeline.get_timeline_interface()
        self._rgb_byte_provider = omni.ui.ByteImageProvider()

        self._assets_root_path = get_assets_root_path()

        if self._assets_root_path is None:
            carb.log_error("Could not fine Isaac Sim assets folder")
            self._simulation_app.close()
            sys.exit()

    def _get_the_scene(self):

        prim = get_prim_at_path(f"/World/{self._args.scene_prim_name}")
        if not prim.IsValid():
            prim = define_prim(f"/World/{self._args.scene_prim_name}", "Xform") # 수정 - 테스트 필요
            asset_path = self._assets_root_path + self._args.scene_path
            prim.GetReferences().AddReference(asset_path)

    def _save_image(self, dims, file_name, folder_name):

        from PIL import Image
        rotate_image_angle = self._args.rotate_image_angle
        self._im = Image.frombytes("RGBA", (dims.x, dims.y), bytes(self._image))
        self._im = self._im.rotate(-rotate_image_angle, expand=True)
        self._image = list(self._im.tobytes())

        image_width, image_height = self._im.width, self._im.height

        im = Image.frombytes("RGBA", (image_width, image_height), bytes(self._image))
        print("Saving occupancy map image to", folder_name + "/" + file_name)

        return im.save(folder_name + "/" + file_name)

    def run(self):
        self._get_the_scene()
        self._world.reset()
        
        ##############################################
        # occupant grid map - 2d metric map
        context = omni.usd.get_context()
        self._stage = context.get_stage()
        self._physx = omni.physx.acquire_physx_interface()

        generator = _occupancy_map.Generator(self._physx, context.get_stage_id())
        #generator.update_settings(0.05, 4, 5, 6)
        generator.set_transform((0, 0, 0), (-2.00, -2.00, 0), (2.00, 2.00, 0))
        generator.generate2d()

        self._timeline.play()
        update_location(self._om, (0, 0, 0), (-12, -18, 0.1), (12, 20.81808, 0.62))
        cell_size = 0.05
        self._om.set_cell_size(cell_size)

        self._om.generate()
        self._timeline.stop()

        scale = cell_size
        top_left, top_right, bottom_left, bottom_right, image_coords = compute_coordinates(self._om, scale)

        dims = self._om.get_dimensions()

        self._image = generate_image(self._om, [0, 0, 0, 255], [127, 127, 127, 255], [255, 255, 255, 255])

        # save_image
        file_name = self._args.image_file
        folder_name = self._directory

        self._save_image(dims, file_name, folder_name)

        return dims, top_left, top_right, bottom_left, bottom_right, image_coords

####################################################################################
# optimal path planning
class opt_path_planning(object):
    def __init__(self, args, directory, now) -> None:
        self._args = args
        self._directory = directory
        self._now = now
    
    def _get_marker_controlled_watershd(self, img, params):
        # convert to gray image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # convert to binary image
        ret, thresh = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        img_for_ws = cv2.cvtColor(opening.copy(), cv2.COLOR_GRAY2BGR)

        # sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # distance transform
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        norm_dist = cv2.normalize(dist_transform, dist_transform, 0, 1, cv2.NORM_MINMAX)

        # finding sure foreground area
        ret, sure_fg = cv2.threshold(dist_transform, 0.4*dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)

        # finding unknown region
        unknown = cv2.subtract(sure_bg, sure_fg)

        # marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)

        # add one to all labels so that sure background is not 0, but 1
        markers = markers + 1

        # now, marker the region of unknown with zero
        markers[unknown == 1] = 0

        # watershed algorithm
        ws = cv2.watershed(img_for_ws, markers)
        img_for_ws[markers == -1] = [255, 0, 0]

        # add safety margin as erosion
        size = 1+2*params["safety_distance"]
        print("Add safety distance with erosion and kernel:", (size, size))
        kernel_sd = np.ones((size, size), np.uint8)
        img_with_erosion = cv2.erode(opening.copy(), kernel_sd).astype(np.int32)
        img_with_erosion *= ws

        sep_colors = list()
        for miter in range(ret):
            if miter == 0:
                sep_colors.append((255, 255, 255))
            else:
                r_rand, g_rand, b_rand = random.randint(0, 230), random.randint(0, 230), random.randint(0, 230)
                sep_colors.append((r_rand, g_rand, b_rand))

        gp_zone = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)

        for i in range(markers.shape[0]):
            for j in range(markers.shape[1]):
                index = markers[i, j]
                if index > 0 and index <= len(img):
                    gp_zone[i, j, :] = sep_colors[index - 1]

        # save distance transform image
        save_file_name1 = f"Distance_Transfer_{self._now.year}_{self._now.month}_{self._now.day}_{self._now.hour}.png"
        try:
            plt.imshow(norm_dist)
            plt.savefig(f"{self._directory}/output/{save_file_name1}")
        except Exception:
            print("No image for Distance transfer")
            pass

        # save 
        save_file_name2 = f"Watershed_SetZone_{self._now.year}_{self._now.month}_{self._now.day}_{self._now.hour}.png"
        try:
            plt.imshow(gp_zone)
            plt.savefig(f"{self._directory}/output/{save_file_name2}")
        except Exception:
            print("No image for Watershed")

        plt.close()

        return ws, img_with_erosion, dist_transform, gp_zone

    def _get_find_bridge_nodes(self, ws, dist_transform, gp_zone):
        max_row, max_col = ws.shape
        edges = np.where(ws == -1)
        adjacent = np.eye(ws.max()+1)
        adjacent_edges: Dict[Tuple, List] = {}
        for edge_row, edge_col in zip(edges[0], edges[1]):
            neighbors = set()
            if edge_row-1 >= 0:
                neighbors.add(ws[edge_row-1, edge_col])
            if edge_row+1 < max_row:
                neighbors.add(ws[edge_row+1, edge_col])
            if edge_col-1 >= 0:
                neighbors.add(ws[edge_row, edge_col-1])
            if edge_col+1 < max_col:
                neighbors.add(ws[edge_row, edge_col+1])

            neighbors.discard(1)
            neighbors.discard(-1)
            neighbors.discard(0)

            if len(neighbors) != 2:
                continue

            n1, n2 = list(neighbors)
            if n2 < n1:
                n1, n2 = n2, n1

            adjacent[n1, n2] = 1

            if (n1, n2) in adjacent_edges:
                adjacent_edges[(n1, n2)].append((edge_row, edge_col))
            else:
                adjacent_edges[(n1, n2)] = [(edge_row, edge_col)]

        # print("Adjacent matrix", adjacent)
        bridge_nodes: Dict[Tuple, List] = {}
        bridge_edges: Dict[Tuple, List] = {}
        for adj_marker, edge in adjacent_edges.items():
            border = np.zeros(shape=ws.shape, dtype="uint8")
            for pixel in edge:
                border[pixel] = 255
            ret, connected_edges = cv2.connectedComponents(border)

            for i in range(1, connected_edges.max() + 1):
                connected_edge = np.where(connected_edges == i)
                connected_edge = list(zip(connected_edge[1], connected_edge[0]))
                bridge_pixel = max(connected_edge, key=lambda x: dist_transform[x[1], x[0]])

                if adj_marker in bridge_nodes:
                    bridge_nodes[adj_marker].append(bridge_pixel)
                    bridge_edges[adj_marker].append(connected_edge)

                else:
                    bridge_nodes[adj_marker] = [bridge_pixel]
                    bridge_edges[adj_marker] = [connected_edge]
        
        # bridge_nodes_axis = list()
        # [bridge_nodes_axis.extend(val) for _, val in bridge_nodes.items()]

        # plt.imshow(gp_zone)
        # plt.scatter(*zip(*bridge_nodes_axis), s=20, c='#ffa500', alpha=0.8)
        # save_file_name3 = f"Bridge_node_{self._now.year}_{self._now.month}_{self._now.day}_{self._now.hour}.png"
        # plt.savefig(f"{self._directory}/output/{save_file_name3}")
        # plt.close()

        return bridge_nodes, bridge_edges

    def _create_rooms(self, ws_erosion, params):
        ws_tmp = ws_erosion.copy()
        segment_envs: Dict[int, Environment] = {}
        largest_rectangles: Dict[int, List] = {}

        centroids = dict()

        for i in range(2, ws_tmp.max() + 1):
            env: Environment = Environment(i)
            segment_envs[i] = env
            largest_rectangles[i], centroid = self._calc_largest_rectangles(ws_tmp, env, params)
            centroids[i] = centroid

        return segment_envs, largest_rectangles, centroids

    def _calc_largest_rectangles(self, ws_erosion, env, params):
        largest_rectangles: List = []
        centroid = Position(0, 0, 0)
        first_loop = True
        is_corridor = [False]
        while True:
            segment = np.where(ws_erosion == env.room_id, 255, 0).astype("uint8")
            # TODO: Changed from cv2.RETR_TREE to EXTERNAL because it is faster and hierarchy doesn't matter
            contours, hierarchy = cv2.findContours(segment, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            contour_areas = list(map(cv2.contourArea, contours))
            c_max_index = np.argmax(contour_areas)
            c_max = contours[c_max_index]

            if first_loop:
                M = cv2.moments(segment)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centroid = Position(cX, cY, 0)

                x, y, w, h = cv2.boundingRect(c_max)
                env.room_bbox = [x, y, w, h]                                        #
                env.floor_bbox = [0, 0, ws_erosion.shape[1], ws_erosion.shape[0]]
                padding = 10
                # interior polygon (= free room) has to be inverted with [::-1] to be clear space in shapely
                walls = Polygon([(x-1-padding, y-1-padding), (x-1-padding, y+h+padding),
                                (x+w+padding, y+h+padding), (x+w+padding, y-1-padding)], [c_max[:, 0, :][::-1]])
                env.add_obstacle(walls)
                obstacle_index = np.where(hierarchy[0, :, 3] == c_max_index)[0]
                [env.add_obstacle(Polygon(contours[index][:, 0, :])) for index in obstacle_index]

            elif np.max(contour_areas) < params["min_contour_area"]:
                # print("New contour too small", cv2.contourArea(c_max))
                break

            rectangle = lir.lir(segment.astype("bool"), c_max[:, 0, :])
            largest_rectangles.append(rectangle)

            path = self._path_from_rectangle(rectangle, params, is_corridor, first_loop)

            if not path.is_empty:
                env.add_path(path)

            x, y, w, h = rectangle
            # Shrink rectangle by x pixel to make touching possible
            shrinkage = 1
            ws_erosion = cv2.rectangle(ws_erosion, (x+shrinkage, y+shrinkage),
                                    (x+w-1-shrinkage, y+h-1-shrinkage), (1), -1)
            first_loop = False

        # segmentation.show_imgs(segment, ws_erosion)
        # env.plot()
        # segmentation.show_imgs(segment, ws_erosion)
        return largest_rectangles, centroid

    def _path_from_rectangle(self, rectangle, params, is_corridor, first_loop):
        x, y, w, h = rectangle

        # TODO: This check is reduntant to "min_contour_area" put them together
        if (w * h) < params["min_roadmap_area"]:
            # print("area for roadmap too small")
            return LineString()

        # Offset of -1 pixel needed because opencv retuns w, h instead of coordinates
        # But shapely takes 1 pixel width edge as infinte thin line,
        # thus rectangles with edge 1 pixel different are not touching (not able to merge).
        # So rectangles are shrinked by 1 pixel in ws_tmp to compensate and make touching possible.

        w -= 1
        h -= 1
        if first_loop:
            if w < params["max_corridor_width"] or h < params["max_corridor_width"]:
                is_corridor[0] = True
                if w < h:
                    point_1 = (x + w//2, y)
                    point_2 = (x + w//2, y+h)
                else:
                    point_1 = (x, y+h//2)
                    point_2 = (x + w, y+h//2)

                # cv2.line(ws, point_1, point_2, (25), 2)
                return LineString([point_1, point_2])

        if not is_corridor[0]:
            point_1 = (x, y)
            point_2 = (x + w, y)
            point_3 = (x + w, y + h)
            point_4 = (x, y + h)

            return LineString([point_1, point_2, point_3, point_4, point_1])
        else:
            return LineString()

    def _create_paths(self, envs, bridge_nodes, bridge_edges, params):
        for room, env in envs.items():
            room_bridge_nodes = {adj_rooms: points for adj_rooms, points in bridge_nodes.items() if room in adj_rooms}
            room_bridge_edges = {adj_rooms: points for adj_rooms, points in bridge_edges.items() if room in adj_rooms}
            self._connect_paths(env, room_bridge_nodes, room_bridge_edges, params)

    def _connect_paths(self, env, bridge_nodes, bridge_edges, params):
        # Get all bridge points of this room to others
        bridge_points = [point for points in bridge_nodes.values() for point in points]
        bridge_points_not_connected = set()
        if not env.scene:
            print("No scene and paths in room", env.room_id)
            bridge_points_not_connected.update(bridge_points)
            return bridge_points_not_connected

        # Remove all bridge edges from walls
        [env.clear_bridge_edges(edge_points) for edge_points in bridge_edges.values()]

        # Connect bridge points between each other if room has no roadmap
        if not env.path:
            print("No path, connect only bridge points in room", env.room_id)
            for p1, p2 in itertools.combinations(bridge_points, 2):
                connection = env.get_valid_connection(Point(p1[0], p1[1]), Point(p2[0], p2[1]))
                if connection is None:
                    bridge_points_not_connected.add(p1)
                    bridge_points_not_connected.add(p2)
                    continue
                else:
                    env.add_path(connection)
                    print("Connection between bridge points added")
            if not env.path:
                print("No path in room", env.room_id)
                bridge_points_not_connected.update(bridge_points)
                return bridge_points_not_connected

        # Merge rectangles to roadmap and connect bridge points to path
        else:
            print("Connecting paths in room", env.room_id)
            result, dangles, cuts, invalids = polygonize_full(env.path)
            union_polygon: Polygon = unary_union(result)
            if isinstance(union_polygon, MultiPolygon):
                max_poly = max(union_polygon.geoms, key=lambda x: x.area)
                env.path = [max_poly.boundary]
            elif isinstance(union_polygon, Polygon):
                env.path = [union_polygon.boundary]
            elif isinstance(union_polygon, GeometryCollection):
                # union polygon is empty. No polygon in original room path
                pass
            else:
                print("unknown shape returned from polygon union")
                raise SHGGeometryError()

            # First try to connect all points with a straight line
            bridge_points_not_connected_directly = []
            for point in bridge_points:
                connection, _ = env.find_shortest_connection(point)
                if connection is not None:
                    env.add_path(connection)
                else:
                    bridge_points_not_connected_directly.append(point)

            # Second try to connect remaining points with a planner
            for point in bridge_points_not_connected_directly:
                connections, _ = self._connect_point_to_path(point, env, params)
                if len(connections) > 0:
                    for connection in connections:
                        env.add_path(connection)
                else:
                    bridge_points_not_connected.add(point)

        # print(len(env.path), "paths in room", env.room_id)
        # env.plot()
        return bridge_points_not_connected

    def _connect_point_to_path(self, point, env, params):

        """ List of connections is always in direction from point to path.
        Every connection is a two point line without collision.
        """
        connection, closest_path = env.find_shortest_connection(point)
        if connection is None:
            path = self._connect_point_with_astar(point, env, params)
            # path = _connect_point_with_rrt(point, env, params)

            if path == [] or path is None:
                print("No connection found for point", point)
                return [], None

            # TODO: Modiefied, check if this is still producing errors somewhre else
            if len(path) == 1:
                connections = [LineString([path[0].pos.xy, point])]
            else:
                connections = [LineString([path[i].pos.xy, path[i+1].pos.xy])
                                for i in range(0, len(path)-1)]  # type: ignore

            return connections, closest_path

        return [connection], closest_path

    def _connect_point_with_astar(self, point, env, params):
        
        config = dict()
        config["heuristic"] = 'euclidean'
        config["w"] = 0.5
        config['max_iterations'] = 10000
        config["smoothing_algorithm"] = "bechtold_glavina"
        config["smoothing_max_iterations"] = 100
        config["smoothing_max_k"] = 50
        planner = astar_planner.AStarPlanner.around_point(
            point, params["max_distance_to_connect_points"], env.scene, config)

        goal_list = self._get_goal_points(env)
        ts = time.time()
        path = planner.plan_with_lists([[point[0], point[1]]], goal_list, True)[0]
        print("Time", time.time() - ts)
        return path  # type: ignore

    def _get_goal_points(self, env):
        room_img = np.zeros((env.floor_bbox[3], env.floor_bbox[2]), np.uint8)
        room_img = self._draw_path(room_img, env, (1,), 1)
        goal_points = np.flip(np.argwhere(room_img == 1))
        return goal_points.tolist()

    def _draw_path(self, img, env, color, tickness):
        for line in env.path:
            if isinstance(line, MultiLineString):
                for string in line.geoms:
                    cv2.polylines(img, [string.coords._coords.astype("int32")], False,  color, tickness)
            else:
                cv2.polylines(img, [line.coords._coords.astype("int32")], False,  color, tickness)

        return img

    def _get_area(self, segment_envs, largest_rectangles, gp_zone):

        for room, rec_env in largest_rectangles.items():
            rec_polygon_area_list = list()

            for miter in range(len(rec_env)):
                if len(rec_env[miter]) > 2:
                    rec_x = rec_env[miter][0]
                    rec_y = rec_env[miter][1]
                    rec_w = rec_env[miter][2]
                    rec_h = rec_env[miter][3]

                    if miter == 0:
                        rgb_val = gp_zone[rec_y + np.uint32(float(rec_h)/2)][rec_x + np.uint32(float(rec_w)/2)]
                        hex_name = rgb_to_hex((rgb_val[0], rgb_val[1], rgb_val[2]))

                    rec_point_1 = (rec_x, rec_y)
                    rec_point_2 = (rec_x + rec_w, rec_y)
                    rec_point_3 = (rec_x + rec_w, rec_y + rec_h)
                    rec_point_4 = (rec_x, rec_y + rec_h)

                    rec_polygon = Polygon([rec_point_1, rec_point_2, rec_point_3, rec_point_4])
                    rec_polygon_area = rec_polygon.area
                else:
                    rec_x = rec_env[miter][0]
                    rec_y = rec_env[miter][1]

                    rec_polygon_area = 1
                    if miter == 0:
                        rgb_val = gp_zone[rec_y][rec_x]
                        hex_name = rgb_to_hex((rgb_val[0], rgb_val[1], rgb_val[2]))

                rec_polygon_area_list.append(rec_polygon_area)

            segment_envs[room].area = rec_polygon_area_list
            try:
                segment_envs[room].rgb = hex_name
            except Exception:
                segment_envs[room].rgb = rgb_to_hex((255, 255, 255))

        return segment_envs

    def _get_path_positions(self, segment_envs, src_room, tgt_room):

        Path_pos = dict()

        x_val_1 = round(segment_envs[int(src_room)].room_bbox[0] + segment_envs[int(src_room)].room_bbox[2] / 2)
        y_val_1 = round(segment_envs[int(src_room)].room_bbox[1] + segment_envs[int(src_room)].room_bbox[3] / 2)

        Path_pos[src_room] = (y_val_1, x_val_1) # 입력 필요

        x_val_2 = round(segment_envs[int(tgt_room)].room_bbox[0] + segment_envs[int(tgt_room)].room_bbox[2] / 2)
        y_val_2 = round(segment_envs[int(tgt_room)].room_bbox[1] + segment_envs[int(tgt_room)].room_bbox[3] / 4)

        Path_pos[tgt_room] = (y_val_2, x_val_2) # 입력 필요

        return Path_pos

    def _get_global_map(self, Path_pos, bridge_nodes, node_path, src_room, tgt_room):

        Path_scenarios = dict()

        for miter in range(len(node_path)):
            
            if miter == 0:
                src_node = [Path_pos[src_room]]

                min_tgt_node = min(int(node_path[miter]), int(node_path[miter+1]))
                max_tgt_node = max(int(node_path[miter]), int(node_path[miter+1]))
                tgt_node = bridge_nodes[(min_tgt_node, max_tgt_node)] # 낮은 값으로 정렬 필요
                tgt_node = [(tgt_node[0][1], tgt_node[0][0])]

            elif miter == len(node_path) - 1:
                min_src_node = min(int(node_path[miter - 1]), int(node_path[miter]))
                max_src_node = max(int(node_path[miter - 1]), int(node_path[miter]))
                src_node = bridge_nodes[(min_src_node, max_src_node)]
                src_node = [(src_node[0][1], src_node[0][0])]
                tgt_node = [Path_pos[tgt_room]]

            else:
                min_src_node = min(int(node_path[miter - 1]), int(node_path[miter]))
                max_src_node = max(int(node_path[miter - 1]), int(node_path[miter]))
                src_node = bridge_nodes[(min_src_node, max_src_node)]
                src_node = [(src_node[0][1], src_node[0][0])]

                min_tgt_node = min(int(node_path[miter]), int(node_path[miter+1]))
                max_tgt_node = max(int(node_path[miter]), int(node_path[miter+1]))
                tgt_node = bridge_nodes[(min_tgt_node, max_tgt_node)]
                tgt_node = [(tgt_node[0][1], tgt_node[0][0])]

            Path_scenarios[node_path[miter]] = {'src_node': src_node, 'tgt_node': tgt_node}

        return Path_scenarios

    def _get_network_graph(self, zone_list, centroids, src_room, tgt_room):

        G = nx.Graph()
        for nodes in zone_list:
            Point2D = collections.namedtuple('Point2D', ['x', 'y'])
            p1 = Point2D(x=centroids[int(nodes[0])].xy[0], y=centroids[int(nodes[0])].xy[1])
            p2 = Point2D(x=centroids[int(nodes[1])].xy[0], y=centroids[int(nodes[1])].xy[1])

            a_length = p1.x - p2.x  # 선 a의 길이
            b_length = p1.y - p2.y  # 선 b의 길이
            c_length = round(math.sqrt((a_length * a_length) + (b_length * b_length)) / 10)

            G.add_edge(nodes[0], nodes[1], weight=c_length)

        node_length, node_path = nx.single_source_dijkstra(G=G, source=src_room, target=tgt_room)

        print(f"Shortest path: {node_path}")
        print(f"Path length: {node_length}")

        return node_length, node_path

    def _get_path_planning_per_scenarios(self, node_path, Path_scenarios, img, gp_zone):
        # using the astar for path planning
        opt_path = dict()
        opt_search = 0
        ogm_path =[(None, None)]

        path_x_point, path_y_point = list(), list()
        astar = AStar("euclidean", img) # img, manhattan

        for opt_room_id in node_path:
            src_point = Path_scenarios[opt_room_id]['src_node']
            tgt_point = Path_scenarios[opt_room_id]['tgt_node']

            scenario_iter = 0
            room_path, room_path_len = list(), list()

            if len(tgt_point) > 1:
                while 1:
                    if opt_search < 1:
                        path_local, _ = astar.searching(src_point[0], tgt_point[scenario_iter])
                    else:
                        path_local, _ = astar.searching(src_point[-1], tgt_point[scenario_iter])
                    room_path.append(path_local)
                    room_path_len.append(len(path_local))

                    scenario_iter += 1
                    if len(tgt_point) >= scenario_iter:
                        break

                min_argument = np.argmin(room_path_len)
                ogm_path = list(reversed(room_path[min_argument]))
                opt_search += 1
            else:
                room_path, _ = astar.searching(src_point[0], tgt_point[0])
                ogm_path = list(reversed(room_path))

            print("*---------------------------------------------------*")
            print(f"main_opt_path for {opt_room_id}: {ogm_path}")

            opt_path[opt_room_id] = ogm_path
            path_xy_split = list(zip(*ogm_path))

            path_x_point.append(path_xy_split[1])
            path_y_point.append(path_xy_split[0])

        main_opt_path = opt_path.copy()
        print(f"main_opt_path : {main_opt_path}")

        plt.imshow(gp_zone)
        for opt_room_id in node_path:
            path_lists_1 = main_opt_path[opt_room_id]
            path_lists_2 = list(zip(*path_lists_1))
            plt.scatter(x=path_lists_2[1], y=path_lists_2[0], s=1, c='#ffa500', alpha=0.8)

        save_file_name4 = f"Path_planning_{self._now.year}_{self._now.month}_{self._now.day}_{self._now.hour}.png"
        plt.savefig(f"{self._directory}/output/{save_file_name4}")
        plt.close()

        return main_opt_path

    def run(self):

        ######################################################
        # segment envs
        img = cv2.imread(f"{self._directory}/{self._args.image_file}")
        params = Parameter(f"{self._directory}/{self._args.conf_file}").params

        ws, ws_erosion, dist_transform, gp_zone = self._get_marker_controlled_watershd(img=img, params=params)
        bridge_nodes, bridge_edges = self._get_find_bridge_nodes(ws=ws_erosion, dist_transform=dist_transform, gp_zone=gp_zone)

        segment_envs, largest_rectangles, centroids = self._create_rooms(ws_erosion, params)
        self._create_paths(segment_envs, bridge_nodes, bridge_edges, params)

        segment_envs = self._get_area(segment_envs, largest_rectangles, gp_zone)

        print(f"segment_envs: {segment_envs}")
        ######################################################
        # optimal route planning using topology

        room_list = list()
        [room_list.append(str(room_val)) for room_val, _ in segment_envs.items()]
        zone_list = list()
        [zone_list.append(list(map(str, node_val))) for node_val, _ in bridge_nodes.items()]
        
        if not zone_list:
            Path_scenarios = dict()
            src_room, tgt_room = '2', '2'
            # path positions
            x_val_1 = round(segment_envs[int(src_room)].room_bbox[0] + segment_envs[int(src_room)].room_bbox[2] / 2)
            y_val_1 = round(segment_envs[int(src_room)].room_bbox[1] + segment_envs[int(src_room)].room_bbox[3] / 2)
    
            x_val_2 = round(segment_envs[int(tgt_room)].room_bbox[0] + segment_envs[int(tgt_room)].room_bbox[2] / 2)
            y_val_2 = round(segment_envs[int(tgt_room)].room_bbox[1] + segment_envs[int(tgt_room)].room_bbox[3] / 4)

            # global map
            src_node, tgt_node = [(y_val_1, x_val_1)], [(y_val_2, x_val_2)]
            Path_scenarios['2'] = {'src_node': src_node, 'tgt_node': tgt_node}

            node_path = ['2']
            # optimal path planning
            main_opt_path = self._get_path_planning_per_scenarios(node_path, Path_scenarios, img, gp_zone)
            
        else:
            src_room, tgt_room = zone_list[0], zone_list[-1]
            # path positions
            Path_pos = self._get_path_positions(segment_envs, src_room, tgt_room)
            # network graph
            node_length, node_path = self._get_network_graph(zone_list, centroids, src_room, tgt_room)
            # global map
            Path_scenarios = self._get_global_map(Path_pos, bridge_nodes, node_path, src_room, tgt_room)
            # optimal path planning
            main_opt_path = self._get_path_planning_per_scenarios(node_path, Path_scenarios, img, gp_zone)

        return main_opt_path, node_path, ws_erosion

####################################################################################
# navigation
class navi_runner(object):

    def __init__(self, args, main_opt_path, node_path, simulation_app, ws_erosion,
                 dims, top_left, top_right, bottom_left, bottom_right, image_coords) -> None:

        self._args = args
        self._main_opt_path = main_opt_path
        self._node_path = node_path
        self._simulation_app = simulation_app
        self._ws_erosion = ws_erosion
        self._dims = dims
        self._top_left = top_left
        self._top_right = top_right
        self._bottom_left = bottom_left
        self._bottom_right = bottom_right
        self._image_coords = image_coords

        self._world = World(stage_units_in_meters=1.0)
        self._assets_root_path = get_assets_root_path()
        if self._assets_root_path is None:
            carb.log_error("Could not fine Isaac Sim assets folder")
            self._simulation_app.close()
            sys.exit()

    def _get_the_scene(self):

        prim = get_prim_at_path(f"/World/{self._args.scene_prim_name}")
        if not prim.IsValid():
            prim = define_prim(f"/World/{self._args.scene_prim_name}", "Xform") # 수정 - 테스트 필요
            asset_path = self._assets_root_path + self._args.scene_path
            prim.GetReferences().AddReference(asset_path)

    def _get_the_robot(self):
        self._x_diff = (abs(self._top_left[1]) + abs(self._bottom_left[1]))/self._ws_erosion.shape[0]
        self._y_diff = (abs(self._bottom_left[1]) + abs(self._bottom_right[0]))/self._ws_erosion.shape[1]

        path_lists_1 = self._main_opt_path[self._args.src_room_number]
        path_lists_2 = list(zip(*path_lists_1))

        init_x = self._bottom_left[0] - (path_lists_2[0][0] * self._x_diff)
        init_y = self._bottom_left[1] - (path_lists_2[1][0] * self._y_diff)
         
        rb_asset_path = self._assets_root_path + self._args.robot_path

        self._carter = self._world.scene.add(
            WheeledRobot(
                prim_path=f"/World/{self._args.robot_prim_name}",
                name="my_carter",
                wheel_dof_names=["joint_wheel_left", "joint_wheel_right"],
                create_robot=True,
                usd_path=rb_asset_path,
                position=np.array([init_x, init_y, 0.2]),
            )
        )

        self._state = State(wheel_base=0.4132, x=init_x, y=init_y, yaw=np.radians(10.0), v=0.0)
        
    def _get_the_lidar(self):
        self._lidar = self._world.scene.add(
            RotatingLidarPhysX(prim_path=f"/World/Carter/chassis_link/stereo_cam_right/lidar", name="lidar"))

    def _get_the_controller(self):
        self._controller = DifferentialController(name="simple_control", wheel_radius=0.04295, wheel_base=0.4132)
    
    def _get_the_imu(self):
        self._imu_sensor = self._world.scene.add(
            IMUSensor(
                prim_path="/World/Carter/caster_wheel_left/imu_sensor",
                name=self._args.imu_sensor_name,
                frequency=60,
                translation=np.array([0, 0, 0]),
            )
        )

    def bearing_to_radian(self, cx, cy):
        cyaw_list, bearing_list = list(), list()
        for miter in range(len(cx)-1):
            src_position = [cx[miter], cy[miter]]
            tgt_position = [cx[miter+1], cy[miter+1]]
            
            # radian transfer
            src_position_to_rad = [src_position[0] * math.pi / 180, tgt_position[0] * math.pi / 180]
            tgt_position_to_rad = [src_position[1] * math.pi / 180, tgt_position[1] * math.pi / 180]

            # theta & bearing
            y_val = math.sin(tgt_position_to_rad[1] - tgt_position_to_rad[0]) * math.cos(src_position_to_rad[1])
            x_val = ((math.cos(src_position_to_rad[0]) * math.sin(src_position_to_rad[1])) -
                    (math.sin(src_position_to_rad[0]) * math.cos(src_position_to_rad[1]) * math.cos(tgt_position_to_rad[1] - tgt_position_to_rad[0])))

            theta = math.atan2(y_val, x_val)
            bearing = (theta * 180 / math.pi + 360) % 360

            cyaw_list.append(theta)
            bearing_list.append(bearing)

        return cyaw_list, bearing_list

    def path_tracking(self):
        cx, cy = list(), list()
        path_room_numb = list()

        room_numb = 0
        for room_key in self._node_path:
            path_lists_1 = self._main_opt_path[room_key]
            path_lists_2 = list(zip(*path_lists_1))

            cx_raw = list(self._bottom_left[0] - (np.array(path_lists_2[0]) * self._x_diff))
            cy_raw = list(self._bottom_left[1] - (np.array(path_lists_2[1]) * self._y_diff))
            romm_numb_raw = list(np.ones((len(cx_raw))) * int(room_key))

            room_numb += 1

            if room_numb >= 1:
                cx.extend(cx_raw[5:])
                cy.extend(cy_raw[5:])
                path_room_numb.extend(romm_numb_raw[5:])
            else:
                cx.extend(cx_raw)
                cy.extend(cy_raw)
                path_room_numb.extend(romm_numb_raw)
        
        cyaw, _ = self.bearing_to_radian(cx, cy)
        cyaw.insert(0, 0)

        cyaw, _ = self.bearing_to_radian(cx, cy)
        cyaw.insert(0, 0)
    
        target_rb_speed = self._args.robot_tgt_speed
        last_idx = len(cx) - 1

        target_idx, _ = calc_target_index(self._state, cx, cy)

        target_idx_list, linear_speed_list, angular_speed_list = list(), list(), list()
        trajectory_x_list, trajectory_y_list = list(), list()


        while last_idx > target_idx:
            ai = pid_control(target=target_rb_speed, current=self._state.v)
            di, target_idx =stanley_control(state=self._state, cx=cx, cy=cy, cyaw=cyaw, last_target_idx=target_idx, p=ai)
            self._state.update(acceleration=ai, delta=di, dt=0.01)
            linear_speed = self._state.v
            angular_speed = self._state.w

            target_idx_list.append(target_idx)
            linear_speed_list.append(linear_speed)
            angular_speed_list.append(angular_speed)
            trajectory_x_list.append(self._state.x)
            trajectory_y_list.append(self._state.y)
        
        target_idx_arr = list(dict.fromkeys(target_idx_list))

        base_path = os.getcwd()
        fig, ax = plt.subplots()
        ax.plot(cx, cy, 'r-')
        ax.plot(trajectory_x_list, trajectory_y_list, 'b--')

        fig.savefig(f"{base_path}/standalone_examples/ros_test/trajectory_1.png")
        plt.show()
        plt.close()

        return target_idx, target_idx_arr, target_idx_list, linear_speed_list, angular_speed_list, cx, cy, path_room_numb

    def run(self):
        
        ##############################################
        # initial setting
        self._get_the_scene()       
        self._get_the_robot()
        self._get_the_lidar()
        self._get_the_imu()
        self._get_the_controller()

        self._simulation_app.update() 
        self._world.reset()
        self._lidar.add_depth_data_to_frame()
        self._lidar.add_point_cloud_data_to_frame()
        self._lidar.enable_visualization()

        ##############################################
        # path tracking
        target_idx, target_idx_arr, target_idx_list, linear_speed_list, angular_speed_list, cx, cy, path_room_numb = self.path_tracking()

        i = 0
        threshold_dist = 2
        
        # measured dataset
        nav_x_list, nav_y_list = list(), list()
        output_data = {
            'time_lidar': [],
            'step_lidar': [],
            'space_number': [],
            'current_position': [],
            'goal_position': [],
            'current_dist_to_goal': [],
            'linear_velocity': [],
            'angular_velocity': [],
            'depth_lidar': [],
            'point_cloud_lidar': [],
            'lin_acc_imu': [],
            'ang_vel_imu': [],
            'orientation_imu': []}

        while simulation_app.is_running():
            self._world.step(render=True)
            for miter in range(len(target_idx_arr)):
                if self._world.is_playing():
                    if self._world.current_time_step_index == 0:
                        self._world.reset()
                        self._controller.reset()

                    if miter == len(target_idx_arr)-1:
                        tgt_idx_pos = target_idx_list.index(target_idx_arr[miter])
                    else:
                        tgt_idx_pos = target_idx_list.index(target_idx_arr[miter+1]) - 1

                    linear_speed_arr = linear_speed_list[tgt_idx_pos]
                    angular_speed_arr = angular_speed_list[tgt_idx_pos]

                    while 1:
                        pre_robot_position, _ = self._carter.get_world_pose()
                        self._carter.apply_wheel_actions(self._controller.forward(command=[linear_speed_arr, angular_speed_arr]))
                        self._simulation_app.update()

                        current_robot_position, _ = self._carter.get_world_pose()   # current position
                        goal_position = np.array([cx[i], cy[i], pre_robot_position[-1]])    # goal_position

                        pre_dist_to_goal = np.linalg.norm(goal_position - pre_robot_position)
                        current_dist_to_goal = np.linalg.norm(goal_position - current_robot_position)   # current dist to goal
                        linear_vel = self._carter.get_linear_velocity()     # lienar velocity
                        angular_vel = self._carter.get_angular_velocity()

                        # lidar
                        out_lidar_val = self._lidar.get_current_frame()
                        time_lidar = out_lidar_val['time']
                        physics_step_lidar = out_lidar_val['physics_step']
                        depth_lidar = out_lidar_val['depth']
                        point_cloud_lidar = out_lidar_val['point_cloud']
                        
                        # imu
                        out_imu_val = self._imu_sensor.get_current_frame()
                        lin_acc_imu = out_imu_val['lin_acc']
                        ang_vel_imu = out_imu_val['ang_vel']
                        orientation_imu = out_imu_val['orientation']
                        
                        output_data['time_lidar'].append(time_lidar)
                        output_data['step_lidar'].append(physics_step_lidar)
                        output_data['space_number'].append(path_room_numb[i])
                        output_data['current_position'].append(current_robot_position)
                        output_data['goal_position'].append(goal_position)
                        output_data['current_dist_to_goal'].append(current_dist_to_goal)
                        output_data['linear_velocity'].append(linear_vel)
                        output_data['angular_velocity'].append(angular_vel)
                        output_data['depth_lidar'].append(depth_lidar)
                        output_data['point_cloud_lidar'].append(point_cloud_lidar)
                        output_data['lin_acc_imu'].append(lin_acc_imu)
                        output_data['ang_vel_imu'].append(ang_vel_imu)
                        output_data['orientation_imu'].append(orientation_imu)

                        nav_x_list.append(current_robot_position[0])
                        nav_y_list.append(current_robot_position[1])

                        if i >= target_idx_list[tgt_idx_pos]:
                            break
                        else:
                            if (current_dist_to_goal < threshold_dist):
                                i += 1
                                iter_count = 0
                            else:
                                iter_count += 1

                        print("*---------------------------------------------------*")
                        print(f"iteration number = [{i}]")
                        print(f"linear speed = [{linear_speed_arr}]")
                        print(f"angular speed = [{angular_speed_arr}]")
                        print(f"goal_position = [{goal_position}]")
                        print(f"current_dist_to_goal= [{current_dist_to_goal}]")

            if i >= target_idx:
                break

def main():

    now = datetime.now()
    base_path = os.getcwd()

    parser = argparse.ArgumentParser(description='robot information')
    parser.add_argument('--rb_deviceId', default='isaac_test') # 수정: [00.00.00.FA.86.09, isaac_test]
    parser.add_argument('--directory', default=f"{base_path}/standalone_examples/ros_test/data/")
    parser.add_argument("--scene_prim_name", default="Warehouse")
    parser.add_argument("--scene_path", default="/Isaac/Environments/Simple_Warehouse/warehouse.usd")
    parser.add_argument("--robot_path", default="/Isaac/Robots/Carter/nova_carter_sensors.usd")
    parser.add_argument("--robot_prim_name", default="nova_carter_sensors")
    parser.add_argument("--robot_name", default="nova_carter")
    parser.add_argument("--imu_sensor_name", default="kt_imu")

    parser.add_argument('--src_room_number', default='2') 
    parser.add_argument('--tgt_room_number', default='2')
    parser.add_argument("--robot_tgt_speed", default=0.05)

    parser.add_argument("--rotate_image_angle", default=90)
    parser.add_argument("--image_file", default='test_map.png')
    parser.add_argument('--conf_file', default='map_params.yaml')
    parser.add_argument('--path_sim', default='astar') 

    args, _ = parser.parse_known_args()
    
    directory = f"{args.directory}{args.rb_deviceId}"
    world = World(stage_units_in_meters=1.0)

    ##############################################
    # ogm based image save
    dims, top_left, top_right, bottom_left, bottom_right, image_coords = \
        ogm_runner(args=args, directory=directory, simulation_app=simulation_app, world=world).run()

    ##############################################
    # optimal path planning
    main_opt_path, node_path, ws_erosion = opt_path_planning(args=args, directory=directory, now=now).run()

    ##############################################
    # Navigation
    runner = navi_runner(args, main_opt_path, node_path, simulation_app, ws_erosion, 
                        dims, top_left, top_right, bottom_left, bottom_right, image_coords)
    runner._world.reset()
    runner.run()
    simulation_app.close()

    

    print("hello world")

if __name__ == "__main__":
    main()



