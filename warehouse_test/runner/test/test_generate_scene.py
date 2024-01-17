
import argparse
import os

import numpy as np
from omni.isaac.kit import SimulationApp

# 'http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/2023.1.1'

def main(args):
    simulation_app = SimulationApp()

    # Late import because of runtime modules
    from omni.isaac.core import World
    from omni.isaac.core.utils.stage import close_stage
    from omni.isaac.scene_blox.generation.test_scene_generator import SceneGenerator
    from omni.isaac.scene_blox.grid_utils import config
    from omni.isaac.scene_blox.grid_utils.grid import Grid
    from omni.isaac.scene_blox.grid_utils.grid_constraints import GridConstraints
    from omni.isaac.scene_blox.grid_utils.tile import tile_loader
    from omni.isaac.scene_blox.grid_utils.tile_superposition import TileSuperposition

    tiles, weights = tile_loader(args.grid_config)
    constraints = None
    if args.constraints_config is not None:
        constraints = GridConstraints.from_yaml(args.constraints_config, args.rows, args.cols)
    superposition = TileSuperposition(tiles, weights)

    # Instantiate global RNG and set its seed (if args.seed == None, fresh entropy used)
    global_rng = config.GlobalRNG()
    global_rng.rng = np.random.default_rng(args.seed)

    generator = SceneGenerator(args.generation_config, args.collisions)

    rows = args.rows
    cols = args.cols

    grid = Grid(rows, cols, superposition)
    for i in range(args.variants):
        world = World(stage_units_in_meters=args.units_in_meters)
        success = False
        max_tries = 100
        current = 0
        while not success and current < max_tries:
            success = grid.solve(constraints, args.display)
            current += 1
            if not success:
                print("retry")
                grid.reset(superposition)
                if constraints is not None:
                    constraints.reset()
        if not success:
            print("Could not solve grid")
            continue
        print(f"Grid {i} solved in {current} tries")
        scene_path = os.path.join(f"{args.save_path}/{i}", f"generated_{i}.usd")
        world.reset()
        if args.collisions:
            print("Checking for colllisions while generating")
        generator.generate_scene(grid, world, scene_path)
        grid.reset(superposition)
        close_stage()
        world.clear_instance()

    simulation_app.close()


if __name__ == "__main__":

    # WAR for issue where script fails if run from non-standard path
    script_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", default=f"{script_path}/tools/scene_blox/src/output", help="Folder where the scenes will be generated")
    parser.add_argument("--variants", type=int, default=1, help="Number of variants of the scenes to be generated")
    parser.add_argument(
        "--grid_config",
        default=script_path + "/tools/scene_blox/parameters/warehouse/tile_config.yaml",
        help="Path to the yaml containing the combination rules and tile size",
    )
    parser.add_argument(
        "--generation_config",
        default=script_path + "/tools/scene_blox/parameters/warehouse/tile_generation.yaml",
        help="Path to the yaml containing generation configuration (base tile usd and randomization)",
    )
    parser.add_argument(
        "--constraints_config",
        default=script_path + "/tools/scene_blox/parameters/warehouse/constraints.yaml",
        help="Path to the yaml with the initial grid constraints",
    )

    parser.add_argument("--rows", type=int, default=11, help="Number of rows for the generated grids")
    parser.add_argument("--cols", type=int, default=15, help="Number of cols for the generated grids")
    parser.add_argument("--display", action="store_true", help="Add a display showing the grid solving process")
    parser.add_argument(
        "--collisions",
        action="store_true",
        help="Check for collisions on objects generated that have both collisions and rigid body physics enabled",
    )
    parser.add_argument(
        "--units_in_meters", default=1.0, type=float, help="Set the scene unit conversion (important for physics scene)"
    )
    parser.add_argument("--seed", type=int, help="Seed for random number generator")
    args, unknown = parser.parse_known_args()

    # workaround for agg backend which does not allow display to work
    import matplotlib

    if matplotlib.rcParams["backend"] == "agg":
        print("non-gui agg backend detected for matplotlib, setting display to false")
        args.display = True

    main(args)
