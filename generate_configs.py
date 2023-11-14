import numpy as np
import pyflex
from copy import deepcopy
from softgym.utils.pyflex_utils import center_object
from tqdm import tqdm
import pickle
from softgym.envs.flex_utils import update_camera, set_cloth3d_scene, get_state, set_square_scene, get_current_covered_area
import argparse
import os
import random
from softgym.envs.cloth_env import ClothEnv
from Policy.demonstrator import Demonstrator

def get_square_default_config():
    cam_pos, cam_angle = np.array([0, 1.0, 0]), np.array([0 * np.pi, -90 / 180.0 * np.pi, 0])
    config = {
        "ClothPos": [0, 0, 0],
        "ClothSize": [55, 55],
        "ClothStiff": [2.0, 0.5, 1.0],
        "mass": 0.5,
        "camera_name": "default_camera",
        "camera_params": {
            "default_camera": {
                "pos": cam_pos,
                "angle": cam_angle,
                "width": 720,
                "height": 720,
            }
        },
        "flip_mesh": 0,
    }
    return config

def generate_square_configs(cloth_dimx, cloth_dimy):
    max_wait_step = 1000  # Maximum number of steps waiting for the cloth to stablize
    stable_vel_threshold = 0.2  # Cloth stable when all particles' vel are smaller than this
    default_config = get_square_default_config()

    config = deepcopy(default_config)
    update_camera(config["camera_params"], config["camera_name"])
    config["ClothSize"] = [cloth_dimx, cloth_dimy]

    set_square_scene(config)
    pos = pyflex.get_positions().reshape(-1, 4)
    pos[:, :3] -= np.mean(pos, axis=0)[:3]
    pos[:, 1] = 0.005
    pos[:, 3] = 1
    pyflex.set_positions(pos.flatten())
    pyflex.set_velocities(np.zeros_like(pos))
    for _ in range(5):  # In case if the cloth starts in the air
        pyflex.step()

    for _ in range(max_wait_step):
        pyflex.step()
        curr_vel = pyflex.get_velocities()
        if np.alltrue(np.abs(curr_vel) < stable_vel_threshold):
            break
    
    max_area = get_current_covered_area()
    state = get_state(config["camera_params"])
    state["max_area"] = max_area

    center_object()
    return deepcopy(config), deepcopy(state)


def generate_crumpled(cloth3d, configs, states):
    max_wait_step = 1000  # Maximum number of steps waiting for the cloth to stablize
    stable_vel_threshold = 0.2  # Cloth stable when all particles' vel are smaller than this

    policy = Demonstrator["DummyTask"]()
    env = ClothEnv(
        gui=False,
        cloth3d=cloth3d,
        dump_visualizations=False,
        pick_speed=policy.pick_speed,
        move_speed=policy.move_speed,
        place_speed=policy.place_speed,
        lift_height=policy.lift_height,
    )

    crumpled_configs, crumpled_states = [], []

    for config, state in tqdm(zip(configs, states), total=len(configs)):
        env.reset(config=config, state=state, lift_height=random.uniform(0.5, 1.5))
        particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3]
        num_points = particle_pos.shape[0]
        pick_index = np.random.randint(num_points)
        pick_pos = particle_pos[pick_index]
        env.pick_and_drop(pick_pos)


        for _ in range(max_wait_step):
            pyflex.step()
            curr_vel = pyflex.get_velocities()
            if np.alltrue(np.abs(curr_vel) < stable_vel_threshold):
                break

        center_object()
        crumpled_configs.append(deepcopy(config))
        curr_state = get_state(config["camera_params"])
        curr_state["max_area"] = state["max_area"]
        crumpled_states.append(deepcopy(curr_state))
    
    return crumpled_configs, crumpled_states


def get_cloth3d_default_config():
    cam_pos, cam_angle = np.array([0, 1.0, 0]), np.array([0 * np.pi, -90 / 180.0 * np.pi, 0])
    config = {
        "pos": [0, 0, 0],
        "scale": -1,
        "rot": 0,
        "vel": [0.0, 0.0, 0.0],
        "stiff": 1.0,
        "mass": 0.5 / (40 * 40),
        "radius": 0.00625,
        "camera_name": "default_camera",
        "camera_params": {
            "default_camera": {
                "pos": cam_pos,
                "angle": cam_angle,
                "width": 720,
                "height": 720,
            }
        },
        "cloth_type": 0,
        "cloth_index": 0,
    }
    return config


def generate_cloth3d_configs(cloth_type, cloth_index):
    max_wait_step = 300  # Maximum number of steps waiting for the cloth to stablize
    stable_vel_threshold = 0.2  # Cloth stable when all particles' vel are smaller than this
    default_config = get_cloth3d_default_config()

    config = deepcopy(default_config)
    update_camera(config["camera_params"], config["camera_name"])
    if cloth_type == "Tshirt":
        config["cloth_type"] = 0
    elif cloth_type == "Trousers":
        config["cloth_type"] = 1
    config["cloth_index"] = cloth_index

    set_cloth3d_scene(config)
    for _ in range(5):  # In case if the cloth starts in the air
        pyflex.step()

    for _ in range(max_wait_step):
        pyflex.step()
        curr_vel = pyflex.get_velocities()
        if np.alltrue(np.abs(curr_vel) < stable_vel_threshold):
            break
    
    center_object()

    max_area = get_current_covered_area()
    state = get_state(config["camera_params"])
    state["max_area"] = max_area

    return deepcopy(config), deepcopy(state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Cached Configs.")
    parser.add_argument("--cloth_type", type=str, default="Tshirt", help="choose square, rectangular or cloth3d cloth")
    parser.add_argument("--min_length", type=int, default=45, help="min length of square/rectangular cloth")
    parser.add_argument("--max_length", type=int, default=60, help="max length of square/rectangular cloth")
    parser.add_argument("--crumpled", action="store_true", help="make it crumpled")
    parser.add_argument("--num_configs", type=int, default=1000, help="Num of configs")
    parser.add_argument("--name", type=str, default="", help="Name of filename")
    args = parser.parse_args()

    assert args.cloth_type in ["Square", "Rectangular", "Tshirt", "Trousers"]

    if args.crumpled:
        save_file = os.path.join("configs", args.cloth_type + "_crumpled"  + "_" + args.name+ ".pkl")
    else:
        save_file = os.path.join("configs", args.cloth_type + args.name+ ".pkl")
    
    generated_configs = []
    generated_states = []

    os.makedirs("configs", exist_ok=True)
    num_objs = {"Tshirt": 44, "Trousers":34}

    pyflex.init(True, True, 720, 720)
    if args.cloth_type == "Square":
        for tqdm_i in tqdm(range(args.num_configs)):
            cloth_dim = random.randint(args.min_length, args.max_length)
            generated_config, generated_state = generate_square_configs(cloth_dim, cloth_dim)
            generated_configs.append(generated_config)
            generated_states.append(generated_state)
        if args.crumpled:
            print("Make cloth crumpled. Starting...")
            generated_configs, generated_states = generate_crumpled(cloth3d=False, configs=generated_configs, states=generated_states)

    elif args.cloth_type == "Rectangular":
        for tqdm_i in tqdm(range(args.num_configs)):
            cloth_dimx = random.randint(args.min_length, args.max_length)
            cloth_dimy = int(np.random.uniform(0.7, 0.9) * cloth_dimx)
            # cloth_dimy = int(0.8 * cloth_dimx)
            generated_config, generated_state = generate_square_configs(cloth_dimx, cloth_dimy)
            generated_configs.append(generated_config)
            generated_states.append(generated_state)
        if args.crumpled:
            print("Make cloth crumpled. Starting...")
            generated_configs, generated_states = generate_crumpled(cloth3d=False, configs=generated_configs, states=generated_states)
        
    else:
        with open(os.path.join(os.getenv("CLOTH3D_PATH"), "keypoints", args.cloth_type + "_keypoints.pkl"), "rb") as f:
            keypoints = pickle.load(f)["keypoints"]
        generated_keypoints = []
        for tqdm_i in tqdm(range(args.num_configs)):
            cloth_index = random.randint(0, num_objs[args.cloth_type] - 1)
            generated_config, generated_state = generate_cloth3d_configs(args.cloth_type, cloth_index)
            generated_configs.append(generated_config)
            generated_states.append(generated_state)
            generated_keypoints.append(keypoints[cloth_index])
        if args.crumpled:
            print("Make cloth crumpled. Starting...")
            generated_configs, generated_states = generate_crumpled(cloth3d=True, configs=generated_configs, states=generated_states)


    data = {"configs": generated_configs, "states": generated_states}
    if args.cloth_type != "Square" and args.cloth_type != "Rectangular":
        data["keypoints"] = generated_keypoints

    with open(save_file, "wb+") as f:
        pickle.dump(data, f)
