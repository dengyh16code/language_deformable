import numpy as np
import pickle
import pyflex
from softgym.envs.cloth_env import ClothEnv
from tqdm import tqdm
import imageio
import os
from Policy.demonstrator import Demonstrator
from utils.visual import get_pixel_coord_from_world, action_viz
import argparse
from softgym.envs.flex_utils import move_to_pos, rotate_particles,  get_current_covered_area


def generate_demos(cached_path, save_root, task, gui, cloth3d, randomize_pose, num_demonstrations):
    # set policy & env
    policy = Demonstrator[task]()
    env = ClothEnv(
        gui=gui,
        cloth3d=cloth3d,
        dump_visualizations=False,
        pick_speed=policy.pick_speed,
        move_speed=policy.move_speed,
        place_speed=policy.place_speed,
        lift_height=policy.lift_height,
    )

    # load configs
    with open(cached_path, "rb") as f:
        config_data = pickle.load(f)
    configs = config_data["configs"]
    states = config_data["states"]
    print("load {} configs from {}".format(len(configs), cached_path))

    # file
    task_root = os.path.join(save_root, task)
    os.makedirs(task_root, exist_ok=True)
    dirs = os.listdir(task_root)
    if dirs == []:
        max_index = 0
    else:
        existed_index = np.array(dirs).astype(np.int)
        max_index = existed_index.max() + 1

    for i in tqdm(range(num_demonstrations)):
        # reset env
        rand_idx = np.random.randint(len(configs))
        config = configs[rand_idx]
        state = states[rand_idx]
        env.reset(config=config, state=state)

        if cloth3d:
            keypoints_index = config_data["keypoints"][rand_idx]
        else:
            keypoints_index = env.get_square_keypoints_idx()

        # randomization
        if randomize_pose:
            max_wait_step = 300  # Maximum number of steps waiting for the cloth to stablize
            stable_vel_threshold = 0.2  # Cloth stable when all particles' vel are smaller than this
            if task == "StraightFold":
            #fix simulation bugs of square
                random_angle = np.random.uniform(-80, 80)
            elif cloth3d:
                random_angle = np.random.uniform(-40, 40)
            else:
                random_angle = np.random.uniform(0, 40)
            #print("random angle:",random_angle)
            random_pos_move = np.random.uniform(low=-0.02, high=0.02, size=(3,))
            rotate_particles([0, random_angle, 0])
            random_pos_move[1] = 0
            move_to_pos(random_pos_move)
            for _ in range(max_wait_step):
                pyflex.step()
                curr_vel = pyflex.get_velocities()
                if np.alltrue(np.abs(curr_vel) < stable_vel_threshold):
                    break

        # save file
        rgb_folder = os.path.join(task_root, str(max_index + i), "rgb")
        depth_folder = os.path.join(task_root, str(max_index + i), "depth")
        os.makedirs(rgb_folder, exist_ok=True)
        os.makedirs(depth_folder, exist_ok=True)
        pick_pixels = []
        place_pixels = []
        instructions = []
        rgbs = []
        primitives = []
        unseen_flags = []


        # initial observation
        action_index = 0
        rgb, depth = env.render_image()
        imageio.imwrite(os.path.join(rgb_folder, str(action_index) + ".png"), rgb)
        depth = depth * 255
        depth = depth.astype(np.uint8)
        imageio.imwrite(os.path.join(depth_folder, str(action_index) + ".png"), depth)
        rgbs.append(rgb)
        
        if task == "StraightFold":
            #fix simulation bugs of square
            pick_idxs, place_idxs, gammas, action_instructions, action_primitives, action_unseen_flags = policy.get_action_instruction(random_angle)
        else:
            pick_idxs, place_idxs, gammas, action_instructions, action_primitives, action_unseen_flags = policy.get_action_instruction()
        for pick_idx, place_idx, gamma in zip(pick_idxs, place_idxs, gammas):   
            keypoints_pos = env.get_keypoints(keypoints_index)
            pick_pos = keypoints_pos[pick_idx]
            place_pos = keypoints_pos[place_idx]
            place_pos = pick_pos + gamma * (place_pos - pick_pos)

            pick_pixel = get_pixel_coord_from_world(pick_pos, depth.shape, env.camera_params)
            place_pixel = get_pixel_coord_from_world(place_pos, depth.shape, env.camera_params)
            env.pick_and_place_single(pick_pos.copy(), place_pos.copy())            
            action_index += 1
            # save
            rgb, depth = env.render_image()
            imageio.imwrite(os.path.join(rgb_folder, str(action_index) + ".png"), rgb)
            depth = depth * 255
            depth = depth.astype(np.uint8)
            imageio.imwrite(os.path.join(depth_folder, str(action_index) + ".png"), depth)
            pick_pixels.append(pick_pixel)
            place_pixels.append(place_pixel)
            rgbs.append(rgb)
        
        primitives += action_primitives
        instructions += action_instructions
        unseen_flags += action_unseen_flags

        # print(primitives)
        # print(instructions)
        # print(unseen_flags)
        # print("...............")

        with open(os.path.join(task_root, str(max_index + i), "info.pkl"), "wb+") as f:
            data = {"pick": pick_pixels, 
                    "place": place_pixels,
                    "primitive": primitives, 
                    "instruction": instructions,
                    "unseen_flags": unseen_flags,}
            pickle.dump(data, f)


        # action viz
        save_folder_viz = os.path.join(task_root, str(max_index + i), "viz")
        os.makedirs(save_folder_viz, exist_ok=True)
        num_actions = len(pick_pixels)
        for act in range(num_actions + 1):
            if act < num_actions:
                img = action_viz(rgbs[act], pick_pixels[act], place_pixels[act])
            else:
                img = rgbs[act]
            imageio.imwrite(os.path.join(save_folder_viz, str(act) + ".png"), img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Demonstrations")
    parser.add_argument("--task", type=str, default="StraightFold", help="choose task")
    parser.add_argument("--cloth_type", type=str, default="Rectangular", help="choose square cloth or cloth3d cloth")
    parser.add_argument("--gui", action="store_true", help="run with/without gui")
    parser.add_argument("--randomize_pose", action="store_true", help="for squre cloth only")
    parser.add_argument("--num_demonstrations", type=int, default=1000, help="number of demonstrations")
    args = parser.parse_args()
    
    cached_path = os.path.join("configs", args.cloth_type + ".pkl")
    save_root = "raw_data"
    cloth3d = False if (args.cloth_type == "Square" or args.cloth_type == "Rectangular") else True
    generate_demos(cached_path, save_root, args.task, args.gui, cloth3d, args.randomize_pose, args.num_demonstrations)