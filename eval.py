import argparse
from utils.build_model import get_configs
from softgym.envs.cloth_env import ClothEnv
from utils.visual import action_viz, get_pixel_coord_from_world, get_world_coord_from_pixel
import os
from Policy.demonstrator import Demonstrator
import pickle
import numpy as np
from tqdm import tqdm
from softgym.envs.flex_utils import move_to_pos, rotate_particles
import pyflex
from Policy.agent import Agents
import imageio

def initial_state(random_angle):
    max_wait_step = 300  # Maximum number of steps waiting for the cloth to stablize
    stable_vel_threshold = 0.2  # Cloth stable when all particles' vel are smaller than this
    rotate_particles([0, random_angle, 0])
    for _ in range(max_wait_step):
        pyflex.step()
        curr_vel = pyflex.get_velocities()
        if np.alltrue(np.abs(curr_vel) < stable_vel_threshold):
            break


def main():
    parser = argparse.ArgumentParser(description="Evaluate")
    parser.add_argument("--task", type=str, default="SquareTriangle", help="choose task")
    parser.add_argument("--cloth_type", type=str, default="Square", help="choose square cloth or cloth3d cloth")
    parser.add_argument("--gui", action="store_true", help="run with/without gui")
    parser.add_argument("--randomize_pose", action="store_true", help="for squre cloth only")
    parser.add_argument("--model", type=str, help="Evaluate which model")
    parser.add_argument("--agent_model", type=str, help="Evaluate which trained agent model")
    parser.add_argument("--predictor_model", type=str, help="Evaluate which trained predictor model")
    parser.add_argument("--num_eval", type=int, default=50, help="number of eval instances")
    args = parser.parse_args()

    cached_path = os.path.join("configs", args.cloth_type + ".pkl")
    save_root = "eval"
    cloth3d = False if (args.cloth_type == "Square" or args.cloth_type == "Rectangular") else True
    model_config_path = os.path.join("train", "train_configs", args.model + ".yaml")
    configs = get_configs(model_config_path)
    agent_trained_model_path = os.path.join(
        "train",
        "trained_models",
        configs["save_model_name"],
        "model",
        args.agent_model + ".pth",
    )
    predictor_trained_model_path = os.path.join(
        "train",
        "trained_models",
        configs["save_model_name"],
        "predictor",
        args.predictor_model + ".pth",
    )
    task = args.task

    # set task, env & agent
    demonstrator = Demonstrator[task]()
    agent = Agents[configs["type"]](configs)
    agent.load(agent_trained_model_path, predictor_trained_model_path)
    env = ClothEnv(
        gui=args.gui,
        cloth3d=cloth3d,
        dump_visualizations=False,
        pick_speed=demonstrator.pick_speed,
        move_speed=demonstrator.move_speed,
        place_speed=demonstrator.place_speed,
        lift_height=demonstrator.lift_height,
    )
    # load configs
    with open(cached_path, "rb") as f:
        config_data = pickle.load(f)
    cached_configs = config_data["configs"]
    cached_states = config_data["states"]
    print("load {} configs from {}".format(len(cached_configs), cached_path))

    # file
    if args.model == "All":
        task_root = os.path.join(save_root, "Multi", task)
    else:
        task_root = os.path.join(save_root, "single", task)
    os.makedirs(task_root, exist_ok=True)
    dirs = os.listdir(task_root)
    if dirs == []:
        max_index = 0
    else:
        existed_index = np.array(dirs).astype(np.int)
        max_index = existed_index.max() + 1

    for i in tqdm(range(args.num_eval)):
        rand_idx = np.random.randint(len(cached_configs))
        config = cached_configs[rand_idx]
        state = cached_states[rand_idx]
           
        if task == "StraightFold":
            #fix simulation bugs of square
            random_angle = np.random.uniform(-80, 80)
        elif cloth3d:           
            random_angle = np.random.uniform(-40, 40)
        else:
            random_angle = np.random.uniform(0, 40)

        
        # save file dir
        rgb_folder = os.path.join(task_root, str(max_index + i), "rgb")
        depth_folder = os.path.join(task_root, str(max_index + i), "depth")
        save_folder_viz = os.path.join(task_root, str(max_index + i), "viz")     
        save_folder = os.path.join(task_root, str(max_index + i))
        os.makedirs(rgb_folder, exist_ok=True)
        os.makedirs(depth_folder, exist_ok=True)
        os.makedirs(save_folder_viz, exist_ok=True)
        
        # reset env
        env.reset(config=config, state=state)
        if cloth3d:
            keypoints_index = config_data["keypoints"][rand_idx]
        else:
            keypoints_index = env.get_square_keypoints_idx()
        if args.randomize_pose:
            initial_state(random_angle)
        
        # oracle excuted
        if task == "StraightFold":
            eval_seen_instructions,  eval_unseen_instructions, eval_unseen_tasks = demonstrator.get_eval_instruction(random_angle)
        else: 
            eval_seen_instructions,  eval_unseen_instructions, eval_unseen_tasks = demonstrator.get_eval_instruction() 
        eval_datas = [eval_seen_instructions, eval_unseen_instructions,eval_unseen_tasks]
        eval_name_list = ["si","usi","ut"]

        eval_results =  {
        "si":[],
        "usi":[],
        "ut":[],
        } 

        eval_flags =  {
        "si":[],
        "usi":[],
        "ut":[],
        } 
               
        for eval_index in range(3):  #eval seen instructions, unseen instructions, unseent task
            eval_data = eval_datas[eval_index]
            eval_name = eval_name_list[eval_index]
            print("eval_stage:", eval_name)
            pick_idxs = eval_data["pick"]
            place_idxs = eval_data["place"]
            gammas = eval_data["gammas"]
            unseen_flags = eval_data["flags"]
            instructions = eval_data["instructions"]
            oracle_results = []
            model_results = []
                                                   
            # reset env
            env.reset(config=config, state=state)
            if args.randomize_pose:
                initial_state(random_angle)
            #oracle
            action_index = 0      
            for pick_idx, place_idx, gamma in zip(pick_idxs, place_idxs, gammas):      
                keypoints_pos = env.get_keypoints(keypoints_index)
                pick_pos = keypoints_pos[pick_idx]
                place_pos = keypoints_pos[place_idx]
                place_pos = pick_pos + gamma * (place_pos - pick_pos)
                env.pick_and_place_single(pick_pos.copy(), place_pos.copy()) 
                action_index += 1
                # save
                rgb, _ = env.render_image()
                imageio.imwrite(os.path.join(rgb_folder, eval_name + "_ori_" + str(action_index) + ".png"), rgb)
                particle_pos = pyflex.get_positions().reshape(-1,4)[:,:3]
                oracle_results.append(particle_pos)

            # reset env
            env.reset(config=config, state=state)
            if args.randomize_pose:
                initial_state(random_angle)

            # model excuted 
            pick_pixels = []
            place_pixels = []
            rgbs = []

            # initial observation
            action_index = 0
            rgb, depth = env.render_image()
            depth_save = depth.copy() * 255
            depth_save = depth_save.astype(np.uint8)
            imageio.imwrite(os.path.join(depth_folder, eval_name+"_"+str(action_index) + ".png"), depth_save)
            imageio.imwrite(os.path.join(rgb_folder, eval_name+"_"+str(action_index) + ".png"), rgb)
            rgbs.append(rgb)

            # test begin
            for pick_idx, place_idx, gamma, instruction, unseen_flag in zip(pick_idxs, place_idxs, gammas, instructions, unseen_flags):     
                print("task: " + instruction)
                if eval_index < 2: # eval seen instructions, unseen instructions,
                    if unseen_flag == 1: #oracle execute action
                        keypoints_pos = env.get_keypoints(keypoints_index)
                        pick_pos = keypoints_pos[pick_idx]
                        place_pos = keypoints_pos[place_idx]
                        place_pos = pick_pos + gamma * (place_pos - pick_pos)
                        pick_pixel = get_pixel_coord_from_world(pick_pos, depth.shape, env.camera_params)
                        place_pixel = get_pixel_coord_from_world(place_pos, depth.shape, env.camera_params)
                    else: #model execute action
                        pick_pixel, place_pixel, success_prediction = agent.get_action(instruction, depth)
                        pick_pos = get_world_coord_from_pixel(pick_pixel, depth, env.camera_params)
                        place_pos = get_world_coord_from_pixel(place_pixel, depth, env.camera_params)
                else: #eval unseen tasks
                    if unseen_flag == 0: #oracle execute action
                        keypoints_pos = env.get_keypoints(keypoints_index)
                        pick_pos = keypoints_pos[pick_idx]
                        place_pos = keypoints_pos[place_idx]
                        place_pos = pick_pos + gamma * (place_pos - pick_pos)
                        pick_pixel = get_pixel_coord_from_world(pick_pos, depth.shape, env.camera_params)
                        place_pixel = get_pixel_coord_from_world(place_pos, depth.shape, env.camera_params)
                    else: #model execute action
                        pick_pixel, place_pixel, success_prediction = agent.get_action(instruction, depth)
                        pick_pos = get_world_coord_from_pixel(pick_pixel, depth, env.camera_params)
                        place_pos = get_world_coord_from_pixel(place_pixel, depth, env.camera_params)

                env.pick_and_place_single(pick_pos.copy(), place_pos.copy()) #take action

                # render & update frames & save
                action_index += 1
                rgb, depth = env.render_image()
                depth_save = depth.copy() * 255
                depth_save = depth_save.astype(np.uint8)
                imageio.imwrite(os.path.join(rgb_folder, eval_name+"_"+str(action_index) + ".png"), rgb)
                imageio.imwrite(os.path.join(depth_folder, eval_name+"_"+str(action_index) + ".png"), depth_save)
                rgbs.append(rgb)
                pick_pixels.append(pick_pixel)
                place_pixels.append(place_pixel)
                particle_pos = pyflex.get_positions().reshape(-1,4)[:,:3]
                model_results.append(particle_pos) 
        
            #record results
            eval_results[eval_name]= [oracle_results,model_results]
            eval_flags[eval_name]= unseen_flags

            # action viz
            num_actions = len(pick_pixels)
            for act in range(num_actions + 1):
                if act < num_actions:
                    img = action_viz(rgbs[act], pick_pixels[act], place_pixels[act])
                else:
                    img = rgbs[act]
                imageio.imwrite(os.path.join(save_folder_viz, eval_name+"_"+str(act) + ".png"), img)
        
        #save results
        with open(os.path.join(save_folder, "resu.pkl"), "wb+") as f:
            data = {"eval_results": eval_results, 
                    "eval_flags": eval_flags}
            pickle.dump(data, f)



if __name__ == "__main__":
    main()
