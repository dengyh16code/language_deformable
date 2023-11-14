import os
import pickle
import imageio
import numpy as np
import argparse

Done = np.array([0, 0])

def create_dataset(root, tasks, save_path, with_done, use_rgb, n_demos):
    depths = []
    picks = []
    places = []
    instructions = []
    success = []
    rgbs = []
    total_num = 0
    each_task_num = 0
    seen_num = 0
    
    if "All" in tasks:
        tasks = os.listdir(root)
        print("Load All Tasks: ", tasks)
    
    for task in tasks:
        each_task_num = 0
        task_path = os.path.join(root, task)
        trajs = os.listdir(task_path)
        for traj in trajs:
            if each_task_num >= n_demos:
                break 
            traj_path = os.path.join(task_path, traj)
            # actions & instructions
            with open(os.path.join(traj_path, "info.pkl"), "rb") as f:
                data = pickle.load(f)
                pick_pixels = data["pick"]
                place_pixels = data["place"]
                langs = data["instruction"]
                prims = data["primitive"]
                unseens = data["unseen_flags"]
            num_actions = len(pick_pixels)
            total_num += num_actions
            each_task_num += 1
            depth_path = os.path.join(task_path, traj, "depth")
            rgb_path = os.path.join(task_path, traj, "rgb")

            if not with_done:
                i = 0
                while i < num_actions:
                    unseen = unseens[i]
                    if not unseen:
                        seen_num += 1
                        # insert actions & instructions
                        picks.append(pick_pixels[i])
                        places.append(place_pixels[i])
                        instructions.append(langs[i])
                        success.append(0)

                        # observations
                        depths.append(imageio.imread(os.path.join(depth_path, str(i) + ".png")))
                        if use_rgb:
                            rgbs.append(imageio.imread(os.path.join(rgb_path, str(i) + ".png")))

                    i = i+1 #next step
                      
            else:
                i = 0
                while i < num_actions:
                    prim = prims[i]
                    unseen = unseens[i]
                    if prim == "single":
                        if not unseen:
                            seen_num += 1
                            # insert actions & instructions
                            picks.append(pick_pixels[i])
                            places.append(place_pixels[i])
                            instructions.append(langs[i])
                            success.append(0)
                            
                            picks.append(Done)
                            places.append(Done)
                            instructions.append(langs[i])
                            success.append(1)
                            
                            # observations
                            depths.append(imageio.imread(os.path.join(depth_path, str(i) + ".png")))
                            depths.append(imageio.imread(os.path.join(depth_path, str(i+1) + ".png")))
                            if use_rgb:
                                rgbs.append(imageio.imread(os.path.join(rgb_path, str(i) + ".png")))
                                rgbs.append(imageio.imread(os.path.join(rgb_path, str(i+1) + ".png")))

                        i = i+1
                        
                    elif prim == "multi":
                            # insert actions & instructions
                        if not unseen:
                            seen_num += 2
                            picks.append(pick_pixels[i])
                            places.append(place_pixels[i])
                            instructions.append(langs[i])
                            success.append(0)

                            picks.append(pick_pixels[i+1])
                            places.append(place_pixels[i+1])
                            instructions.append(langs[i+1])
                            success.append(0)

                            picks.append(Done)
                            places.append(Done)
                            instructions.append(langs[i+1])
                            success.append(1)

                            # observations
                            depths.append(imageio.imread(os.path.join(depth_path, str(i) + ".png")))
                            depths.append(imageio.imread(os.path.join(depth_path, str(i+1) + ".png")))
                            depths.append(imageio.imread(os.path.join(depth_path, str(i+2) + ".png")))
                            if use_rgb:
                                rgbs.append(imageio.imread(os.path.join(rgb_path, str(i) + ".png")))
                                rgbs.append(imageio.imread(os.path.join(rgb_path, str(i+1) + ".png")))
                                rgbs.append(imageio.imread(os.path.join(rgb_path, str(i+2) + ".png")))

                        i = i+2
   
    assert len(depths)==len(picks)==len(places)==len(instructions)==len(success)
    print("build {} seen tasks from {} tasks".format(seen_num, total_num))
    
    # save
    dataset = {
        "depth": depths,
        "pick": picks,
        "place": places,
        "instruction": instructions,
        "success": success,
        #"primitive": primitives
    }

    if use_rgb:
        dataset.update({"rgbs": rgbs})
    with open(save_path, "wb+") as f:
        pickle.dump(dataset, f)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset")
    parser.add_argument("--tasks", type=str, help="choose single task / all task(all)")
    parser.add_argument("--with_done", action="store_true", help="choose with_success or not")
    parser.add_argument("--use_rgb", action="store_true", help="choose with inst feature or not")
    parser.add_argument("--root", type=str, default="/raw_data")
    parser.add_argument("--save_path_root", type=str, default="/data")
    parser.add_argument("--n_demos", type=int, default=100, help="num of demos")
    args = parser.parse_args()
    
    if args.tasks == "All":
        if args.with_done:
            save_path = os.path.join(args.save_path_root, "All_done_100.pkl")
        else:
            save_path = os.path.join(args.save_path_root, "All_100.pkl")
    else:
        if args.with_done:
            save_path = os.path.join(args.save_path_root, args.tasks+"_done"+".pkl")
        else:
            save_path = os.path.join(args.save_path_root, args.tasks+".pkl")
 
    tasks = [args.tasks]
    create_dataset(args.root, tasks , save_path, args.with_done, args.use_rgb, args.n_demos)




