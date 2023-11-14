import sys

sys.path.append("")
import numpy as np
import pyflex
from softgym.action_space.action_space import PickerPickPlace
from softgym.envs.flex_utils import set_cloth3d_scene, set_square_scene, get_current_covered_area
from copy import deepcopy
import cv2
import imageio
from tqdm import tqdm

class ClothEnv:
    def __init__(self, 
                gui=False, 
                dump_visualizations=False, 
                cloth3d=True, 
                pick_speed=5e-3, 
                move_speed=5e-3, 
                place_speed=5e-3, 
                lift_height=0.1,
                fling_speed=8e-3,
                render_dim=224, 
                particle_radius=0.00625):
        
        # environment state variables
        self.grasp_states = [False, False]
        self.particle_radius = particle_radius
        self.image_dim = render_dim

        # visualizations
        self.gui = gui
        self.dump_visualizations = dump_visualizations
        self.gui_render_freq = 2
        self.gui_step = 0

        # setup env
        self.cloth3d = cloth3d
        self.setup_env()

        # primitives parameters
        self.grasp_height = self.action_tool.picker_radius
        self.default_speed = 1e-2
        self.reset_pos = [[0.5, 0.2, 0.5], [-0.5, 0.2, 0.5]]
        self.default_pos = [-0.5, 0.2, 0.5]
        self.pick_speed = pick_speed
        self.move_speed = move_speed
        self.place_speed = place_speed
        self.lift_height = lift_height
        self.fling_speed = fling_speed

    def setup_env(self):
        pyflex.init(not self.gui, True, 720, 720)
        self.action_tool = PickerPickPlace(
            num_picker=2,
            particle_radius=self.particle_radius,
            picker_threshold=0.005,
            picker_low=(-10.0, 0.0, -10.0),
            picker_high=(10.0, 10.0, 10.0),
        )
        if self.dump_visualizations:
            self.frames = []

    def reset(self, config, state, **kwargs):
        self.current_config = deepcopy(config)
        if self.cloth3d:
            set_cloth3d_scene(config=config, state=state)
        else:
            set_square_scene(config=config, state=state)
        self.camera_params = deepcopy(state["camera_params"])

        self.action_tool.reset(self.reset_pos[0])
        self.step_simulation()
        self.set_grasp(False)
        if self.dump_visualizations:
            self.frames = []

        if bool(kwargs):
            for key, val in kwargs.items():
                if key in ["pick_speed", "move_speed", "place_speed", "lift_height", "fling_speed"]:
                    exec(f"self.{key} = {str(val)}")
        
        self.max_area = state["max_area"]

    def step_simulation(self):
        pyflex.step()
        if self.gui and self.gui_step % self.gui_render_freq == 0:
            pyflex.render()
        self.gui_step += 1

    def set_grasp(self, grasp):
        self.grasp_states = [grasp] * len(self.grasp_states)

    def render_image(self):
        rgb, depth = pyflex.render()
        rgb = rgb.reshape((720, 720, 4))[::-1, :, :3]
        depth = depth.reshape((720, 720))[::-1]
        rgb = cv2.resize(rgb, (self.image_dim, self.image_dim), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (self.image_dim, self.image_dim), interpolation=cv2.INTER_LINEAR)
        return rgb, depth

    def render_gif(self, path):
        with imageio.get_writer(path, mode="I", fps=30) as writer:
            for frame in tqdm(self.frames):
                writer.append_data(frame)

    #################################################
    ######################Picker#####################
    #################################################
    def movep(self, pos, speed=None, limit=1000, min_steps=None, eps=1e-4):
        if speed is None:
            speed = 0.1
        target_pos = np.array(pos)
        for step in range(limit):
            curr_pos = self.action_tool._get_pos()[0]
            deltas = [(targ - curr) for targ, curr in zip(target_pos, curr_pos)]
            dists = [np.linalg.norm(delta) for delta in deltas]
            if all([dist < eps for dist in dists]) and\
                    (min_steps is None or step > min_steps):
                return
            action = []
            for targ, curr, delta, dist, gs in zip(target_pos, curr_pos, deltas, dists, self.grasp_states):
                if dist < speed:
                    action.extend([*targ, float(gs)])
                else:
                    delta = delta/dist
                    action.extend([*(curr+delta*speed), float(gs)])
    
            action = np.array(action)
            self.action_tool.step(action, step_sim_fn=self.step_simulation)
            if self.dump_visualizations:
                self.frames.append(self.render_image()[0])

    # single arm primitive, default use picker1 for manipulation
    def pick_and_place_single(self, pick_pos, place_pos):
        pick_pos[1] = self.grasp_height
        place_pos[1] = self.grasp_height

        prepick_pos = pick_pos.copy()
        prepick_pos[1] = self.lift_height

        preplace_pos = place_pos.copy()
        preplace_pos[1] = self.lift_height

        # execute action
        self.movep([prepick_pos, self.default_pos], speed=0.5)
        self.movep([pick_pos, self.default_pos], speed=0.005) 
        self.set_grasp(True)
        self.movep([prepick_pos, self.default_pos], speed=self.pick_speed)
        self.movep([preplace_pos, self.default_pos], speed=self.move_speed)
        self.movep([place_pos, self.default_pos], speed=self.place_speed)
        self.set_grasp(False)
        self.movep([preplace_pos, self.default_pos], speed=0.5)

        # reset
        self.movep(self.reset_pos, speed=0.5)

    # pick and drop
    def pick_and_drop(self, pick_pos):
        pick_pos[1] = self.grasp_height
        prepick_pos = pick_pos.copy()
        prepick_pos[1] = self.lift_height

        # execute action
        self.movep([prepick_pos, self.default_pos], speed=0.5)
        self.movep([pick_pos, self.default_pos], speed=0.005) 
        self.set_grasp(True)
        self.movep([prepick_pos, self.default_pos], speed=self.pick_speed)
        self.set_grasp(False)

        # reset
        self.movep(self.reset_pos, speed=0.5)

    # dual arm primitive
    def pick_and_place_dual(self, pick_pos_left, place_pos_left, pick_pos_right, place_pos_right):
        pick_pos_left[1] = self.grasp_height
        place_pos_left[1] = self.grasp_height        
        pick_pos_right[1] = self.grasp_height
        place_pos_right[1] = self.grasp_height

        prepick_pos_left = pick_pos_left.copy()
        prepick_pos_left[1] = self.lift_height
        prepick_pos_right = pick_pos_right.copy()
        prepick_pos_right[1] = self.lift_height

        preplace_pos_left = place_pos_left.copy()
        preplace_pos_left[1] = self.lift_height
        preplace_pos_right = place_pos_right.copy()
        preplace_pos_right[1] = self.lift_height

        # execute action
        self.movep([prepick_pos_left, prepick_pos_right], speed=0.5)
        self.movep([pick_pos_left, pick_pos_right], speed=0.005) 
        self.set_grasp(True)
        self.movep([prepick_pos_left, prepick_pos_right], speed=self.pick_speed)
        self.movep([preplace_pos_left, preplace_pos_right], speed=self.move_speed)
        self.movep([place_pos_left, place_pos_right], speed=self.place_speed)
        self.set_grasp(False)
        self.movep([preplace_pos_left, preplace_pos_right], speed=0.5)

        # reset
        self.movep(self.reset_pos, speed=0.5)

    def pick_and_fling(self, pick_pos_left, pick_pos_right):
        pick_pos_left[1] = self.grasp_height
        pick_pos_right[1] = self.grasp_height

        prepick_pos_left = pick_pos_left.copy()
        prepick_pos_left[1] = self.lift_height

        prepick_pos_right = pick_pos_right.copy()
        prepick_pos_right[1] = self.lift_height

        # grasp distance
        dist = np.linalg.norm(np.array(prepick_pos_left) - np.array(prepick_pos_right))
        
        # pick cloth
        self.movep([prepick_pos_left, prepick_pos_right])
        self.movep([pick_pos_left, pick_pos_right])
        self.set_grasp(True)

        # prelift & stretch
        self.movep([[-dist / 2, 0.3, -0.3], [dist / 2, 0.3, -0.3]], speed=5e-3)
        if not self.is_cloth_grasped():
            return False
        dist = self.stretch_cloth(grasp_dist=dist, max_grasp_dist=0.4, fling_height=0.5)

        # lift
        fling_height = self.lift_cloth(grasp_dist=dist, fling_height=0.5)
        
        # fling
        self.fling(dist=dist, fling_height=fling_height, fling_speed=self.fling_speed)

        # reset
        self.movep(self.reset_pos, speed=0.5)


    def fling(self, dist, fling_height, fling_speed):
        # fling
        self.movep([[-dist/2, fling_height, -0.2],
                    [dist/2, fling_height, -0.2]], speed=fling_speed)
        self.movep([[-dist/2, fling_height, 0.2],
                    [dist/2, fling_height, 0.2]], speed=fling_speed)
        self.movep([[-dist/2, fling_height, 0.2],
                    [dist/2, fling_height, 0.2]], speed=1e-2, min_steps=4)
        
        # lower & flatten
        self.movep([[-dist/2, self.grasp_height*2, 0.2],
                    [dist/2, self.grasp_height*2, 0.2]], speed=fling_speed)

        self.movep([[-dist/2, self.grasp_height, 0],
                    [dist/2, self.grasp_height, 0]], speed=fling_speed)
        
        self.movep([[-dist/2, self.grasp_height, -0.2],
                    [dist/2, self.grasp_height, -0.2]], speed=5e-3)
        
        # release
        self.set_grasp(False)

        if self.dump_visualizations:
            self.movep(
                [[-dist/2, self.grasp_height*2, -0.2],
                 [dist/2, self.grasp_height*2, -0.2]], min_steps=10)

    def stretch_cloth(self, grasp_dist, fling_height=0.7, max_grasp_dist=0.7, increment_step=0.02):
        # lift cloth in the air
        left, right = self.action_tool._get_pos()[0]
        left[1] = fling_height
        right[1] = fling_height
        midpoint = (left + right)/2
        direction = left - right
        direction = direction / np.linalg.norm(direction)
        self.movep([left, right], speed=5e-4, min_steps=20)
        stable_steps = 0
        cloth_midpoint = 1e2
        while True:
            positions = pyflex.get_positions().reshape((-1, 4))[:, :3]
            # get midpoints
            high_positions = positions[positions[:, 1] > fling_height-0.1, ...]
            if (high_positions[:, 0] < 0).all() or \
                    (high_positions[:, 0] > 0).all():
                # single grasp
                return grasp_dist
            positions = [p for p in positions]
            positions.sort(
                key=lambda pos: np.linalg.norm(pos[[0, 2]]-midpoint[[0, 2]]))
            new_cloth_midpoint = positions[0]
            stable = np.linalg.norm(
                new_cloth_midpoint - cloth_midpoint) < 1.5e-2
            if stable:
                stable_steps += 1
            else:
                stable_steps = 0
            stretched = stable_steps > 2
            if stretched:
                return grasp_dist
            cloth_midpoint = new_cloth_midpoint
            grasp_dist += increment_step
            left = midpoint + direction*grasp_dist/2
            right = midpoint - direction*grasp_dist/2
            self.movep([left, right], speed=5e-4)
            if grasp_dist > max_grasp_dist:
                return max_grasp_dist


    def lift_cloth(self, grasp_dist, fling_height: float = 0.7, increment_step: float = 0.05, max_height=0.7):
        while True:
            positions = pyflex.get_positions().reshape((-1, 4))[:, :3]
            heights = positions[:, 1]
            if heights.min() > 0.02:
                return fling_height
            fling_height += increment_step
            self.movep([[-grasp_dist/2, fling_height, -0.3],
                        [grasp_dist/2, fling_height, -0.3]], speed=1e-3)
            if fling_height >= max_height:
                return fling_height

    #################################################
    ###################Ground Truth##################
    #################################################
    # square cloth index looks like the following:
    # 0, 1, ..., cloth_xdim -1
    # ...
    # cloth_xdim * (cloth_ydim -1 ), ..., cloth_xdim * cloth_ydim -1

    # Cloth Keypoints are defined:
    #  0  1  2
    #  3  4  5
    #  6  7  8
    def get_square_keypoints_idx(self):
        """The keypoints are defined as the four corner points of the cloth"""
        dimx, dimy = self.current_config["ClothSize"]
        idx0 = 0
        idx1 = int((dimx - 1) / 2)
        idx2 = dimx - 1
        idx3 = int((dimy - 1) / 2) * dimx
        idx4 = int((dimy - 1) / 2) * dimx + int((dimx - 1) / 2)
        idx5 = int((dimy - 1) / 2) * dimx + dimx - 1
        idx6 = dimx * (dimy - 1)
        idx7 = dimx * (dimy - 1) + int((dimx - 1) / 2)
        idx8 = dimx * dimy - 1
        return [idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7, idx8]

    def get_keypoints(self,keypoints_index):
        particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3]
        keypoint_pos = particle_pos[keypoints_index, :3]
        return keypoint_pos

    def is_cloth_grasped(self):
        positions = pyflex.get_positions().reshape((-1, 4))
        positions = positions[:, :3]
        heights = positions[:, 1]
        return heights.max() > 0.2
