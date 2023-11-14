import os
import pickle
import torch
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms.functional as F
from Model.CLIP import clip
import numpy as np
from utils.visual import get_world_coord_from_pixel
from utils.graph import build_graph, get_sampled_pc, calc_distances
from torch_geometric.data import Data

CAMERA_PARAMS = {
    "default_camera": {
        "pos": np.array([0.0, 0.65, 0.0]),
        "angle": np.array([0.0, -1.57079633, 0.0]),
        "width": 720,
        "height": 720,
    }
}


def make_gaussmap(point, img_size, sigma=5):
    center_x = round(point[0])
    center_y = round(point[1])
    xy_grid = np.arange(0, img_size)
    [x, y] = np.meshgrid(xy_grid, xy_grid)
    dist = (x - center_x) ** 2 + (y - center_y) ** 2
    gauss_map = np.exp(-dist / (2 * sigma * sigma))
    return gauss_map


def preprocess(depth):
    depth = depth / 255
    # generate a mask
    mask = depth.copy()
    mask[mask == depth.max()] = 0
    mask[mask != 0] = 1
    depth = depth * mask
    return depth


class DeformableDataset(Dataset):
    def __init__(self, configs):
        super().__init__()
        self.dataset_path = configs["dataset_path"]
        self.spatial_augment = configs["spatial_augment"]
        self.voxel_size = configs["voxel_size"]
        self.num_nodes = configs["num_nodes"]
        self.neighbor_radius = configs["neighbor_radius"]

        # load dataset
        with open(self.dataset_path, "rb") as f:
            data = pickle.load(f)

        self.depths = data["depth"]
        self.pick_pixels = data["pick"]
        self.place_pixels = data["place"]
        self.instructions = data["instruction"]
        self.success_flags = data["success"]

        self.img_size = self.depths[0].shape[0]

    def __len__(self):
        return len(self.instructions)

    def __getitem__(self, index):
        depth = preprocess(self.depths[index])
        depth = torch.FloatTensor(depth).unsqueeze(0)
        pick_pixel = self.pick_pixels[index]
        place_pixel = self.place_pixels[index]
        instruction = self.instructions[index]
        success_flag = self.success_flags[index]

        # graph
        depth_ori = self.depths[index] / 255
        sampled_pc = get_sampled_pc(
            depth=depth_ori, voxel_size=self.voxel_size, K=self.num_nodes, camera_params=CAMERA_PARAMS
        )
        graph_data = build_graph(sampled_pc, self.neighbor_radius)
        graph_data = Data.from_dict(graph_data)

        # pick point
        pick_pos = get_world_coord_from_pixel(pick_pixel, depth_ori, CAMERA_PARAMS)
        distances = calc_distances(pick_pos, sampled_pc)
        pick_point = torch.tensor(distances == np.min(distances)).float()

        # aug
        if self.spatial_augment:
            angle = np.random.randint(-5, 6)
            dx = np.random.randint(-5, 6)
            dy = np.random.randint(-5, 6)
            depth, pick_pixel, place_pixel = self.aug_spatial(depth, pick_pixel, place_pixel, angle, dx, dy)

        # pick & place heatmap
        pick_heatmap = make_gaussmap(pick_pixel, self.img_size)
        pick_heatmap = torch.FloatTensor(pick_heatmap)
        place_heatmap = make_gaussmap(place_pixel, self.img_size)
        place_heatmap = torch.FloatTensor(place_heatmap)

        # encode instruction
        instruction = clip.tokenize(instruction).squeeze()

        # success
        success_flag = torch.tensor(success_flag).float()

        return (depth, graph_data, pick_point, place_heatmap, instruction, success_flag)

    def aug_spatial(self, img, pick, place, angle, dx, dy):
        img = F.affine(img, angle=angle, translate=(dx, dy), scale=1.0, shear=0)
        pick = self.aug_pixel(pick.astype(np.float64)[None, :], -angle, dx, dy, size=self.img_size - 1)
        pick = pick.squeeze().astype(int)
        place = self.aug_pixel(place.astype(np.float64)[None, :], -angle, dx, dy, size=self.img_size - 1)
        place = place.squeeze().astype(int)
        return img, pick, place

    def aug_pixel(self, pixel, angle, dx, dy, size):
        rad = np.deg2rad(-angle)
        R = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
        pixel -= size / 2
        pixel = np.dot(R, pixel.T).T
        pixel += size / 2
        pixel[:, 0] += dx
        pixel[:, 1] += dy
        pixel = np.clip(pixel, 0, size)
        return pixel


class DepthOnlyDataset(Dataset):
    def __init__(self, configs):
        super().__init__()
        self.dataset_path = configs["dataset_path"]
        self.spatial_augment = configs["spatial_augment"]

        # load dataset
        with open(self.dataset_path, "rb") as f:
            data = pickle.load(f)

        self.depths = data["depth"]
        self.pick_pixels = data["pick"]
        self.place_pixels = data["place"]
        self.instructions = data["instruction"]
        self.success_flags = data["success"]

        self.img_size = self.depths[0].shape[0]

    def __len__(self):
        return len(self.instructions)

    def __getitem__(self, index):
        depth = preprocess(self.depths[index])
        depth = torch.FloatTensor(depth).unsqueeze(0)
        pick_pixel = self.pick_pixels[index]
        place_pixel = self.place_pixels[index]
        instruction = self.instructions[index]
        success_flag = self.success_flags[index]

        # aug
        if self.spatial_augment:
            angle = np.random.randint(-5, 6)
            dx = np.random.randint(-5, 6)
            dy = np.random.randint(-5, 6)
            depth, pick_pixel, place_pixel = self.aug_spatial(depth, pick_pixel, place_pixel, angle, dx, dy)

        # pick & place heatmap
        pick_heatmap = make_gaussmap(pick_pixel, self.img_size)
        pick_heatmap = torch.FloatTensor(pick_heatmap)
        place_heatmap = make_gaussmap(place_pixel, self.img_size)
        place_heatmap = torch.FloatTensor(place_heatmap)

        # encode instruction
        instruction = clip.tokenize(instruction).squeeze()

        # success
        success_flag = torch.tensor(success_flag).float()

        return (depth, depth, pick_heatmap, place_heatmap, instruction, success_flag)

    def aug_spatial(self, img, pick, place, angle, dx, dy):
        img = F.affine(img, angle=angle, translate=(dx, dy), scale=1.0, shear=0)
        pick = self.aug_pixel(pick.astype(np.float64)[None, :], -angle, dx, dy, size=self.img_size - 1)
        pick = pick.squeeze().astype(int)
        place = self.aug_pixel(place.astype(np.float64)[None, :], -angle, dx, dy, size=self.img_size - 1)
        place = place.squeeze().astype(int)
        return img, pick, place

    def aug_pixel(self, pixel, angle, dx, dy, size):
        rad = np.deg2rad(-angle)
        R = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
        pixel -= size / 2
        pixel = np.dot(R, pixel.T).T
        pixel += size / 2
        pixel[:, 0] += dx
        pixel[:, 1] += dy
        pixel = np.clip(pixel, 0, size)
        return pixel
