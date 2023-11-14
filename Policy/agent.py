from utils.build_model import build_model, build_predictor
import torch
from Model.CLIP import clip
import matplotlib.pyplot as plt
import numpy as np
from utils.visual import get_pixel_coord_from_world, nearest_to_mask
from utils.graph import build_graph, get_sampled_pc
from torch_geometric.data import Data

CAMERA_PARAMS = {
    "default_camera": {
        "pos": np.array([0.0, 1.0, 0.0]),
        "angle": np.array([0.0, -1.57079633, 0.0]),
        "width": 720,
        "height": 720,
    }
}


def get_mask(depth):
    mask = depth.copy()
    mask[mask > 0.996] = 0
    mask[mask != 0] = 1
    return mask


def preprocess(depth):
    mask = get_mask(depth)
    depth = depth * mask
    return depth


def make_gaussmap(point, img_size, sigma=5):
    center_x = round(point[0])
    center_y = round(point[1])
    xy_grid = np.arange(0, img_size)
    [x, y] = np.meshgrid(xy_grid, xy_grid)
    dist = (x - center_x) ** 2 + (y - center_y) ** 2
    gauss_map = np.exp(-dist / (2 * sigma * sigma))
    return gauss_map


class Agent:
    def __init__(self, configs):
        self.configs = configs
        self.model = build_model(self.configs)
        self.model.eval()
        self.success_predictor = build_predictor(self.configs)
        self.success_predictor.eval()

    def load(self, agent_model_path, predictor_model_path):
        self.model.load_model(agent_model_path)
        self.success_predictor.load_model(predictor_model_path)

    def get_action(self):
        raise NotImplementedError()


class AgentGraph(Agent):
    def __init__(self, configs):
        super().__init__(configs)

    def get_action(self, text, depth):
        # text
        text = clip.tokenize(text)
        # graph
        depth_ori = depth.copy()
        sampled_pc = get_sampled_pc(
            depth=depth_ori,
            voxel_size=self.configs["voxel_size"],
            K=self.configs["num_nodes"],
            camera_params=CAMERA_PARAMS,
        )
        graph_data = build_graph(sampled_pc, self.configs["neighbor_radius"])
        graph_data = Data.from_dict(graph_data)
        graph_data["batch"] = torch.zeros(graph_data["x"].size(0), dtype=torch.long, device="cuda")
        graph_data["u"] = torch.zeros([1, 128], device="cuda")

        text, graph_data = (
            text.to(self.configs["device"]),
            graph_data.to(self.configs["device"]),
        )
        with torch.no_grad():
            pick_prob, place_prob, all_head = self.model(text, graph_data)

        # pick
        pick_node_index = torch.argmax(pick_prob)
        pick_pos = sampled_pc[pick_node_index]

        # place
        place_node_index = torch.argmax(place_prob)
        place_pos = sampled_pc[place_node_index]

        # success
        success_predicton_val = self.success_predictor(all_head)
        success_predicton = success_predicton_val > 0
        # print("predict val: {}, prediction: {}".format(success_predicton_val.item(), success_predicton))

        pick_pixel = get_pixel_coord_from_world(pick_pos, depth.shape, CAMERA_PARAMS)
        place_pixel = get_pixel_coord_from_world(place_pos, depth.shape, CAMERA_PARAMS)
        plt.imshow(depth_ori)
        plt.plot(pick_pixel[0], pick_pixel[1], "ro")
        plt.plot(place_pixel[0], place_pixel[1], "bo")
        plt.savefig("test/evaltest.png")
        plt.close()

        return pick_pos, place_pos, success_predicton, pick_pixel, place_pixel


class AgentDepthGraph(Agent):
    def __init__(self, configs):
        super().__init__(configs)
        print("Agent: DepthGraph")

    def get_action(self, text, depth):
        # text
        text = clip.tokenize(text)
        # graph
        depth_ori = depth.copy()
        sampled_pc = get_sampled_pc(
            depth=depth_ori,
            voxel_size=self.configs["voxel_size"],
            K=self.configs["num_nodes"],
            camera_params=CAMERA_PARAMS,
        )
        graph_data = build_graph(sampled_pc, self.configs["neighbor_radius"])
        graph_data = Data.from_dict(graph_data)
        graph_data["batch"] = torch.zeros(graph_data["x"].size(0), dtype=torch.long, device="cuda")
        graph_data["u"] = torch.zeros([1, 128], device="cuda")

        depth_masked = preprocess(depth)
        depth_masked = torch.FloatTensor(depth_masked).unsqueeze(0).unsqueeze(0)
        depth_masked, text, graph_data = (
            depth_masked.to(self.configs["device"]),
            text.to(self.configs["device"]),
            graph_data.to(self.configs["device"]),
        )
        with torch.no_grad():
            pick_probs, pred_heatmaps, all_heads = self.model(text, depth_masked, graph_data)

        # pick
        pick_node_index = torch.argmax(pick_probs)
        pick_pos = sampled_pc[pick_node_index]
        pick_pixel = get_pixel_coord_from_world(pick_pos, depth.shape, CAMERA_PARAMS)

        # place
        placemap = torch.sigmoid(pred_heatmaps)
        placemap = placemap.detach().cpu().numpy()
        place_pixel = np.array(np.unravel_index(placemap.argmax(), placemap.shape))
        place_pixel[0], place_pixel[1] = place_pixel[1], place_pixel[0]

        # success
        success_prediction_val = self.success_predictor(all_heads)
        success_prediction = success_prediction_val > 0
        #print("predict val: {}, prediction: {}".format(success_prediction_val.item(), success_prediction))

        # _, axes = plt.subplots(1, 2)
        # axes[0].imshow(depth_ori)
        # axes[0].plot(pick_pixel[0], pick_pixel[1], "ro")
        # axes[0].plot(place_pixel[0], place_pixel[1], "bo")
        # axes[1].imshow(placemap)
        # plt.savefig("test/evaltest.png")

        return pick_pixel, place_pixel, success_prediction


class AgentDepth(Agent):
    def __init__(self, configs):
        super().__init__(configs)
        print("Agent: DepthOnly")

    def get_action(self, text, depth):
        text = clip.tokenize(text)
        depth_ori = depth.copy()
        depth_masked = preprocess(depth)
        depth_masked = torch.FloatTensor(depth_masked).unsqueeze(0).unsqueeze(0)
        depth_masked, text = (depth_masked.to(self.configs["device"]), text.to(self.configs["device"]))
        with torch.no_grad():
            pred_pick_heatmap, pred_place_heatmap, all_heads = self.model(text, depth_masked)

        # pick
        pickmap = torch.sigmoid(pred_pick_heatmap)
        pickmap = pickmap.detach().cpu().numpy()
        pick_pixel = np.array(np.unravel_index(pickmap.argmax(), pickmap.shape))
        mask = get_mask(depth_ori)
        pick_pixel_mask = nearest_to_mask(pick_pixel[0], pick_pixel[1], mask)
        pick_pixel[0], pick_pixel[1] = pick_pixel_mask[1], pick_pixel_mask[0]

        # place
        placemap = torch.sigmoid(pred_place_heatmap)
        placemap = placemap.detach().cpu().numpy()
        place_pixel = np.array(np.unravel_index(placemap.argmax(), placemap.shape))
        place_pixel[0], place_pixel[1] = place_pixel[1], place_pixel[0]

        # success
        success_prediction_val = self.success_predictor(all_heads)
        success_prediction = success_prediction_val > 0
        #print("predict val: {}, prediction: {}".format(success_prediction_val.item(), success_prediction))

        # _, axes = plt.subplots(1, 4)
        # axes[0].imshow(depth_ori)
        # axes[0].plot(pick_pixel[0], pick_pixel[1], "ro")
        # axes[0].plot(place_pixel[0], place_pixel[1], "bo")
        # axes[1].imshow(pickmap)
        # axes[1].plot(pick_pixel[0], pick_pixel[1], "ro")
        # axes[2].imshow(placemap)
        # axes[2].plot(place_pixel[0], place_pixel[1], "ro")
        # axes[3].imshow(depth_masked.squeeze().detach().cpu().numpy())
        # plt.savefig("test/test.png")

        return pick_pixel, place_pixel, success_prediction


Agents = {
    "graphdepth": AgentDepthGraph,
    "depthonly": AgentDepth,
}
