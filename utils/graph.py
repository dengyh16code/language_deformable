from scipy import spatial
import numpy as np
import torch
#import pcl
import open3d as o3d
from utils.visual import get_world_coords_from_pixels


def compute_edge_attr(normalized_vox_pc, neighbor_radius):
    point_tree = spatial.cKDTree(normalized_vox_pc)
    undirected_neighbors = np.array(list(point_tree.query_pairs(neighbor_radius, p=2))).T

    if len(undirected_neighbors) > 0:
        dist_vec = normalized_vox_pc[undirected_neighbors[0, :]] - normalized_vox_pc[undirected_neighbors[1, :]]
        dist = np.linalg.norm(dist_vec, axis=1, keepdims=True)
        edge_attr = np.concatenate([dist_vec, dist], axis=1)
        edge_attr_reverse = np.concatenate([-dist_vec, dist], axis=1)

        # Generate directed edge list and corresponding edge attributes
        edges = torch.from_numpy(np.concatenate([undirected_neighbors, undirected_neighbors[::-1]], axis=1))
        edge_attr = torch.from_numpy(np.concatenate([edge_attr, edge_attr_reverse]))
    else:
        print("number of distance edges is 0! adding fake edges")
        edges = np.zeros((2, 2), dtype=np.uint8)
        edges[0][0] = 0
        edges[1][0] = 1
        edges[0][1] = 0
        edges[1][1] = 2
        edge_attr = np.zeros((2, 4), dtype=np.float32)
        edges = torch.from_numpy(edges).bool()
        edge_attr = torch.from_numpy(edge_attr)
        print("shape of edges: ", edges.shape)
        print("shape of edge_attr: ", edge_attr.shape)

    return edges, edge_attr


def build_graph(vox_pc, neighbor_radius):
    """
    Input:
    pointcloud

    Return:
    node_attr: N x (vel_history x 3)
    edges: 2 x E, the edges
    edge_attr: E x edge_feature_dim
    """
    normalized_vox_pc = vox_pc - np.mean(vox_pc, axis=0)
    node_attr = torch.from_numpy(normalized_vox_pc)
    edges, edge_attr = compute_edge_attr(normalized_vox_pc, neighbor_radius)

    return {"x": node_attr, "edge_index": edges, "edge_attr": edge_attr}


def voxelize_pointcloud(pointcloud, voxel_size):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pointcloud)
    downpcd = cloud.voxel_down_sample(voxel_size)  
    # cloud = pcl.PointCloud(pointcloud)
    # sor = cloud.make_voxel_grid_filter()
    # sor.set_leaf_size(voxel_size, voxel_size, voxel_size)   
    # pointcloud = sor.filter()
    return np.asarray(downpcd.points).astype(np.float32)


def get_mask(depth):
    mask = depth.copy()
    mask[mask > 0.996] = 0
    mask[mask != 0] = 1
    return mask


def get_sampled_pc(depth, voxel_size, K, camera_params):
    mask = get_mask(depth)
    world_coordinates = get_world_coords_from_pixels(depth, camera_params)
    world_coords = world_coordinates[:, :, :3].reshape((-1, 3))
    pointcloud = world_coords[mask.flatten() > 0].astype(np.float32)
    vox_pc = voxelize_pointcloud(pointcloud, voxel_size)
    sampled_pc = fps(vox_pc, K).astype(np.float32)

    return sampled_pc


def calc_distances(p0, points):
    return ((p0 - points) ** 2).sum(axis=1)


def fps(pts, K):
    farthest_pts = np.zeros((K, 3))
    farthest_pts[0] = pts[np.random.randint(len(pts))]
    distances = calc_distances(farthest_pts[0], pts)
    for i in range(1, K):
        farthest_pts[i] = pts[np.argmax(distances)]
        distances = np.minimum(distances, calc_distances(farthest_pts[i], pts))
    return farthest_pts
