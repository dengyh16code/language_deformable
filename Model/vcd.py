import torch
import torch_scatter
from torch_geometric.nn import MetaLayer
from scipy import spatial
import numpy as np
from torch_geometric.data import Data

# ================== Encoder ================== #
class NodeEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=128):
        super(NodeEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_size),
            torch.nn.ReLU(inplace=True),
            # torch.nn.LayerNorm(self.hidden_size),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(inplace=True),
            # torch.nn.LayerNorm(self.hidden_size),
            torch.nn.Linear(self.hidden_size, self.output_size),
        )

    def forward(self, node_state):
        out = self.model(node_state)
        return out


class EdgeEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=128):
        super(EdgeEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_size),
            torch.nn.ReLU(inplace=True),
            # torch.nn.LayerNorm(self.hidden_size),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(inplace=True),
            # torch.nn.LayerNorm(self.hidden_size),
            torch.nn.Linear(self.hidden_size, self.output_size),
        )

    def forward(self, edge_properties):
        out = self.model(edge_properties)
        return out


class Encoder(torch.nn.Module):
    def __init__(self, node_input_size, edge_input_size, hidden_size=128, output_size=128):
        super(Encoder, self).__init__()
        self.node_input_size = node_input_size
        self.edge_input_size = edge_input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.node_encoder = NodeEncoder(self.node_input_size, self.hidden_size, self.output_size)
        self.edge_encoder = EdgeEncoder(self.edge_input_size, self.hidden_size, self.output_size)

    def forward(self, node_states, edge_properties):
        node_embedding = self.node_encoder(node_states)
        edge_embedding = self.edge_encoder(edge_properties)
        return node_embedding, edge_embedding


# ================== Processor ================== #
class EdgeModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=128):
        super(EdgeModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_size),
            torch.nn.ReLU(inplace=True),
            # torch.nn.LayerNorm(self.hidden_size),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(inplace=True),
            # torch.nn.LayerNorm(self.hidden_size),
            torch.nn.Linear(self.hidden_size, self.output_size),
        )

    def forward(self, src, dest, edge_attr, u, batch):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        # u_expanded = u.expand([src.size()[0], -1])
        # model_input = torch.cat([src, dest, edge_attr, u_expanded], 1)
        # out = self.model(model_input)
        model_input = torch.cat([src, dest, edge_attr, u[batch]], 1)
        out = self.model(model_input)
        return out


class NodeModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=128):
        super(NodeModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_size),
            torch.nn.ReLU(inplace=True),
            # torch.nn.LayerNorm(self.hidden_size),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(inplace=True),
            # torch.nn.LayerNorm(self.hidden_size),
            torch.nn.Linear(self.hidden_size, self.output_size),
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        _, edge_dst = edge_index
        edge_attr_aggregated = torch_scatter.scatter_add(edge_attr, edge_dst, dim=0, dim_size=x.size(0))
        model_input = torch.cat([x, edge_attr_aggregated, u[batch]], dim=1)
        out = self.model(model_input)
        return out


class GlobalModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=128):
        super(GlobalModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_size),
            torch.nn.ReLU(inplace=True),
            # torch.nn.LayerNorm(self.hidden_size),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(inplace=True),
            # torch.nn.LayerNorm(self.hidden_size),
            torch.nn.Linear(self.hidden_size, self.output_size),
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        node_attr_mean = torch_scatter.scatter_mean(x, batch, dim=0)
        edge_attr_mean = torch_scatter.scatter_mean(edge_attr, batch[edge_index[0]], dim=0)
        model_input = torch.cat([u, node_attr_mean, edge_attr_mean], dim=1)
        out = self.model(model_input)
        assert out.shape == u.shape
        return out


class GNBlock(torch.nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=128, use_global=True, global_size=128):
        super(GNBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        if use_global:
            self.model = MetaLayer(
                EdgeModel(self.input_size[0], self.hidden_size, self.output_size),
                NodeModel(self.input_size[1], self.hidden_size, self.output_size),
                GlobalModel(self.input_size[2], self.hidden_size, global_size),
            )
        else:
            self.model = MetaLayer(
                EdgeModel(self.input_size[0], self.hidden_size, self.output_size),
                NodeModel(self.input_size[1], self.hidden_size, self.output_size),
                None,
            )

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        x, edge_attr, u = self.model(x, edge_index, edge_attr, u, batch)
        return x, edge_attr, u


class Processor(torch.nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=128, use_global=True, global_size=128, layers=10):
        """
        :param input_size: A list of size to edge model, node model and global model
        """
        super(Processor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.use_global = use_global
        self.global_size = global_size
        self.gns = torch.nn.ModuleList(
            [
                GNBlock(self.input_size, self.hidden_size, self.output_size, self.use_global, global_size=global_size)
                for _ in range(layers)
            ]
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        # def forward(self, data):
        # x, edge_index, edge_attr, u, batch = data.node_embedding, data.neighbors, data.edge_embedding, data.global_feat, data.batch
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        if len(u.shape) == 1:
            u = u[None]
        if edge_index.shape[1] < 10:
            print("--------debug info---------")
            print("small number of edges")
            print("x.shape: ", x.shape)
            print("edge_index.shape: ", edge_index.shape)
            print("edge_attr.shape: ", edge_attr.shape, flush=True)
            print("--------------------------")

        x_new, edge_attr_new, u_new = x, edge_attr, u
        for gn in self.gns:
            x_res, edge_attr_res, u_res = gn(x_new, edge_index, edge_attr_new, u_new, batch)
            x_new = x_new + x_res
            edge_attr_new = edge_attr_new + edge_attr_res
            u_new = u_new + u_res
        return x_new, edge_attr_new, u_new

# todo: uncomment
class GNN(torch.nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, proc_layer, global_size):
        super(GNN, self).__init__()
        self.use_global = True if global_size > 1 else False
        embed_dim = 128
        self.dyn_models = torch.nn.ModuleDict(
            {
                "encoder": Encoder(node_input_dim, edge_input_dim, output_size=embed_dim),
                "processor": Processor(
                    [
                        3 * embed_dim + global_size,
                        2 * embed_dim + global_size,
                        2 * embed_dim + global_size,
                    ],
                    use_global=self.use_global,
                    layers=proc_layer,
                    global_size=global_size,
                ),
            }
        )

    def forward(self, data):
        node_embedding, edge_embedding = self.dyn_models["encoder"](data["x"], data["edge_attr"])
        x_nxt, _, _ = self.dyn_models["processor"](
            node_embedding, data["edge_index"], edge_embedding, u=data["u"], batch=data["batch"]
        )

        return x_nxt

    def load_model(self, model_path,device):
        ckpt = torch.load(model_path,map_location=device)
        for k in ["encoder", "processor"]:
            self.dyn_models[k].load_state_dict(ckpt[k])
        print("Loaded trained GNN model for {}".format(model_path))


# todo: comment

# class Decoder(torch.nn.Module):
#     def __init__(self, input_size=128, hidden_size=128, output_size=3):
#         super(Decoder, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.model = torch.nn.Sequential(
#             torch.nn.Linear(self.input_size, self.hidden_size),
#             torch.nn.ReLU(inplace=True),
#             # torch.nn.LayerNorm(self.hidden_size),
#             torch.nn.Linear(self.hidden_size, self.hidden_size),
#             torch.nn.ReLU(inplace=True),
#             # torch.nn.LayerNorm(self.hidden_size),
#             torch.nn.Linear(self.hidden_size, self.output_size),
#         )

#     def forward(self, node_feat, res=None):
#         out = self.model(node_feat)
#         if res is not None:
#             out = out + res
#         return out


# class GNN(torch.nn.Module):
#     def __init__(self, node_input_dim, edge_input_dim, proc_layer, global_size, decoder_output_dim):
#         super(GNN, self).__init__()
#         self.use_global = True if global_size > 1 else False
#         embed_dim = 128
#         self.dyn_models = torch.nn.ModuleDict(
#             {
#                 "encoder": Encoder(node_input_dim, edge_input_dim, output_size=embed_dim),
#                 "processor": Processor(
#                     [
#                         3 * embed_dim + global_size,
#                         2 * embed_dim + global_size,
#                         2 * embed_dim + global_size,
#                     ],
#                     use_global=self.use_global,
#                     layers=proc_layer,
#                     global_size=global_size,
#                 ),
#                 "decoder": Decoder(output_size=decoder_output_dim),
#             }
#         )

#     def forward(self, data):
#         """data should be a dictionary containing the following dict
#         edge_index: Edge index 2 x E
#         x: Node feature
#         edge_attr: Edge feature
#         gt_accel: Acceleration label for each node
#         x_batch: Batch index
#         """

#         node_embedding, edge_embedding = self.dyn_models["encoder"](data["x"], data["edge_attr"])
#         _, e_nxt, _ = self.dyn_models["processor"](
#             node_embedding, data["edge_index"], edge_embedding, u=data["u"], batch=data["x_batch"]
#         )

#         return self.dyn_models["decoder"](e_nxt)

#     def load_model(self, model_path):
#         ckpt = torch.load(model_path)
#         for k, v in self.dyn_models.items():
#             self.dyn_models[k].load_state_dict(ckpt[k])
#         print("Loaded trained model for {}".format(model_path))


# class EdgePredictor:
#     def __init__(self, neighbor_radius):
#         self.neighbor_radius = neighbor_radius
#         self.model = GNN(node_input_dim=3, edge_input_dim=4, proc_layer=10, global_size=128, decoder_output_dim=1)
#         self.model = self.model.to(device="cuda")

#     def compute_edge_attr(self, normalized_vox_pc):
#         point_tree = spatial.cKDTree(normalized_vox_pc)
#         undirected_neighbors = np.array(list(point_tree.query_pairs(self.neighbor_radius, p=2))).T

#         if len(undirected_neighbors) > 0:
#             dist_vec = normalized_vox_pc[undirected_neighbors[0, :]] - normalized_vox_pc[undirected_neighbors[1, :]]
#             dist = np.linalg.norm(dist_vec, axis=1, keepdims=True)
#             edge_attr = np.concatenate([dist_vec, dist], axis=1)
#             edge_attr_reverse = np.concatenate([-dist_vec, dist], axis=1)

#             # Generate directed edge list and corresponding edge attributes
#             edges = torch.from_numpy(np.concatenate([undirected_neighbors, undirected_neighbors[::-1]], axis=1))
#             edge_attr = torch.from_numpy(np.concatenate([edge_attr, edge_attr_reverse]))
#         else:
#             print("number of distance edges is 0! adding fake edges")
#             edges = np.zeros((2, 2), dtype=np.uint8)
#             edges[0][0] = 0
#             edges[1][0] = 1
#             edges[0][1] = 0
#             edges[1][1] = 2
#             edge_attr = np.zeros((2, 4), dtype=np.float32)
#             edges = torch.from_numpy(edges).bool()
#             edge_attr = torch.from_numpy(edge_attr)
#             print("shape of edges: ", edges.shape)
#             print("shape of edge_attr: ", edge_attr.shape)

#         return edges, edge_attr

#     def build_graph(self, normalized_vox_pc):
#         """
#         data: positions, picked_points, picked_point_positions, scene_params
#         downsample: whether to downsample the graph
#         test: if False, we are in the training mode, where we know exactly the picked point and its movement
#             if True, we are in the test mode, we have to infer the picked point in the (downsampled graph) and compute
#                 its movement.

#         return:
#         node_attr: N x (vel_history x 3)
#         edges: 2 x E, the edges
#         edge_attr: E x edge_feature_dim
#         """
#         node_attr = torch.from_numpy(normalized_vox_pc)
#         edges, edge_attr = self.compute_edge_attr(normalized_vox_pc)

#         return {"x": node_attr, "edge_index": edges, "edge_attr": edge_attr}

#     def infer_egdes(self, normalized_vox_pc):
#         d = self.build_graph(normalized_vox_pc)
#         data = Data.from_dict(d)
#         with torch.no_grad():
#             data["x_batch"] = torch.zeros(data["x"].size(0), dtype=torch.long, device="cuda")
#             data["u"] = torch.zeros([1, 128], device="cuda")
#             for key in ["x", "edge_index", "edge_attr"]:
#                 data[key] = data[key].to(device="cuda")
#             pred_mesh_edge_logits = self.model(data)

#         pred_mesh_edge_logits = pred_mesh_edge_logits.cpu().numpy()
#         pred_mesh_edge = pred_mesh_edge_logits > 0
#         edges = data["edge_index"].detach().cpu().numpy()
#         senders = []
#         receivers = []
#         num_edges = edges.shape[1]
#         for e_idx in range(num_edges):
#             if pred_mesh_edge[e_idx]:
#                 senders.append(int(edges[0][e_idx]))
#                 receivers.append(int(edges[1][e_idx]))

#         mesh_edges = np.vstack([senders, receivers])
#         return pred_mesh_edge, mesh_edges
