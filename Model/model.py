import numpy as np
import torch
from torch import nn
from Model.CLIP import clip

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from Model.utils import pair, init_weights
from Model.vcd import GNN

from utils.visual import get_matrix_world_to_camera, get_pixel_coord_from_world
from softgym.utils.gemo_utils import intrinsic_from_fov
import matplotlib.pyplot as plt


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ConvDecoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = int(input_dim)
        self.decoder_net = self._init_decoder()

    def _init_decoder(self):
        intermediate_channels1 = int(self.input_dim / 2)
        intermediate_channels2 = int(self.input_dim / 4)
        in_channels = [
            self.input_dim,
            intermediate_channels1,
            intermediate_channels1,
            intermediate_channels2,
            intermediate_channels2,
        ]
        out_channels = [
            intermediate_channels1,
            intermediate_channels1,
            intermediate_channels2,
            intermediate_channels2,
            1,
        ]
        modules = []
        for i, (in_channel, out_channel) in enumerate(zip(in_channels, out_channels)):
            modules.append(
                nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=(0, 0))
            )
            if i != 4:
                modules.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))

        return nn.Sequential(*modules)

    def forward(self, x):
        return self.decoder_net(x)


#################################################
##################Depth Only#####################
#################################################
class DepthOnly(nn.Module):
    def __init__(
        self,
        device="cuda",
        image_size=224,
        patch_size=16,
        dim=512,
        depth=8,
        heads=16,
        mlp_ratio=4,
        channels=1,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.device = device

        # images
        self.image_height, self.image_width = pair(image_size)
        self.patch_height, self.patch_width = pair(patch_size)
        assert (
            self.image_height % self.patch_height == 0 and self.image_width % self.patch_width == 0
        ), "Image dimensions must be divisible by the patch size."
        num_patches = (self.image_height // self.patch_height) * (self.image_width // self.patch_width)
        patch_dim = channels * self.patch_height * self.patch_width
        self.image_to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=self.patch_height, p2=self.patch_width),
            nn.Linear(patch_dim, dim),
        )
        self.image_pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.image_token = nn.Parameter(torch.randn(1, 1, dim))
        self.image_dropout = nn.Dropout(emb_dropout)
        self.mlp_dim = dim * mlp_ratio
        self.dim_head = int(dim / heads)

        # text
        self.text_token = nn.Parameter(torch.randn(1, 1, dim))
        self.text_pos_embedding = nn.Parameter(torch.randn(1, 78, dim))
        self.text_dropout = nn.Dropout(emb_dropout)
        self.clip_encoder, _ = clip.load("RN50", device=self.device)

        # type embedding
        self.token_type_embeddings = nn.Embedding(2, dim)  # 0 for text, 1 for depth image
        self.token_type_embeddings.apply(init_weights)

        # transformer encoder
        self.transformer_encoder = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=self.dim_head,
            mlp_dim=self.mlp_dim,
            dropout=dropout,
        )

        # pick decoder
        self.pick_decoder = ConvDecoder(dim)

        # place decoder
        self.place_decoder = ConvDecoder(dim)

        # frozen clip & graph_encoder
        self.frozen_clip()

    def load_model(self, model_path):
        self.load_state_dict(torch.load(model_path)["model"])
        print(f"load agent from {model_path}")

    def frozen_clip(self):
        for param in self.named_parameters():
            if "clip_encoder" in param[0]:
                param[1].requires_grad = False

    def frozen_all(self):
        for param in self.named_parameters():
            param[1].requires_grad = False

    def encode_image(self, img):
        x = self.image_to_patch_embedding(img)
        b, n, _ = x.shape
        image_tokens = repeat(self.image_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((image_tokens, x), dim=1)
        x += self.image_pos_embedding[:, : (n + 1)]
        x = self.image_dropout(x)
        return x

    def encode_text(self, text):
        x = self.clip_encoder.encode_text_with_embeddings(text)
        b, n, _ = x.shape
        text_tokens = repeat(self.text_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((text_tokens, x), dim=1)
        x += self.text_pos_embedding[:, : (n + 1)]
        x = self.text_dropout(x)
        return x

    def reshape_output(self, x, input_dim):
        x = x.view(
            x.size(0),
            int(self.image_height // self.patch_height),
            int(self.image_width // self.patch_width),
            input_dim,
        )
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, text, img):
        # encode text and image
        x_text = self.encode_text(text)
        x_image = self.encode_image(img)

        # type encoding
        text_type_embeddings = self.token_type_embeddings(
            torch.full((x_text.shape[0], x_text.shape[1]), 0, device=self.device).long()
        )
        image_type_embeddings = self.token_type_embeddings(
            torch.full((x_image.shape[0], x_image.shape[1]), 1, device=self.device).long()
        )

        x_text += text_type_embeddings
        x_image += image_type_embeddings

        # concatenate
        x = torch.cat((x_text, x_image), dim=1)

        # transformer encoder
        x = self.transformer_encoder(x)

        # task success prediction
        text_first_token_features = x[:, 0, :]
        image_first_token_features = x[:, x_text.shape[1], :]
        all_head = torch.cat((text_first_token_features, image_first_token_features), dim=1)

        # pick & place heatmap
        image_features = x[:, -x_image.shape[1] : -1, :]
        image_features = self.reshape_output(image_features, x.shape[-1])
        pick_heatmap = self.pick_decoder(image_features).squeeze()
        place_heatmap = self.place_decoder(image_features).squeeze()

        return pick_heatmap, place_heatmap, all_head


class DepthSuccessPredictor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(dim * 2, dim * 2), nn.Tanh(), nn.Linear(dim * 2, 1))

    def forward(self, x):
        return self.head(x).squeeze()

    def load_model(self, model_path):
        self.load_state_dict(torch.load(model_path)["model"])
        print(f"load success predictor from {model_path}")


#################################################
##################Graph Only#####################
#################################################
class GraphOnly(nn.Module):
    def __init__(
        self,
        device="cuda",
        dim=512,
        depth=8,
        heads=16,
        mlp_ratio=4,
        dropout=0.0,
        emb_dropout=0.0,
        graph_encoder_path=None,
        num_nodes=200,
    ):
        super().__init__()
        self.device = device

        # attention
        self.mlp_dim = dim * mlp_ratio
        self.dim_head = int(dim / heads)

        # text
        self.text_token = nn.Parameter(torch.randn(1, 1, dim))
        self.text_pos_embedding = nn.Parameter(torch.randn(1, 78, dim))
        self.text_dropout = nn.Dropout(emb_dropout)
        self.clip_encoder, _ = clip.load("RN50", device=self.device)

        # graph
        self.graph_encoder = GNN(node_input_dim=3, edge_input_dim=4, proc_layer=10, global_size=128).to(device=self.device)
        self.graph_encoder.load_model(graph_encoder_path,self.device)
        self.graph_project = nn.Linear(128, dim)
        self.graph_token = nn.Parameter(torch.randn(1, 1, dim))
        self.num_nodes = num_nodes

        # type embedding
        self.token_type_embeddings = nn.Embedding(2, dim)  # 0 for text, 1 for graph
        self.token_type_embeddings.apply(init_weights)

        # transformer encoder
        self.transformer_encoder = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=self.dim_head,
            mlp_dim=self.mlp_dim,
            dropout=dropout,
        )

        # # pick decoder (node classification)
        # self.pick_decoder = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

        # # place decoder (node classification)
        # self.place_decoder = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

        # pick & place decoder
        self.decoder = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2))

        # frozen clip & graph_encoder
        self.frozen_clip()
        self.frozen_graph_encoder()

    def load_model(self, model_path):
        self.load_state_dict(torch.load(model_path)["model"])
        print(f"load agent from {model_path}")

    def frozen_clip(self):
        for param in self.named_parameters():
            if "clip_encoder" in param[0]:
                param[1].requires_grad = False

    def frozen_graph_encoder(self):
        for param in self.named_parameters():
            if "dyn_models" in param[0]:
                param[1].requires_grad = False

    def frozen_all(self):
        for param in self.named_parameters():
            param[1].requires_grad = False

    def encode_text(self, text):
        x = self.clip_encoder.encode_text_with_embeddings(text)
        b, n, _ = x.shape
        text_tokens = repeat(self.text_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((text_tokens, x), dim=1)
        x += self.text_pos_embedding[:, : (n + 1)]
        x = self.text_dropout(x)
        return x

    def encode_graph(self, graph):
        # add initial global features
        batch_size = int(graph["x"].shape[0] / self.num_nodes)
        global_features = torch.zeros(batch_size, 128, dtype=torch.float32, device=self.device)
        graph["u"] = global_features
        # encode
        x = self.graph_encoder(graph)
        x = torch.reshape(x, (-1, self.num_nodes, 128))
        x = self.graph_project(x)
        b, _, _ = x.shape
        graph_tokens = repeat(self.graph_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((graph_tokens, x), dim=1)
        return x

    def forward(self, text, graph):
        # encode text and image
        x_text = self.encode_text(text)
        x_graph = self.encode_graph(graph)

        # type encoding
        text_type_embeddings = self.token_type_embeddings(
            torch.full((x_text.shape[0], x_text.shape[1]), 0, device=self.device).long()
        )
        graph_type_embeddings = self.token_type_embeddings(
            torch.full((x_graph.shape[0], x_graph.shape[1]), 1, device=self.device).long()
        )

        x_text += text_type_embeddings
        x_graph += graph_type_embeddings

        # concatenate
        x = torch.cat((x_text, x_graph), dim=1)

        # transformer encoder
        x = self.transformer_encoder(x)

        # task success prediction
        text_first_token_features = x[:, 0, :]
        graph_first_token_features = x[:, x_text.shape[1], :]
        all_head = torch.cat((text_first_token_features, graph_first_token_features), dim=1)

        # pick & place point predict
        graph_features = x[:, -x_graph.shape[1] + 1 :, :]
        probs = self.decoder(graph_features)

        pick_prob = probs[:, :, 0]
        place_prob = probs[:, :, 1]

        # pick_prob = self.pick_decoder(graph_features).squeeze()
        # place_prob = self.place_decoder(graph_features).squeeze()

        return pick_prob, place_prob, all_head


class GraphSuccessPredictor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.head = nn.Sequential(nn.LayerNorm(dim * 2), nn.Linear(dim * 2, 1))

    def forward(self, x):
        return self.head(x).squeeze()

    def load_model(self, model_path):
        self.load_state_dict(torch.load(model_path)["model"])
        print(f"load success predictor from {model_path}")


#################################################
##################Graph Depth####################
#################################################
class GraphDepth(nn.Module):
    def __init__(
        self,
        device="cuda",
        image_size=224,
        patch_size=16,
        dim=512,
        depth=8,
        heads=16,
        mlp_ratio=4,
        channels=1,
        dropout=0.0,
        emb_dropout=0.0,
        graph_encoder_path=None,
        num_nodes=200,
    ):
        super().__init__()
        self.device = device

        # images
        self.image_height, self.image_width = pair(image_size)
        self.patch_height, self.patch_width = pair(patch_size)
        assert (
            self.image_height % self.patch_height == 0 and self.image_width % self.patch_width == 0
        ), "Image dimensions must be divisible by the patch size."
        num_patches = (self.image_height // self.patch_height) * (self.image_width // self.patch_width)
        patch_dim = channels * self.patch_height * self.patch_width
        self.image_to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=self.patch_height, p2=self.patch_width),
            nn.Linear(patch_dim, dim),
        )
        self.image_pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.image_token = nn.Parameter(torch.randn(1, 1, dim))
        self.image_dropout = nn.Dropout(emb_dropout)
        self.mlp_dim = dim * mlp_ratio
        self.dim_head = int(dim / heads)

        # text
        self.text_token = nn.Parameter(torch.randn(1, 1, dim))
        self.text_pos_embedding = nn.Parameter(torch.randn(1, 78, dim))
        self.text_dropout = nn.Dropout(emb_dropout)
        self.clip_encoder, _ = clip.load("RN50", device=self.device)

        # graph
        self.graph_encoder = GNN(node_input_dim=3, edge_input_dim=4, proc_layer=10, global_size=128).to(device=self.device)
        self.graph_encoder.load_model(graph_encoder_path,self.device)
        self.graph_project = nn.Linear(128, dim)
        self.graph_token = nn.Parameter(torch.randn(1, 1, dim))
        self.num_nodes = num_nodes

        # type embedding
        self.token_type_embeddings = nn.Embedding(3, dim)  # 0 for text, 1 for depth image, 2 for graph
        self.token_type_embeddings.apply(init_weights)

        # transformer encoder
        self.transformer_encoder = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=self.dim_head,
            mlp_dim=self.mlp_dim,
            dropout=dropout,
        )

        # pick decoder (node classification)
        self.pick_decoder = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

        # place decoder (heatmap)
        self.place_decoder = ConvDecoder(dim)

        # frozen clip & graph_encoder
        self.frozen_clip()
        self.frozen_graph_encoder()

    def load_model(self, model_path):
        self.load_state_dict(torch.load(model_path)["model"])
        print(f"load agent from {model_path}")

    def frozen_clip(self):
        for param in self.named_parameters():
            if "clip_encoder" in param[0]:
                param[1].requires_grad = False

    def frozen_graph_encoder(self):
        for param in self.named_parameters():
            if "dyn_models" in param[0]:
                param[1].requires_grad = False

    def frozen_all(self):
        for param in self.named_parameters():
            param[1].requires_grad = False

    def encode_image(self, img):
        x = self.image_to_patch_embedding(img)
        b, n, _ = x.shape
        image_tokens = repeat(self.image_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((image_tokens, x), dim=1)
        x += self.image_pos_embedding[:, : (n + 1)]
        x = self.image_dropout(x)
        return x

    def encode_text(self, text):
        x = self.clip_encoder.encode_text_with_embeddings(text)
        b, n, _ = x.shape
        text_tokens = repeat(self.text_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((text_tokens, x), dim=1)
        x += self.text_pos_embedding[:, : (n + 1)]
        x = self.text_dropout(x)
        return x

    def encode_graph(self, graph):
        # add initial global features
        batch_size = int(graph["x"].shape[0] / self.num_nodes)
        global_features = torch.zeros(batch_size, 128, dtype=torch.float32, device=self.device)
        graph["u"] = global_features
        # encode
        x = self.graph_encoder(graph)
        x = torch.reshape(x, (-1, self.num_nodes, 128))
        x = self.graph_project(x)
        b, n, _ = x.shape
        graph_tokens = repeat(self.graph_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((graph_tokens, x), dim=1)
        return x

    def reshape_output(self, x, input_dim):
        x = x.view(
            x.size(0),
            int(self.image_height // self.patch_height),
            int(self.image_width // self.patch_width),
            input_dim,
        )
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, text, img, graph):
        # encode text and image
        x_text = self.encode_text(text)
        x_image = self.encode_image(img)
        x_graph = self.encode_graph(graph)

        # type encoding
        text_type_embeddings = self.token_type_embeddings(
            torch.full((x_text.shape[0], x_text.shape[1]), 0, device=self.device).long()
        )
        image_type_embeddings = self.token_type_embeddings(
            torch.full((x_image.shape[0], x_image.shape[1]), 1, device=self.device).long()
        )
        graph_type_embeddings = self.token_type_embeddings(
            torch.full((x_graph.shape[0], x_graph.shape[1]), 2, device=self.device).long()
        )

        x_text += text_type_embeddings
        x_image += image_type_embeddings
        x_graph += graph_type_embeddings

        # concatenate
        x = torch.cat((x_text, x_image, x_graph), dim=1)

        # transformer encoder
        x = self.transformer_encoder(x)

        # task success prediction
        text_first_token_features = x[:, 0, :]
        image_first_token_features = x[:, x_text.shape[1], :]
        graph_first_token_features = x[:, x_text.shape[1] + x_image.shape[1], :]
        all_heads = torch.cat(
            (text_first_token_features, image_first_token_features, graph_first_token_features), dim=1
        )

        # pick point predict
        graph_features = x[:, -x_graph.shape[1] + 1 :, :]
        pick_probs = self.pick_decoder(graph_features).squeeze(2)

        # place heatmap
        image_features = x[:, x_text.shape[1] + 1 : x_text.shape[1] + x_image.shape[1], :]
        image_features = self.reshape_output(image_features, x.shape[-1])
        place_heatmaps = self.place_decoder(image_features).squeeze()

        return pick_probs, place_heatmaps, all_heads


class GraphDepthSuccessPredictor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(dim * 3, dim * 3), nn.Tanh(), nn.Linear(dim * 3, 1))

    def forward(self, x):
        return self.head(x).squeeze()

    def load_model(self, model_path):
        self.load_state_dict(torch.load(model_path)["model"])
        print(f"load success predictor from {model_path}")


#################################################
###############Pick Conditioned##################
#################################################
# class GraphDepth(nn.Module):
#     def __init__(
#         self,
#         device="cuda",
#         image_size=224,
#         patch_size=16,
#         dim=512,
#         depth=8,
#         heads=16,
#         mlp_ratio=4,
#         channels=1,
#         dropout=0.0,
#         emb_dropout=0.0,
#         graph_encoder_path=None,
#         num_nodes=200,
#     ):
#         super().__init__()
#         self.device = device

#         # images
#         self.image_height, self.image_width = pair(image_size)
#         self.patch_height, self.patch_width = pair(patch_size)
#         assert (
#             self.image_height % self.patch_height == 0 and self.image_width % self.patch_width == 0
#         ), "Image dimensions must be divisible by the patch size."
#         num_patches = (self.image_height // self.patch_height) * (self.image_width // self.patch_width)
#         patch_dim = channels * self.patch_height * self.patch_width
#         self.image_to_patch_embedding = nn.Sequential(
#             Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=self.patch_height, p2=self.patch_width),
#             nn.Linear(patch_dim, dim),
#         )
#         self.image_pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
#         self.image_token = nn.Parameter(torch.randn(1, 1, dim))
#         self.image_dropout = nn.Dropout(emb_dropout)
#         self.mlp_dim = dim * mlp_ratio
#         self.dim_head = int(dim / heads)

#         # text
#         self.text_token = nn.Parameter(torch.randn(1, 1, dim))
#         self.text_pos_embedding = nn.Parameter(torch.randn(1, 78, dim))
#         self.text_dropout = nn.Dropout(emb_dropout)
#         self.clip_encoder, _ = clip.load("RN50", device=self.device)

#         # graph
#         self.graph_encoder = GNN(node_input_dim=3, edge_input_dim=4, proc_layer=10, global_size=128)
#         self.graph_encoder.load_model(graph_encoder_path)
#         self.graph_project = nn.Linear(128, dim)
#         self.graph_token = nn.Parameter(torch.randn(1, 1, dim))
#         self.num_nodes = num_nodes

#         # type embedding
#         self.token_type_embeddings = nn.Embedding(3, dim)  # 0 for text, 1 for depth image, 2 for graph
#         self.token_type_embeddings.apply(init_weights)

#         # transformer encoder
#         self.transformer_encoder = Transformer(
#             dim=dim,
#             depth=depth,
#             heads=heads,
#             dim_head=self.dim_head,
#             mlp_dim=self.mlp_dim,
#             dropout=dropout,
#         )

#         # pick decoder (node classification)
#         self.pick_decoder = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

#         # place decoder (heatmap)
#         self.place_decoder_1 = ConvDecoder(dim)
#         self.place_decoder_2 = nn.Sequential(
#             nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=(0, 0)),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=(0, 0)),
#         )

#         # camera
#         self.CAMERA_PARAMS = {
#             "default_camera": {
#                 "pos": np.array([0.0, 0.65, 0.0]),
#                 "angle": np.array([0.0, -1.57079633, 0.0]),
#                 "width": 720,
#                 "height": 720,
#             }
#         }
#         self.matrix_world_to_camera = get_matrix_world_to_camera(self.CAMERA_PARAMS)
#         self.matrix_world_to_camera = torch.FloatTensor(self.matrix_world_to_camera).cuda()
#         self.K = intrinsic_from_fov(224, 224, 45)
#         self.K = torch.FloatTensor(self.K).cuda()

#         # frozen clip & graph_encoder
#         self.frozen_clip()
#         self.frozen_graph_encoder()

#     def load_model(self, model_path):
#         self.load_state_dict(torch.load(model_path)["model"])
#         print(f"load agent from {model_path}")

#     def frozen_clip(self):
#         for param in self.named_parameters():
#             if "clip_encoder" in param[0]:
#                 param[1].requires_grad = False

#     def frozen_graph_encoder(self):
#         for param in self.named_parameters():
#             if "dyn_models" in param[0]:
#                 param[1].requires_grad = False

#     def frozen_all(self):
#         for param in self.named_parameters():
#             param[1].requires_grad = False

#     def encode_image(self, img):
#         x = self.image_to_patch_embedding(img)
#         b, n, _ = x.shape
#         image_tokens = repeat(self.image_token, "1 1 d -> b 1 d", b=b)
#         x = torch.cat((image_tokens, x), dim=1)
#         x += self.image_pos_embedding[:, : (n + 1)]
#         x = self.image_dropout(x)
#         return x

#     def encode_text(self, text):
#         x = self.clip_encoder.encode_text_with_embeddings(text)
#         b, n, _ = x.shape
#         text_tokens = repeat(self.text_token, "1 1 d -> b 1 d", b=b)
#         x = torch.cat((text_tokens, x), dim=1)
#         x += self.text_pos_embedding[:, : (n + 1)]
#         x = self.text_dropout(x)
#         return x

#     def encode_graph(self, graph):
#         # add initial global features
#         batch_size = int(graph["x"].shape[0] / self.num_nodes)
#         global_features = torch.zeros(batch_size, 128, dtype=torch.float32, device=self.device)
#         graph["u"] = global_features
#         # encode
#         x = self.graph_encoder(graph)
#         x = torch.reshape(x, (-1, self.num_nodes, 128))
#         x = self.graph_project(x)
#         b, n, _ = x.shape
#         graph_tokens = repeat(self.graph_token, "1 1 d -> b 1 d", b=b)
#         x = torch.cat((graph_tokens, x), dim=1)
#         return x

#     def reshape_output(self, x, input_dim):
#         x = x.view(
#             x.size(0),
#             int(self.image_height // self.patch_height),
#             int(self.image_width // self.patch_width),
#             input_dim,
#         )
#         x = x.permute(0, 3, 1, 2).contiguous()
#         return x

#     def get_embbeding(self, text, img, graph):
#         # encode text and image
#         x_text = self.encode_text(text)
#         x_image = self.encode_image(img)
#         x_graph = self.encode_graph(graph)

#         # type encoding
#         text_type_embeddings = self.token_type_embeddings(
#             torch.full((x_text.shape[0], x_text.shape[1]), 0, device=self.device).long()
#         )
#         image_type_embeddings = self.token_type_embeddings(
#             torch.full((x_image.shape[0], x_image.shape[1]), 1, device=self.device).long()
#         )
#         graph_type_embeddings = self.token_type_embeddings(
#             torch.full((x_graph.shape[0], x_graph.shape[1]), 2, device=self.device).long()
#         )

#         x_text += text_type_embeddings
#         x_image += image_type_embeddings
#         x_graph += graph_type_embeddings

#         # concatenate
#         x = torch.cat((x_text, x_image, x_graph), dim=1)

#         # transformer encoder
#         x = self.transformer_encoder(x)

#         return x, x_text.shape, x_image.shape, x_graph.shape

#     def get_pick_point(self, x, graph_shape):
#         # pick point predict
#         graph_features = x[:, -graph_shape[1] + 1 :, :]
#         pick_prob = self.pick_decoder(graph_features).squeeze()
#         pick_node_index = torch.argmax(pick_prob)

#         return pick_node_index

#     def get_place_heatmap(self, x, pick_heatmap, text_shape, image_shape):
#         image_features = x[:, text_shape[1] + 1 : text_shape[1] + image_shape[1], :]
#         image_features = self.reshape_output(image_features, x.shape[-1])
#         pick_heatmap = pick_heatmap.unsqueeze(0).unsqueeze(0)
#         place_heatmap = self.place_decoder_1(image_features)
#         place_heatmap1 = place_heatmap.squeeze().detach().cpu().numpy()
#         heatmaps = torch.cat((pick_heatmap, place_heatmap), dim=1)
#         place_heatmap = self.place_decoder_2(heatmaps).squeeze()
#         place_heatmap2 = place_heatmap.squeeze().detach().cpu().numpy()
#         _, axes = plt.subplots(1, 2)
#         axes[0].imshow(place_heatmap1)
#         axes[1].imshow(place_heatmap2)
#         plt.savefig("test/2heatmaps.png")
#         return place_heatmap

#     def get_all_heads(self, x, text_shape, image_shape):
#         text_first_token_features = x[:, 0, :]
#         image_first_token_features = x[:, text_shape[1], :]
#         graph_first_token_features = x[:, text_shape[1] + image_shape[1], :]
#         all_heads = torch.cat(
#             (text_first_token_features, image_first_token_features, graph_first_token_features), dim=1
#         )
#         return all_heads

#     def get_pixels_coord_from_worlds(self, coords):
#         world_coordinates = torch.cat([coords, torch.ones((coords.shape[0], 1)).cuda()], dim=1)
#         camera_coordinates = torch.mm(self.matrix_world_to_camera, world_coordinates.transpose(0, 1))
#         camera_coordinates = camera_coordinates.transpose(0, 1)

#         x, y, depth = camera_coordinates[:, 0], camera_coordinates[:, 1], camera_coordinates[:, 2]
#         u = x * self.K[0, 0] / depth + self.K[0, 2]
#         v = y * self.K[1, 1] / depth + self.K[1, 2]

#         pixels = torch.cat((v.unsqueeze(1), u.unsqueeze(1)), dim=1)

#         return pixels

#     def make_gaussmaps(self, pixels, sigma=5):
#         center_x = pixels[:, 0]
#         center_y = pixels[:, 1]
#         xy_grid = torch.arange(0, 224)
#         x, y = torch.meshgrid(xy_grid, xy_grid, indexing="ij")
#         # align dimensions
#         x = x.unsqueeze(0).repeat(center_x.shape[0], 1, 1).cuda()
#         y = y.unsqueeze(0).repeat(center_y.shape[0], 1, 1).cuda()
#         center_x = center_x.unsqueeze(1).unsqueeze(1).repeat(1, 224, 224).cuda()
#         center_y = center_y.unsqueeze(1).unsqueeze(1).repeat(1, 224, 224).cuda()
#         # cacl gauusianmap
#         dists = (x - center_x) ** 2 + (y - center_y) ** 2
#         gauss_maps = torch.exp(-dists / (2 * sigma * sigma))
#         return gauss_maps

#     def forward(self, text, img, graph, sampled_pc):
#         # encode text and image
#         x_text = self.encode_text(text)
#         x_image = self.encode_image(img)
#         x_graph = self.encode_graph(graph)

#         # type encoding
#         text_type_embeddings = self.token_type_embeddings(
#             torch.full((x_text.shape[0], x_text.shape[1]), 0, device=self.device).long()
#         )
#         image_type_embeddings = self.token_type_embeddings(
#             torch.full((x_image.shape[0], x_image.shape[1]), 1, device=self.device).long()
#         )
#         graph_type_embeddings = self.token_type_embeddings(
#             torch.full((x_graph.shape[0], x_graph.shape[1]), 2, device=self.device).long()
#         )

#         x_text += text_type_embeddings
#         x_image += image_type_embeddings
#         x_graph += graph_type_embeddings

#         # concatenate
#         x = torch.cat((x_text, x_image, x_graph), dim=1)

#         # transformer encoder
#         x = self.transformer_encoder(x)

#         # task success prediction
#         text_first_token_features = x[:, 0, :]
#         image_first_token_features = x[:, x_text.shape[1], :]
#         graph_first_token_features = x[:, x_text.shape[1] + x_image.shape[1], :]
#         all_heads = torch.cat(
#             (text_first_token_features, image_first_token_features, graph_first_token_features), dim=1
#         )

#         # pick point predict
#         graph_features = x[:, -x_graph.shape[1] + 1 :, :]
#         pick_probs = self.pick_decoder(graph_features).squeeze(2)

#         # get pick-condintioned heatmap
#         pick_index = torch.argmax(pick_probs, dim=1)

#         sampled_pc_gather = rearrange(sampled_pc, "b n c -> (b n) c")
#         pick_index = torch.arange(0, pick_index.shape[0]).cuda() * self.num_nodes + pick_index
#         pick_pos = sampled_pc_gather[pick_index]
#         pick_pixels = self.get_pixels_coord_from_worlds(pick_pos)
#         pick_heatmaps = self.make_gaussmaps(pick_pixels)

#         # place heatmap
#         image_features = x[:, x_text.shape[1] + 1 : x_text.shape[1] + x_image.shape[1], :]
#         image_features = self.reshape_output(image_features, x.shape[-1])
#         pick_heatmaps = pick_heatmaps.unsqueeze(1)
#         place_heatmaps1 = self.place_decoder_1(image_features)
#         heatmaps = torch.cat((pick_heatmaps, place_heatmaps1), dim=1)
#         place_heatmaps2 = self.place_decoder_2(heatmaps).squeeze()
#         _, axes = plt.subplots(1, 3)
#         axes[0].imshow(pick_heatmaps.squeeze().detach().cpu().numpy())
#         axes[1].imshow(place_heatmaps1.squeeze().detach().cpu().numpy())
#         axes[2].imshow(place_heatmaps2.squeeze().detach().cpu().numpy())
#         plt.savefig("test/3heatmaps.png")

#         return pick_probs, place_heatmaps2, all_heads


# class SuccessPredictor(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.head = nn.Sequential(nn.Linear(dim * 3, dim * 3), nn.Tanh(), nn.Linear(dim * 3, 1))

#     def forward(self, x):
#         return self.head(x).squeeze()

#     def load_model(self, model_path):
#         self.load_state_dict(torch.load(model_path)["model"])
#         print(f"load success predictor from {model_path}")
