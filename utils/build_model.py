import yaml
from Model.model import GraphDepth, GraphDepthSuccessPredictor, DepthOnly, DepthSuccessPredictor
import torch


def get_configs(filepath):
    with open(filepath, "rb") as f:
        configs = yaml.safe_load(f)
    return configs


def build_model(configs):
    assert configs["type"] in ["graphdepth", "depthonly"]
    if configs["type"] == "graphdepth":
        model = GraphDepth(
            device=configs["device"],
            image_size=configs["image_size"],
            patch_size=configs["patch_size"],
            dim=configs["dim"],
            depth=configs["depth"],
            heads=configs["heads"],
            mlp_ratio=configs["mlp_ratio"],
            channels=configs["channels"],
            dropout=configs["dropout"],
            emb_dropout=configs["emb_dropout"],
            graph_encoder_path=configs["graph_encoder_path"],
            num_nodes=configs["num_nodes"],
        ).to(device=configs["device"])
    elif configs["type"] == "depthonly":
        model = DepthOnly(
            device=configs["device"],
            image_size=configs["image_size"],
            patch_size=configs["patch_size"],
            dim=configs["dim"],
            depth=configs["depth"],
            heads=configs["heads"],
            mlp_ratio=configs["mlp_ratio"],
            channels=configs["channels"],
            dropout=configs["dropout"],
            emb_dropout=configs["emb_dropout"],
        ).to(device=configs["device"])
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters (with clip's parameters) in model.")
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} trainable parameters in model.")
    if configs["continue_train"]:
        model.load_model(configs["checkpoint_path"])
        print("load checkpoint from {}".format(configs["checkpoint_path"]))
    return model


def build_predictor(configs):
    assert configs["type"] in ["graphdepth", "depthonly"]
    if configs["type"] == "graphdepth":
        predictor = GraphDepthSuccessPredictor(dim=configs["dim"]).to(device=configs["device"])
    elif configs["type"] == "depthonly":
        predictor = DepthSuccessPredictor(dim=configs["dim"]).to(device=configs["device"])
    total_params = sum(p.numel() for p in predictor.parameters())
    print(f"{total_params:,} total parameters in predictor.")
    total_trainable_params = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} trainable parameters in predictor.")
    if configs["continue_train"]:
        predictor.load_model(configs["checkpoint_path"])
        print("load checkpoint predictor from {}".format(configs["checkpoint_path"]))
    return predictor


def construct_optimizer(model, configs):
    params_non_frozen = filter(lambda p: p.requires_grad, model.parameters())  # filter clip params
    # adam optimizer
    optimizer = torch.optim.Adam(
        params_non_frozen,
        lr=float(configs["lr"]),
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=float(configs["weight_decay"]),
    )
    if configs["continue_train"]:
        optimizer.load_state_dict(torch.load(configs["checkpoint_path"])["optimizer"])
        print("load checkpoint optimizer from {}".format(configs["checkpoint_path"]))
    return optimizer
