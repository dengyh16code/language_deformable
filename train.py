import torch
import torch.nn as nn
from utils.build_model import construct_optimizer, get_configs, build_model, build_predictor
from utils.dataset import DeformableDataset, DepthOnlyDataset
from torch_geometric.loader import DataLoader
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse


def train_net(configs):
    # get params
    num_workers = configs["num_workers"]
    batch_size = configs["batch_size"]
    epochs = configs["epochs"]
    dataset_path = configs["dataset_path"]
    device = configs["device"]
    train_success_head = configs["train_success_head"]
    continue_train = configs["continue_train"]
    model_type = configs['type']
    

    # create folder to save trained model
    save_model_name = configs["save_model_name"]
    if train_success_head:
        save_folder = os.path.join("train/trained_models", save_model_name, "predictor")
    else:
        save_folder = os.path.join("train/trained_models", save_model_name, "model")
    os.makedirs(save_folder, exist_ok=True)

    # epoch
    if continue_train:
        already_trained_epochs = configs["already_trained_epochs"]
    else:
        already_trained_epochs = 0

    # create summarywritter tensorboard
    writer = SummaryWriter(log_dir=save_folder)
    print(
        "batch size:{} | epochs:{} | dataset:{} | device:{} | train predictor:{} | continue train:{} | already trained epochs: {}".format(
            batch_size, epochs, dataset_path, device, train_success_head, continue_train, already_trained_epochs
        )
    )

    # create network & optimizer
    model = build_model(configs)
    if train_success_head:
        # build model
        trained_model_path = configs["trained_model_path"]
        model.load_model(trained_model_path)
        model.frozen_all()
        # build predictor
        success_predictor = build_predictor(configs)
        optimizer = construct_optimizer(success_predictor, configs)
    else:
        optimizer = construct_optimizer(model, configs)
        
    # loss function
    loss_fn = nn.BCEWithLogitsLoss()

    # Dataset & Dataloader
    if model_type == 'depthonly':
        dataset = DepthOnlyDataset(configs=configs)
    elif model_type == "graphdepth":
        dataset = DeformableDataset(configs=configs)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    num_batches = len(dataloader)

    # Model train
    for epoch in tqdm(range(already_trained_epochs, already_trained_epochs + epochs)):
        pbar = tqdm(dataloader)
        # train head
        if train_success_head:
            model.eval()
            total_predict_loss = 0
            for batch, (depths, graph_datas, _, _, instructions, success_flags) in enumerate(pbar):
                depths, graph_datas, instructions, success_flags = (
                    depths.to(device),
                    graph_datas.to(device),
                    instructions.to(device),
                    success_flags.to(device),
                )

                # forward & loss
                with torch.no_grad():
                    if model_type == "depthonly":
                        _, _, all_heads = model(instructions, depths)
                    elif model_type == "graphdepth":
                        _, _, all_heads = model(instructions, depths,graph_datas)
                
                success_predictions = success_predictor(all_heads)
                loss_predict = loss_fn(success_predictions, success_flags)

                # backpropagation
                optimizer.zero_grad()
                loss_predict.backward()
                optimizer.step()

                # print loss
                total_predict_loss += loss_predict.item()
                pbar.set_description("average loss:{}".format(total_predict_loss / (batch + 1)))

            # logger
            average_predict_loss = total_predict_loss / num_batches
            writer.add_scalar(tag="average_predict_loss", scalar_value=average_predict_loss, global_step=epoch)

            # save model
            if (epoch + 1) % 10 == 0:
                model_path = os.path.join(save_folder, "epoch{}.pth".format(epoch))
                model_state_dict = {
                    "model": success_predictor.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(model_state_dict, model_path)
        # train model
        else:
            model.train()
            total_loss = 0
            total_pick_loss = 0
            total_place_loss = 0
            for batch, (depths, graph_datas, pick_heatmaps, place_heatmaps, instructions, _) in enumerate(pbar):
                depths, graph_datas, pick_heatmaps, place_heatmaps, instructions = (
                    depths.to(device),
                    graph_datas.to(device),
                    pick_heatmaps.to(device),
                    place_heatmaps.to(device),
                    instructions.to(device),
                )
                # forward & loss
                if model_type == "depthonly":
                    pred_pick_heatmaps, pred_place_heatmaps, _ = model(instructions, depths)
               
                elif model_type == "graphdepth":
                    pred_pick_heatmaps, pred_place_heatmaps, _ = model(instructions, depths, graph_datas)
                
                loss_pick = loss_fn(pred_pick_heatmaps, pick_heatmaps)
                loss_place = loss_fn(pred_place_heatmaps, place_heatmaps)
                loss = loss_pick + loss_place

                # backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # print loss
                total_loss += loss.item()
                total_pick_loss += loss_pick.item()
                total_place_loss += loss_place.item()
                pbar.set_description("average loss:{}".format(total_loss / (batch + 1)))

            # logger
            average_loss = total_loss / num_batches
            average_pick_loss = total_pick_loss / num_batches
            average_place_loss = total_place_loss / num_batches
            writer.add_scalar(tag="average_loss", scalar_value=average_loss, global_step=epoch)
            writer.add_scalar(tag="average_pick_loss", scalar_value=average_pick_loss, global_step=epoch)
            writer.add_scalar(tag="average_place_loss", scalar_value=average_place_loss, global_step=epoch)

            # save model
            if (epoch + 1) % 10 == 0:
                model_path = os.path.join(save_folder, "epoch{}.pth".format(epoch))
                model_state_dict = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(model_state_dict, model_path)

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--config_path", type=str, help="train configs path")
    args = parser.parse_args()
    config_path = os.path.join("train/train_configs", args.config_path + ".yaml")
    configs = get_configs(config_path)
    train_net(configs=configs)
