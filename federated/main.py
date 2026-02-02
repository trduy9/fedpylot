import math
import logging
import time
import argparse
import os
import shutil
import sys
import random
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "yolov7"))

import pandas as pd
import yaml

from node import Client, Server

# Try to import predefined clients if user created them externally
try:
    # expected: predefined_clients.py contains "clients = [Client(...), ...]"
    from predefined_clients import clients as predefined_clients_list
except Exception:
    predefined_clients_list = None


def federated_loop(server: Server, clients: list, nrounds: int, epochs: int,
                   saving_path: str, architecture: str, pretrained_weights: str,
                   data: str, bsz_train: int, bsz_val: int, imgsz: int,
                   conf_thres: float, iou_thres: float, cfg: str, hyp: str,
                   workers: int, selection_ratio: float = 1.0, 
                   oort_start_round: int = 0, use_random: bool = False) -> None:

    
    # Register clients to Oort if used
    if server.use_oort and not use_random:
        for client in clients:
            server.oort_sampler.register_client(
                client.rank,
                client.nsamples,
                duration=2000
            )
    
    # Server initializes model
    print("Initializing model on server...")
    server.initialize_model(pretrained_weights)
    server.post_init_update(data=data, cfg=cfg, hyp=hyp, imgsz=imgsz)
    
    total_clients = len(clients)
    print(f"[Main] Total clients = {total_clients}")
    
    if use_random:
        print(f"[Main] Selection method: RANDOM")
        print(f"[Main] Selection ratio = {selection_ratio} ({int(total_clients * selection_ratio)} clients per round)")
        print(f"[Main] Round 0: ALL clients train {epochs} epochs to initialize weights")
    elif server.use_oort:
        print(f"[Main] Selection method: OORT")
        print(f"[Main] Oort start round: {oort_start_round}")
        print(f"[Main] Selection ratio = {selection_ratio}")
    else:
        print(f"[Main] Selection method: ALL CLIENTS")
    
    for kround in range(nrounds):
        print(f"\n{'='*50}")
        print(f"Round {kround + 1}/{nrounds}")
        print(f"{'='*50}")
        
        # Round 0: share initial weights with clients
        if kround == 0:
            initial_weights = server.get_weights(metadata=True)
            for client in clients:
                client.set_weights(initial_weights, metadata=True)
                client.post_init_update(data=data, cfg=cfg, hyp=hyp, imgsz=imgsz)
        
        # CLIENT SELECTION
        
        num_to_select = max(1, int(total_clients * selection_ratio))
        
        # Random selection
        if use_random and kround < 5:
            selected_clients = clients  # ALL clients
            print(f"\n[Warmup Round {kround + 1}/5] Training ALL {len(clients)} clients with {epochs} epochs")
            print(f"  Purpose: Initialize weights for future random selection")
        
       
        elif use_random:
            selected_clients = random.sample(clients, num_to_select)
            selected_ranks = [c.rank for c in selected_clients]
            print(f"\n[Random Selection] Selected {len(selected_clients)}/{total_clients} clients")
            print(f"[Random Selection] Selected client ranks: {selected_ranks}")
            
        # Oort selection
        elif server.use_oort and kround >= oort_start_round:
            feasible = [c.rank for c in clients]
            print(f"\n[Oort] round={kround} selecting num_to_select={num_to_select}")
            selected_ranks = server.oort_sampler.select_clients(
                num_clients=num_to_select,
                feasible_clients=feasible,
                round_num=kround
            )
            print(f"[Oort DEBUG] selected_ranks = {selected_ranks}")
            selected_clients = [c for c in clients if c.rank in selected_ranks]
            print(f"[Oort] Selected {len(selected_clients)}/{len(clients)} clients -> {[c.rank for c in selected_clients]}")
        
        # All clients
        else:
            selected_clients = clients
            if server.use_oort:
                print(f"\n[Oort] Warmup/start phase (round {kround}): using all clients")
        
        updates = []
        nsamples_list = []
        client_durations = []
        client_losses = []
        client_utilities = []
        
        for client in selected_clients:
            print(f"\n--- Training Client {client.rank} ---")
            
            train_start = time.time()
            
            client.train(
                nrounds=nrounds,
                kround=kround,
                epochs=epochs,
                architecture=architecture,
                data=data,
                bsz_train=bsz_train,
                imgsz=imgsz,
                cfg=cfg,
                hyp=hyp,
                workers=workers,
                saving_path=saving_path
            )
            
            train_duration = time.time() - train_start
            client_durations.append(train_duration)
            update = client.get_update()
            updates.append(update)
            nsamples_list.append(client.nsamples)
            
            # Extract loss for Oort (if used)
            if server.use_oort and not use_random:
                try:
                    client_loss = client.extract_losses(
                        saving_path=saving_path,
                        epochs=epochs,
                        nrounds=kround 
                    )
                except Exception as e:
                    print(f"  Warning: Could not extract loss: {e}")
                    client_loss = 2.0 
                
                client_losses.append(client_loss)
                # utility = math.sqrt(max(client_loss, 1e-10)) * client.nsamples
                # client_utilities.append(utility)
                
                print(f"  Loss: {client_loss:.8f}")
                print(f"  Duration: {train_duration:.4f}s")
                print(f"  Samples: {client.nsamples}")
                # print(f"  Utility: {utility:.4f}")
                
                if kround >= oort_start_round:
                    server.oort_sampler.update_client(
                        client.rank,
                        loss=client_loss,
                        duration=train_duration,
                        round_num=kround
                    )
            else:
                print(f"  Duration: {train_duration:.4f}s")
                print(f"  Samples: {client.nsamples}")
        
        # Server aggregation
        print(f"\n--- Server Aggregation ---")
        server.aggregate(updates, nsamples_list)
        server.reparameterize(architecture)
        
        # Server evaluation
        print(f"\n--- Server Evaluation ---")
        server.test(kround, saving_path, data, bsz_val, imgsz, conf_thres, iou_thres)
        
        # Broadcast new weights to ALL clients (not just selected ones)
        new_weights = server.get_weights(metadata=False)
        for client in clients:
            client.set_weights(new_weights, metadata=False)


def gather_analytics(saving_path: str, clients: list, server=None) -> None:
    """Gather local analytics from clients; if server (with Oort) provided, save Oort utilities too."""
    os.makedirs(f'{saving_path}/run/local-analytics/', exist_ok=True)
    
    for client in clients:
        rank = client.rank
        try:
            df_lr = pd.read_csv(f'{saving_path}/run/train-client{rank}/optim_params.csv')
            df_loss = pd.read_csv(f'{saving_path}/run/train-client{rank}/training_losses.csv')
            
            df_lr.to_csv(f'{saving_path}/run/local-analytics/optim_params_{rank}.csv', index=False)
            df_loss.to_csv(f'{saving_path}/run/local-analytics/training_losses_{rank}.csv', index=False)
            
            if os.path.exists(f'{saving_path}/run/train-client{rank}/opt.yaml'):
                with open(f'{saving_path}/run/train-client{rank}/opt.yaml') as f:
                    save_yaml = yaml.load(f, Loader=yaml.SafeLoader)
                with open(f'{saving_path}/run/local-analytics/opt_{rank}.yaml', 'w') as f:
                    yaml.dump(save_yaml, f)
        except Exception as e:
            print(f"Warning: Could not gather analytics for client {rank}: {e}")
    
    # Save Oort utilities if server provided
    if server is not None and getattr(server, 'use_oort', False) and hasattr(server, 'oort_sampler'):
        try:
            utilities = {}
            for client_id, info in server.oort_sampler.selector.totalArms.items():
                utilities[client_id] = {
                    'reward': info.get('reward', 0.0),
                    'duration': info.get('duration', 0.0),
                    'count': info.get('count', 0),
                    'last_selected_round': info.get('time_stamp', -1)
                }
            df_util = pd.DataFrame.from_dict(utilities, orient='index')
            df_util.to_csv(f'{saving_path}/run/local-analytics/oort_utilities.csv')
            print(f"[INFO] Oort utilities saved for {len(utilities)} clients.")
            
            server.oort_sampler.selector.export_utility_pivot(
                f'{saving_path}/run/local-analytics/client_utilities_pivot.csv'
            )
            print(f"[INFO] Oort utility pivot CSV exported.")
            
        except Exception as e:
            print(f"Warning: Could not gather Oort utilities: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nrounds', type=int, default=30, help='number of communication rounds')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs per round')
    parser.add_argument('--server-opt', type=str, default='fedavg', help='aggregation algorithm')
    parser.add_argument('--server-lr', type=float, default=1., help='server learning rate')
    parser.add_argument('--tau', type=float, default=1e-3, help='server adaptivity')
    parser.add_argument('--beta', type=float, default=0.1, help='server momentum')
    parser.add_argument('--architecture', type=str, default='yolov7', help='model architecture')
    parser.add_argument('--weights', type=str, required=True, help='path to pretrained weights')
    parser.add_argument('--data', type=str, required=True, help='*.yaml path')
    parser.add_argument('--bsz-train', type=int, default=16, help='batch size for training')
    parser.add_argument('--bsz-val', type=int, default=16, help='batch size for evaluation')
    parser.add_argument('--img', type=int, default=640, help='image size')
    parser.add_argument('--conf', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--cfg', type=str, default='yolov7/cfg/training/yolov7.yaml', help='model.yaml path')
    parser.add_argument('--hyp', type=str, required=True, help='hyperparameters path')
    parser.add_argument('--workers', type=int, default=4, help='number of workers')
    
    # ===== CLIENT SELECTION PARAMETERS =====
    parser.add_argument('--use-random', action='store_true', 
                        help='Use RANDOM client selection instead of Oort')
    parser.add_argument('--use-oort', action='store_true', 
                        help='Enable Oort client selection')
    parser.add_argument('--selection-ratio', type=float, default=1.0, 
                        help='Fraction of clients to select each round (0.0-1.0)')
    parser.add_argument('--oort-start-round', type=int, default=1, 
                        help='Round index to start Oort client selection (if using Oort)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Validation
    if args.use_random and args.use_oort:
        raise ValueError("Cannot use both --use-random and --use-oort at the same time")
    
    # Initialize server
    print("Initializing server...")
    server = Server(args.server_opt, args.server_lr, args.tau, args.beta, 
                   use_oort=args.use_oort and not args.use_random)
    server.get_device_info()
    
    # Initialize clients: detect from dataset folder
    print("Detecting clients from dataset folder...")
    with open(args.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)
    
    train_root = data_dict.get('train')
    if train_root is None:
        raise RuntimeError("data yaml must contain 'train' path (where client subfolders are).")
    
    client_dirs = sorted([d for d in os.listdir(train_root) if d.startswith("client")])
    clients = []
    
    for idx, dirname in enumerate(client_dirs, start=1):
        client = Client(rank=idx)
        client.get_device_info()
        img_path = os.path.join(train_root, dirname, "images")
        try:
            client.nsamples = len(os.listdir(img_path))
        except Exception:
            client.nsamples = 1
        clients.append(client)
    
    print(f"Detected {len(clients)} clients from {train_root}.")
    
    # Validate selection ratio
    if not 0.0 < args.selection_ratio <= 1.0:
        raise ValueError(f"selection_ratio must be in (0.0, 1.0], got {args.selection_ratio}")
    
    # Create saving folder
    saving_path = 'experiments'
    os.makedirs(saving_path, exist_ok=True)
    os.makedirs(f'{saving_path}/weights/', exist_ok=True)
    os.makedirs(f'{saving_path}/run/', exist_ok=True)
    
    # Determine selection method
    if args.use_random:
        selection_method = 'random'
    elif args.use_oort:
        selection_method = 'oort'
    else:
        selection_method = 'all_clients'
    
    # Save configuration
    with open(f'{saving_path}/config.txt', 'w') as f:
        f.write(f'nrounds: {args.nrounds}\n')
        f.write(f'epochs: {args.epochs}\n')
        f.write(f'total_clients: {len(clients)}\n')
        f.write(f'selection_method: {selection_method}\n')
        f.write(f'selection_ratio: {args.selection_ratio}\n')
        f.write(f'clients_per_round: {max(1, int(len(clients) * args.selection_ratio))}\n')
        if args.use_oort:
            f.write(f'oort_start_round: {args.oort_start_round}\n')
        f.write(f'server opt: {args.server_opt}\n')
        f.write(f'server learning rate: {args.server_lr}\n')
        if args.server_opt == 'fedavgm':
            f.write(f'beta: {args.beta}\n')
        if args.server_opt in ['fedadagrad', 'fedadam', 'fedyogi']:
            f.write(f'tau: {args.tau}\n')
        f.write(f'architecture: {args.architecture}\n')
        f.write(f'weights: {args.weights}\n')
        f.write(f'data: {args.data}\n')
        f.write(f'batch size (train): {args.bsz_train}\n')
        f.write(f'batch size (eval): {args.bsz_val}\n')
        f.write(f'img: {args.img}\n')
        f.write(f'cfg: {args.cfg}\n')
        f.write(f'hyp: {args.hyp}\n')
    
    try:
        shutil.copy(args.cfg, saving_path)
        shutil.copy(args.hyp, saving_path)
        shutil.copy(args.data, saving_path)
    except Exception as e:
        print(f"Warning: Could not copy config files: {e}")
    
    # Run federated learning
    print("\n" + "="*60)
    if args.use_random:
        print("Starting Federated Learning with RANDOM CLIENT SELECTION")
    elif args.use_oort:
        print("Starting Federated Learning with OORT CLIENT SELECTION")
    else:
        print("Starting Federated Learning with ALL CLIENTS")
    print(f"Total clients: {len(clients)}")
    print(f"Selection ratio: {args.selection_ratio}")
    print(f"Clients per round: {max(1, int(len(clients) * args.selection_ratio))}")
    print("="*60 + "\n")
    
    federated_loop(
        server=server,
        clients=clients,
        nrounds=args.nrounds,
        epochs=args.epochs,
        saving_path=saving_path,
        architecture=args.architecture,
        pretrained_weights=args.weights,
        data=args.data,
        bsz_train=args.bsz_train,
        bsz_val=args.bsz_val,
        imgsz=args.img,
        conf_thres=args.conf,
        iou_thres=args.iou,
        cfg=args.cfg,
        hyp=args.hyp,
        workers=args.workers,
        selection_ratio=args.selection_ratio,
        oort_start_round=args.oort_start_round,
        use_random=args.use_random
    )
    
    # Gather analytics (and save Oort utilities if used)
    print("\nGathering analytics...")
    gather_analytics(saving_path, clients, server=server)
    
    print("\nFederated learning completed!")