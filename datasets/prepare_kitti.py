# # FedPylot by Cyprien Quéméneur, GPL-3.0 license

# import argparse
# import os
# from PIL import Image
# import random
# import shutil
# from tqdm import tqdm
# from datasets_utils import create_directories, archive_directories, get_distribution_dataframe, convert_bbox

# KITTI_TRAIN_SIZE = 7481
# DEFAULT_CLASS_MAP = {
#     'Car': 0,
#     'Van': 1,
#     'Truck': 2,
#     'Pedestrian': 3,
#     'Person_sitting': 4,
#     'Cyclist': 5,
#     'Tram': 6,
#     'Misc': 7
# }


# def get_iid_splits(nclients: int, val_frac: float) -> dict:
#     """Return a dictionary to store IID and balanced mapping of KITTI data split."""
#     random.seed(0)
#     client_frac = (1 - val_frac) / nclients
#     indices = list(range(KITTI_TRAIN_SIZE))
#     client_split_size = int(KITTI_TRAIN_SIZE * client_frac)
#     splits = {}
#     # Create the client splits
#     for k in range(1, nclients + 1):
#         client_data = random.sample(indices, client_split_size)
#         for index in client_data:
#             splits[index] = f'client{k}'
#             indices.remove(index)
#     # Create the server split
#     for index in indices:
#         splits[index] = 'server'
#     return splits


# def process_kitti(img_path: str, label_path: str, target_path: str, data: str, class_map: dict, nclients: int,
#                   val_frac: float, tar: bool) -> None:
#     """Convert KITTI annotations and split the data among the server and clients."""
#     print('Converting annotations and splitting data...')
#     create_directories(target_path, nclients)
#     splits = get_iid_splits(nclients, val_frac)
#     objects_distribution = get_distribution_dataframe(data, nclients)
#     # Iterate over KITTI training labels
#     for fname in tqdm(os.listdir(label_path)):
#         # Create target file
#         destination = splits[int(fname[:-4])]
#         objects_distribution.loc['Samples', destination] += 1
#         with (open(f'{target_path}/{destination}/labels/{fname}', 'w') as target_file):
#             # Open KITTI training label
#             with open(f'{label_path}/{fname}', 'r') as label_file:
#                 # Open KITTI corresponding image and extract image width and height
#                 with open(f'{img_path}/{fname[:-3]}png', 'rb') as img_file:
#                     img = Image.open(img_file)
#                     img_width, img_height = img.size
#                 # Copy the image to its destination without deleting the original file
#                 shutil.copyfile(f'{img_path}/{fname[:-3]}png', f'{target_path}/{destination}/images/{fname[:-3]}png')
#                 # Iterate over KITTI training label lines
#                 for line in label_file.readlines():
#                     line = line.split()
#                     obj_type, _, _, _, bbox_left, bbox_top, bbox_right, bbox_bottom, *_ = line
#                     # Skip line with DontCare type
#                     if obj_type == 'DontCare':
#                         continue
#                     # Convert KITTI training label line to YOLO format [class_id, x, y, w, h]
#                     class_id = class_map[obj_type]
#                     x, y, w, h = convert_bbox(
#                         bbox_left=float(bbox_left),
#                         bbox_top=float(bbox_top),
#                         bbox_right=float(bbox_right),
#                         bbox_bottom=float(bbox_bottom),
#                         img_width=img_width,
#                         img_height=img_height
#                     )
#                     # Write processed label line to target file
#                     target_file.write(f'{class_id} {x} {y} {w} {h}\n')
#                     # Update object distribution
#                     objects_distribution.loc[obj_type, destination] += 1
#     # Save objects distribution
#     objects_distribution.to_csv(f'{target_path}/objects_distribution.csv')
#     # Archive the directories of the federated participants
#     if tar:
#         print('Archiving...')
#         archive_directories(target_path, nclients)


# if __name__ == '__main__':
#     args = argparse.ArgumentParser()
#     args.add_argument('--img-path', type=str, default='datasets/data_object_image_2/training/image_2', help='path to images')
#     args.add_argument('--label-path', type=str, default='datasets/data_object_label_2/training/label_2', help='path to labels')
#     args.add_argument('--target-path', type=str, default='datasets/kitti', help='path to target directory')
#     args.add_argument('--data', type=str, default='data/kitti.yaml', help='path to data yaml file')
#     args.add_argument('--class-map', type=dict, default=DEFAULT_CLASS_MAP, help='map between annotations, should match yaml file')
#     args.add_argument('--nclients', type=int, default=5, help='number of clients in federated experiment')
#     args.add_argument('--val-frac', type=float, default=0.25, help='fraction of data held by the server for validation')
#     args.add_argument('--tar', action='store_true', help='archive the directories of the federated participants')
#     args = args.parse_args()
#     process_kitti(args.img_path, args.label_path, args.target_path, args.data, args.class_map, args.nclients, args.val_frac, args.tar)

import argparse
import os
from PIL import Image
import random
import shutil
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from datasets_utils import create_directories, archive_directories, get_distribution_dataframe, convert_bbox

KITTI_TRAIN_SIZE = 7481
DEFAULT_CLASS_MAP = {
    'Car': 0,
    'Van': 1,
    'Truck': 2,
    'Pedestrian': 3,
    'Person_sitting': 4,
    'Cyclist': 5,
    'Tram': 6,
    'Misc': 7
}


def get_labels_from_kitti(label_path: str, class_map: dict) -> dict:
    """Trích xuất class chủ đạo cho mỗi ảnh từ KITTI labels."""
    image_labels = {}
    for fname in os.listdir(label_path):
        img_id = int(fname[:-4])
        class_counts = defaultdict(int)
        
        with open(f'{label_path}/{fname}', 'r') as f:
            for line in f.readlines():
                obj_type = line.split()[0]
                if obj_type != 'DontCare' and obj_type in class_map:
                    class_counts[class_map[obj_type]] += 1
        
        # Gán class có nhiều objects nhất, hoặc random nếu không có object
        if class_counts:
            image_labels[img_id] = max(class_counts, key=class_counts.get)
        else:
            image_labels[img_id] = random.choice(list(class_map.values()))
    
    return image_labels


def get_iid_splits(nclients: int, val_frac: float) -> dict:
    """Return a dictionary to store IID and balanced mapping of KITTI data split."""
    random.seed(0)
    client_frac = (1 - val_frac) / nclients
    indices = list(range(KITTI_TRAIN_SIZE))
    client_split_size = int(KITTI_TRAIN_SIZE * client_frac)
    splits = {}
    # Create the client splits
    for k in range(1, nclients + 1):
        client_data = random.sample(indices, client_split_size)
        for index in client_data:
            splits[index] = f'client{k}'
            indices.remove(index)
    # Create the server split
    for index in indices:
        splits[index] = 'server'
    return splits


def get_dirichlet_splits(label_path: str, class_map: dict, nclients: int, val_frac: float, alpha: float) -> dict:
    """
    Chia dữ liệu theo phân phối Dirichlet (Non-IID).
    
    Args:
        label_path: đường dẫn đến labels KITTI
        class_map: mapping từ tên class sang ID
        nclients: số lượng clients
        val_frac: tỷ lệ validation cho server
        alpha: tham số Dirichlet
    
    Returns:
        dict: {image_id: 'client1'/'client2'/.../'server'}
    """
    print(f'Creating Dirichlet Non-IID splits with alpha={alpha}...')
    random.seed(0)
    np.random.seed(0)
    
    # Bước 1: Lấy labels cho tất cả ảnh
    image_labels = get_labels_from_kitti(label_path, class_map)
    n_classes = len(class_map)
    
    # Bước 2: Nhóm ảnh theo class
    class_indices = defaultdict(list)
    for img_id, label in image_labels.items():
        class_indices[label].append(img_id)
    
    # Shuffle
    for label in class_indices:
        random.shuffle(class_indices[label])
    
    # Bước 3: Tách validation set
    val_indices = []
    train_class_indices = defaultdict(list)
    
    for class_id in range(n_classes):
        indices = class_indices[class_id]
        n_val = int(len(indices) * val_frac)
        val_indices.extend(indices[:n_val])
        train_class_indices[class_id] = indices[n_val:]
    
    # Bước 4: Chia training data theo Dirichlet
    client_indices = [[] for _ in range(nclients)]
    
    for class_id in range(n_classes):
        indices = train_class_indices[class_id]
        if len(indices) == 0:
            continue
        
        # Sinh phân phối Dirichlet
        proportions = np.random.dirichlet([alpha] * nclients)
        proportions = proportions / proportions.sum()
        splits = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        
        # Gán cho clients
        for client_id, idx_subset in enumerate(np.split(indices, splits)):
            client_indices[client_id].extend(idx_subset)
    
    # Bước 5: Tạo dict kết quả
    splits = {}
    for client_id, indices in enumerate(client_indices):
        for idx in indices:
            splits[idx] = f'client{client_id + 1}'
    
    for idx in val_indices:
        splits[idx] = 'server'
    
    # In thống kê
    print(f'\nData distribution:')
    print(f'Server (validation): {len(val_indices)} samples')
    for i, indices in enumerate(client_indices):
        print(f'Client {i+1}: {len(indices)} samples')
    
    # In phân phối classes
    print(f'\nClass distribution per client:')
    inv_class_map = {v: k for k, v in class_map.items()}
    
    for client_id in range(nclients):
        client_imgs = client_indices[client_id]
        class_dist = defaultdict(int)
        for img_id in client_imgs:
            class_dist[image_labels[img_id]] += 1
        
        print(f'\nClient {client_id + 1}:')
        for class_id, count in sorted(class_dist.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(client_imgs)) * 100
            print(f'  {inv_class_map[class_id]:20s}: {count:4d} ({percentage:5.1f}%)')
    
    return splits


def process_kitti(img_path: str, label_path: str, target_path: str, data: str, class_map: dict, nclients: int,
                  val_frac: float, partition: str, alpha: float, tar: bool) -> None:
    """Convert KITTI annotations and split the data among the server and clients."""
    print('Converting annotations and splitting data...')
    create_directories(target_path, nclients)
    
    # Chọn phương pháp chia dữ liệu
    if partition == 'iid':
        splits = get_iid_splits(nclients, val_frac)
    elif partition == 'dirichlet':
        splits = get_dirichlet_splits(label_path, class_map, nclients, val_frac, alpha)
    else:
        raise ValueError(f'Unknown partition strategy: {partition}. Choose "iid" or "dirichlet"')
    
    objects_distribution = get_distribution_dataframe(data, nclients)
    
    # Iterate over KITTI training labels
    for fname in tqdm(os.listdir(label_path)):
        img_id = int(fname[:-4])
        destination = splits[img_id]
        objects_distribution.loc['Samples', destination] += 1
        
        with open(f'{target_path}/{destination}/labels/{fname}', 'w') as target_file:
            # Open KITTI training label
            with open(f'{label_path}/{fname}', 'r') as label_file:
                # Open KITTI corresponding image and extract image width and height
                with open(f'{img_path}/{fname[:-3]}png', 'rb') as img_file:
                    img = Image.open(img_file)
                    img_width, img_height = img.size
                # Copy the image to its destination without deleting the original file
                shutil.copyfile(f'{img_path}/{fname[:-3]}png', f'{target_path}/{destination}/images/{fname[:-3]}png')
                # Iterate over KITTI training label lines
                for line in label_file.readlines():
                    line = line.split()
                    obj_type, _, _, _, bbox_left, bbox_top, bbox_right, bbox_bottom, *_ = line
                    # Skip line with DontCare type
                    if obj_type == 'DontCare':
                        continue
                    # Convert KITTI training label line to YOLO format [class_id, x, y, w, h]
                    class_id = class_map[obj_type]
                    x, y, w, h = convert_bbox(
                        bbox_left=float(bbox_left),
                        bbox_top=float(bbox_top),
                        bbox_right=float(bbox_right),
                        bbox_bottom=float(bbox_bottom),
                        img_width=img_width,
                        img_height=img_height
                    )
                    # Write processed label line to target file
                    target_file.write(f'{class_id} {x} {y} {w} {h}\n')
                    # Update object distribution
                    objects_distribution.loc[obj_type, destination] += 1
    
    # Save objects distribution
    objects_distribution.to_csv(f'{target_path}/objects_distribution.csv')
    print(f'\nObjects distribution saved to {target_path}/objects_distribution.csv')
    
    # Archive the directories of the federated participants
    if tar:
        print('Archiving...')
        archive_directories(target_path, nclients)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--img-path', type=str, default='datasets/data_object_image_2/training/image_2', help='path to images')
    args.add_argument('--label-path', type=str, default='datasets/data_object_label_2/training/label_2', help='path to labels')
    args.add_argument('--target-path', type=str, default='datasets/kitti', help='path to target directory')
    args.add_argument('--data', type=str, default='data/kitti.yaml', help='path to data yaml file')
    args.add_argument('--class-map', type=dict, default=DEFAULT_CLASS_MAP, help='map between annotations, should match yaml file')
    args.add_argument('--nclients', type=int, default=5, help='number of clients in federated experiment')
    args.add_argument('--val-frac', type=float, default=0.25, help='fraction of data held by the server for validation')
    args.add_argument('--partition', type=str, default='iid', choices=['iid', 'dirichlet'], 
                      help='data partitioning strategy: iid or dirichlet')
    args.add_argument('--alpha', type=float, default=0.5, 
                      help='Dirichlet alpha parameter (smaller = more non-IID). Only used when partition=dirichlet')
    args.add_argument('--tar', action='store_true', help='archive the directories of the federated participants')
    args = args.parse_args()
    
    process_kitti(args.img_path, args.label_path, args.target_path, args.data, args.class_map, 
                  args.nclients, args.val_frac, args.partition, args.alpha, args.tar)