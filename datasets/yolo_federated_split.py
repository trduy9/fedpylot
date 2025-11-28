import argparse
import os
import random
import shutil
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import pandas as pd


DEFAULT_CLASS_MAP = {
    'motorbike': 0,
    'car': 1,
    'bus': 2,
    'truck': 3
}

# Reverse map for display
INV_CLASS_MAP = {v: k for k, v in DEFAULT_CLASS_MAP.items()}


def create_directories(target_path: str, nclients: int) -> None:
    """Tạo cấu trúc thư mục cho server và clients."""
    # Tạo thư mục cho server
    os.makedirs(f'{target_path}/server/images', exist_ok=True)
    os.makedirs(f'{target_path}/server/labels', exist_ok=True)
    
    # Tạo thư mục cho clients
    for i in range(1, nclients + 1):
        os.makedirs(f'{target_path}/client{i}/images', exist_ok=True)
        os.makedirs(f'{target_path}/client{i}/labels', exist_ok=True)
    
    print(f'Created directories for server and {nclients} clients at {target_path}')


def archive_directories(target_path: str, nclients: int) -> None:
    """Nén các thư mục thành file tar."""
    # Archive server
    shutil.make_archive(f'{target_path}/server', 'tar', target_path, 'server')
    
    # Archive clients
    for i in range(1, nclients + 1):
        shutil.make_archive(f'{target_path}/client{i}', 'tar', target_path, f'client{i}')
    
    print('Archived all directories')


def get_distribution_dataframe(nclients: int) -> pd.DataFrame:
    """Tạo DataFrame để theo dõi phân phối objects."""
    columns = ['server'] + [f'client{i}' for i in range(1, nclients + 1)]
    rows = ['Samples'] + list(INV_CLASS_MAP.values())
    
    df = pd.DataFrame(0, index=rows, columns=columns)
    return df


def get_image_files(img_path: str) -> list:
    """Lấy danh sách các file ảnh."""
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = []
    
    for fname in os.listdir(img_path):
        ext = os.path.splitext(fname)[1].lower()
        if ext in valid_extensions:
            image_files.append(fname)
    
    return sorted(image_files)


def get_labels_from_yolo(label_path: str, image_files: list, n_classes: int) -> dict:
    """Trích xuất class chủ đạo cho mỗi ảnh từ YOLO labels."""
    image_labels = {}
    
    for img_file in image_files:
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_filepath = os.path.join(label_path, label_file)
        
        class_counts = defaultdict(int)
        
        if os.path.exists(label_filepath):
            with open(label_filepath, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if line:
                        parts = line.split()
                        class_id = int(parts[0])
                        if 0 <= class_id < n_classes:
                            class_counts[class_id] += 1
        
        # Gán class có nhiều objects nhất, hoặc random nếu không có object
        if class_counts:
            image_labels[img_file] = max(class_counts, key=class_counts.get)
        else:
            image_labels[img_file] = random.choice(list(range(n_classes)))
    
    return image_labels


def get_iid_splits(image_files: list, nclients: int, val_frac: float) -> dict:
    """Chia dữ liệu theo IID (mỗi client có phân phối giống nhau)."""
    print('Creating IID splits...')
    random.seed(0)
    
    n_samples = len(image_files)
    client_frac = (1 - val_frac) / nclients
    indices = list(range(n_samples))
    random.shuffle(indices)
    
    client_split_size = int(n_samples * client_frac)
    splits = {}
    
    # Chia cho clients
    for k in range(1, nclients + 1):
        client_data = indices[:client_split_size]
        indices = indices[client_split_size:]
        for idx in client_data:
            splits[image_files[idx]] = f'client{k}'
    
    # Phần còn lại cho server (validation)
    for idx in indices:
        splits[image_files[idx]] = 'server'
    
    return splits


def get_dirichlet_splits(label_path: str, image_files: list, n_classes: int, 
                         nclients: int, val_frac: float, alpha: float) -> dict:
    """
    Chia dữ liệu theo phân phối Dirichlet (Non-IID).
    
    Args:
        label_path: đường dẫn đến labels YOLO
        image_files: danh sách file ảnh
        n_classes: số lượng classes
        nclients: số lượng clients
        val_frac: tỷ lệ validation cho server
        alpha: tham số Dirichlet (nhỏ hơn = non-IID hơn)
    
    Returns:
        dict: {image_file: 'client1'/'client2'/.../'server'}
    """
    print(f'Creating Dirichlet Non-IID splits with alpha={alpha}...')
    random.seed(0)
    np.random.seed(0)
    
    # Bước 1: Lấy labels cho tất cả ảnh
    image_labels = get_labels_from_yolo(label_path, image_files, n_classes)
    
    # Bước 2: Nhóm ảnh theo class
    class_indices = defaultdict(list)
    for img_file, label in image_labels.items():
        class_indices[label].append(img_file)
    
    # Shuffle
    for label in class_indices:
        random.shuffle(class_indices[label])
    
    # Bước 3: Tách validation set (stratified)
    val_files = []
    train_class_indices = defaultdict(list)
    
    for class_id in range(n_classes):
        indices = class_indices[class_id]
        n_val = int(len(indices) * val_frac)
        val_files.extend(indices[:n_val])
        train_class_indices[class_id] = indices[n_val:]
    
    # Bước 4: Chia training data theo Dirichlet
    client_files = [[] for _ in range(nclients)]
    
    for class_id in range(n_classes):
        files = train_class_indices[class_id]
        if len(files) == 0:
            continue
        
        # Sinh phân phối Dirichlet
        proportions = np.random.dirichlet([alpha] * nclients)
        proportions = proportions / proportions.sum()
        splits_idx = (np.cumsum(proportions) * len(files)).astype(int)[:-1]
        
        # Gán cho clients
        file_splits = np.split(np.array(files), splits_idx)
        for client_id, file_subset in enumerate(file_splits):
            client_files[client_id].extend(file_subset.tolist())
    
    # Bước 5: Tạo dict kết quả
    splits = {}
    for client_id, files in enumerate(client_files):
        for f in files:
            splits[f] = f'client{client_id + 1}'
    
    for f in val_files:
        splits[f] = 'server'
    
    # In thống kê
    print(f'\n{"="*60}')
    print('DATA DISTRIBUTION SUMMARY')
    print(f'{"="*60}')
    print(f'Server (validation): {len(val_files)} samples')
    for i, files in enumerate(client_files):
        print(f'Client {i+1}: {len(files)} samples')
    
    # In phân phối classes cho mỗi client
    print(f'\n{"="*60}')
    print('CLASS DISTRIBUTION PER CLIENT')
    print(f'{"="*60}')
    
    for client_id in range(nclients):
        client_imgs = client_files[client_id]
        if len(client_imgs) == 0:
            print(f'\nClient {client_id + 1}: No samples')
            continue
            
        class_dist = defaultdict(int)
        for img_file in client_imgs:
            class_dist[image_labels[img_file]] += 1
        
        print(f'\nClient {client_id + 1}:')
        for class_id in range(n_classes):
            count = class_dist[class_id]
            percentage = (count / len(client_imgs)) * 100 if len(client_imgs) > 0 else 0
            print(f'  {INV_CLASS_MAP[class_id]:15s}: {count:5d} ({percentage:5.1f}%)')
    
    # In phân phối cho server
    print(f'\nServer (validation):')
    server_class_dist = defaultdict(int)
    for img_file in val_files:
        server_class_dist[image_labels[img_file]] += 1
    
    for class_id in range(n_classes):
        count = server_class_dist[class_id]
        percentage = (count / len(val_files)) * 100 if len(val_files) > 0 else 0
        print(f'  {INV_CLASS_MAP[class_id]:15s}: {count:5d} ({percentage:5.1f}%)')
    
    return splits


def process_yolo_dataset(img_path: str, label_path: str, target_path: str, 
                         nclients: int, val_frac: float, partition: str, 
                         alpha: float, tar: bool, n_classes: int = 4) -> None:
    """
    Xử lý và chia dataset YOLO cho federated learning.
    
    Args:
        img_path: đường dẫn đến thư mục images
        label_path: đường dẫn đến thư mục labels
        target_path: đường dẫn đến thư mục đích
        nclients: số lượng clients
        val_frac: tỷ lệ validation
        partition: phương pháp chia ('iid' hoặc 'dirichlet')
        alpha: tham số Dirichlet
        tar: có nén thư mục không
        n_classes: số lượng classes
    """
    print(f'\n{"="*60}')
    print('YOLO FEDERATED DATASET SPLITTER')
    print(f'{"="*60}')
    print(f'Image path: {img_path}')
    print(f'Label path: {label_path}')
    print(f'Target path: {target_path}')
    print(f'Number of clients: {nclients}')
    print(f'Validation fraction: {val_frac}')
    print(f'Partition strategy: {partition}')
    if partition == 'dirichlet':
        print(f'Dirichlet alpha: {alpha}')
    print(f'{"="*60}\n')
    
    # Tạo thư mục
    create_directories(target_path, nclients)
    
    # Lấy danh sách ảnh
    image_files = get_image_files(img_path)
    print(f'Found {len(image_files)} images')
    
    if len(image_files) == 0:
        print('ERROR: No images found!')
        return
    
    # Chọn phương pháp chia dữ liệu
    if partition == 'iid':
        splits = get_iid_splits(image_files, nclients, val_frac)
    elif partition == 'dirichlet':
        splits = get_dirichlet_splits(label_path, image_files, n_classes, 
                                      nclients, val_frac, alpha)
    else:
        raise ValueError(f'Unknown partition strategy: {partition}. Choose "iid" or "dirichlet"')
    
    # Tạo DataFrame theo dõi phân phối
    objects_distribution = get_distribution_dataframe(nclients)
    
    # Copy files và đếm objects
    print('\nCopying files...')
    for img_file in tqdm(image_files):
        destination = splits[img_file]
        objects_distribution.loc['Samples', destination] += 1
        
        # Copy image
        src_img = os.path.join(img_path, img_file)
        dst_img = os.path.join(target_path, destination, 'images', img_file)
        shutil.copyfile(src_img, dst_img)
        
        # Copy và xử lý label
        label_file = os.path.splitext(img_file)[0] + '.txt'
        src_label = os.path.join(label_path, label_file)
        dst_label = os.path.join(target_path, destination, 'labels', label_file)
        
        if os.path.exists(src_label):
            # Đọc và copy label, đồng thời đếm objects
            with open(src_label, 'r') as f_in:
                with open(dst_label, 'w') as f_out:
                    for line in f_in.readlines():
                        line = line.strip()
                        if line:
                            parts = line.split()
                            class_id = int(parts[0])
                            if 0 <= class_id < n_classes:
                                f_out.write(line + '\n')
                                class_name = INV_CLASS_MAP.get(class_id, f'class_{class_id}')
                                if class_name in objects_distribution.index:
                                    objects_distribution.loc[class_name, destination] += 1
        else:
            # Tạo file label trống nếu không tồn tại
            open(dst_label, 'w').close()
    
    # Lưu phân phối objects
    distribution_path = os.path.join(target_path, 'objects_distribution.csv')
    objects_distribution.to_csv(distribution_path)
    print(f'\nObjects distribution saved to {distribution_path}')
    
    # In bảng phân phối
    print(f'\n{"="*60}')
    print('OBJECTS DISTRIBUTION')
    print(f'{"="*60}')
    print(objects_distribution.to_string())
    
    # Tạo file YAML cho mỗi participant
    create_yaml_files(target_path, nclients, n_classes)
    
    # Nén thư mục nếu cần
    if tar:
        print('\nArchiving directories...')
        archive_directories(target_path, nclients)
    
    print(f'\n{"="*60}')
    print('DONE!')
    print(f'{"="*60}')


def create_yaml_files(target_path: str, nclients: int, n_classes: int) -> None:
    """Tạo file YAML config cho mỗi participant."""
    
    yaml_template = """# YOLO Dataset Configuration
path: {path}
train: images
val: images

# Classes
nc: {nc}
names:
  0: motorbike
  1: car
  2: bus
  3: truck
"""
    
    # Tạo YAML cho server
    server_yaml = yaml_template.format(
        path=os.path.abspath(os.path.join(target_path, 'server')),
        nc=n_classes
    )
    with open(os.path.join(target_path, 'server', 'data.yaml'), 'w') as f:
        f.write(server_yaml)
    
    # Tạo YAML cho clients
    for i in range(1, nclients + 1):
        client_yaml = yaml_template.format(
            path=os.path.abspath(os.path.join(target_path, f'client{i}')),
            nc=n_classes
        )
        with open(os.path.join(target_path, f'client{i}', 'data.yaml'), 'w') as f:
            f.write(client_yaml)
    
    print(f'Created YAML config files for server and {nclients} clients')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split YOLO dataset for Federated Learning')
    
    parser.add_argument('--img-path', type=str, required=True,
                        help='Path to images directory')
    parser.add_argument('--label-path', type=str, required=True,
                        help='Path to labels directory')
    parser.add_argument('--target-path', type=str, default='federated_dataset',
                        help='Path to target directory (default: federated_dataset)')
    parser.add_argument('--nclients', type=int, default=5,
                        help='Number of clients in federated experiment (default: 5)')
    parser.add_argument('--val-frac', type=float, default=0.2,
                        help='Fraction of data held by server for validation (default: 0.2)')
    parser.add_argument('--partition', type=str, default='iid', 
                        choices=['iid', 'dirichlet'],
                        help='Data partitioning strategy: iid or dirichlet (default: iid)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Dirichlet alpha parameter - smaller = more non-IID (default: 0.5)')
    parser.add_argument('--tar', action='store_true',
                        help='Archive the directories of federated participants')
    parser.add_argument('--n-classes', type=int, default=4,
                        help='Number of classes (default: 4)')
    
    args = parser.parse_args()
    
    process_yolo_dataset(
        img_path=args.img_path,
        label_path=args.label_path,
        target_path=args.target_path,
        nclients=args.nclients,
        val_frac=args.val_frac,
        partition=args.partition,
        alpha=args.alpha,
        tar=args.tar,
        n_classes=args.n_classes
    )