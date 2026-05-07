import numpy as np
import argparse
import warnings
import os
import os.path as osp
from copy import deepcopy
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def parse_args():
    parser = argparse.ArgumentParser(description='Add Noise to YOLO Annotation')
    parser.add_argument('--data-root', default='data/yolo', help='root path to YOLO dataset')
    parser.add_argument('--split', default='train', type=str, help='split of dataset (train/val/test)')
    parser.add_argument('--gamma', default='0.1', type=str, help="variance of noise added to bbox, e.g., '0.1'/'0.05'")
    parser.add_argument(
        '--force-replace', action='store_true', help='if set, new annotation will be generated no matter '
                                                     'whether it exists')
    parser.add_argument(
        '-o', '--suffix', type=str, default='noise', help='add suffix to the output folder name'
    )
    parser.add_argument(
        '--display', action='store_true', help='whether to display comparison')
    parser.add_argument(
        '--num-display', type=int, default=10, help='number of images to display')
    args = parser.parse_args()

    return args


class MovingAvg(object):
    def __init__(self):
        self.list = []
        self.count = 0

    def add(self, item):
        self.list.append(item)
        self.count += 1

    def clear(self):
        self.list = []
        self.count = 0

    def avg(self):
        if self.count == 0:
            warnings.warn('Count = 0!')
            return 0
        else:
            return sum(self.list) / self.count


def get_image_size(img_path):
    """Get image size from image file"""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    height, width = img.shape[:2]
    return width, height


def yolo_to_xyxy(x_center, y_center, w, h, img_width, img_height):
    """Convert YOLO format (normalized) to absolute xyxy format"""
    x_center_abs = x_center * img_width
    y_center_abs = y_center * img_height
    w_abs = w * img_width
    h_abs = h * img_height
    
    x1 = x_center_abs - w_abs / 2
    y1 = y_center_abs - h_abs / 2
    x2 = x_center_abs + w_abs / 2
    y2 = y_center_abs + h_abs / 2
    
    return x1, y1, x2, y2


def xyxy_to_yolo(x1, y1, x2, y2, img_width, img_height):
    """Convert absolute xyxy format to YOLO format (normalized)"""
    w_abs = x2 - x1
    h_abs = y2 - y1
    x_center_abs = x1 + w_abs / 2
    y_center_abs = y1 + h_abs / 2
    
    x_center = x_center_abs / img_width
    y_center = y_center_abs / img_height
    w = w_abs / img_width
    h = h_abs / img_height
    
    return x_center, y_center, w, h


def check_bbox_valid(x1, y1, x2, y2, img_width, img_height):
    """Check if bbox is valid"""
    # Check intersection with image
    inter_x1 = max(0, x1)
    inter_y1 = max(0, y1)
    inter_x2 = min(img_width, x2)
    inter_y2 = min(img_height, y2)
    
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    
    if inter_w * inter_h == 0:
        return False
    
    w = x2 - x1
    h = y2 - y1
    
    if w < 1 or h < 1:
        return False
    
    return True


def add_noise_single_bbox(x_center, y_center, w, h, img_width, img_height, gamma, max_attempts=100):
    """
    Add noise to a single bounding box in YOLO format
    
    Args:
        x_center, y_center, w, h: YOLO normalized coordinates
        img_width, img_height: Image dimensions
        gamma: Noise level
        max_attempts: Maximum attempts to generate valid noisy bbox
    
    Returns:
        Tuple of (noisy_coords, original_coords) in YOLO normalized format
        Returns None if cannot generate valid bbox after max_attempts
    """
    # Convert to absolute coordinates
    x1, y1, x2, y2 = yolo_to_xyxy(x_center, y_center, w, h, img_width, img_height)
    
    # Store original bbox
    original_bbox = (x_center, y_center, w, h)
    
    # Calculate width and height in absolute coordinates
    w_abs = x2 - x1
    h_abs = y2 - y1
    
    # Try to add noise until we get a valid bbox
    for attempt in range(max_attempts):
        # Add noise to each corner independently
        _x1 = x1 + np.random.randn() * w_abs * gamma
        _y1 = y1 + np.random.randn() * h_abs * gamma
        _x2 = x2 + np.random.randn() * w_abs * gamma
        _y2 = y2 + np.random.randn() * h_abs * gamma
        
        # Ensure x1 < x2 and y1 < y2
        if _x1 >= _x2:
            _x1, _x2 = _x2, _x1
        if _y1 >= _y2:
            _y1, _y2 = _y2, _y1
        
        # Check if valid
        if check_bbox_valid(_x1, _y1, _x2, _y2, img_width, img_height):
            # Clip to image boundaries
            _x1 = max(0, min(_x1, img_width))
            _y1 = max(0, min(_y1, img_height))
            _x2 = max(0, min(_x2, img_width))
            _y2 = max(0, min(_y2, img_height))
            
            # Convert back to YOLO format
            noisy_bbox = xyxy_to_yolo(_x1, _y1, _x2, _y2, img_width, img_height)
            
            return noisy_bbox, original_bbox
    
    # If cannot generate valid bbox, return original
    warnings.warn(f"Cannot generate valid noisy bbox after {max_attempts} attempts. Using original bbox.")
    return original_bbox, original_bbox


def add_noise_to_label_file(label_path, img_path, output_path, gamma):
    """
    Add noise to all bboxes in a YOLO label file
    
    Args:
        label_path: Path to original label file
        img_path: Path to corresponding image file
        output_path: Path to output noisy label file
        gamma: Noise level
    
    Returns:
        Dictionary containing statistics
    """
    # Get image dimensions
    img_width, img_height = get_image_size(img_path)
    
    # Read label file
    if not osp.exists(label_path):
        # No annotations for this image
        with open(output_path, 'w') as f:
            pass  # Create empty file
        return {'num_bboxes': 0, 'num_failed': 0}
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    if len(lines) == 0:
        # Empty label file
        with open(output_path, 'w') as f:
            pass
        return {'num_bboxes': 0, 'num_failed': 0}
    
    new_lines = []
    num_failed = 0
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split()
        if len(parts) < 5:
            warnings.warn(f"Invalid line in {label_path}: {line}")
            continue
        
        class_id = parts[0]
        x_center, y_center, w, h = map(float, parts[1:5])
        
        # Add noise
        noisy_bbox, original_bbox = add_noise_single_bbox(
            x_center, y_center, w, h, img_width, img_height, gamma
        )
        
        if noisy_bbox == original_bbox:
            num_failed += 1
        
        # Format: class_id x_center y_center w h original_x original_y original_w original_h
        # Store original bbox for evaluation
        new_line = f"{class_id} {noisy_bbox[0]:.6f} {noisy_bbox[1]:.6f} {noisy_bbox[2]:.6f} {noisy_bbox[3]:.6f}"
        
        # Optionally add original bbox as comment
        new_line += f" # orig: {original_bbox[0]:.6f} {original_bbox[1]:.6f} {original_bbox[2]:.6f} {original_bbox[3]:.6f}\n"
        
        new_lines.append(new_line)
    
    # Write to output file
    with open(output_path, 'w') as f:
        f.writelines(new_lines)
    
    return {'num_bboxes': len(new_lines), 'num_failed': num_failed}


def get_file_list(data_root, split):
    """
    Custom for dataset with image_2 and label_2
    """
    img_dir = osp.join(data_root, 'image_2')
    label_dir = osp.join(data_root, 'label_2')

    if not osp.exists(img_dir):
        raise FileNotFoundError(f"Image dir not found: {img_dir}")
    if not osp.exists(label_dir):
        raise FileNotFoundError(f"Label dir not found: {label_dir}")

    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    img_files = [
        f for f in os.listdir(img_dir)
        if any(f.lower().endswith(ext) for ext in img_extensions)
    ]

    return img_dir, label_dir, sorted(img_files)


def add_noise_to_dataset(data_root, split, gamma, output_label_dir):
    """
    Add noise to entire YOLO dataset
    """
    print(f"Processing {split} split with gamma={gamma}...")
    
    # Get file lists
    img_dir, label_dir, img_files = get_file_list(data_root, split)
    
    print(f"Found {len(img_files)} images in {img_dir}")
    print(f"Labels directory: {label_dir}")
    
    # Create output directories
 
    
    os.makedirs(output_label_dir, exist_ok=True)
    
    print(f"Output directory: {output_label_dir}")
    
    # Statistics
    total_bboxes = 0
    total_failed = 0
    
    # Process each image
    for img_filename in tqdm(img_files, desc="Adding noise"):
        # Get corresponding label filename
        label_filename = osp.splitext(img_filename)[0] + '.txt'
        
        img_path = osp.join(img_dir, img_filename)
        label_path = osp.join(label_dir, label_filename)
        output_label_path = osp.join(output_label_dir, label_filename)
        
        # Add noise
        stats = add_noise_to_label_file(label_path, img_path, output_label_path, gamma)
        
        total_bboxes += stats['num_bboxes']
        total_failed += stats['num_failed']
    
    print(f"\nProcessing complete!")
    print(f"Total bboxes: {total_bboxes}")
    print(f"Failed to add noise: {total_failed} ({100*total_failed/max(total_bboxes, 1):.2f}%)")
    
    return output_label_dir


def visualize_comparison(data_root, split, original_label_dir, noisy_label_dir, num_display=10):
    """
    Visualize comparison between original and noisy bboxes
    """
    print(f"\nGenerating visualization...")
    
    # Get file lists
    img_dir, _, img_files = get_file_list(data_root, split)
    
    # Limit number of images to display
    img_files = img_files[:min(num_display, len(img_files))]
    
    # Create output directory
    output_dir = osp.join(data_root, 'visualization')
    os.makedirs(output_dir, exist_ok=True)
    
    for img_filename in tqdm(img_files, desc="Visualizing"):
        label_filename = osp.splitext(img_filename)[0] + '.txt'
        
        img_path = osp.join(img_dir, img_filename)
        original_label_path = osp.join(original_label_dir, label_filename)
        noisy_label_path = osp.join(noisy_label_dir, label_filename)
        
        # Read image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_height, img_width = img.shape[:2]
        
        # Create figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot original
        axes[0].imshow(img)
        axes[0].set_title('Original Annotations', fontsize=14)
        axes[0].axis('off')
        
        if osp.exists(original_label_path):
            with open(original_label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        _, x_center, y_center, w, h = parts[:5]
                        x_center, y_center, w, h = map(float, [x_center, y_center, w, h])
                        
                        x1, y1, x2, y2 = yolo_to_xyxy(x_center, y_center, w, h, img_width, img_height)
                        
                        rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                                       linewidth=2, edgecolor='green', facecolor='none')
                        axes[0].add_patch(rect)
        
        # Plot noisy
        axes[1].imshow(img)
        axes[1].set_title('Noisy Annotations', fontsize=14)
        axes[1].axis('off')
        
        if osp.exists(noisy_label_path):
            with open(noisy_label_path, 'r') as f:
                for line in f:
                    # Remove comment part
                    line = line.split('#')[0].strip()
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        _, x_center, y_center, w, h = parts[:5]
                        x_center, y_center, w, h = map(float, [x_center, y_center, w, h])
                        
                        x1, y1, x2, y2 = yolo_to_xyxy(x_center, y_center, w, h, img_width, img_height)
                        
                        rect = Rectangle((x1, y1), x2-x1, y2-y1,
                                       linewidth=2, edgecolor='red', facecolor='none')
                        axes[1].add_patch(rect)
        
        plt.tight_layout()
        output_path = osp.join(output_dir, f'comparison_{osp.splitext(img_filename)[0]}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Visualizations saved to {output_dir}")


def main():
    args = parse_args()
    
    gamma = float(args.gamma)
    data_root = args.data_root
    split = args.split
    suffix = args.suffix
    
    if not osp.exists(data_root):
        raise FileNotFoundError(f"Data root not found: {data_root}")
    
    # Get label directory
    _, label_dir, _ = get_file_list(data_root, split)
    
    # Create output directory
    output_label_dir = label_dir + f'_{suffix}'

    
    # Check if already exists
    if osp.exists(output_label_dir) and not args.force_replace:
        warnings.warn(f"'{output_label_dir}' already exists. Use --force-replace to overwrite.")
        
        if args.display:
            visualize_comparison(data_root, split, label_dir, output_label_dir, args.num_display)
        return
    
    # Add noise to dataset
    output_label_dir = add_noise_to_dataset(
    data_root, split, gamma, output_label_dir
)
    
    # Visualize if requested
    if args.display:
        visualize_comparison(data_root, split, label_dir, output_label_dir, args.num_display)
    
    print("\nDone!")


if __name__ == '__main__':
    main()