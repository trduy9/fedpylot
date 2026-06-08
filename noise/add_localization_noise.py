import numpy as np
import argparse
import warnings
import os
import os.path as osp
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def parse_args():
    parser = argparse.ArgumentParser(description='Add Noise to YOLO Annotation')

    parser.add_argument(
        '--data-root',
        default='data/yolo',
        help='root path to dataset'
    )

    parser.add_argument(
        '--gamma',
        default='0.05',
        type=float,
        help='noise level'
    )

    parser.add_argument(
        '--force-replace',
        action='store_true',
        help='overwrite output folder if exists'
    )

    parser.add_argument(
        '-o',
        '--suffix',
        type=str,
        default='noise',
        help='suffix for output folder'
    )
    parser.add_argument('--strategy', default='log_scale',
                    choices=['log_scale', 'center_scale', 'dndetr', 'corner'],
                    help='noise strategy')
    parser.add_argument('--overflow', default='clip_scale',
                    choices=['clip_scale', 'clip_shift', 'reject'],
                    help='how to handle boxes that overflow image boundary')

    parser.add_argument(
        '--display',
        action='store_true',
        help='visualize comparison'
    )

    parser.add_argument(
        '--num-display',
        type=int,
        default=10,
        help='number of visualization images'
    )

    return parser.parse_args()


# =========================================================
# Utils
# =========================================================

def get_image_size(img_path):
    img = cv2.imread(img_path)

    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    h, w = img.shape[:2]
    return w, h


def yolo_to_xyxy(xc, yc, w, h, img_w, img_h):

    xc *= img_w
    yc *= img_h
    w *= img_w
    h *= img_h

    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2

    return x1, y1, x2, y2


def xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h):

    w = x2 - x1
    h = y2 - y1

    xc = x1 + w / 2
    yc = y1 + h / 2

    return (
        xc / img_w,
        yc / img_h,
        w / img_w,
        h / img_h
    )


def check_bbox_valid(x1, y1, x2, y2, img_w, img_h):

    inter_x1 = max(0, x1)
    inter_y1 = max(0, y1)

    inter_x2 = min(img_w, x2)
    inter_y2 = min(img_h, y2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)

    if inter_w * inter_h == 0:
        return False

    if (x2 - x1) < 1 or (y2 - y1) < 1:
        return False

    return True


# =========================================================
# Noise
# =========================================================

def add_noise_single_bbox(x_center, y_center, w, h,
                          img_width, img_height,
                          gamma,
                          strategy='log_scale',
                          overflow='clip_scale',
                          max_attempts=100):
    """
    Thêm noise vào một bounding box.

    Args:
        x_center, y_center, w, h : YOLO normalized coords
        img_width, img_height    : kích thước ảnh (pixel)
        gamma                    : mức độ noise
                                   - log_scale / center_scale: độ lệch chuẩn (~% thay đổi)
                                   - dndetr: dùng làm cả lambda1 và lambda2
                                   - corner: noise per-corner theo pixel
        strategy                 : 'log_scale'    — giữ center, noise w/h theo log-Gaussian
                                   'center_scale' — giữ center, noise w/h tuyến tính
                                   'dndetr'       — uniform noise center + scale (DN-DETR style)
                                   'corner'       — noise 4 góc độc lập (code gốc)
        overflow                 : xử lý khi box tràn ra ngoài ảnh
                                   'clip_scale' — thu nhỏ w/h để vừa khít, giữ center
                                   'clip_shift' — giữ w/h, dịch box vào trong
                                   'reject'     — thử lại, nếu vẫn fail thì dùng box gốc
        max_attempts             : số lần thử lại tối đa (dùng cho strategy='corner'
                                   và overflow='reject')

    Returns:
        (noisy_bbox, original_bbox) — tuple of (x_center, y_center, w, h) normalized
    """
    original_bbox = (x_center, y_center, w, h)

    # ------------------------------------------------------------------
    # Strategy: corner — noise 4 góc độc lập (code gốc)
    # Đây là case đặc biệt, xử lý riêng vì không theo center+scale flow
    # ------------------------------------------------------------------
    if strategy == 'corner':
        x1, y1, x2, y2 = yolo_to_xyxy(x_center, y_center, w, h, img_width, img_height)
        w_abs = x2 - x1
        h_abs = y2 - y1
        for _ in range(max_attempts):
            _x1 = x1 + np.random.randn() * w_abs * gamma
            _y1 = y1 + np.random.randn() * h_abs * gamma
            _x2 = x2 + np.random.randn() * w_abs * gamma
            _y2 = y2 + np.random.randn() * h_abs * gamma
            if _x1 >= _x2: _x1, _x2 = _x2, _x1
            if _y1 >= _y2: _y1, _y2 = _y2, _y1
            _x1 = max(0, min(_x1, img_width))
            _y1 = max(0, min(_y1, img_height))
            _x2 = max(0, min(_x2, img_width))
            _y2 = max(0, min(_y2, img_height))
            if check_bbox_valid(_x1, _y1, _x2, _y2, img_width, img_height):
                return xyxy_to_yolo(_x1, _y1, _x2, _y2, img_width, img_height), original_bbox
        warnings.warn(f"'corner': cannot generate valid bbox after {max_attempts} attempts. Using original.")
        return original_bbox, original_bbox

    # ------------------------------------------------------------------
    # Strategy: log_scale, center_scale, dndetr
    # Tất cả đều theo flow: tạo noisy_x_center, noisy_y_center, noisy_w, noisy_h
    # rồi xử lý overflow chung bên dưới
    # ------------------------------------------------------------------
    def _sample(strategy):
        """Trả về (noisy_x_center, noisy_y_center, noisy_w, noisy_h)"""
        if strategy == 'log_scale':
            # Center giữ nguyên, w/h noise theo log-Gaussian
            # exp(N(0, gamma^2)): gamma=0.1 → ±10%, gamma=0.3 → ±30%
            noisy_w = w * np.exp(np.random.randn() * gamma)
            noisy_h = h * np.exp(np.random.randn() * gamma)
            return x_center, y_center, noisy_w, noisy_h

        elif strategy == 'center_scale':
            # Center giữ nguyên, w/h noise tuyến tính
            # Đơn giản hơn log_scale, đối xứng tuyệt đối
            noisy_w = w * (1.0 + np.random.randn() * gamma)
            noisy_h = h * (1.0 + np.random.randn() * gamma)
            # Clamp để tránh w/h âm
            noisy_w = max(noisy_w, 1.0 / img_width)
            noisy_h = max(noisy_h, 1.0 / img_height)
            return x_center, y_center, noisy_w, noisy_h

        elif strategy == 'dndetr':
            # gamma dùng làm cả lambda1 (center shift) lẫn lambda2 (scale)
            # Center shift: Δx ~ Uniform(-gamma/2 * w, +gamma/2 * w)
            # Đảm bảo center mới luôn nằm trong box gốc khi gamma < 1
            delta_x = np.random.uniform(-gamma / 2 * w, gamma / 2 * w)
            delta_y = np.random.uniform(-gamma / 2 * h, gamma / 2 * h)
            noisy_x_center = x_center + delta_x
            noisy_y_center = y_center + delta_y
            # Scale: w_new ~ Uniform((1-gamma)*w, (1+gamma)*w)
            noisy_w = np.random.uniform((1 - gamma) * w, (1 + gamma) * w)
            noisy_h = np.random.uniform((1 - gamma) * h, (1 + gamma) * h)
            return noisy_x_center, noisy_y_center, noisy_w, noisy_h

        else:
            raise ValueError(f"Unknown strategy: '{strategy}'. "
                             f"Choose from: 'log_scale', 'center_scale', 'dndetr', 'corner'.")

    # ------------------------------------------------------------------
    # Overflow handling
    # ------------------------------------------------------------------
    def _apply_overflow(nx_c, ny_c, nw, nh, overflow):
        """
        Xử lý trường hợp box tràn ra ngoài [0, 1].
        Trả về (x1, y1, x2, y2) normalized, đã xử lý overflow.
        """
        x1 = nx_c - nw / 2
        y1 = ny_c - nh / 2
        x2 = nx_c + nw / 2
        y2 = ny_c + nh / 2
        out_of_bounds = (x1 < 0 or y1 < 0 or x2 > 1.0 or y2 > 1.0)

        if not out_of_bounds:
            return x1, y1, x2, y2

        if overflow == 'clip_scale':
            # Thu nhỏ w/h để box vừa khít image, giữ center cố định
            # Center không bị lệch, nhưng scale bị cap
            max_w = 2 * min(nx_c, 1.0 - nx_c)
            max_h = 2 * min(ny_c, 1.0 - ny_c)
            nw = min(nw, max_w)
            nh = min(nh, max_h)
            x1 = nx_c - nw / 2
            y1 = ny_c - nh / 2
            x2 = nx_c + nw / 2
            y2 = ny_c + nh / 2

        elif overflow == 'clip_shift':
            # Giữ nguyên w/h, dịch box vào trong — center lệch nhẹ
            # Hành vi giống augmentation thực tế (albumentations style)
            x1 = max(0.0, x1)
            y1 = max(0.0, y1)
            x2 = min(1.0, x1 + nw)
            y2 = min(1.0, y1 + nh)
            x1 = max(0.0, x2 - nw)
            y1 = max(0.0, y2 - nh)

        return x1, y1, x2, y2

    # ------------------------------------------------------------------
    # Main flow
    # ------------------------------------------------------------------
    if overflow == 'reject':
        # Thử lại cho đến khi ra box hợp lệ
        for _ in range(max_attempts):
            nx_c, ny_c, nw, nh = _sample(strategy)
            nw = max(nw, 1.0 / img_width)
            nh = max(nh, 1.0 / img_height)
            x1 = nx_c - nw / 2
            y1 = ny_c - nh / 2
            x2 = nx_c + nw / 2
            y2 = ny_c + nh / 2
            if x1 >= 0 and y1 >= 0 and x2 <= 1.0 and y2 <= 1.0:
                x1_abs = x1 * img_width
                y1_abs = y1 * img_height
                x2_abs = x2 * img_width
                y2_abs = y2 * img_height
                if check_bbox_valid(x1_abs, y1_abs, x2_abs, y2_abs, img_width, img_height):
                    return xyxy_to_yolo(x1_abs, y1_abs, x2_abs, y2_abs, img_width, img_height), original_bbox
        warnings.warn(f"'reject': cannot generate valid bbox after {max_attempts} attempts. Using original.")
        return original_bbox, original_bbox

    else:
        # clip_scale hoặc clip_shift: luôn có output
        nx_c, ny_c, nw, nh = _sample(strategy)
        nw = max(nw, 1.0 / img_width)
        nh = max(nh, 1.0 / img_height)
        x1, y1, x2, y2 = _apply_overflow(nx_c, ny_c, nw, nh, overflow)

        x1_abs = x1 * img_width
        y1_abs = y1 * img_height
        x2_abs = x2 * img_width
        y2_abs = y2 * img_height

        if not check_bbox_valid(x1_abs, y1_abs, x2_abs, y2_abs, img_width, img_height):
            warnings.warn("Final bbox invalid after overflow handling. Using original.")
            return original_bbox, original_bbox

        return xyxy_to_yolo(x1_abs, y1_abs, x2_abs, y2_abs, img_width, img_height), original_bbox


# =========================================================
# Dataset
# =========================================================

def get_file_list(data_root):
    """
    Dataset structure:

    data_root/
        image_2/
        label_2/
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


def add_noise_to_label_file(
    label_path,
    img_path,
    output_path,
    gamma,
    strategy,
    overflow
):

    img_width, img_height = get_image_size(img_path)

    if not osp.exists(label_path):

        with open(output_path, 'w'):
            pass

        return {
            'num_bboxes': 0,
            'num_failed': 0
        }

    with open(label_path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    num_failed = 0

    for line in lines:

        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) < 5:
            continue

        class_id = parts[0]

        x_center, y_center, w, h = map(
            float,
            parts[1:5]
        )

        noisy_bbox, original_bbox = add_noise_single_bbox(
              x_center, y_center, w, h,
              img_width, img_height,
              gamma,
              strategy=strategy,
              overflow=overflow,
          )

        if noisy_bbox == original_bbox:
            num_failed += 1

        new_line = (
            f"{class_id} "
            f"{noisy_bbox[0]:.6f} "
            f"{noisy_bbox[1]:.6f} "
            f"{noisy_bbox[2]:.6f} "
            f"{noisy_bbox[3]:.6f}\n"
        )

        new_lines.append(new_line)

    with open(output_path, 'w') as f:
        f.writelines(new_lines)

    return {
        'num_bboxes': len(new_lines),
        'num_failed': num_failed
    }


def add_noise_to_dataset(
    data_root,
    gamma,
    output_label_dir,
    strategy,
    overflow
):

    print(f"Processing dataset with gamma={gamma}")

    img_dir, label_dir, img_files = get_file_list(data_root)

    print(f"Found {len(img_files)} images")
    print(f"Image dir: {img_dir}")
    print(f"Label dir: {label_dir}")

    os.makedirs(output_label_dir, exist_ok=True)

    print(f"Output dir: {output_label_dir}")

    total_bboxes = 0
    total_failed = 0

    for img_filename in tqdm(img_files, desc='Adding noise'):

        label_filename = osp.splitext(img_filename)[0] + '.txt'

        img_path = osp.join(img_dir, img_filename)

        label_path = osp.join(label_dir, label_filename)

        output_label_path = osp.join(
            output_label_dir,
            label_filename
        )

        stats = add_noise_to_label_file(
            label_path,
            img_path,
            output_label_path,
            gamma,
            strategy,
            overflow
        )

        total_bboxes += stats['num_bboxes']
        total_failed += stats['num_failed']

    print("\nProcessing complete!")
    print(f"Total bboxes: {total_bboxes}")

    print(
        f"Failed: {total_failed} "
        f"({100 * total_failed / max(total_bboxes, 1):.2f}%)"
    )

    return output_label_dir


# =========================================================
# Visualization
# =========================================================

def visualize_comparison(
    data_root,
    original_label_dir,
    noisy_label_dir,
    num_display=10
):

    print("\nGenerating visualization...")

    img_dir, _, img_files = get_file_list(data_root)

    img_files = img_files[:min(num_display, len(img_files))]

    output_dir = osp.join(data_root, 'visualization')

    os.makedirs(output_dir, exist_ok=True)

    for img_filename in tqdm(img_files, desc='Visualizing'):

        label_filename = osp.splitext(img_filename)[0] + '.txt'

        img_path = osp.join(img_dir, img_filename)

        original_label_path = osp.join(
            original_label_dir,
            label_filename
        )

        noisy_label_path = osp.join(
            noisy_label_dir,
            label_filename
        )

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_height, img_width = img.shape[:2]

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Original
        axes[0].imshow(img)
        axes[0].set_title('Original')
        axes[0].axis('off')

        if osp.exists(original_label_path):

            with open(original_label_path, 'r') as f:

                for line in f:

                    parts = line.strip().split()

                    if len(parts) < 5:
                        continue

                    _, xc, yc, w, h = parts[:5]

                    xc, yc, w, h = map(
                        float,
                        [xc, yc, w, h]
                    )

                    x1, y1, x2, y2 = yolo_to_xyxy(
                        xc,
                        yc,
                        w,
                        h,
                        img_width,
                        img_height
                    )

                    rect = Rectangle(
                        (x1, y1),
                        x2 - x1,
                        y2 - y1,
                        linewidth=2,
                        edgecolor='green',
                        facecolor='none'
                    )

                    axes[0].add_patch(rect)

        # Noisy
        axes[1].imshow(img)
        axes[1].set_title('Noisy')
        axes[1].axis('off')

        if osp.exists(noisy_label_path):

            with open(noisy_label_path, 'r') as f:

                for line in f:

                    parts = line.strip().split()

                    if len(parts) < 5:
                        continue

                    _, xc, yc, w, h = parts[:5]

                    xc, yc, w, h = map(
                        float,
                        [xc, yc, w, h]
                    )

                    x1, y1, x2, y2 = yolo_to_xyxy(
                        xc,
                        yc,
                        w,
                        h,
                        img_width,
                        img_height
                    )

                    rect = Rectangle(
                        (x1, y1),
                        x2 - x1,
                        y2 - y1,
                        linewidth=2,
                        edgecolor='red',
                        facecolor='none'
                    )

                    axes[1].add_patch(rect)

        plt.tight_layout()

        save_path = osp.join(
            output_dir,
            f"{osp.splitext(img_filename)[0]}.png"
        )

        plt.savefig(save_path, dpi=150)
        plt.close()

    print(f"Visualization saved to: {output_dir}")


# =========================================================
# Main
# =========================================================

def main():

    args = parse_args()

    data_root = args.data_root
    gamma = args.gamma
    suffix = args.suffix

    if not osp.exists(data_root):
        raise FileNotFoundError(
            f"Data root not found: {data_root}"
        )

    _, label_dir, _ = get_file_list(data_root)

    output_label_dir = label_dir + f'_{suffix}'

    if osp.exists(output_label_dir) and not args.force_replace:

        warnings.warn(
            f"{output_label_dir} already exists!"
        )

        if args.display:

            visualize_comparison(
                data_root,
                label_dir,
                output_label_dir,
                args.num_display
            )

        return

    output_label_dir = add_noise_to_dataset(
        data_root,
        gamma,
        output_label_dir,
        args.strategy,
        args.overflow
    )

    if args.display:

        visualize_comparison(
            data_root,
            label_dir,
            output_label_dir,
            args.num_display
        )

    print("\nDone!")


if __name__ == '__main__':
    main()