import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
import torch.multiprocessing as mp

from .utils import normalize_function_mm


def worker(
    gpu_id,
    positions,
    model_state_dict,
    model_class,
    image_path,
    view_size,
    downsample_factors,
    nclass,
    height,
    width,
):
    """
    Worker function to process partial image blocks on a specified GPU.

    Args:
        gpu_id (int): GPU ID.
        positions (list): List of image block coordinates (i, j) to process.
        model_state_dict (dict): Model state dictionary.
        model_class (type): Model class.
        image_path (str): Input image path.
        view_size (int): Target size of image blocks.
        downsample_factors (tuple): Multi-scale scaling factors.
        nclass (int): Number of classes.
        height (int): Image height.
        width (int): Image width.

    Returns:
        tuple: (assembled_output_gpu, count_gpu) Processing result of this GPU.
    """
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)

    image0 = cv2.imread(image_path)
    if image0 is None:
        raise ValueError(f"Worker {gpu_id} cannot read image {image_path}")

    size = view_size
    d1, d2, d3 = downsample_factors
    s1 = d1 * size
    s2 = d2 * size
    s3 = d3 * size
    pad_size = (s3 - s1) // 2
    delta23 = (s3 - s2) // 2

    image_padded = np.pad(
        image0,
        pad_width=((pad_size, pad_size), (pad_size, pad_size), (0, 0)),
        mode="reflect",
    )

    model = model_class(backbone="swin_l", nclass=2, isContext=True, pretrain=False)
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()

    assembled_output_gpu = np.zeros((nclass, height, width), dtype=np.float32)
    count_gpu = np.zeros((height, width), dtype=np.float32)

    with torch.no_grad():
        for i, j in tqdm(
            positions, desc="Processing segmentation", total=len(positions)
        ):

            img1_patch = image0[i : i + s1, j : j + s1, :]
            img2_patch = image_padded[
                i + delta23 : i + delta23 + s2, j + delta23 : j + delta23 + s2, :
            ]
            img3_patch = image_padded[i : i + s3, j : j + s3, :]

            img1_patch_resized = cv2.resize(
                img1_patch, (size, size), interpolation=cv2.INTER_LINEAR
            )
            img2_patch_resized = cv2.resize(
                img2_patch, (size, size), interpolation=cv2.INTER_LINEAR
            )
            img3_patch_resized = cv2.resize(
                img3_patch, (size, size), interpolation=cv2.INTER_LINEAR
            )

            img1_tensor = normalize_function_mm(img1_patch_resized, device)
            img2_tensor = normalize_function_mm(img2_patch_resized, device)
            img3_tensor = normalize_function_mm(img3_patch_resized, device)

            outputs = model((img1_tensor, img2_tensor, img3_tensor, None))
            outputs = outputs.cpu().numpy()[0]

            output_resized = cv2.resize(
                outputs.transpose(1, 2, 0), (s1, s1), interpolation=cv2.INTER_LINEAR
            ).transpose(2, 0, 1)

            assembled_output_gpu[:, i : i + s1, j : j + s1] += output_resized
            count_gpu[i : i + s1, j : j + s1] += 1

    return assembled_output_gpu, count_gpu


def seg_predict_api(
    model,
    image_path,
    result_dir,
    view_size=512,
    downsample_factors=(1, 3, 6),
    nclass=2,
    device="cuda",
    num_gpus=1,
):
    """
    Perform image segmentation prediction using multi-GPU or single-GPU.

    Args:
        model: Trained PyTorch model.
        image_path (str): Input image path.
        result_dir (str): Directory to save output results.
        view_size (int): Target size of image blocks (default 512).
        downsample_factors (tuple): Multi-scale scaling factors (default (1, 3, 6)).
        nclass (int): Number of classes (default 2).
        device (str): Device for single-GPU mode (default "cuda").
        num_gpus (int): Number of GPUs to use (default 1).

    Returns:
        tuple: (result_path, pred) - Save path and prediction mask.
    """
    model.eval()
    with torch.no_grad():

        name, _ = os.path.splitext(os.path.basename(image_path))
        result_path = os.path.join(result_dir, f"{name}.png")
        if os.path.exists(result_path):
            print(f"Skipping {name}, result already exists.")
            pred = cv2.imread(result_path, cv2.IMREAD_GRAYSCALE)
            return result_path, pred

        image0 = cv2.imread(image_path)
        if image0 is None:
            print(f"Cannot read image {image_path}, skipping.")
            return None, None
        height, width, _ = image0.shape

        size = view_size
        d1, d2, d3 = downsample_factors
        s1 = d1 * size
        s2 = d2 * size
        s3 = d3 * size

        positions = []
        for i in range(0, height - s1 // 2 + 1, s1 // 2):
            if i + s1 > height:
                i = height - s1
            for j in range(0, width - s1 // 2 + 1, s1 // 2):
                if j + s1 > width:
                    j = width - s1
                positions.append((i, j))

        assembled_output = np.zeros((nclass, height, width), dtype=np.float32)
        count = np.zeros((height, width), dtype=np.float32)

        if num_gpus > 1:
            mp.set_start_method("spawn", force=True)
            positions_split = np.array_split(positions, num_gpus)
            model_state_dict = model.state_dict()
            model_class = type(model)

            with mp.Pool(processes=num_gpus) as pool:

                results = pool.starmap(
                    worker,
                    [
                        (
                            gpu_id,
                            pos,
                            model_state_dict,
                            model_class,
                            image_path,
                            view_size,
                            downsample_factors,
                            nclass,
                            height,
                            width,
                        )
                        for gpu_id, pos in enumerate(positions_split)
                    ],
                )

            for assembled_output_gpu, count_gpu in results:
                assembled_output += assembled_output_gpu
                count += count_gpu
        else:
            pad_size = (s3 - s1) // 2
            delta23 = (s3 - s2) // 2
            image_padded = np.pad(
                image0,
                pad_width=((pad_size, pad_size), (pad_size, pad_size), (0, 0)),
                mode="reflect",
            )

            model.to(device)

            for i in tqdm(
                range(0, height - s1 // 2 + 1, s1 // 2),
                desc="Processing segmentation",
                total=(height - s1 // 2 + 1) // (s1 // 2),
            ):
                if i + s1 > height:
                    i = height - s1
                for j in range(0, width - s1 // 2 + 1, s1 // 2):
                    if j + s1 > width:
                        j = width - s1

                    img1_patch = image0[i : i + s1, j : j + s1, :]
                    img2_patch = image_padded[
                        i + delta23 : i + delta23 + s2,
                        j + delta23 : j + delta23 + s2,
                        :,
                    ]
                    img3_patch = image_padded[i : i + s3, j : j + s3, :]

                    img1_patch_resized = cv2.resize(
                        img1_patch, (size, size), interpolation=cv2.INTER_LINEAR
                    )
                    img2_patch_resized = cv2.resize(
                        img2_patch, (size, size), interpolation=cv2.INTER_LINEAR
                    )
                    img3_patch_resized = cv2.resize(
                        img3_patch, (size, size), interpolation=cv2.INTER_LINEAR
                    )

                    img1_tensor = normalize_function_mm(img1_patch_resized, device)
                    img2_tensor = normalize_function_mm(img2_patch_resized, device)
                    img3_tensor = normalize_function_mm(img3_patch_resized, device)

                    outputs = model((img1_tensor, img2_tensor, img3_tensor, None))
                    outputs = outputs.cpu().detach().numpy()[0]

                    output_resized = cv2.resize(
                        outputs.transpose(1, 2, 0),
                        (s1, s1),
                        interpolation=cv2.INTER_LINEAR,
                    ).transpose(2, 0, 1)

                    assembled_output[:, i : i + s1, j : j + s1] += output_resized
                    count[i : i + s1, j : j + s1] += 1

        count[count == 0] = 1
        assembled_output /= count
        pred = np.argmax(assembled_output, axis=0).astype(np.uint8)
        pred[pred == 1] = 255

        cv2.imwrite(result_path, pred)
        print(f"save mask predict result {name} to {result_path}")

        return result_path, pred