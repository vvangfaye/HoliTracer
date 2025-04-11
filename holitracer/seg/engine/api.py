import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
import torch.multiprocessing as mp

from .utils import normalize_function_mm

def worker(gpu_id, positions, model_state_dict, model_class, image_path, view_size, downsample_factors, nclass, height, width):
    """
    Worker函数，在指定GPU上处理部分图像块。

    Args:
        gpu_id (int): GPU ID。
        positions (list): 要处理的图像块坐标 (i, j) 列表。
        model_state_dict (dict): 模型的状态字典。
        model_class (type): 模型的类。
        image_path (str): 输入图像路径。
        view_size (int): 图像块的目标尺寸。
        downsample_factors (tuple): 多尺度缩放因子。
        nclass (int): 类别数。
        height (int): 图像高度。
        width (int): 图像宽度。

    Returns:
        tuple: (assembled_output_gpu, count_gpu) 该GPU的处理结果。
    """
    # 设置当前worker的设备
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)

    # 读取并填充图像
    image0 = cv2.imread(image_path)
    if image0 is None:
        raise ValueError(f"Worker {gpu_id} 无法读取图像 {image_path}")

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

    # 创建模型实例并加载状态字典
    model = model_class(backbone='swin_l',nclass=2,isContext=True,pretrain=False)
    model.load_state_dict(model_state_dict)  # 加载权重
    model.to(device)  # 移动到指定GPU
    model.eval()

    # 初始化输出数组
    assembled_output_gpu = np.zeros((nclass, height, width), dtype=np.float32)
    count_gpu = np.zeros((height, width), dtype=np.float32)

    with torch.no_grad():          
        for (i, j) in tqdm(positions,
                          desc="Processing segmentation",
                          total=len(positions)):
            # 提取图像块
            img1_patch = image0[i:i + s1, j:j + s1, :]
            img2_patch = image_padded[i + delta23:i + delta23 + s2, j + delta23:j + delta23 + s2, :]
            img3_patch = image_padded[i:i + s3, j:j + s3, :]

            # 调整图像块大小
            img1_patch_resized = cv2.resize(img1_patch, (size, size), interpolation=cv2.INTER_LINEAR)
            img2_patch_resized = cv2.resize(img2_patch, (size, size), interpolation=cv2.INTER_LINEAR)
            img3_patch_resized = cv2.resize(img3_patch, (size, size), interpolation=cv2.INTER_LINEAR)

            # 归一化并转换为张量
            img1_tensor = normalize_function_mm(img1_patch_resized, device)
            img2_tensor = normalize_function_mm(img2_patch_resized, device)
            img3_tensor = normalize_function_mm(img3_patch_resized, device)

            # 运行模型
            outputs = model((img1_tensor, img2_tensor, img3_tensor, None))
            outputs = outputs.cpu().numpy()[0]  # 形状: (ncls, h_out, w_out)

            # 调整输出大小
            output_resized = cv2.resize(
                outputs.transpose(1, 2, 0),
                (s1, s1),
                interpolation=cv2.INTER_LINEAR
            ).transpose(2, 0, 1)

            # 累加结果
            assembled_output_gpu[:, i:i + s1, j:j + s1] += output_resized
            count_gpu[i:i + s1, j:j + s1] += 1

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
    使用多GPU或单GPU进行图像分割预测。

    Args:
        model: 已训练的PyTorch模型。
        image_path (str): 输入图像路径。
        result_dir (str): 输出结果保存目录。
        view_size (int): 图像块目标尺寸（默认512）。
        downsample_factors (tuple): 多尺度缩放因子（默认(1, 3, 6)）。
        nclass (int): 类别数（默认2）。
        device (str): 单GPU模式下的设备（默认"cuda"）。
        num_gpus (int): 使用GPU数量（默认1）。

    Returns:
        tuple: (result_path, pred) - 保存路径和预测掩码。
    """
    model.eval()
    with torch.no_grad():
        # 检查结果是否已存在
        name, _ = os.path.splitext(os.path.basename(image_path))
        result_path = os.path.join(result_dir, f"{name}.png")
        if os.path.exists(result_path):
            print(f"跳过 {name}，结果已存在。")
            pred = cv2.imread(result_path, cv2.IMREAD_GRAYSCALE)
            return result_path, pred

        # 读取图像
        image0 = cv2.imread(image_path)
        if image0 is None:
            print(f"无法读取图像 {image_path}，跳过。")
            return None, None
        height, width, _ = image0.shape

        # 计算图像块大小
        size = view_size
        d1, d2, d3 = downsample_factors
        s1 = d1 * size
        s2 = d2 * size
        s3 = d3 * size

        # 生成所有图像块位置
        positions = []
        for i in range(0, height - s1 // 2 + 1, s1 // 2):
            if i + s1 > height:
                i = height - s1
            for j in range(0, width - s1 // 2 + 1, s1 // 2):
                if j + s1 > width:
                    j = width - s1
                positions.append((i, j))

        # 初始化输出数组
        assembled_output = np.zeros((nclass, height, width), dtype=np.float32)
        count = np.zeros((height, width), dtype=np.float32)

        if num_gpus > 1:
            # 多GPU处理
            mp.set_start_method('spawn', force=True)  # 使用'spawn'启动方式
            positions_split = np.array_split(positions, num_gpus)
            model_state_dict = model.state_dict()  # 获取模型状态字典
            model_class = type(model)  # 获取模型类

            with mp.Pool(processes=num_gpus) as pool:
                
                results = pool.starmap(
                    worker,
                    [(gpu_id, pos, model_state_dict, model_class, image_path, view_size, downsample_factors, nclass, height, width)
                     for gpu_id, pos in enumerate(positions_split)]
                )

            # 合并所有GPU的结果
            for assembled_output_gpu, count_gpu in results:
                assembled_output += assembled_output_gpu
                count += count_gpu
        else:
            # 单GPU处理
            pad_size = (s3 - s1) // 2
            delta23 = (s3 - s2) // 2
            image_padded = np.pad(
                image0,
                pad_width=((pad_size, pad_size), (pad_size, pad_size), (0, 0)),
                mode="reflect",
            )

            model.to(device)  # 确保模型在正确设备上

            for i in tqdm(range(0, height - s1 // 2 + 1, s1 // 2),
                          desc="Processing segmentation",
                          total=(height - s1 // 2 + 1) // (s1 // 2)):
                if i + s1 > height:
                    i = height - s1
                for j in range(0, width - s1 // 2 + 1, s1 // 2):
                    if j + s1 > width:
                        j = width - s1

                    # 提取图像块
                    img1_patch = image0[i:i + s1, j:j + s1, :]
                    img2_patch = image_padded[i + delta23:i + delta23 + s2, j + delta23:j + delta23 + s2, :]
                    img3_patch = image_padded[i:i + s3, j:j + s3, :]

                    # 调整图像块大小
                    img1_patch_resized = cv2.resize(img1_patch, (size, size), interpolation=cv2.INTER_LINEAR)
                    img2_patch_resized = cv2.resize(img2_patch, (size, size), interpolation=cv2.INTER_LINEAR)
                    img3_patch_resized = cv2.resize(img3_patch, (size, size), interpolation=cv2.INTER_LINEAR)

                    # 归一化并转换为张量
                    img1_tensor = normalize_function_mm(img1_patch_resized, device)
                    img2_tensor = normalize_function_mm(img2_patch_resized, device)
                    img3_tensor = normalize_function_mm(img3_patch_resized, device)

                    # 运行模型
                    outputs = model((img1_tensor, img2_tensor, img3_tensor, None))
                    outputs = outputs.cpu().detach().numpy()[0]

                    # 调整输出大小
                    output_resized = cv2.resize(
                        outputs.transpose(1, 2, 0),
                        (s1, s1),
                        interpolation=cv2.INTER_LINEAR
                    ).transpose(2, 0, 1)

                    # 累加结果
                    assembled_output[:, i:i + s1, j:j + s1] += output_resized
                    count[i:i + s1, j:j + s1] += 1

        # 最终处理预测结果
        count[count == 0] = 1  # 避免除以零
        assembled_output /= count
        pred = np.argmax(assembled_output, axis=0).astype(np.uint8)
        pred[pred == 1] = 255

        # 保存结果
        cv2.imwrite(result_path, pred)
        print(f"已保存 {name} 的预测结果到 {result_path}")

        return result_path, pred
