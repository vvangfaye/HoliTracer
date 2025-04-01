import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
from queue import Queue
from threading import Thread

from .utils import normalize_function_mm


# predict api  
def seg_predict_api(
    model,
    image_path,
    result_dir,
    view_size=512,
    downsample_factors=(1, 3, 6),
    nclass=2,
    device="cuda",
):
    """Performs prediction on input images."""
    model.eval()
    with torch.no_grad():
        # If distributed, split the images among processes

        name, _ = os.path.splitext(os.path.basename(image_path))
        result_path = os.path.join(result_dir, f"{name}.png")
        
        if os.path.exists(result_path):
            print(f"Skipping {name}, already exists.")
            pred = cv2.imread(result_path, cv2.IMREAD_GRAYSCALE)
            return result_path, pred
        image0 = cv2.imread(image_path)
        
        if image0 is None:
            print(f"Failed to read image {image_path}, skipping.")
            return
        height, width, _ = image0.shape

        # Get parameters
        size = view_size  # e.g., 512
        downsample_factors = downsample_factors  # tuple of 3 ints
        d1, d2, d3 = downsample_factors
        s1 = d1 * size
        s2 = d2 * size
        s3 = d3 * size
        pad_size = (s3 - s1) // 2
        delta23 = (s3 - s2) // 2

        # Pad the image to accommodate larger patches
        image_padded = np.pad(
            image0,
            pad_width=((pad_size, pad_size), (pad_size, pad_size), (0, 0)),
            mode="reflect",  # Use 'reflect' or 'constant' padding as needed
        )

        # Initialize output arrays
        ncls = nclass  # Number of classes
        assembled_output = np.zeros((ncls, height, width), dtype=np.float32)
        count = np.zeros((height, width), dtype=np.float32)

        # Iterate over the image to extract patches
        for i in tqdm(range(0, height - s1 // 2 + 1, s1 // 2), 
              desc="Processing segmentation", 
              total=(height - s1 // 2 + 1) // (s1 // 2)):
            if i + s1 > height:
                i = height - s1
            for j in range(0, width - s1 // 2 + 1, s1 // 2):
                if j + s1 > width:
                    j = width - s1

                # Extract patches at different scales
                img1_patch = image0[i : i + s1, j : j + s1, :]
                img2_patch = image_padded[
                    i + delta23 : i + delta23 + s2,
                    j + delta23 : j + delta23 + s2,
                    :,
                ]
                img3_patch = image_padded[i : i + s3, j : j + s3, :]

                # Resize patches to the desired size
                img1_patch_resized = cv2.resize(
                    img1_patch, (size, size), interpolation=cv2.INTER_LINEAR
                )
                img2_patch_resized = cv2.resize(
                    img2_patch, (size, size), interpolation=cv2.INTER_LINEAR
                )
                img3_patch_resized = cv2.resize(
                    img3_patch, (size, size), interpolation=cv2.INTER_LINEAR
                )
                
                # visualization attention map
                save_name = None

                img1_tensor = normalize_function_mm(img1_patch_resized, device)
                img2_tensor = normalize_function_mm(img2_patch_resized, device)
                img3_tensor = normalize_function_mm(img3_patch_resized, device)

                # Run the model
                outputs = model((img1_tensor, img2_tensor, img3_tensor, save_name))
                outputs = (
                    outputs.cpu().detach().numpy()
                )  # Shape: (1, ncls, h_out, w_out)
                outputs = outputs[0]  # Remove batch dimension

                # Resize outputs to s1 size if necessary
                output_resized = cv2.resize(
                    outputs.transpose(1, 2, 0),
                    (s1, s1),
                    interpolation=cv2.INTER_LINEAR,
                )
                output_resized = output_resized.transpose(
                    2, 0, 1
                )  # Shape: (ncls, s1, s1)
                pred_patch = np.argmax(output_resized, axis=0).astype(np.uint8)
                pred_patch[pred_patch == 1] = 255

                # Add outputs to assembled_output
                assembled_output[:, i : i + s1, j : j + s1] += output_resized
                count[i : i + s1, j : j + s1] += 1

        # Avoid division by zero
        count[count == 0] = 1
        assembled_output /= count

        # Take argmax over classes to get predicted labels
        pred = np.argmax(assembled_output, axis=0).astype(np.uint8)

        # Map class indices to labels if necessary
        # For example, if class 1 is 255 in the mask
        pred[pred == 1] = 255

        # Save the predicted mask
        cv2.imwrite(result_path, pred)

        print(f"Saved prediction for {name} at {result_path}")
        
        return result_path, pred


def seg_predict_api_multi(
    model,
    image_path,
    result_dir,
    view_size=512,
    downsample_factors=(1, 3, 6),
    nclass=2,
    devices=["cuda:0", "cuda:1"],  # List of devices to use, e.g. ["cuda:0", "cuda:1"]
):
    """
    Performs prediction on input images using multiple GPUs in parallel.
    
    Args:
        model: The segmentation model (should be compatible with DataParallel or DistributedDataParallel)
        image_path: Path to the input image
        result_dir: Directory to save results
        view_size: Size of the view (patch after resizing)
        downsample_factors: Tuple of 3 integers for multi-scale processing
        nclass: Number of classes
        devices: List of GPU devices to use (e.g., ["cuda:0", "cuda:1"])
    
    Returns:
        Tuple of (result_path, prediction)
    """
    
    # Set default devices if not provided
    if devices is None:
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            devices = [f"cuda:{i}" for i in range(num_gpus)]
        else:
            devices = ["cpu"]
    
    print(f"Using devices: {devices}")
    
    # Function to normalize images (assuming normalize_function_mm is defined elsewhere)
    def normalize_function_mm(img, device):
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0).to(device)
        return img
    
    # Create replica of model on each device
    models = {}
    for device in devices:
        model_copy = type(model)(**model.__dict__).to(device)
        model_copy.load_state_dict(model.state_dict())
        model_copy.eval()
        models[device] = model_copy
    
    # Prepare the image
    name, _ = os.path.splitext(os.path.basename(image_path))
    result_path = os.path.join(result_dir, f"{name}.png")
    
    if os.path.exists(result_path):
        print(f"Skipping {name}, already exists.")
        pred = cv2.imread(result_path, cv2.IMREAD_GRAYSCALE)
        return result_path, pred
    
    image0 = cv2.imread(image_path)
    if image0 is None:
        print(f"Failed to read image {image_path}, skipping.")
        return None, None
    
    height, width, _ = image0.shape
    
    # Get parameters
    size = view_size  # e.g., 512
    downsample_factors = downsample_factors  # tuple of 3 ints
    d1, d2, d3 = downsample_factors
    s1 = d1 * size
    s2 = d2 * size
    s3 = d3 * size
    pad_size = (s3 - s1) // 2
    delta23 = (s3 - s2) // 2
    
    # Pad the image
    image_padded = np.pad(
        image0,
        pad_width=((pad_size, pad_size), (pad_size, pad_size), (0, 0)),
        mode="reflect",
    )
    
    # Initialize output arrays
    ncls = nclass
    assembled_output = np.zeros((ncls, height, width), dtype=np.float32)
    count = np.zeros((height, width), dtype=np.float32)
    
    # Create a list of all patches to process
    patches_to_process = []
    for i in range(0, height - s1 // 2 + 1, s1 // 2):
        if i + s1 > height:
            i = height - s1
        for j in range(0, width - s1 // 2 + 1, s1 // 2):
            if j + s1 > width:
                j = width - s1
            patches_to_process.append((i, j))
    
    # Create patch processing queue
    task_queue = Queue()
    result_queue = Queue()
    for patch in patches_to_process:
        task_queue.put(patch)
    
    # Define worker function to process patches
    def worker(device_id, task_queue, result_queue):
        device = devices[device_id % len(devices)]
        model_device = models[device]
        
        while not task_queue.empty():
            try:
                i, j = task_queue.get(block=False)
                
                # Extract patches at different scales
                img1_patch = image0[i : i + s1, j : j + s1, :]
                img2_patch = image_padded[
                    i + delta23 : i + delta23 + s2,
                    j + delta23 : j + delta23 + s2,
                    :,
                ]
                img3_patch = image_padded[i : i + s3, j : j + s3, :]
                
                # Resize patches
                img1_patch_resized = cv2.resize(
                    img1_patch, (size, size), interpolation=cv2.INTER_LINEAR
                )
                img2_patch_resized = cv2.resize(
                    img2_patch, (size, size), interpolation=cv2.INTER_LINEAR
                )
                img3_patch_resized = cv2.resize(
                    img3_patch, (size, size), interpolation=cv2.INTER_LINEAR
                )
                
                # Convert to tensors
                save_name = None
                img1_tensor = normalize_function_mm(img1_patch_resized, device)
                img2_tensor = normalize_function_mm(img2_patch_resized, device)
                img3_tensor = normalize_function_mm(img3_patch_resized, device)
                
                # Run inference
                with torch.no_grad():
                    outputs = model_device((img1_tensor, img2_tensor, img3_tensor, save_name))
                    outputs = outputs.cpu().detach().numpy()
                
                # Resize outputs to s1 size
                outputs = outputs[0]  # Remove batch dimension
                output_resized = cv2.resize(
                    outputs.transpose(1, 2, 0),
                    (s1, s1),
                    interpolation=cv2.INTER_LINEAR,
                )
                output_resized = output_resized.transpose(2, 0, 1)  # Shape: (ncls, s1, s1)
                
                # Put results in queue
                result_queue.put((i, j, output_resized))
                task_queue.task_done()
            except Exception as e:
                print(f"Error in worker {device_id}: {e}")
                task_queue.task_done()
    
    # Start worker threads for each GPU
    num_workers = len(devices) * 2  # 2 workers per GPU for better utilization
    threads = []
    for i in range(num_workers):
        t = Thread(target=worker, args=(i, task_queue, result_queue))
        t.daemon = True
        t.start()
        threads.append(t)
    
    # Process results as they come in
    with tqdm(total=len(patches_to_process), desc="Processing patches") as pbar:
        processed = 0
        while processed < len(patches_to_process):
            try:
                i, j, output_resized = result_queue.get(timeout=1.0)
                
                # Add outputs to assembled_output
                assembled_output[:, i : i + s1, j : j + s1] += output_resized
                count[i : i + s1, j : j + s1] += 1
                
                processed += 1
                pbar.update(1)
            except:
                # Check if all workers are done
                if all(not t.is_alive() for t in threads):
                    break
    
    # Wait for all threads to complete
    for t in threads:
        t.join()
    
    # Avoid division by zero
    count[count == 0] = 1
    assembled_output /= count
    
    # Take argmax over classes to get predicted labels
    pred = np.argmax(assembled_output, axis=0).astype(np.uint8)
    
    # Map class indices to labels
    pred[pred == 1] = 255
    
    # Save the predicted mask
    cv2.imwrite(result_path, pred)
    print(f"Saved prediction for {name} at {result_path}")
    
    return result_path, pred
        
