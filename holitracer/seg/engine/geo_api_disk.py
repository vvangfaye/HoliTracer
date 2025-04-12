import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
import torch.multiprocessing as mp
import rasterio # <-- Added import
from rasterio.windows import Window # <-- Added import

from .utils import normalize_function_mm

def worker(gpu_id, positions, model_state_dict, model_class, image_path, view_size, downsample_factors, nclass, height, width, src_profile):
    """
    Worker function modified for windowed reading from large TIF files.

    Args:
        gpu_id (int): GPU ID.
        positions (list): List of top-left coordinates (row, col) for s1 blocks.
        model_state_dict (dict): Model's state dictionary.
        model_class (type): Model's class.
        image_path (str): Path to the input TIF image.
        view_size (int): Target size for image patches.
        downsample_factors (tuple): Multi-scale downsampling factors.
        nclass (int): Number of classes.
        height (int): Original image height.
        width (int): Original image width.
        src_profile (dict): Rasterio profile of the source image (for data type).

    Returns:
        tuple: (assembled_output_gpu, count_gpu) Processing results for this GPU.
    """
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)

    # Image dimensions and patch sizes
    size = view_size
    d1, d2, d3 = downsample_factors
    s1 = d1 * size
    s2 = d2 * size
    s3 = d3 * size
    pad_size = (s3 - s1) // 2 # Conceptual padding size
    delta23 = (s3 - s2) // 2 # Offset for s2 patch relative to s3 patch top-left

    # Create model instance and load state dict
    # Ensure the model class initialization matches the original usage
    model = model_class(backbone='swin_l', nclass=nclass, isContext=True, pretrain=False) # Example init
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()

    # Initialize output arrays for this worker
    # Note: These could still be large if the portion assigned to a worker is large.
    # Consider memory mapping or writing partial results if necessary.
    assembled_output_gpu = np.zeros((nclass, height, width), dtype=np.float32)
    count_gpu = np.zeros((height, width), dtype=np.float32)

    # Use 'with' statement for rasterio dataset
    try:
        with rasterio.open(image_path) as src:
            with torch.no_grad():
                for (i, j) in tqdm(positions,
                                  desc=f"GPU {gpu_id} Processing",
                                  total=len(positions),
                                  leave=False): # Better progress bar for multiple workers

                    # Calculate window coordinates for reading, handling boundaries
                    # Window format: Window(col_off, row_off, width, height)
                    # Need coordinates relative to the *original* image for reading.
                    # The conceptual padding is handled by `boundless=True` in src.read

                    # Window for img1 (s1 size, centered at i, j)
                    win1 = Window(j, i, s1, s1)

                    # Window for img2 (s2 size)
                    # Top-left in conceptual padded space: (i + delta23, j + delta23)
                    # Top-left relative to original image: (i + delta23 - pad_size, j + delta23 - pad_size)
                    win2_col_off = j + delta23 - pad_size
                    win2_row_off = i + delta23 - pad_size
                    win2 = Window(win2_col_off, win2_row_off, s2, s2)

                    # Window for img3 (s3 size)
                    # Top-left in conceptual padded space: (i, j)
                    # Top-left relative to original image: (i - pad_size, j - pad_size)
                    win3_col_off = j - pad_size
                    win3_row_off = i - pad_size
                    win3 = Window(win3_col_off, win3_row_off, s3, s3)

                    # Read image patches directly from the file using rasterio
                    # boundless=True reads data outside the actual raster extent, filling with `src.nodata` or 0
                    # Transpose because rasterio reads (bands, rows, cols), need (rows, cols, bands) for cv2/numpy
                    img1_patch = src.read(window=win1, boundless=True, fill_value=src.profile.get('nodata', 0)).transpose(1, 2, 0)
                    img2_patch = src.read(window=win2, boundless=True, fill_value=src.profile.get('nodata', 0)).transpose(1, 2, 0)
                    img3_patch = src.read(window=win3, boundless=True, fill_value=src.profile.get('nodata', 0)).transpose(1, 2, 0)

                    # Ensure patches have 3 dimensions (H, W, C) even if original is grayscale
                    if img1_patch.ndim == 2: img1_patch = np.expand_dims(img1_patch, axis=-1)
                    if img2_patch.ndim == 2: img2_patch = np.expand_dims(img2_patch, axis=-1)
                    if img3_patch.ndim == 2: img3_patch = np.expand_dims(img3_patch, axis=-1)
                    # If original image is grayscale, ensure it has 3 channels if model expects it
                    # This might require repeating the channel: np.repeat(patch, 3, axis=-1)
                    # Adapt this based on your model's input requirement and image type
                    num_channels = src.count
                    if num_channels == 1 and model.input_channels == 3: # Example check
                         img1_patch = np.repeat(img1_patch, 3, axis=-1)
                         img2_patch = np.repeat(img2_patch, 3, axis=-1)
                         img3_patch = np.repeat(img3_patch, 3, axis=-1)


                    # Resize patches
                    img1_patch_resized = cv2.resize(img1_patch.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)
                    img2_patch_resized = cv2.resize(img2_patch.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)
                    img3_patch_resized = cv2.resize(img3_patch.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)
                     # Ensure resized patches also have 3 dims if needed after resize
                    if img1_patch_resized.ndim == 2: img1_patch_resized = np.expand_dims(img1_patch_resized, axis=-1)
                    if img2_patch_resized.ndim == 2: img2_patch_resized = np.expand_dims(img2_patch_resized, axis=-1)
                    if img3_patch_resized.ndim == 2: img3_patch_resized = np.expand_dims(img3_patch_resized, axis=-1)


                    # Normalize and convert to tensor (ensure normalize_function handles potential different data types from TIF)
                    img1_tensor = normalize_function_mm(img1_patch_resized, device)
                    img2_tensor = normalize_function_mm(img2_patch_resized, device)
                    img3_tensor = normalize_function_mm(img3_patch_resized, device)

                    # Run model
                    outputs = model((img1_tensor, img2_tensor, img3_tensor, None))
                    outputs = outputs.cpu().numpy()[0] # Shape: (ncls, h_out, w_out)

                    # Resize output back to s1 size
                    output_resized = cv2.resize(
                        outputs.transpose(1, 2, 0), # (h, w, ncls) for resize
                        (s1, s1),
                        interpolation=cv2.INTER_LINEAR
                    )
                    # Handle case where resize might remove the channel dim if nclass=1
                    if output_resized.ndim == 2:
                       output_resized = np.expand_dims(output_resized, axis=-1)
                    output_resized = output_resized.transpose(2, 0, 1) # Back to (ncls, h, w)


                    # Accumulate results in the worker's arrays
                    # Careful with slicing end indices
                    h_slice = slice(i, min(i + s1, height))
                    w_slice = slice(j, min(j + s1, width))
                    h_out_slice = slice(0, h_slice.stop - h_slice.start)
                    w_out_slice = slice(0, w_slice.stop - w_slice.start)

                    assembled_output_gpu[:, h_slice, w_slice] += output_resized[:, h_out_slice, w_out_slice]
                    count_gpu[h_slice, w_slice] += 1

    except Exception as e:
        print(f"Error in worker {gpu_id} for image {image_path}: {e}")
        # Return empty arrays or raise exception depending on desired error handling
        return np.zeros((nclass, height, width), dtype=np.float32), np.zeros((height, width), dtype=np.float32)


    return assembled_output_gpu, count_gpu


def seg_geo_predict_api(
    model,
    image_path,
    result_dir,
    view_size=512,
    downsample_factors=(1, 3, 6),
    nclass=2,
    device="cuda", # Only used for single-GPU mode fallback
    num_gpus=1,
):
    """
    Image segmentation prediction using windowed reading for large TIF files.

    Args:
        model: Trained PyTorch model (will be used in single-GPU mode or state_dict passed to workers).
        image_path (str): Path to the input TIF image.
        result_dir (str): Directory to save the output mask.
        view_size (int): Target size for image patches (default 512).
        downsample_factors (tuple): Multi-scale factors (default (1, 3, 6)).
        nclass (int): Number of output classes (default 2).
        device (str): Device for single-GPU mode (default "cuda").
        num_gpus (int): Number of GPUs to use (default 1).
        step_factor (int): Determines overlap. Step size is s1 // step_factor (default 2 means 50% overlap).

    Returns:
        tuple: (result_path, pred) - Path to saved mask and the prediction mask (numpy array).
               Returns (None, None) on failure.
    """
    os.makedirs(result_dir, exist_ok=True)
    name, _ = os.path.splitext(os.path.basename(image_path))
    result_path = os.path.join(result_dir, f"{name}.png") # Consider saving as TIF for large outputs

    # --- Check if result exists ---
    # (Keep this part as is, but consider checking for a .tif if you change output format)
    # if os.path.exists(result_path):
    #     print(f"Skipping {name}, result already exists at {result_path}")
    #     try:
    #         # Decide how to handle existing results: read or just return path?
    #         # Reading large result might also be memory intensive
    #         pred = cv2.imread(result_path, cv2.IMREAD_GRAYSCALE) # Or rasterio.open(result_path).read(1)
    #         return result_path, pred
    #     except Exception as e:
    #          print(f"Could not read existing result {result_path}: {e}")
    #          return result_path, None # Return path even if read fails

    # --- Read Image Metadata using Rasterio ---
    try:
        with rasterio.open(image_path) as src:
            height = src.height
            width = src.width
            src_profile = src.profile # Keep metadata for saving output
            print(f"Processing image: {src_profile}")
    except Exception as e:
        print(f"An unexpected error occurred opening {image_path}: {e}")
        return None, None


    # --- Calculate Patch Sizes and Step ---
    size = view_size
    d1, d2, d3 = downsample_factors
    s1 = d1 * size
    s2 = d2 * size # Needed for patch calculation logic, even if not directly used in step
    s3 = d3 * size # Needed for patch calculation logic
    step = s1 // 2 # Sliding window step size

    # --- Generate Patch Positions ---
    positions = []
    for i in range(0, height, step):
        # Adjust last step start position to prevent exceeding bounds unnecessarily
        if i + s1 > height:
            i = height - s1
            if i < 0: i = 0 # Handle cases where image is smaller than s1
        for j in range(0, width, step):
            if j + s1 > width:
                j = width - s1
                if j < 0: j = 0 # Handle cases where image is smaller than s1
            positions.append((i, j))
        if i == height - s1: # Break if last row was adjusted
            break
    if not positions:
         print(f"Error: No valid positions generated for image size {height}x{width} and patch size {s1}x{s1}.")
         return None, None
    # Remove duplicate positions that might occur due to boundary adjustments
    positions = sorted(list(set(positions)))
    print(f"Total positions to process: {len(positions)}")


    # --- Initialize Full Output Arrays ---
    # Potentially large, consider alternatives (memory mapping, chunked writing) if necessary
    try:
        assembled_output = np.zeros((nclass, height, width), dtype=np.float32)
        count = np.zeros((height, width), dtype=np.float32)
    except MemoryError:
         print(f"Error: Not enough RAM to allocate full output arrays ({nclass}x{height}x{width} float32).")
         print("Consider reducing image size, using fewer classes, or implementing chunked writing.")
         return None, None

    # --- Processing Logic (Multi-GPU or Single-GPU) ---
    available_gpus = torch.cuda.device_count()
    actual_num_gpus = min(num_gpus, available_gpus)
    if actual_num_gpus <= 0:
        print("Error: No CUDA GPUs available or requested num_gpus <= 0.")
        # Optionally add CPU fallback here if desired
        return None, None

    if actual_num_gpus > 1 and len(positions) > actual_num_gpus: # Check if multiprocessing is beneficial
        print(f"Using {actual_num_gpus} GPUs for processing.")
        # Ensure model is on CPU before getting state_dict to avoid issues
        model.cpu()
        model_state_dict = model.state_dict()
        model_class = type(model)

        # Split positions among workers
        positions_split = np.array_split(positions, actual_num_gpus)

        # Required for CUDA multiprocessing
        try:
             mp.set_start_method('spawn', force=True)
        except RuntimeError as e:
             print(f"Note: Could not set multiprocessing start method to 'spawn'. Using default. ({e})")

        with mp.Pool(processes=actual_num_gpus) as pool:
            results = pool.starmap(
                worker,
                [(gpu_id, pos.tolist(), model_state_dict, model_class, image_path, view_size, downsample_factors, nclass, height, width, src_profile)
                 for gpu_id, pos in enumerate(positions_split)]
            )

        # Aggregate results from workers
        for assembled_output_gpu, count_gpu in results:
            # Add results carefully, ensuring shapes match
            assembled_output += assembled_output_gpu
            count += count_gpu

    else:
        # Single GPU processing (or fallback if num_gpus=1 or few positions)
        print(f"Using single GPU: {device}")
        target_device = torch.device(device if torch.cuda.is_available() else "cpu")
        model.to(target_device)
        model.eval() # Ensure model is in eval mode

        # Use the worker logic but directly in the main process
        # Initialize local arrays (redundant but follows worker structure)
        assembled_output_gpu = np.zeros((nclass, height, width), dtype=np.float32)
        count_gpu = np.zeros((height, width), dtype=np.float32)

        try:
            with rasterio.open(image_path) as src:
                with torch.no_grad():
                    for (i, j) in tqdm(positions, desc="Single GPU Processing", total=len(positions)):
                        # --- Patch extraction (copied & adapted from worker) ---
                        size = view_size
                        d1, d2, d3 = downsample_factors
                        s1 = d1 * size
                        s2 = d2 * size
                        s3 = d3 * size
                        pad_size = (s3 - s1) // 2
                        delta23 = (s3 - s2) // 2

                        win1 = Window(j, i, s1, s1)
                        win2_col_off = j + delta23 - pad_size
                        win2_row_off = i + delta23 - pad_size
                        win2 = Window(win2_col_off, win2_row_off, s2, s2)
                        win3_col_off = j - pad_size
                        win3_row_off = i - pad_size
                        win3 = Window(win3_col_off, win3_row_off, s3, s3)

                        img1_patch = src.read(window=win1, boundless=True, fill_value=src.profile.get('nodata', 0)).transpose(1, 2, 0)
                        img2_patch = src.read(window=win2, boundless=True, fill_value=src.profile.get('nodata', 0)).transpose(1, 2, 0)
                        img3_patch = src.read(window=win3, boundless=True, fill_value=src.profile.get('nodata', 0)).transpose(1, 2, 0)

                        if img1_patch.ndim == 2: img1_patch = np.expand_dims(img1_patch, axis=-1)
                        if img2_patch.ndim == 2: img2_patch = np.expand_dims(img2_patch, axis=-1)
                        if img3_patch.ndim == 2: img3_patch = np.expand_dims(img3_patch, axis=-1)

                        num_channels = src.count
                        if num_channels == 1 and model.input_channels == 3: # Example check
                            img1_patch = np.repeat(img1_patch, 3, axis=-1)
                            img2_patch = np.repeat(img2_patch, 3, axis=-1)
                            img3_patch = np.repeat(img3_patch, 3, axis=-1)


                        img1_patch_resized = cv2.resize(img1_patch.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)
                        img2_patch_resized = cv2.resize(img2_patch.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)
                        img3_patch_resized = cv2.resize(img3_patch.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)

                        if img1_patch_resized.ndim == 2: img1_patch_resized = np.expand_dims(img1_patch_resized, axis=-1)
                        if img2_patch_resized.ndim == 2: img2_patch_resized = np.expand_dims(img2_patch_resized, axis=-1)
                        if img3_patch_resized.ndim == 2: img3_patch_resized = np.expand_dims(img3_patch_resized, axis=-1)

                        img1_tensor = normalize_function_mm(img1_patch_resized, target_device)
                        img2_tensor = normalize_function_mm(img2_patch_resized, target_device)
                        img3_tensor = normalize_function_mm(img3_patch_resized, target_device)

                        outputs = model((img1_tensor, img2_tensor, img3_tensor, None))
                        outputs = outputs.cpu().numpy()[0] # No detach needed in torch.no_grad()

                        output_resized = cv2.resize(outputs.transpose(1, 2, 0), (s1, s1), interpolation=cv2.INTER_LINEAR)
                        if output_resized.ndim == 2:
                           output_resized = np.expand_dims(output_resized, axis=-1)
                        output_resized = output_resized.transpose(2, 0, 1)


                        h_slice = slice(i, min(i + s1, height))
                        w_slice = slice(j, min(j + s1, width))
                        h_out_slice = slice(0, h_slice.stop - h_slice.start)
                        w_out_slice = slice(0, w_slice.stop - w_slice.start)

                        assembled_output_gpu[:, h_slice, w_slice] += output_resized[:, h_out_slice, w_out_slice]
                        count_gpu[h_slice, w_slice] += 1
            # Assign results back to main variables
            assembled_output = assembled_output_gpu
            count = count_gpu

        except Exception as e:
            print(f"Error during single GPU processing: {e}")
            # Consider more specific error handling or re-raising
            return None, None


    # --- Finalize Prediction ---
    # Avoid division by zero for areas potentially not covered
    count[count == 0] = 1
    assembled_output /= count

    # Get final prediction mask
    pred = np.argmax(assembled_output, axis=0).astype(np.uint8)

    # Remap class indices if needed (e.g., background 0, foreground 1 -> 255)
    pred[pred == 1] = 255 # Assuming class 1 is the target, remap to 255 for typical mask visualization


    # --- Save Result using Rasterio (preserves georeferencing if input had it) ---
    print(f"Saving prediction mask to {result_path}...")
    try:
        # Update profile for output: single band, uint8, maybe compression
        profile = src_profile # Start with input profile
        profile.update(dtype=rasterio.uint8, count=1, nodata=0) # Output is single band Uint8, set nodata if appropriate
        # Optional: Add compression
        profile['compress'] = 'lzw' # Example compression

        with rasterio.open(result_path.replace('.png', '.tif'), 'w', **profile) as dst: # Save as TIF
            dst.write(pred, 1) # Write prediction to the first band
        print(f"Successfully saved prediction TIF to {result_path.replace('.png', '.tif')}")
        # Keep the original png path for return consistency if needed, or update return path
        result_path = result_path.replace('.png', '.tif')

    except Exception as e:
        print(f"Error saving output TIF: {e}")
        print("Attempting fallback save as PNG...")
        try:
            # Fallback to cv2.imwrite if rasterio fails or PNG is preferred
            png_result_path = os.path.join(result_dir, f"{name}.png")
            cv2.imwrite(png_result_path, pred)
            print(f"Successfully saved prediction PNG to {png_result_path}")
            result_path = png_result_path # Update result path to PNG
        except Exception as e2:
            print(f"Error saving output PNG: {e2}")
            return None, None # Failed to save result

    # Clean up large arrays if memory is critical
    del assembled_output, count
    # If you created memory maps, close them here

    return result_path, pred