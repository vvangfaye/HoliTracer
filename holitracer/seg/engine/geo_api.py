import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
import torch.multiprocessing as mp
import rasterio # Still useful for reading TIFs robustly into memory

from .utils import normalize_function_mm

def worker(gpu_id, positions, model_state_dict, model_class,
           image0, image_padded, # Pass NumPy arrays instead of path
           view_size, downsample_factors, nclass, height, width):
    """
    Worker function that processes patches from pre-loaded image arrays.

    Args:
        gpu_id (int): GPU ID.
        positions (list): List of top-left coordinates (row, col) for s1 blocks.
        model_state_dict (dict): Model's state dictionary.
        model_class (type): Model's class.
        image0 (np.ndarray): The original image data (H, W, C).
        image_padded (np.ndarray): The padded image data (H_pad, W_pad, C).
        view_size (int): Target size for image patches.
        downsample_factors (tuple): Multi-scale downsampling factors.
        nclass (int): Number of classes.
        height (int): Original image height.
        width (int): Original image width.

    Returns:
        tuple: (assembled_output_gpu, count_gpu) Processing results for this GPU.
    """
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)

    # Image dimensions and patch sizes (calculated from inputs)
    size = view_size
    d1, d2, d3 = downsample_factors
    s1 = d1 * size
    s2 = d2 * size
    s3 = d3 * size
    # These offsets are relative to the *padded* image's coordinate system
    pad_size = (s3 - s1) // 2 # Amount of padding on each side
    delta23 = (s3 - s2) // 2 # Offset for s2 patch relative to s3 patch top-left in padded

    # Create model instance and load state dict
    model = model_class(backbone='swin_l', nclass=nclass, isContext=True, pretrain=False) # Example init
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()

    # Initialize output arrays for this worker
    # Allocating these per worker might still consume significant memory.
    assembled_output_gpu = np.zeros((nclass, height, width), dtype=np.float32)
    count_gpu = np.zeros((height, width), dtype=np.float32)

    # --- No file opening needed here, process directly from passed arrays ---
    try:
        with torch.no_grad():
            for (i, j) in tqdm(positions,
                              desc=f"GPU {gpu_id} Processing",
                              total=len(positions),
                              leave=False):

                # --- Extract patches using NumPy slicing from in-memory arrays ---
                # Note: Slicing is relative to the top-left (0,0) of the respective arrays.
                # The 'i, j' coordinates are relative to the *original* image top-left.
                # For padded access, we need to account for the pad_size offset.

                # img1 from original image
                img1_patch = image0[i : i + s1, j : j + s1, :]

                # img2 and img3 from padded image
                # Calculate top-left corner in the padded array's coordinate system
                padded_i = i + pad_size
                padded_j = j + pad_size

                # Extract img3 (s3 x s3) starting from (padded_i, padded_j)
                # This corresponds to the window centered around (i,j) in the original
                img3_patch = image_padded[padded_i : padded_i + s3, padded_j : padded_j + s3, :]

                # Extract img2 (s2 x s2) starting offset by delta23 within the s3 window
                img2_patch = image_padded[padded_i + delta23 : padded_i + delta23 + s2,
                                          padded_j + delta23 : padded_j + delta23 + s2, :]

                # --- Resize, Normalize, Predict (Same as before) ---

                # Check if patches are valid (e.g., if slicing resulted in empty array near edges)
                if img1_patch.size == 0 or img2_patch.size == 0 or img3_patch.size == 0:
                     print(f"Warning: Skipping empty patch at ({i}, {j})")
                     continue

                # Ensure patches have 3 dimensions (H, W, C)
                # (This check might be less necessary if image0 is guaranteed HWC)
                if img1_patch.ndim == 2: img1_patch = np.expand_dims(img1_patch, axis=-1)
                if img2_patch.ndim == 2: img2_patch = np.expand_dims(img2_patch, axis=-1)
                if img3_patch.ndim == 2: img3_patch = np.expand_dims(img3_patch, axis=-1)
                # Handle grayscale to 3-channel if necessary (check image0.shape[-1])
                # num_channels = image0.shape[-1]
                # if num_channels == 1 and model.input_channels == 3: # Example check
                #      img1_patch = np.repeat(img1_patch, 3, axis=-1)
                #      img2_patch = np.repeat(img2_patch, 3, axis=-1)
                #      img3_patch = np.repeat(img3_patch, 3, axis=-1)

                # Resize patches
                img1_patch_resized = cv2.resize(img1_patch, (size, size), interpolation=cv2.INTER_LINEAR)
                img2_patch_resized = cv2.resize(img2_patch, (size, size), interpolation=cv2.INTER_LINEAR)
                img3_patch_resized = cv2.resize(img3_patch, (size, size), interpolation=cv2.INTER_LINEAR)
                 # Ensure resized patches also have 3 dims if needed after resize
                if img1_patch_resized.ndim == 2: img1_patch_resized = np.expand_dims(img1_patch_resized, axis=-1)
                if img2_patch_resized.ndim == 2: img2_patch_resized = np.expand_dims(img2_patch_resized, axis=-1)
                if img3_patch_resized.ndim == 2: img3_patch_resized = np.expand_dims(img3_patch_resized, axis=-1)

                # Normalize and convert to tensor
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
                if output_resized.ndim == 2:
                   output_resized = np.expand_dims(output_resized, axis=-1)
                output_resized = output_resized.transpose(2, 0, 1) # Back to (ncls, h, w)

                # Accumulate results
                h_slice = slice(i, min(i + s1, height))
                w_slice = slice(j, min(j + s1, width))
                h_out_slice = slice(0, h_slice.stop - h_slice.start)
                w_out_slice = slice(0, w_slice.stop - w_slice.start)

                assembled_output_gpu[:, h_slice, w_slice] += output_resized[:, h_out_slice, w_out_slice]
                count_gpu[h_slice, w_slice] += 1

    except Exception as e:
        print(f"Error in worker {gpu_id}: {e}")
        # Return empty arrays or raise exception
        # Ensure shape matches expected return format
        return np.zeros((nclass, height, width), dtype=np.float32), np.zeros((height, width), dtype=np.float32)

    return assembled_output_gpu, count_gpu


# --- Main API Function (Loads image to memory) ---
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
    Image segmentation prediction loading the **entire TIF image into memory**.
    WARNING: This may fail due to MemoryError for large images.

    Args:
        model: Trained PyTorch model.
        image_path (str): Path to the input TIF image.
        result_dir (str): Directory to save the output mask.
        view_size (int): Target size for image patches (default 512).
        downsample_factors (tuple): Multi-scale factors (default (1, 3, 6)).
        nclass (int): Number of output classes (default 2).
        device (str): Device for single-GPU mode (default "cuda").
        num_gpus (int): Number of GPUs to use (default 1).

    Returns:
        tuple: (result_path, pred) - Path to saved mask and the prediction mask.
               Returns (None, None) on failure.
    """
    print("\n" + "="*30)
    print(" WARNING: Running in 'load-to-memory' mode.")
    print(" This will load the entire image into RAM.")
    print(" Use the previous version for large images (> RAM).")
    print("="*30 + "\n")

    os.makedirs(result_dir, exist_ok=True)
    name, _ = os.path.splitext(os.path.basename(image_path))
    # Decide output format (TIF preserves geo-info, PNG is common)
    save_as_tif = True # Set to False to save as PNG
    result_filename = f"{name}.tif" if save_as_tif else f"{name}.png"
    result_path = os.path.join(result_dir, result_filename)

    # --- Check if result already exists ---
    if os.path.exists(result_path):
        print(f"Skipping {name}, result already exists.")
        pred = cv2.imread(result_path, cv2.IMREAD_GRAYSCALE)
        return result_path, pred
    else:
        print(f"Processing {name}...")

    # --- Load Full Image and Pad using Rasterio and NumPy ---
    try:
        print(f"Loading image {image_path} into memory...")
        # Use rasterio to read robustly, then transpose
        with rasterio.open(image_path) as src:
            image0 = src.read() # Reads as (bands, height, width)
             # Transpose to (height, width, bands) common for image processing
            if image0.ndim == 3:
                 image0 = image0.transpose(1, 2, 0)
            elif image0.ndim == 2: # Handle grayscale case
                 image0 = np.expand_dims(image0, axis=-1) # Add channel dim -> (H, W, 1)
            else:
                 raise ValueError(f"Unexpected number of dimensions in image: {image0.ndim}")

            height, width = src.height, src.width
            src_profile = src.profile # Keep metadata for saving output
            print(f"Image loaded: {src_profile['width']}x{src_profile['height']} with {src_profile['count']} bands.")

        # Calculate padding size (based on largest scale s3)
        size = view_size
        d1, d2, d3 = downsample_factors
        s1 = d1 * size
        # s2 = d2 * size # Not directly needed here, but calculated in worker
        s3 = d3 * size
        pad_size = (s3 - s1) // 2

        print(f"Padding image with {pad_size} pixels...")
        # Use reflect padding as in the original user code example
        image_padded = np.pad(
            image0,
            pad_width=((pad_size, pad_size), (pad_size, pad_size), (0, 0)),
            mode="reflect",
        )
        print(f"Padded image shape: {image_padded.shape}")

    except MemoryError:
         print("\n" + "*"*40)
         print(f"FATAL: MemoryError occurred while loading or padding {image_path}.")
         print(f"The image is too large to fit into RAM ({image0.nbytes / 1e9:.2f} GB approx for raw image).")
         print("Please use the previous version of the code that uses windowed reading.")
         print("*"*40 + "\n")
         return None, None
    except rasterio.RasterioIOError as e:
        print(f"Error opening image {image_path} with rasterio: {e}")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred during image loading/padding: {e}")
        return None, None

    # --- Calculate Patch Sizes and Step ---
    # (s1, s2, s3 are calculated above and in worker)
    step = s1 // 2 # 50% overlap

    # --- Generate Patch Positions (Same as before) ---
    positions = []
    for i in range(0, height, step):
        if i + s1 > height: i = max(0, height - s1)
        for j in range(0, width, step):
            if j + s1 > width: j = max(0, width - s1)
            positions.append((i, j))
        if i == max(0, height - s1): break # Break after processing the potentially adjusted last row
    positions = sorted(list(set(positions)))
    if not positions:
         print(f"Error: No valid positions generated.")
         return None, None
    print(f"Total positions to process: {len(positions)}")

    # --- Initialize Full Output Arrays (Still requires memory) ---
    try:
        assembled_output = np.zeros((nclass, height, width), dtype=np.float32)
        count = np.zeros((height, width), dtype=np.float32)
    except MemoryError:
        print(f"Error: Not enough RAM to allocate full output arrays ({nclass}x{height}x{width} float32).")
        # No easy fallback here if the output itself is too large
        return None, None

    # --- Processing Logic (Multi-GPU or Single-GPU) ---
    available_gpus = torch.cuda.device_count()
    actual_num_gpus = min(num_gpus, available_gpus)
    if actual_num_gpus <= 0:
        print("Error: No CUDA GPUs available or requested num_gpus <= 0.")
        return None, None

    if actual_num_gpus > 1 and len(positions) > actual_num_gpus:
        print(f"Using {actual_num_gpus} GPUs for processing.")
        model.cpu() # Move model to CPU before getting state dict
        model_state_dict = model.state_dict()
        model_class = type(model)

        positions_split = np.array_split(positions, actual_num_gpus)

        try:
             mp.set_start_method('spawn', force=True)
        except RuntimeError as e:
             print(f"Note: Could not set multiprocessing start method to 'spawn'. ({e})")

        # --- Pass loaded image arrays to workers ---
        # WARNING: Passing large arrays can be slow/memory intensive depending on backend.
        print("Distributing work to GPU processes...")
        with mp.Pool(processes=actual_num_gpus) as pool:
            results = pool.starmap(
                worker,
                [(gpu_id, pos.tolist(), model_state_dict, model_class,
                  image0, image_padded, # Pass the actual data
                  view_size, downsample_factors, nclass, height, width)
                 for gpu_id, pos in enumerate(positions_split)]
            )
        print("Aggregating results from GPU processes...")
        # Aggregate results
        for assembled_output_gpu, count_gpu in results:
            assembled_output += assembled_output_gpu
            count += count_gpu

    else:
        # Single GPU processing
        print(f"Using single GPU: {device}")
        target_device = torch.device(device if torch.cuda.is_available() else "cpu")
        if str(target_device) == "cpu": print("Warning: CUDA not available, using CPU.")
        model.to(target_device)
        model.eval()

        # Get necessary params calculated earlier
        size = view_size
        d1, d2, d3 = downsample_factors
        s1 = d1 * size
        s2 = d2 * size
        s3 = d3 * size
        pad_size = (s3 - s1) // 2
        delta23 = (s3 - s2) // 2

        try:
            with torch.no_grad():
                for (i, j) in tqdm(positions, desc="Single GPU Processing", total=len(positions)):
                    # --- Patch extraction using NumPy slicing ---
                    img1_patch = image0[i : i + s1, j : j + s1, :]
                    padded_i = i + pad_size
                    padded_j = j + pad_size
                    img3_patch = image_padded[padded_i : padded_i + s3, padded_j : padded_j + s3, :]
                    img2_patch = image_padded[padded_i + delta23 : padded_i + delta23 + s2,
                                              padded_j + delta23 : padded_j + delta23 + s2, :]

                    if img1_patch.size == 0 or img2_patch.size == 0 or img3_patch.size == 0: continue

                    # --- Resize, Normalize, Predict ---
                    if img1_patch.ndim == 2: img1_patch = np.expand_dims(img1_patch, axis=-1)
                    if img2_patch.ndim == 2: img2_patch = np.expand_dims(img2_patch, axis=-1)
                    if img3_patch.ndim == 2: img3_patch = np.expand_dims(img3_patch, axis=-1)
                    # Grayscale handling if needed...

                    img1_patch_resized = cv2.resize(img1_patch, (size, size), interpolation=cv2.INTER_LINEAR)
                    img2_patch_resized = cv2.resize(img2_patch, (size, size), interpolation=cv2.INTER_LINEAR)
                    img3_patch_resized = cv2.resize(img3_patch, (size, size), interpolation=cv2.INTER_LINEAR)

                    if img1_patch_resized.ndim == 2: img1_patch_resized = np.expand_dims(img1_patch_resized, axis=-1)
                    if img2_patch_resized.ndim == 2: img2_patch_resized = np.expand_dims(img2_patch_resized, axis=-1)
                    if img3_patch_resized.ndim == 2: img3_patch_resized = np.expand_dims(img3_patch_resized, axis=-1)

                    img1_tensor = normalize_function_mm(img1_patch_resized, target_device)
                    img2_tensor = normalize_function_mm(img2_patch_resized, target_device)
                    img3_tensor = normalize_function_mm(img3_patch_resized, target_device)

                    outputs = model((img1_tensor, img2_tensor, img3_tensor, None))
                    outputs = outputs.cpu().numpy()[0]

                    output_resized = cv2.resize(outputs.transpose(1, 2, 0), (s1, s1), interpolation=cv2.INTER_LINEAR)
                    if output_resized.ndim == 2:
                       output_resized = np.expand_dims(output_resized, axis=-1)
                    output_resized = output_resized.transpose(2, 0, 1)

                    # --- Accumulate results ---
                    h_slice = slice(i, min(i + s1, height))
                    w_slice = slice(j, min(j + s1, width))
                    h_out_slice = slice(0, h_slice.stop - h_slice.start)
                    w_out_slice = slice(0, w_slice.stop - w_slice.start)

                    assembled_output[:, h_slice, w_slice] += output_resized[:, h_out_slice, w_out_slice]
                    count[h_slice, w_slice] += 1
        except Exception as e:
            print(f"Error during single GPU processing: {e}")
            # Clean up potentially large arrays on error
            del assembled_output, count, image0, image_padded
            return None, None

    # --- Finalize Prediction ---
    print("Finalizing prediction...")
    count[count == 0] = 1
    assembled_output /= count
    pred = np.argmax(assembled_output, axis=0).astype(np.uint8)
    pred[pred == 1] = 255 # Remap class 1 to 255 for visualization

    # --- Save Result ---
    print(f"Saving prediction mask to {result_path}...")
    try:
        if save_as_tif:
            profile = src_profile # Use profile loaded earlier
            profile.update(dtype=rasterio.uint8, count=1, nodata=0) # Update for output
            profile['compress'] = 'lzw' # Optional compression
            with rasterio.open(result_path, 'w', **profile) as dst:
                dst.write(pred, 1)
        else: # Save as PNG
             # Check if prediction is grayscale or needs conversion
            if pred.ndim == 3 and pred.shape[-1] == 1:
                 pred_save = pred[:, :, 0] # Select first channel if it's (H, W, 1)
            elif pred.ndim == 2:
                 pred_save = pred # Already (H, W)
            else:
                 print(f"Warning: Prediction has unexpected shape {pred.shape} for PNG saving. Attempting anyway.")
                 pred_save = pred # May fail if not HxW or HxWx1/3/4

            cv2.imwrite(result_path, pred_save)

        print(f"Successfully saved prediction to {result_path}")

    except Exception as e:
        print(f"Error saving output file: {e}")
        # Clean up large arrays even if saving fails
        del assembled_output, count, image0, image_padded, pred
        return None, None

    # --- Clean Up ---
    print("Cleaning up large arrays...")
    del assembled_output, count, image0, image_padded # Explicitly delete large arrays

    print("Processing complete.")
    return result_path, pred