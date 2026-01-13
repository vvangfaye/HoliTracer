import torch


def compute_polygon_angles(points, valid_mask):
    """
    Calculate the angle (radians) of each vertex of the polygon, only calculated for valid points and when their previous/next neighbors are also valid.
    Otherwise, the angle at that position is 0.

    Args:
        points:     (B, N, 2)  Polygon vertex coordinates (if closed polygon, use circular indexing)
        valid_mask: (B, N)     1/0 indicates whether the vertex is valid

    Return:
        angles:     (B, N)     Angle of each point (0~π). If the point or its neighbors are invalid, output 0.
    """
    device = points.device
    B, N, _ = points.shape

    # (1) Circular indexing
    idx = torch.arange(N, device=device).unsqueeze(0).repeat(B, 1)
    idx_left = (idx - 1) % N
    idx_right = (idx + 1) % N

    # (2) Find neighbor coordinates
    p_left = torch.gather(
        points, 1, idx_left.unsqueeze(-1).expand(-1, -1, 2)
    )  # (B, N, 2)
    p_right = torch.gather(points, 1, idx_right.unsqueeze(-1).expand(-1, -1, 2))

    # (3) Construct adjacent valid point mask: current point, previous point, next point must all be valid
    valid_left = torch.gather(valid_mask, 1, idx_left)
    valid_right = torch.gather(valid_mask, 1, idx_right)
    valid_mask_3 = valid_mask * valid_left * valid_right  # (B, N)

    # (4) Construct vectors v1, v2
    v1 = points - p_left  # (B, N, 2)
    v2 = p_right - points  # (B, N, 2)

    # (5) Calculate norm (add a small min to ensure division doesn't explode)
    v1_norm = v1.norm(dim=-1).clamp(min=1e-8)  # (B, N)
    v2_norm = v2.norm(dim=-1).clamp(min=1e-8)

    # (6) Dot product
    dot_prod = (v1 * v2).sum(dim=-1)  # (B, N)
    cos_theta = dot_prod / (v1_norm * v2_norm + 1e-8)

    # (7) Clamp to [-1, 1], then acos
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    angles = torch.acos(cos_theta)  # (B, N), in [0, π]

    # (8) Directly set "invalid points (or neighbors)" to 0
    angles = angles * valid_mask_3

    return angles


def compute_polygon_angles_with_loop(points, valid_mask):
    """
    Calculate the angle (unit radians) of each vertex of the polygon.

    Convention:
      1) If valid_mask is all 1, no circular processing:
         - i.e., point i's left neighbor is i-1, right neighbor is i+1, but endpoints 0 and N-1 angles are not calculated (because they lack valid previous/next neighbors).
      2) If valid_mask is not all 1, do circular processing:
         - Use (i-1)%M and (i+1)%M indices as neighbors, but calculate the angle only when valid_mask of i, i-1, i+1 are all 1.

    Args:
        points:     (B, N, 2)  - Polygon vertex coordinates (sorted)
        valid_mask: (B, N)     - Int / Bool, 1 indicates valid point, 0 indicates invalid

    Returns:
        angles:     (B, N)     - Angle corresponding to each vertex (0 ~ π),
                                  Output 0 for invalid or uncalculable points.
    """
    device = points.device
    B, N, _ = points.shape
    # Initialize angles to 0, convenient to keep 0 for uncalculable or invalid points
    angles = torch.zeros(B, N, device=device, dtype=points.dtype)

    # Check if all points are valid
    all_valid = torch.all(valid_mask.bool(), dim=1)  # (B,) Judge for each batch

    # =========== Process samples where all_valid is True ===========
    # For this batch of data, do not do circular processing, only calculate angles in [1..N-2] interval
    # (Because 0 and N-1 have no valid left/right neighbors)
    # This part can be processed using a loop/mask or batch processing
    # Below demonstrates batch processing idea

    if all_valid.any():
        # Extract indices where all_valid=True in all rows
        idx_full = torch.nonzero(all_valid).squeeze(-1)  # Shape like [b1, b2, ...]
        if len(idx_full) > 0:
            # Get points / valid_mask corresponding to these samples
            # Since split processing is needed when batch_size > 1, do a simple gather here
            # (More efficient way is to concatenate those batches first, process, then scatter back)
            # For clarity, use direct for-loop here
            for b_idx in idx_full:
                # Extract all vertices of this batch
                single_points = points[b_idx]  # (N, 2)
                # If N < 3, skip directly (cannot calculate angle)
                if single_points.shape[0] < 3:
                    continue

                # Calculate angles for middle interval [1..N-2]
                # p_left: i-1, p_mid: i, p_right: i+1
                # As long as i-1, i, i+1 are all within [1..N-2] range
                # shape -> (N-2, 2)
                p_left = single_points[0 : N - 2]
                p_mid = single_points[1 : N - 1]
                p_right = single_points[2:N]

                # Vector
                v1 = p_left - p_mid  # shape: (N-2, 2)
                v2 = p_right - p_mid  # shape: (N-2, 2)

                # dot & norms
                dot_prod = (v1 * v2).sum(dim=-1)  # (N-2,)
                norms = v1.norm(dim=-1).clamp(min=1e-8) * v2.norm(dim=-1).clamp(
                    min=1e-8
                )  # (N-2,)
                cos_theta = dot_prod / (norms + 1e-8)
                cos_theta = torch.clamp(cos_theta, min=-1.0, max=1.0)
                sub_angles = torch.acos(cos_theta)  # (N-2,)

                # Save back to angles[b_idx, 1..N-1)
                angles[b_idx, 1 : N - 1] = sub_angles

    # =========== Process samples where all_valid is False (circular) ===========
    if not torch.all(all_valid):
        # Get batch indices that are not all valid points
        idx_ring = torch.nonzero(~all_valid).squeeze(-1)
        if len(idx_ring) > 0:
            # Similarly, demonstrate a simple for-loop here
            for b in idx_ring:
                single_points = points[b]  # (N, 2)
                single_valid = valid_mask[b]  # (N,)

                # Collect all valid point indices
                valid_indices = torch.nonzero(single_valid).squeeze(-1)  # shape: (M,)

                M = len(valid_indices)
                if M < 3:
                    # Less than 3 valid points, cannot calculate angle, keep all 0
                    continue

                # Circular traversal
                for k in range(M):
                    i = valid_indices[k].item()  # Current point index
                    i_l = valid_indices[(k - 1) % M].item()  # Left neighbor
                    i_r = valid_indices[(k + 1) % M].item()  # Right neighbor

                    # Vector
                    v1 = single_points[i_l] - single_points[i]  # (2,)
                    v2 = single_points[i_r] - single_points[i]  # (2,)

                    # dot & norm
                    dot = torch.dot(v1, v2)
                    norm = v1.norm().clamp(min=1e-8) * v2.norm().clamp(min=1e-8)
                    cos_theta = dot / (norm + 1e-8)
                    cos_theta = torch.clamp(cos_theta, min=-1.0, max=1.0)
                    angle = torch.acos(cos_theta)

                    # Save to angles
                    angles[b, i] = angle

    return angles