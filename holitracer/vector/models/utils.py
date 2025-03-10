import torch


def img_poly_to_can_poly(img_poly, valid_mask):
    """
    Convert image polygon coordinates to canonical form by subtracting minimum x,y values

    Args:
        img_poly: Tensor of shape (B, num_points, 2) containing point coordinates
        valid_mask: Tensor of shape (B, num_points) containing 1 for valid points, 0 for invalid

    Returns:
        Tensor of shape (B, num_points, 2) with normalized coordinates
    """
    if len(img_poly) == 0:
        return torch.zeros_like(img_poly)

    # Create mask for coordinate dimension
    coord_mask = valid_mask.unsqueeze(-1).expand_as(img_poly)  # (B, num_points, 2)

    # Create a large value tensor for masked values
    max_val = torch.finfo(img_poly.dtype).max
    masked_poly = torch.where(
        coord_mask.bool(),
        img_poly,
        torch.tensor(max_val, dtype=img_poly.dtype, device=img_poly.device),
    )

    # Calculate min values for x and y coordinates separately
    x_min = torch.min(masked_poly[..., 0], dim=1)[0]  # (B,)
    y_min = torch.min(masked_poly[..., 1], dim=1)[0]  # (B,)

    # Create output tensor
    can_poly = img_poly.clone()

    # Subtract min values only from valid points
    can_poly[..., 0] = torch.where(
        valid_mask.bool(), can_poly[..., 0] - x_min.unsqueeze(-1), can_poly[..., 0]
    )
    can_poly[..., 1] = torch.where(
        valid_mask.bool(), can_poly[..., 1] - y_min.unsqueeze(-1), can_poly[..., 1]
    )

    return can_poly

def compute_polygon_angles(points, valid_mask):
    """
    计算多边形每个顶点的夹角（弧度），仅对有效点且其前/后邻均有效时才计算。
    否则该位置角度为 0。
    
    Args:
        points:     (B, N, 2)  多边形顶点坐标(若是闭合多边形，做环状索引)
        valid_mask: (B, N)     1/0表示该顶点是否有效

    Return:
        angles:     (B, N)     每个点的夹角(0~π)。如果该点或其邻点无效，则输出0。
    """
    device = points.device
    B, N, _ = points.shape

    # (1) 做环状索引
    idx = torch.arange(N, device=device).unsqueeze(0).repeat(B, 1)
    idx_left = (idx - 1) % N
    idx_right = (idx + 1) % N

    # (2) 找到邻点坐标
    p_left = torch.gather(points, 1, idx_left.unsqueeze(-1).expand(-1, -1, 2))   # (B, N, 2)
    p_right = torch.gather(points, 1, idx_right.unsqueeze(-1).expand(-1, -1, 2))

    # (3) 构造相邻有效点掩码: 当前点、前一点、后一点都要有效
    valid_left  = torch.gather(valid_mask, 1, idx_left)
    valid_right = torch.gather(valid_mask, 1, idx_right)
    valid_mask_3 = valid_mask * valid_left * valid_right  # (B, N)

    # (4) 构造向量 v1, v2
    v1 = points - p_left   # (B, N, 2)
    v2 = p_right - points  # (B, N, 2)

    # (5) 计算范数 (加个较小min保证除法不爆炸)
    v1_norm = v1.norm(dim=-1).clamp(min=1e-8)   # (B, N)
    v2_norm = v2.norm(dim=-1).clamp(min=1e-8)

    # (6) 点积
    dot_prod = (v1 * v2).sum(dim=-1)  # (B, N)
    cos_theta = dot_prod / (v1_norm * v2_norm + 1e-8)

    # (7) clamp到[-1,1]之间，再acos
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    angles = torch.acos(cos_theta)   # (B, N), in [0, π]

    # (8) 对“无效的点(或者邻点)”直接置0
    angles = angles * valid_mask_3

    return angles

def compute_polygon_angles_with_loop(points, valid_mask):
    """
    计算多边形每个顶点的夹角（单位弧度）。

    约定:
      1) 若 valid_mask 全为1，则不做环状处理:
         - 即点 i 的左邻居为 i-1、右邻居为 i+1，但 0 和 N-1 两个端点不计算角度(因为其没有有效的前后邻居)。
      2) 若 valid_mask 不全是1，则做环状:
         - 使用 (i-1)%M 和 (i+1)%MS 索引做邻居，但仅当 i、i-1、i+1 的 valid_mask 都为1 时，才计算该点的角度。

    Args:
        points:     (B, N, 2)  - 多边形顶点坐标(已排好序)
        valid_mask: (B, N)     - Int / Bool，1表示该点有效，0表示无效

    Returns:
        angles:     (B, N)     - 每个顶点对应的夹角(0 ~ π)，
                                  对于无效或无法计算的点则输出0。
    """
    device = points.device
    B, N, _ = points.shape
    # 初始化角度为0，方便对无法计算或无效点直接保持0
    angles = torch.zeros(B, N, device=device, dtype=points.dtype)

    # 检查是否所有点都是有效的
    all_valid = torch.all(valid_mask.bool(), dim=1)  # (B,) 对每个batch判断

    # =========== 处理 all_valid 为 True 的样本 ===========
    # 对于该 batch 的数据，不做环状，只在 [1..N-2] 区间内计算角度
    # （因为 0 和 N-1 没有有效的左右邻点）
    # 这部分我们可以用一个循环/掩码或分批处理
    # 下面演示分批处理的思路

    if all_valid.any():
        # 取出所有行里all_valid=True的indices
        idx_full = torch.nonzero(all_valid).squeeze(-1)  # 形如 [b1, b2, ...]
        if len(idx_full) > 0:
            # 拿到这些样本对应的 points / valid_mask
            # 由于 batch_size>1 时需要切分处理，这里做一个简单 gather
            # （更高效的做法是先把那些 batch 拼在一起处理，最后再散回去）
            # 为演示清晰，这里直接 for-loop
            for b_idx in idx_full:
                # 取出该 batch 的所有顶点
                single_points = points[b_idx]  # (N, 2)
                # 如果 N < 3, 直接跳过(无法计算角度)
                if single_points.shape[0] < 3:
                    continue

                # 计算中间区间 [1..N-2] 的角度
                # p_left: i-1, p_mid: i, p_right: i+1
                # 只要 i-1, i, i+1 都在 [1..N-2] 范围内即可
                # shape -> (N-2, 2)
                p_left = single_points[0 : N - 2]
                p_mid = single_points[1 : N - 1]
                p_right = single_points[2:N]

                # 向量
                v1 = p_left - p_mid  # shape: (N-2, 2)
                v2 = p_right - p_mid  # shape: (N-2, 2)

                # dot & norms
                dot_prod = (v1 * v2).sum(dim=-1)  # (N-2,)
                norms = v1.norm(dim=-1).clamp(min=1e-8) * v2.norm(dim=-1).clamp(min=1e-8)  # (N-2,)
                cos_theta = dot_prod / (norms + 1e-8)
                cos_theta = torch.clamp(cos_theta, min=-1.0, max=1.0)
                sub_angles = torch.acos(cos_theta)  # (N-2,)

                # 存回 angles[b_idx, 1..N-1)
                angles[b_idx, 1 : N - 1] = sub_angles

    # =========== 处理 all_valid 为 False 的样本 (做环状) ===========
    if not torch.all(all_valid):
        # 拿到那些不全是有效点的 batch indices
        idx_ring = torch.nonzero(~all_valid).squeeze(-1)
        if len(idx_ring) > 0:
            # 同理，这里演示一个简单 for-loop
            for b in idx_ring:
                single_points = points[b]      # (N, 2)
                single_valid  = valid_mask[b]  # (N,)

                # 收集所有有效点的索引
                valid_indices = torch.nonzero(single_valid).squeeze(-1)  # shape: (M,)

                M = len(valid_indices)
                if M < 3:
                    # 有效点不足3，无法计算角度，全部保持0
                    continue

                # 环状遍历
                for k in range(M):
                    i   = valid_indices[k].item()                # 当前点索引
                    i_l = valid_indices[(k - 1) % M].item()      # 左邻居
                    i_r = valid_indices[(k + 1) % M].item()      # 右邻居

                    # 向量
                    v1 = single_points[i_l] - single_points[i]   # (2,)
                    v2 = single_points[i_r] - single_points[i]   # (2,)

                    # dot & norm
                    dot = torch.dot(v1, v2)
                    norm = v1.norm().clamp(min=1e-8) * v2.norm().clamp(min=1e-8)
                    cos_theta = dot / (norm + 1e-8)
                    cos_theta = torch.clamp(cos_theta, min=-1.0, max=1.0)
                    angle = torch.acos(cos_theta)

                    # 存到 angles 里
                    angles[b, i] = angle

    return angles

    
import torch

def compute_polygon_angles_with_loop_n_neighbors(points, valid_mask, n):
    """
    计算多边形每个顶点的夹角（单位弧度），考虑左右各 n 个邻居。

    约定:
        1) 若 valid_mask 全为1，则不做环状处理:
            - 点 i 的左邻居为 [i-n, ..., i-1] (不足n个则有多少算多少)、右邻居为 [i+1, ..., i+n]，但 0 ~ n-1 和 N-n ~ N-1 两个端点不计算角度(因为其没有有效的前后邻居)。
        2) 若 valid_mask 不全是1，则做环状:
            - 使用 (i-k)%M 和 (i+k)%M 索引做邻居，但仅当 i 以及左右各n个邻居都有效时，才计算该点的角度。

    Args:
        points:     (B, N, 2)  - 多边形顶点坐标(已排好序)
        valid_mask: (B, N)     - Int / Bool，1表示该点有效，0表示无效
        n:          int        - 考虑左右邻居的数量

    Returns:
        angles:     (B, N)     - 每个顶点对应的夹角(0 ~ π)，
                                对于无效或无法计算的点则输出0。
    """
    device = points.device
    B, N, _ = points.shape
    # 初始化角度为0，方便对无法计算或无效点直接保持0
    angles = torch.zeros(B, N, device=device, dtype=points.dtype)

    # 检查是否所有点都是有效的
    all_valid = torch.all(valid_mask.bool(), dim=1)  # (B,) 对每个batch判断

    # =========== 处理 all_valid 为 True 的样本 ===========
    # 对于该 batch 的数据，不做环状，只在 [n..N-n-1] 区间内计算角度
    # （因为 0 ~ n-1 和 N-n ~ N-1 没有有效的左右邻点）

    if all_valid.any():
        # 取出所有行里all_valid=True的indices
        idx_full = torch.nonzero(all_valid).squeeze(-1)  # 形如 [b1, b2, ...]
        if len(idx_full) > 0:
            for b_idx in idx_full:
                # 取出该 batch 的所有顶点
                single_points = points[b_idx]  # (N, 2)
                # 如果 N < 2*n+1, 直接跳过(无法计算角度)
                if single_points.shape[0] < 2 * n + 1:
                    continue

                # 计算中间区间 [n..N-n-1] 的角度
                for i in range(n, N - n):
                    # 计算左边 n 个点和右边 n 个点的向量
                    v_left = single_points[i - n : i] - single_points[i]  # (n, 2)
                    v_right = single_points[i + 1 : i + n + 1] - single_points[i]  # (n, 2)
                    
                    # 计算每个左边点和右边点的夹角
                    angles_ij = []
                    for j in range(n):
                      for k in range(n):
                        dot_prod = (v_left[j] * v_right[k]).sum()
                        norms = v_left[j].norm().clamp(min=1e-8) * v_right[k].norm().clamp(min=1e-8)
                        cos_theta = dot_prod / (norms + 1e-8)
                        cos_theta = torch.clamp(cos_theta, min=-1.0, max=1.0)
                        angles_ij.append(torch.acos(cos_theta))
                    
                    # 取最小夹角
                    min_angle = min(angles_ij)
                    
                    # 存回 angles[b_idx, i]
                    angles[b_idx, i] = min_angle


    # =========== 处理 all_valid 为 False 的样本 (做环状) ===========
    if not torch.all(all_valid):
        # 拿到那些不全是有效点的 batch indices
        idx_ring = torch.nonzero(~all_valid).squeeze(-1)
        if len(idx_ring) > 0:
            for b in idx_ring:
                single_points = points[b]      # (N, 2)
                single_valid = valid_mask[b]  # (N,)

                # 收集所有有效点的索引
                valid_indices = torch.nonzero(single_valid).squeeze(-1)  # shape: (M,)

                M = len(valid_indices)
                if M < 2 * n + 1:
                    # 有效点不足 2*n+1，无法计算角度，全部保持0
                    continue

                # 环状遍历
                for k in range(M):
                    i = valid_indices[k].item()  # 当前点索引

                    # 检查左右各n个邻居是否有效
                    left_neighbors_valid = all(single_valid[(valid_indices[(k - j) % M]).item()] for j in range(1, n + 1))
                    right_neighbors_valid = all(single_valid[(valid_indices[(k + j) % M]).item()] for j in range(1, n + 1))
                    
                    if not (left_neighbors_valid and right_neighbors_valid):
                        continue

                    # 计算左边 n 个点和右边 n 个点的向量
                    v_left = torch.stack([single_points[valid_indices[(k - j) % M].item()] - single_points[i] for j in range(1, n+1)]) # (n, 2)
                    v_right = torch.stack([single_points[valid_indices[(k + j) % M].item()] - single_points[i] for j in range(1, n+1)]) # (n, 2)

                    # 计算每个左边点和右边点的夹角
                    angles_ij = []
                    for j in range(n):
                      for l in range(n):
                        dot = torch.dot(v_left[j], v_right[l])
                        norm = v_left[j].norm().clamp(min=1e-8) * v_right[l].norm().clamp(min=1e-8)
                        cos_theta = dot / (norm + 1e-8)
                        cos_theta = torch.clamp(cos_theta, min=-1.0, max=1.0)
                        angles_ij.append(torch.acos(cos_theta))

                    # 取最小夹角
                    min_angle = min(angles_ij)
                    
                    # 存到 angles 里
                    angles[b, i] = min_angle

    return angles
