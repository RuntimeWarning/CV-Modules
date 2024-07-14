import numpy as np
from matplotlib.colors import hsv_to_rgb


'''
Paper: `Deep learning for precipitation nowcasting: a benchmark and a new model`
'''
def flow_to_img(flow_dat, max_displacement=None):
    """Convert optical flow data to HSV images

    Parameters
    ----------
    flow_dat : np.ndarray
        Shape: (seq_len, 2, H, W)
    max_displacement : float or None

    Returns
    -------
    rgb_dat : np.ndarray
        Shape: (seq_len, 3, H, W)
    """
    assert flow_dat.ndim == 4
    flow_scale = np.square(flow_dat).sum(axis=1, keepdims=True)
    flow_x = flow_dat[:, :1, :, :]
    flow_y = flow_dat[:, 1:, :, :]
    flow_angle = np.arctan2(flow_y, flow_x)
    flow_angle[flow_angle < 0] += np.pi * 2
    v = np.ones((flow_dat.shape[0], 1, flow_dat.shape[2], flow_dat.shape[3]),
                dtype=np.float32)
    if max_displacement is None:
        flow_scale_max = np.sqrt(flow_scale.max())
    else:
        flow_scale_max = max_displacement
    h = flow_angle / (2 * np.pi)
    s = np.sqrt(flow_scale) / flow_scale_max

    hsv_dat = np.concatenate((h, s, v), axis=1)
    rgb_dat = hsv_to_rgb(hsv_dat.transpose((0, 2, 3, 1))).transpose((0, 3, 1, 2))
    return rgb_dat


'''
Paper: `A Dynamic Multi-Scale Voxel Flow Network for Video Prediction`
'''
def flow2rgb(flow_map_np):
    h, w, _ = flow_map_np.shape
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
        
    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1])
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]
    return rgb_map.clip(0, 1)