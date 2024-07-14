import numpy as np


'''
Paper: `Deep learning for precipitation nowcasting: a benchmark and a new model`
'''
def merge_rgba_cv2(front_img, back_img):
    """Merge the front image with the background image using the `Painter's algorithm`

    Parameters
    ----------
    front_img : np.ndarray
    back_img : np.ndarray

    Returns
    -------
    result_img : np.ndarray
    """
    assert front_img.shape == back_img.shape
    if front_img.dtype == np.uint8:
        front_img = front_img.astype(np.float32) / 255.0
    if back_img.dtype == np.uint8:
        back_img =  back_img.astype(np.float32) / 255.0
    result_img = np.zeros(front_img.shape, dtype=np.float32)
    result_img[:, :, 3] = front_img[:, :, 3] + back_img[:, :, 3] * (1 - front_img[:, :, 3])
    result_img[:, :, :3] = (front_img[:, :, :3] * front_img[:, :, 3:] +
                            back_img[:, :, :3] * back_img[:, :, 3:] * (1 - front_img[:, :, 3:])) /\
                           result_img[:, :, 3:]
    result_img = (result_img * 255.0).astype(np.uint8)
    return result_img