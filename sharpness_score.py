import h5py
import numpy as np


def gradient(x):
    # tf.image.image_gradients(image)

    # gradient step=1
    r = np.pad(x, ((0, 0),(0, 0),(0, 0),(0, 1)))[:, :, :, 1:]
    # r = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
    b = np.pad(x, ((0, 0),(0, 0),(0, 1),(0, 0)))[:, :, 1:, :]
    # b = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]
    dx, dy = np.abs(r - x), np.abs(b - x)
    dx[:, :, :, -1] = 0
    dy[:, :, -1, :] = 0

    return dx, dy


if __name__ == '__main__':
    # fake depth image
    data_path = r"E:\代码和数据集\SmaAt-UNet\train_test_2016-2019_input-length_12_img-ahead_6_rain-threshhold_50.h5"
    dataset = h5py.File(data_path, 'r', rdcc_nbytes=1024**3)["train"]['images']
    image = dataset[0][0]*47.83
    input = image[np.newaxis, np.newaxis, :, :]
    image = dataset[0][10]*47.83
    predict = image[np.newaxis, np.newaxis, :, :]
    input_dx, input_dy = gradient(input)
    gdl_input = input_dx + input_dy
    predict_dx, predict_dy = gradient(predict)
    gdl_predict = predict_dx + predict_dy
    sharpness = 10 * np.log10(255 ** 2 / np.mean(np.abs(gdl_input - gdl_predict)))
    print(sharpness)