import numpy as np
import cv2.cv2 as cv
import pandas as pd
import yuvio


# Read first Frame Y of video
def ReadFirstFrameY_of_Video(videoPath, videoW, videoH, yuvForm):
    image = yuvio.imread(videoPath, videoW, videoH, yuvForm)
    y = image.y
    return y

# Split a frame(image) to a 4D array
def imageSplit(image, img_h, img_w, subimg_h, subimg_w):
    array = np.lib.stride_tricks.as_strided(
        image,
        shape=(img_h // subimg_h, img_w // subimg_w, subimg_h, subimg_w),  # rows, cols
        strides=image.itemsize * np.array([subimg_h * img_w, subimg_w, img_w, 1])
    )
    return array


# Transfer a 4D array to a 2D array
def reshapeSplit(array):
    imgCU_h, imgCU_w, subimg_h, subimg_w = array.shape
    # print(imgCU_h, imgCU_w, subimg_h, subimg_w)
    reshapedArray = np.reshape(array, (imgCU_h * imgCU_w, subimg_h * subimg_w))
    return reshapedArray


# Using Sobel filter to an image
def Sobel_Filter(image):
    ddepth = cv.CV_16S
    # image = cv.GaussianBlur(image, (3, 3), 0)
    Sx = cv.Sobel(image, ddepth, 1, 0)
    Sy = cv.Sobel(image, ddepth, 0, 1)
    abs_gradX = cv.convertScaleAbs(Sx)
    abs_grady = cv.convertScaleAbs(Sy)
    grad = cv.addWeighted(abs_gradX, 0.5, abs_grady, 0.5, 0)
    return grad


# Using laplacian filter to an image
def Laplacian_Filter(image):
    ddepth = cv.CV_16S
    kernal_size = 3
    image = cv.GaussianBlur(image, (3, 3), 0)
    dst = cv.Laplacian(image, ddepth, ksize=kernal_size)
    abs_dst = cv.convertScaleAbs(dst)
    return abs_dst

# Transfer a 2D array to a csv with Mean Std Skew Kurt
def Array2DTo_MSSK(data):
    df1 = pd.DataFrame(data)
    mean = df1.mean(axis=1)
    std = df1.std(axis=1)
    skew = df1.skew(axis=1)
    kurt = df1.kurt(axis=1)

    # print("mean:\n", mean)
    # print("std:\n", std)
    # print("skew:\n", skew)
    # print("kurt:\n", kurt)

    result = pd.concat([mean, std, skew, kurt], axis=1)
    result.columns = ['mean', 'std', 'skew', 'kurt']
    # print(result)
    # result.to_csv(path, 'w')
    return result


# Extract video frames to array
def getVideoArray(videoPath, videoW, videoH, yuvForm):
    yuv_frames = yuvio.mimread(videoPath, videoW, videoH, yuvForm)

    img_h = videoH
    img_w = videoW
    subimg_h, subimg_w = 128, 128

    #初始化 initialization
    yuv_frame = yuv_frames.pop()
    y = yuv_frame.y
    # grad = sobelFilter.SobelFilter(y)
    ##################### Divide frame and produce 2d matrix ############################
    array = imageSplit(y, img_h, img_w, subimg_h, subimg_w)
    ra0 = reshapeSplit(array)

    length = yuv_frames.__len__()
    for i in range(length):
        yuv_frame = yuv_frames.pop()
        y = yuv_frame.y
        # grad = sobelFilter.SobelFilter(y)
        ##################### Divide frame and produce 2d matrix ############################
        array = imageSplit(y, img_h, img_w, subimg_h, subimg_w)
        ra = reshapeSplit(array)
        ra0 = np.row_stack((ra0, ra))

    return ra0


# Extract video frames to array, But these frames are applied SobelF
def getSobel_VideoArray(videoPath, videoW, videoH, yuvForm):
    yuv_frames = yuvio.mimread(videoPath, videoW, videoH, yuvForm)

    img_h = videoH
    img_w = videoW
    subimg_h, subimg_w = 128, 128

    # 初始化
    yuv_frame = yuv_frames.pop()
    y = yuv_frame.y
    grad = Sobel_Filter(y)
    ##################### Divide frame and produce 2d matrix ############################
    array = imageSplit(grad, img_h, img_w, subimg_h, subimg_w)
    ra0 = reshapeSplit(array)

    length = yuv_frames.__len__()
    for i in range(length):
        yuv_frame = yuv_frames.pop()
        y = yuv_frame.y
        grad = Sobel_Filter(y)
        ##################### Divide frame and produce 2d matrix ############################
        array = imageSplit(grad, img_h, img_w, subimg_h, subimg_w)
        ra = reshapeSplit(array)
        ra0 = np.row_stack((ra0, ra))

    return ra0


# Extract video Frame to array, but these frames are applied LaplacianFilter
def getLaplacian_VideoArray(videoPath, videoW, videoH, yuvForm):
    yuv_frames = yuvio.mimread(videoPath, videoW, videoH, yuvForm)

    img_h = videoH
    img_w = videoW
    subimg_h, subimg_w = 128, 128

    # 初始化
    yuv_frame = yuv_frames.pop()
    y = yuv_frame.y
    grad = Laplacian_Filter(y)
    ##################### Divide frame and produce 2d matrix ############################
    array = imageSplit(grad, img_h, img_w, subimg_h, subimg_w)
    ra0 = reshapeSplit(array)

    length = yuv_frames.__len__()
    for i in range(length):
        yuv_frame = yuv_frames.pop()
        y = yuv_frame.y
        grad = Laplacian_Filter(y)
        ##################### Divide frame and produce 2d matrix ############################
        array = imageSplit(grad, img_h, img_w, subimg_h, subimg_w)
        ra = reshapeSplit(array)
        ra0 = np.row_stack((ra0, ra))

    return ra0




if __name__ == '__main__':
    videoPath = "video/BasketballPass_416x240_50.yuv"
    videoW = 416
    videoH = 240
    yuvForm = "yuv420p"
    Nomalarray = getVideoArray(videoPath, videoW, videoH, yuvForm)
    # print(Nomalarray.shape)
    sobelarray = getSobel_VideoArray(videoPath, videoW, videoH, yuvForm)
    # print(sobelarray)
    Nresult = Array2DTo_MSSK(Nomalarray)
    Sresult = Array2DTo_MSSK(sobelarray)
    Sresult.columns = ['SobelMean', 'Sobel_std', 'SobelSkew', 'SobelKurt']
    result = pd.concat([Nresult, Sresult], axis=1)
    print(result)