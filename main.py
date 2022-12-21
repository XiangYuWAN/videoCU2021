# -*- coding: utf-8 -*-

import pandas as pd
import videoProcessFunctions as vpf

if __name__ == '__main__':
    videoPath = "video/BasketballPass_416x240_50.yuv"
    videoW = 416
    videoH = 240
    yuvForm = "yuv420p"
    csvPath = "csvFile/basketballCU.csv"

    # get 2D arrays of video with normal, after sobel, after lap, each row of arrays stand for a CU
    normal_array = vpf.getVideoArray(videoPath, videoW, videoH, yuvForm)
    sobel_array = vpf.getSobel_VideoArray(videoPath, videoW, videoH, yuvForm)
    lap_array = vpf.getLaplacian_VideoArray(videoPath, videoW, videoH, yuvForm)

    # calculate each CU's mean, std, skew and kurt
    N_result = vpf.Array2DTo_MSSK(normal_array)
    S_result = vpf.Array2DTo_MSSK(sobel_array)
    L_result = vpf.Array2DTo_MSSK(lap_array)

    # change column name
    S_result.columns = ['Sobel_Mean', 'Sobel_std', 'Sobel_Skew', 'Sobel_Kurt']
    L_result.columns = ['Lap_Mean', 'Lap_std', 'Lap_Skew', 'Lap_Kurt']

    #  merge all these data frame together and produce a csv file
    result = pd.concat([N_result, S_result, L_result], axis=1)
    result.to_csv(csvPath)
    # print(result)