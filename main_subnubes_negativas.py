from icp_with_alignment import icp_from_neighbors
from cloud_management import add_noise_to_cloud, get_minimum_distance
from subclouds_maker import get_bunch_dataset
import copy
import random
import os
import open3d as o3d
import numpy as np
import pandas as pd
from time import time
import glob

if __name__ == "__main__":
    path = "../input/subnubes"
    pattern = "*.ply"
    output_path = '../output/salidasICP/subnubes_negativas_sin_ruido'
    folders = os.listdir(path)
    n_folders = len(folders)
    cloud_names = [] # contendrá una lista de paths

    ##### hiper-parámetros ####
    n_neighbors = 1             # cantidad de vecinos por cada punto de una nube con los que va a intentar alinear
    distances_tolerance = 0.2   # Solo comparará puntos que en ambas nubes estén a distancias similares ) +/- 20%
    threshold_percentage_list = [0.1, 0.2, 0.3, 0.4, 0.5, 1]  # porcentaje de la distancia mínima en la nube a usar como trheshold
    noise_std_dev = 0.05        # Desviación estandar de la gaussiana
                                #   quizás debería ser un porcentaje sobre la distancia mínima
    n_points_min = 5            # número mínimo de puntos para obtener sub-nubes
    for i in range(n_folders):
        for j in range(i+1, n_folders):

            if i == j:
                continue

            print(f"----------- Folder {i} vs. Folder {j} ---------")
            folder_1 = path + '/' + folders[i]
            folder_2 = path + '/' + folders[j]
            clouds_1 = glob.glob(folder_1 + '/' + pattern)
            clouds_2 = glob.glob(folder_2 + '/' + pattern)
            n_clouds_1 = len(clouds_1)
            n_clouds_2 = len(clouds_2)
            frame = np.zeros([n_clouds_1 * n_clouds_2 * len(threshold_percentage_list), 8])
            frame_count = 0
            for k, cloud_1_path in enumerate(clouds_1):
                for l, cloud_2_path in enumerate(clouds_2):
                    for thresh in threshold_percentage_list:
                        cloud_1 = o3d.io.read_point_cloud(cloud_1_path)
                        cloud_2 = o3d.io.read_point_cloud(cloud_2_path)
                        minimun_distance = get_minimum_distance(cloud_1)
                        metric = icp_from_neighbors(cloud_1, cloud_2, minimun_distance * thresh, n_neighbors, np.pi/2,
                                                     distances_tolerance)
                        frame[frame_count, :] = [k, metric[1], l, metric[2], metric[0], metric[3], thresh, 4]
                        frame_count += 1
                        # print(metric[0:-1])
                        # print(i)

            frame = pd.DataFrame(frame,
                                 columns=["subnube1", "tamaño_subnube1", "subnube2", "tamaño_subnube2", "matcheos", "rmse", "radio", "giros"])
            frame = frame.astype({'subnube1': int, 'tamaño_subnube1': int, 'subnube2': int, 'tamaño_subnube2': int, 'matcheos': int, "giros": int})


            frame.to_csv(f'{output_path}/{folders[i]}_{folders[j]}.csv')








