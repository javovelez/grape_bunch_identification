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

if __name__ == "__main__":

    ##### Obtenemos nubes de los racimos #######
    bunchs_paths = "../input/nubes_completas/grapes_paths_6.txt"
    f = open(bunchs_paths)
    clouds = [] # contendrá una lista de nubes
    cloud_names = [] # contendrá una lista de paths

    for bunch in f:
        cloud = o3d.io.read_point_cloud("../input/nubes_completas/"+bunch[0:-1])
        clouds.append(cloud)
        working_directory, video_name = os.path.split(bunch[0:-1])
        file_name_without_extension, extension = os.path.splitext(video_name)
        cloud_names.append(file_name_without_extension)
    n_clouds = len(clouds)
    clouds_c = copy.deepcopy(clouds)  # contiene una copia de la lista de las nubes
                                      # uso para agregarles ruido aleatorio por separado


    ##### hiper-parámetros ####
    n_neighbors = 1             # cantidad de vecinos por cada punto de una nube con los que va a intentar alinear
    distances_tolerance = 0.2   # Solo comparará puntos que en ambas nubes estén a distancias similares ) +/- 20%
    threshold_percentage_list = [0.1,0.2,0.3,0.4,0.5,1]  # porcentaje de la distancia mínima en la nube a usar como trheshold
    angle_step_list = [1/2, 1/3, 1/4, 1/6]     # paso de rotación de la nube "source" alrededor del eje z
    noise_std_dev = 0.05        # Desviación estandar de la gaussiana
                                #   quizás debería ser un porcentaje sobre la distancia mínima
    n_points_min = 5            # número mínimo de puntos para obtener sub-nubes

    ############# Funciones para obtener datasets de subnubes #######
    start_time = time()
    for i, cloud in enumerate(clouds):
        # if i!=14:
        #     continue
        print("-------------------------------------SUBNUBES DE LA NUBE",i,"--------------------------------")

        sub_cloud_pairs = get_bunch_dataset(cloud, n_points_min, cloud_names[i])
        n_sub_clouds_pairs = len(sub_cloud_pairs)
        frame = np.zeros((n_sub_clouds_pairs*len(threshold_percentage_list), 10))
        j = 0
        for pair in sub_cloud_pairs:
            index1, index2, cloud1, cloud2, overlap = pair
            for thresh in threshold_percentage_list:
                minimun_distance = get_minimum_distance(cloud1)
                angle = np.pi/2
                metric = icp_from_neighbors(cloud1, cloud2, minimun_distance * thresh, n_neighbors, angle,
                                                distances_tolerance)
                giros = 4
                frame[j, :] = [index1,metric[1],index2,metric[2],metric[0],overlap,metric[3],thresh,giros,i]
                print(index1, index2, metric[0],overlap)
                j += 1
        print(frame, frame.shape, j)
        data = pd.DataFrame(frame,
                           columns=["subnube1", "tamaño_subnube1", "subnube2", "tamaño_subnube2", "matcheos", "overlap",
                                    "rmse", "radio", "giros", "nube_completa"])
        path = "../output/subnubes_sin_ruido_nube"+str(i)+".csv"
        data.to_csv(path)
    end_time = time()
    print(end_time - start_time)
