
from icp_with_alignment import icp_scaled_and_aligned
from cloud_management import add_noise_to_cloud, get_minimum_distance
import copy
import random
import open3d as o3d
import numpy as np
import pandas as pd
from time import time

if __name__ == "__main__":
    ##### Obtenemos nubes de los racimos #######
    bunchs_paths = "/home/javo/onedrive/Doctorado/3df/repo/identificaci-nDeRacimos/input/nubes_completas/grapes_paths_6.txt"
    f = open(bunchs_paths)
    source_clouds = [] # contendrá una lista de nubes
    for bunch in f:
        cloud = o3d.io.read_point_cloud("../input/nubes_completas/cabernet/"+bunch[0:-1])
        source_clouds.append(cloud)
    n_clouds = len(source_clouds)
    target_clouds = copy.deepcopy(source_clouds)  # contiene una copia de la lista de las nubes
                                      # uso para agregarles ruido aleatorio por separado


    ##### hiper-parámetros ####
    n_neighbors = 1             # cantidad de vecinos por cada punto de una nube con los que va a intentar alinear
    threshold_percentage_list = [0.5] # porcentaje de la distancia mínima en la nube a usar como trheshold
    angle_step_list = [1/4]     # paso de rotación de la nube "source" alrededor del eje z
    noise_std_dev = 0.05        # Desviación estandar de la gaussiana
                                #   quizás debería ser un porcentaje sobre la distancia mínima

    ############# Funciones para obtener datasets de subnubes #######
    #cloud_ds = get_bunch_dataset(clouds[0], n_points_min)
    # Devuelve  tupla (subnube1, subnube2, número de matcheos) los tamaños de las sub-nubes van de n_points_min hasta
    # la cantidad de puntos en la nube

    ################ Funciones para agregar ruido #####
    # for source_cloud in source_clouds:
    #     add_noise_to_cloud(source_cloud, noise_std_dev)
    for target_cloud in target_clouds:
        add_noise_to_cloud(target_cloud, noise_std_dev)

    ################# para eliminar n puntos al azar #############
    # delete_points(cloud, 2)
    # delete_points(cloud, 2)
    #############################################################

    n = 0
    times = np.zeros((n_clouds, n_clouds), dtype=float)
    start_time = time()
    rows = [] 
    for thresh in threshold_percentage_list:
        for step in angle_step_list: 
            for i in range(n_clouds):
                for j in range(i, n_clouds):
                    # o3d.visualization.draw_geometries([clouds_c[j].paint_uniform_color([0, 1, 0]), clouds[i].paint_uniform_color([1, 0, 0])])
                    source = source_clouds[i]
                    target = target_clouds[j]
                    source.scale(random.uniform(0.5, 1.5), (0, 0, 0))
                    angle = np.pi * step
                    metric = icp_scaled_and_aligned(source, target, thresh, n_neighbors, angle,)
                    print(f"matcheos: {metric[0]}, source_idx: {i}, target_idx: {j}, n puntos source: {metric[1]}, n puntos target: {metric[2]}")
