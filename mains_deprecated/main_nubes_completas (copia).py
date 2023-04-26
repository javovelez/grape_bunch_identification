from icp_with_alignment import icp_from_neighbors
from cloud_management import add_noise_to_cloud, get_minimum_distance
import copy
import random
import open3d as o3d
import numpy as np
import pandas as pd
from time import time



if __name__ == "__main__":
    main()
    return 0
    ##### Obtenemos nubes de los racimos #######
    bunchs_paths = "../input/grapes_paths_6.txt"
    f = open(bunchs_paths)
    clouds = [] # contendrá una lista de nubes
    for bunch in f:
        cloud = o3d.io.read_point_cloud("../input/"+bunch[0:-1])
        clouds.append(cloud)
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
    n_points_min = 6            # número mínimo de puntos para obtener sub-nubes

    ############# Funciones para obtener datasets de subnubes #######
    #cloud_ds = get_bunch_dataset(clouds[0], n_points_min)
    # Devuelve  tupla (subnube1, subnube2, número de matcheos) los tamaños de las sub-nubes van de n_points_min hasta
    # la cantidad de puntos en la nube

    ################ Funciones para agregar ruido #####
    for cloud in clouds:
        add_noise_to_cloud(cloud, noise_std_dev)
    for cloud in clouds_c:
        add_noise_to_cloud(cloud, noise_std_dev)

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
                    source = clouds_c[i]
                    target = clouds[j]
                    minimun_distance = get_minimum_distance(source) # lo hice sobre source porque es la que rota sobre z

                    start = time()
                    angle = np.pi * step
                    metric = icp_from_neighbors(source, target, minimun_distance * thresh, n_neighbors, angle,
                                                distances_tolerance)
                    giros = 2/step
                    row = pd.DataFrame([[i,metric[1],j,metric[2],metric[0],metric[3],thresh,giros]],
                     columns=["nube1","tamaño_nube1","nube2","tamaño_nube2","matcheos","rmse","radio","giros"])
                    # Devuelve: (cantidad de matcheos, cantidad de puntos nube source, cantidad de puntos nube target, rmse, conjunto de correspondencia)
                    rows.append(row)
                    end = time()
                    times[i, j] = end - start
                    print(i, j, metric[0])
        data = pd.concat(rows, axis=0)
        path = "nubes_completas_conruido"+str(thresh)+".csv"
        data.to_csv(path)
    end_time = time()
    print(times)
    print(end_time - start_time)

    # results = pd.concat(df_list)
    # results = (results, times)
    # results_file = open("nombre.pickle", "wb")
    # pickle.dump(results, results_file)
    # results_file.close()
    # f.close()