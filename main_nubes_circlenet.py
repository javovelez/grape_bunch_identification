from icp_with_alignment import icp_scaled_and_aligned
from cloud_management import add_noise_to_cloud, get_minimum_distance
import copy
import random
import sys
import argparse
import open3d as o3d
import numpy as np
import pandas as pd
from time import time
from viewer import point_cloud_viewer


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
        description= """Etiqueta nubes 3D a partir de su nombre
        -INPUT:
            --input_dir: folder with inputs
        -OUTPUT:
            --output_dir: Carpeta de salida donde se creará el archivo labels
        """
    )
    parser.add_argument('-i','--input_dir', type=str, required=True)
    parser.add_argument('-o','--output_dir', type=str, required=True)
    args = parser.parse_args(args)

    inputs_path = args.input_dir+"/labels.csv"
    inputs_df = pd.read_csv(inputs_path)
    n_clouds = len(inputs_df.index)
    clouds = {} #dict containing point clouds
    cloud_idx = 0
    clouds_per_video = 29
    lab = 0
    clouds_counter = 0
    a = 0
    b = 6
    while_breaker=10
    name_debug = []
    while cloud_idx < 3*clouds_per_video:
        if while_breaker > 10:
            a+=1
            while_breaker =95
        for name, label in zip(inputs_df["cloud_name"], inputs_df["label"]):
            # print(f'{a}_{a + b}.ply')
            # print(lab)

            if label == lab:
                if name.endswith(f'{a}_{a + b}.ply'):
                    while_breaker -= 1
                    cloud = o3d.io.read_point_cloud(args.input_dir + name)
                    clouds[cloud_idx] = [name, cloud, label]
                    name_debug.append(name)
                    # print(name)
                    # print(cloud_idx)
                    cloud_idx += 1
                    a += 1
                    if a == 19 or a + b == 25:
                        a = 0
                        b += 1
            if cloud_idx == clouds_per_video:
                while_breaker -= 1
                lab = 1
                a = 0
                b = 6
            elif cloud_idx == 2 * clouds_per_video:
                while_breaker -= 1
                lab = 2
                a = 0
                b = 6
            elif cloud_idx == 3 * clouds_per_video:
                while_breaker -= 1
                break
        while_breaker+=1


   #realiza el producto cartesiano entre las nubes
    # df = inputs_df.merge(inputs_df, how='cross')
    # #print(df.colnames())
    # #df.to_csv(args.input_dir+"/df.csv")
    # cloud_names_1 = df["cloud_name_x"].tolist()
    # cloud_names_2 = df["cloud_name_y"].tolist()
    # labels_1 = df["label_x"].tolist()
    # labels_2 = df["label_y"].tolist()

    ##### hiper-parámetros ####
    n_neighbors = 1             # cantidad de vecinos por cada punto de una nube con los que va a intentar alinear
    distances_tolerance = 0.2   # Solo comparará puntos que en ambas nubes estén a distancias similares ) +/- 20%
    threshold_percentage_list = [0.6, 0.7, 0.8]  # porcentaje de la distancia mínima en la nube a usar como trheshold
    angle_step_list = [1/4]     # paso de rotación de la nube "source" alrededor del eje z

    #times = np.zeros((n_clouds, n_clouds), dtype=float)
    start_time = time()
    rows = []
    n_clouds = len(clouds)

    for i, thresh in enumerate(threshold_percentage_list):
        print(f"thresh : {thresh}; iter {i+1} de {len(threshold_percentage_list)}")
        for step in angle_step_list:
            result = np.empty((int(((n_clouds ** 2) / 2) + n_clouds/2), 9), dtype=object)
            counter = 0
            stime=time()
            for i in range(len(clouds)):
                for j in range(i, len(clouds)):
                    cn1 = clouds[i][0]
                    cn2 = clouds[j][0]
                    source = clouds[i][1]
                    target = clouds[j][1]
                    label = clouds[i][2] == clouds[j][2]
                    start = time()
                    angle = np.pi * step
                    metric = icp_scaled_and_aligned(source, target, thresh,
                                                    n_neighbors, angle)
                    giros = 2 / step
                    result[counter, :] = [cn1, metric[1], cn2, metric[2], metric[0], label, metric[3], thresh, giros],

                    # Devuelve: (cantidad de matcheos, cantidad de puntos nube source, cantidad de puntos nube target, rmse, conjunto de correspondencia)

                    # print(counter, n_clouds, metric[0])
                    print(cn1,cn2)
                    print(f"counter: {counter+1}/{int(((len(clouds)**2)/2)+len(clouds)/2)}, matcheos: {metric[0]}")
                    counter += 1

            frame = pd.DataFrame(result, columns=["nube1", "tamaño_nube1", "nube2", "tamaño_nube2", "matcheos", "label", "rmse", "radio", "giros"])
            path = args.output_dir + "/nubes_circlenet"+str(thresh)+".csv"
            frame.to_csv(path)
            bucle_time = time()
            print(bucle_time - start_time)

    # print(times)
    print(end_time - start_time)
    

if __name__ == "__main__":
    main()
    