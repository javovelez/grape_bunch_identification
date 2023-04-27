from icp_with_alignment import icp_scaled_and_aligned
import sys
import argparse
import open3d as o3d
import numpy as np
from time import time
import pandas as pd
from viewer import point_cloud_viewer
from cloud_management import filter_clouds

def get_points_ids(dir, name):
    df = pd.read_csv(dir + name[:-3] +'csv')
    idx_ceros = df[df['Z'] == 0].index
    df = df.drop(idx_ceros)
    return list(df['track_id'])

def save_to_file(result, save_counter, args,dir, thresh):
    frame = pd.DataFrame(result,
                         columns=["nube1", "tamaño_nube1", "nube2", "tamaño_nube2", "matcheos", "overlap", "label",
                                  "rmse", "radio", "giros"])
    path = args.output_dir + dir + "/180_thresh_0.7_radio_" + str(thresh)+ '_' + str(save_counter) +".csv"
    print(f'partial saving in {path}')
    frame.to_csv(path)

def main(args=None):
    if args is None:
        args = sys.argv[1:]
    dirs = [ 'thresh0.7/' ]
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
        description= """Etiqueta nubes 3D a partir de su nombre
        -INPUT:
            --input_dir: folder with inputs
        -OUTPUT:
            --output_dir: Carpeta de salida donde se creará el archivo labels
        """
    )
    parser.add_argument('-i', '--input_dir', type=str, required=True)
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    args = parser.parse_args(args)
    for dir in dirs:
        inputs_path = args.input_dir+dir+"labels.csv"
        inputs_df = pd.read_csv(inputs_path)
        n_clouds = len(inputs_df.index)
        clouds = {} #dict containing point clouds
        cloud_idx = 0
    ####
        for name, label in zip(inputs_df["cloud_name"], inputs_df["label"]):
            cloud = o3d.io.read_point_cloud(args.input_dir+dir + name)
            points_ids = get_points_ids(args.input_dir+dir, name)
            clouds[cloud_idx] = [name, cloud, label, points_ids]
            cloud_idx += 1
    ####


        ##### hiper-parámetros ####
        n_neighbors = 1             # cantidad de vecinos por cada punto de una nube con los que va a intentar alinear
        distances_tolerance = 0.2   # Solo comparará puntos que en ambas nubes estén a distancias similares ) +/- 20%
        threshold_percentage_list = [0.05, 0.1]   # porcentaje de la distancia en la nube a usar como trheshold
        angle_step_list = [1/4]     # paso de rotación de la nube "source" alrededor del eje z
        save_interval = 100
        start_time = time()
        n_clouds = len(clouds)

        clouds = filter_clouds(clouds)
        clouds = filter_clouds(clouds)


        for thresh_idx, thresh in enumerate(threshold_percentage_list):
            for step in angle_step_list:
                result = np.empty((save_interval, 10), dtype=object) #int(((n_clouds ** 2) / 2) + n_clouds/2)
                counter = 0
                local_counter = 0
                save_counter = 0
                stime = time()
                for i in range(len(clouds)):
                    print("##########################################################")
                    for j in range(i, len(clouds)):
                        cn1 = clouds[i][0]
                        cn2 = clouds[j][0]
                        label_1 = clouds[i][2]
                        label_2 = clouds[j][2]
                        overlap = 0
                        source = clouds[i][1]
                        target = clouds[j][1]
                        # if cn1 == '85_VID_20230321_145958.ply':
                        #     point_cloud_viewer([source, target])
                        #     point_cloud_viewer([source])
                        #     point_cloud_viewer([target])

                        label = cn1[:2] == cn2[0:2]
                        start = time()
                        angle = np.pi * step
                        # for debug: comentar metric = icp_scale_and_aligned... y ver si corre hasta el final
                        #descomentar la siguiente línea
                        # metric = icp_scaled_and_aligned(source, target, thresh,n_neighbors, angle, distance_criterion='mean')
                        metric = [ 1, 1, 1, 0, 0, 0]
                        giros = 2 / step




                        result[local_counter, :] = cn1, metric[1], cn2, metric[2], metric[0], overlap, label, metric[3], thresh, giros
                        end_t = time()
                        # Devuelve: (cantidad de matcheos, cantidad de puntos nube source, cantidad de puntos nube target, rmse, conjunto de correspondencia)
                        print(f"thresh: {thresh} ; thresh {thresh_idx + 1} de {len(threshold_percentage_list)}")
                        print(f'{cn1} (n:{metric[1]})', cn2 + f' (n:{metric[2]})')
                        print(f'matcheos: {metric[0]}, fitness: {metric[0]/metric[1]*100:2f}')
                        print(f"    counter: {counter+1}/{int(((len(clouds)**2)/2)+len(clouds)/2)}, overlap: {overlap}") #
                        print(f"    iteration time: {end_t-start} ")
                        local_counter += 1
                        if counter % (save_interval-1) == 0 and counter != 0:
                            save_to_file(result, save_counter, args, dir, thresh)
                            save_counter += 1
                            local_counter = 0
                        counter += 1
                bucle_time = time()
                print(bucle_time - start_time)


    

if __name__ == "__main__":
    main()
