import csv
from icp_with_alignment import icp_scaled_and_aligned
import open3d as o3d
import numpy as np
from time import time
import pandas as pd
from cloud_management import outliers_filter_v2, duplicates_filter_v2


def save_to_file(result, start_row, end_row, output_dir, thresh):
    frame = pd.DataFrame(result,
                         columns=["nube1", "tamaño_nube1", "nube2", "tamaño_nube2", "matcheos", "overlap", "label",
                                  "rmse", "radio", "giros"])
    path = output_dir + "/180_vs_mejoradas_180_thresh_0.7_radio_" + str(thresh) + '_rows_' + str(start_row) + '_' + str(end_row) + ".csv"
    print(f'partial saving in {path}')
    frame.to_csv(path)

def main():
    mejoradas_input_dir = '/data/180_v7/' #  'F:/Escritorio/repo_2023/identificaci-nDeRacimos/src/180_mejoradas_v2/thresh0.7/' # #
    a180_input_dir = '/data/180_v5/' # 'F:/Escritorio/repo_2023/identificaci-nDeRacimos/src/180/thresh0.7/' #'/data/180/thresh0.7/' # 'F:/Escritorio/repo_2023/identificaci-nDeRacimos/input/2023.03_captura_2/180/thresh0.7/'
    output_dir = '/data/output/180_mejoradas_v3/thresh0.7/' # 'F:/Escritorio/repo_2023/identificaci-nDeRacimos/output/2023.03_captura_2/180_vs_mejoradas_180_v2/'# '/data/output/180_mejoradas/thresh0.7/' #
    mejoradas_180_inputs_path = mejoradas_input_dir + "labels.csv"
    a180_inputs_path = a180_input_dir + "labels.csv"
    mejoradas_180_inputs_df = pd.read_csv(mejoradas_180_inputs_path)
    a180_inputs_df = pd.read_csv(a180_inputs_path)
    mejoradas_180_clouds = {}
    a180_clouds = {}
    master = open('master_180_old_vs_new_v7git .csv')#/data/
    master_reader = csv.reader(master)
    start_row = 0 # numbered from0
    end_row = 11963 #
    save_interval = 100
    threshold_percentage_list = [0.1]   # porcentaje de la distancia en la nube a usar como trheshold

    for name, label in zip(mejoradas_180_inputs_df["cloud_name"], mejoradas_180_inputs_df["label"]):
        cloud = o3d.io.read_point_cloud(mejoradas_input_dir + name)
        mejoradas_180_clouds[name] = cloud

    for name, label in zip(a180_inputs_df["cloud_name"], a180_inputs_df["label"]):
        cloud = o3d.io.read_point_cloud(a180_input_dir + name)
        a180_clouds[name] = cloud

    ##### hiper-parámetros ####
    n_neighbors = 1                     # cantidad de vecinos por cada punto de una nube con los que va a intentar alinear
    step = 1/4                          # paso de rotación de la nube "source_cloud" alrededor del eje z
    giros = 2 / step
    start_time = time()
    angle = np.pi * step

    mejoradas_180_clouds = outliers_filter_v2(mejoradas_180_clouds)
    mejoradas_180_clouds = outliers_filter_v2(mejoradas_180_clouds)
    mejoradas_180_clouds = outliers_filter_v2(mejoradas_180_clouds)
    mejoradas_180_clouds = outliers_filter_v2(mejoradas_180_clouds)
    mejoradas_180_clouds = outliers_filter_v2(mejoradas_180_clouds)
    mejoradas_180_clouds = outliers_filter_v2(mejoradas_180_clouds)
    mejoradas_180_clouds = duplicates_filter_v2(mejoradas_180_clouds)

    a180_clouds = outliers_filter_v2(a180_clouds)
    a180_clouds = outliers_filter_v2(a180_clouds)
    a180_clouds = outliers_filter_v2(a180_clouds)
    a180_clouds = outliers_filter_v2(a180_clouds)
    a180_clouds = outliers_filter_v2(a180_clouds)
    a180_clouds = outliers_filter_v2(a180_clouds)
    a180_clouds = duplicates_filter_v2(a180_clouds)

    for thresh_idx, thresh in enumerate(threshold_percentage_list):
        result = np.empty((save_interval, 10), dtype=object)
        counter = 0
        local_counter = 0
        stime = time()
        overlap = 0
        lista = []
        fila = -1
        try:
            next(master_reader, None)
            for i in range(start_row):
                next(master_reader, None)
                fila += 1
            for i in range(end_row - start_row + 1):
                row = next(master_reader, None)
                if row is None:
                    break
                else:
                    cn1, label_1, cn2, label_2 = row[1:]
                    source = a180_clouds[cn1]
                    target = mejoradas_180_clouds[cn2]
                    label = label_1 == label_2
                    start = time()

                    # for debug: comentar metric = icp_scale_and_aligned... y ver si corre hasta el final
                    # descomentar la siguiente línea
                    # metric = [1, 1, 1, 0, 0, 0]
                    metric = icp_scaled_and_aligned(source, target, thresh, n_neighbors, angle, distance_criterion='mean')

                    result[local_counter, :] = cn1, metric[1], cn2, metric[2], metric[0], overlap, label, metric[
                            3], thresh, giros

                    # Devuelve: (cantidad de matcheos, cantidad de puntos nube source_cloud, cantidad de puntos nube target_cloud,
                    # rmse, conjunto de correspondencia)

                    end_t = time()

                    print(f"thresh: {thresh} ; thresh {thresh_idx + 1} de {len(threshold_percentage_list)}")
                    print(f'{cn1} (n:{metric[1]})', cn2 + f' (n:{metric[2]})')
                    print(f'matcheos: {metric[0]}, fitness: {metric[0] / metric[1] * 100:2f}, {label}')
                    print(f'fila: {fila+i+1}')
                    print(f"    counter: {counter + 1}/{end_row-start_row+1}, overlap: {overlap}")
                    print(f"    iteration time: {end_t - start} ")
                    local_counter += 1
                    counter += 1
                    if counter == len(range(end_row - start_row + 1)):
                        save_to_file(result[:local_counter], end_row-local_counter+1, end_row, output_dir, thresh)
                        continue
                    if counter % save_interval == 0:
                        partial_end_row = start_row+i

                        save_to_file(result, start_row+i-save_interval+1, partial_end_row, output_dir, thresh)
                        local_counter = 0
                        result = np.empty((save_interval, 10), dtype=object)


        finally:
            master.close()
    print(f'Tiempo total transcurrido: {time() - start_time}')


if __name__ == "__main__":
    main()
