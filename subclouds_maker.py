import numpy as np
import open3d as o3d
import copy
import os
from cloud_management import conform_point_cloud

def extract_points(pc, center, n_points, type_flag):
    """type_flag: True return poincloud, False return np.array"""
    pc_tree = o3d.geometry.KDTreeFlann(pc)
    points = np.asarray(pc.points)
    # points = np.append(points, [[0, 0, 0]], axis=0)
    [k, idx, dist] = pc_tree.search_knn_vector_3d(center, n_points)
    sub_cloud_points = points[idx, :]
    if type_flag:
        sub_cloud = copy.deepcopy(pc)
        sub_cloud.points = o3d.utility.Vector3dVector(sub_cloud_points)
        return sub_cloud, idx
    else:
        return sub_cloud_points, idx

def get_overlap(idxs_1, idxs_2):
    overlap = 0
    for idx1 in idxs_1:
        for idx2 in idxs_2:
            if idx1 == idx2:
                overlap += 1
    return overlap

def check_ids_repetition(ids, ids_list, n_iter):
    n_ids = len(ids)
    ids = np.sort(ids)
    for i, ids2 in enumerate(ids_list):
        ids2 = np.sort(ids2)
        ids_comparation = ids == ids2
        if sum(ids_comparation) == n_ids:
            return False
        if i == n_iter:
            break
    return True


def check_ids_repetition_final(pc_data_set, ids_lista):
    n_ids = ids_lista.shape[1]
    clouds_len = ids_lista.shape[0]
    idxs_to_remove = []
    ids_list = copy.deepcopy(ids_lista)
    for i, ids in enumerate(ids_list):
        ids = np.sort(ids)
        for j in range(i + 1, clouds_len):
            ids2 = np.sort(np.squeeze(ids_list[j, :]))
            ids_comparation = ids == ids2
            chek = sum(ids_comparation) == n_ids
            # print(chek)
            if chek:
                idxs_to_remove.append(j)
    idxs_to_remove = np.unique(idxs_to_remove)
    # print(idxs_to_remove)
    if idxs_to_remove.size == 0:
        return ids_lista, pc_data_set
    else:
        return np.delete(ids_lista, idxs_to_remove, 0), np.delete(pc_data_set, idxs_to_remove, 2)


def extract_points_variation(pc, center, n_points, type_flag):
    """type_flag: True return poincloud, False return np.array"""
    n_points += 1
    pc_tree = o3d.geometry.KDTreeFlann(pc)
    points = np.asarray(pc.points)
    # points = np.append(points, [[0, 0, 0]], axis=0)
    [k, idx, dist] = pc_tree.search_knn_vector_3d(center, n_points)
    idx.remove(idx[-2])
    sub_cloud_points = points[idx, :]
    if type_flag:
        sub_cloud = copy.deepcopy(pc)
        sub_cloud.points = o3d.utility.Vector3dVector(sub_cloud_points)
        return sub_cloud, idx
    else:
        return sub_cloud_points, idx


def get_sub_cloud_dataset(pc, n_points):
    points = np.asarray(pc.points)
    clouds_qty = len(points)
    points_clouds = np.zeros((n_points, 3, clouds_qty))
    ids_list = np.zeros((clouds_qty, n_points))
    for i, point in enumerate(points):
        sub_cloud_temp, ids_tmp = extract_points(pc, point, n_points, False)
        if check_ids_repetition(ids_tmp, ids_list, i):
            sub_cloud_temp, ids_tmp = extract_points_variation(pc, point, n_points, False)
        points_clouds[:, :, i] = sub_cloud_temp
        ids_list[i, :] = ids_tmp
    # print(ids_list)
    # print(points_clouds.shape)
    ids_list, points_clouds = check_ids_repetition_final(points_clouds, ids_list)
    # print(ids_list)
    # print(points_clouds.shape)
    return points_clouds, ids_list

def get_bunch_dataset(cloud, overlap_min, name, save=False):
    n_points = len(np.asarray(cloud.points))
    clouds = []
    ids_tuple= ()
    for i in range(overlap_min, n_points):
        clouds_ds, ids_list_aux = get_sub_cloud_dataset(cloud, i)
        subcloud_list = [conform_point_cloud(np.squeeze(clouds_ds[:, :, z])) for z in range(clouds_ds.shape[2])]
        for subcloud_aux  in subcloud_list:
            clouds.append(subcloud_aux)
        for ids_aux in ids_list_aux:
            ids_tuple = ids_tuple +  (ids_aux,)
    sub_cloud_pairs =[]
    n_clouds = len(clouds)
    for i in range(n_clouds):
        if save:
            try:
                os.mkdir(name)
            except OSError as exc:
                pass
            o3d.io.write_point_cloud(name+"/"+str(i)+".ply",clouds[i],write_ascii=True)
        for j in range(i,n_clouds):
            overlap = get_overlap(ids_tuple[i], ids_tuple[j])
            sub_cloud_tuple = (i,j,clouds[i], clouds[j], overlap)
            sub_cloud_pairs.append(sub_cloud_tuple)
    return  sub_cloud_pairs