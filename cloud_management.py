import open3d as o3d
import numpy as np
import copy
from statistics import median

def get_minimum_distance(cloud):
    """
    compute the distance between all the pairs of grapes and return the minimun
    distance
    input:
        cloud: PointCloud object (open3d)
    return:
        min_distance: minimun distance (float)
    """
    pc_tree = o3d.geometry.KDTreeFlann(cloud)
    points = np.asarray(cloud.points)
    min_distance = 100
    # points = np.append(points, [[0, 0, 0]], axis=0)
    for point in points:
        _, idx, _ = pc_tree.search_knn_vector_3d(point, 2)
        distance = np.linalg.norm(point - points[idx[1], :])
        if min_distance > distance:
            min_distance = distance
    return min_distance

def get_mean_distance(cloud, n_neighbors=1):
    index_pair_set = set()
    index_pair_list = []
    pc_tree = o3d.geometry.KDTreeFlann(cloud)
    points = np.asarray(cloud.points)
    distances_list = []
    for point in points:
        a, idxs, squared_distances = pc_tree.search_knn_vector_3d(point, n_neighbors + 1)
        del(squared_distances[0])
        distances = np.sqrt(squared_distances)
        for i, distance in enumerate(distances):
            distances_list.append(distance)
    distances_mean = statistics.median(distances_list)
    return distances_mean

def get_median_distance_to_second_neighbord(cloud, n_neighbors=2):
    index_pair_set = set()
    index_pair_list = []
    pc_tree = o3d.geometry.KDTreeFlann(cloud)
    points = np.asarray(cloud.points)
    distances_list = []
    for point in points:
        a, idxs, squared_distances = pc_tree.search_knn_vector_3d(point, n_neighbors + 1)
        # del(squared_distances[0]) #Elimino el punto de referencia de la lista
        # del(squared_distances[0]) #Elimino el primer vecino de la lista
        distance = np.sqrt(squared_distances[2]) #Elijo la distancia al segundo vecino
        # for i, distance in enumerate(distances):
        distances_list.append(distance)
    distances_median = median(distances_list)
    return distances_median

def delete_points(cl, n_points):
    """
    delete n_points from the PointCloud cl
    inputs:
        cl: PointCloud object (open3d)
        n_points: number of points to delete from cl
    """
    points = np.asarray(cl.points)
    idx = np.random.randint(0, len(points), n_points)
    print(f"Removed point index: {idx}")
    points = np.delete(points, idx, axis=0)
    cl.points = o3d.utility.Vector3dVector(points)


def add_noise_to_cloud(cl, std_dev):
    """
    add gaussian noise to a point cloud
    inputs:
        cl: PoinCloud object (open3d)
        std_dev: standard deviation of the gaussian noise to add
    """
    pts = np.asarray(cl.points)
    noise = np.random.normal(loc=0.0, scale=std_dev, size=pts.shape)
    pts = pts + noise
    cl.points = o3d.utility.Vector3dVector(pts)


def get_pairs(pc_points):
    pc_len = len(pc_points)
    for idx1 in range(pc_len):
        for idx2 in range(idx1 + 1, pc_len):
            yield pc_points[[idx1, idx2], :]

def conform_point_cloud(points):
    """
    create a PointCloud object from a matrix
    inputs:
        points: a mumpy matrix with shape (n, 3) (n arbitrary points and x, y, z coordinates)
    return:
        PointCloud object (open3d)
    """
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

def add_points(pc, points_to_add):
    """
    add points to a point cloud
    inputs:
        pc: PointCloud object
        points_to_add: numpy matrix with shape (n, 3)  [[x1 y1 z1][x2 y2 z2]... ]
    return:
        original PointCloud object containing the new points
    """
    points = np.asarray(pc.points)
    points = np.append(points, points_to_add, axis=0)
    return o3d.utility.Vector3dVector(points)

def get_neighbors(pc, n_neighbors):
    pc_tree = o3d.geometry.KDTreeFlann(pc)
    points = np.asarray(pc.points)
    # points = np.append(points, [[0, 0, 0]], axis=0)
    for point in points:
        _, idxs, _ = pc_tree.search_knn_vector_3d(point, n_neighbors + 1)
        yield points[idxs[0], :], points[idxs[1:], :]

def get_neighbors_to_filter(pc, n_neighbors, median_proportion=0.3):
    index_pair_set = set()
    index_pair_list = []
    pc_tree = o3d.geometry.KDTreeFlann(pc)
    points = np.asarray(pc.points)
    distances_list = []
    for point in points:
        a, idxs, squared_distances = pc_tree.search_knn_vector_3d(point, n_neighbors + 1)
        del(squared_distances[0])
        idx_point = idxs.pop(0)
        distances = np.sqrt(squared_distances)
        for i, distance in enumerate(distances):
            set_len_prev = len(index_pair_set)
            idx_pair = tuple(sorted((idx_point, idxs[i])))
            index_pair_set.add(idx_pair)
            set_len_post = len(index_pair_set)
            if set_len_post != set_len_prev:
                distances_list.append(distance)
                index_pair_list.append(idx_pair)
    distances_list.append(0.5)
    index_pair_list.append((1, 12))
    combined_list = zip(distances_list, index_pair_list)
    ordered_list = sorted(combined_list, key=lambda x: x[0])
    ordered_distances_list , ordered_index_pair_list = zip(*combined_list)

    distances_median = statistics.median(ordered_distances_list)
    threshold_index = bisect.bisect(ordered_distances_list,median_proportion*distances_median)
    print("")
    #print(distances_list , len(distances_list))
    #print(statistics.median(distances_list))


if __name__=='__main__':
    cloud_path = '/mnt/datos/onedrive/Doctorado/3df/repo/identificaci-nDeRacimos/input/nubes_completas/bonarda/frames01/VID_20220217_101459.ply'
    pc = o3d.io.read_point_cloud(cloud_path)
    get_neighbors_to_filter(pc, 2)
