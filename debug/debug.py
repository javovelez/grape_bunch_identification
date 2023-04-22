from cloud_management import get_minimum_distance
import copy
import random
import open3d as o3d
import numpy as np
import pandas as pd
from time import time
# from  icp import *
from viewer import point_cloud_viewer



def get_neighbors_generator(pc, n_neighbors):
    """
    for each iteration return a tuple with 2 elements, 1. each point of a the point cloud
    2. the n_neighbors of the given point in 1.
    inputs:
        pc: PointCloud object (open3d)
        n_neighbors: int, number of neighbors to find per point
    return:
        tuple:
            - numpy array with shape (1, 3),are the coordinates of the reference point
            - numpy matrix with shape (n_neighbors, 3), are the coordinates of the
              n_neighbors points
    """
    pc_tree = o3d.geometry.KDTreeFlann(pc)
    points = np.asarray(pc.points)
    # points = np.append(points, [[0, 0, 0]], axis=0)
    for point in points:
        _, idxs, _ = pc_tree.search_knn_vector_3d(point, n_neighbors + 1)
        yield points[idxs[0], :], points[idxs[1:], :]


def compare_distances(dist1, dist2, tolerance):
    """
    return True if dist2 is inside the range of [dist1*tolerance, dis1*(1-tolerance)]
    inputs:
        dist1: float
        dist2: float
        tolerance: float, number reprecenting the tolerance of the distance.
                   must be lower of 1
    return: bool, True, if dist2 is inside the tolerance interval
    """
    lower_end = dist1 * (1 - tolerance)
    top_end = dist1 * (1 + tolerance)
    if lower_end < dist2 < top_end:
        return True
    else:
        return False

def compare_distances_v2(dist1, dist2, threshold):
    """
    return True if the absolute value of the difference of the distances is less than the threshold
    inputs:
        dist1: float
        dist2: float
        threshold: float, maximum allowed difference between dist1 and dist2
    return: bool, True if the absolute value of the difference between dist1 and dist2 is less than the threshold
    """
    d = abs(dist1-dist2)
    if d <= threshold:
        return True
    else:
        return False

def get_RT_in_z_direction(point, neighbor, dist):
    """
    Compute the rotation and the translation to align a pair of points over the z axis
    inputs:
        point: numpy float array with shape (1, 3) first point of the pair
        neighbor: numpy float array with shape (1, 3), second point of the pair
        dist: distance between the points
    output:
        R: numpy float matrix, rotation matrix
        centroid_B: numpy float array, translation
    """
    A = np.asarray([[0., 0., 0.], [0., 0., dist]])
    B = np.asarray([point, neighbor])
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    Am = A - centroid_A
    Bm = B - centroid_B
    H = np.transpose(Am) @ Bm
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # point_cloud_viewer([conform_point_cloud(Am), conform_point_cloud(np.dot(Bm, R))])
    return R, centroid_B


def icp_search_arround_z(source, target, neighbors_distance=0.8, step=np.pi / 10):
    """
    compute ipc algorithm for different rotations of the source cloud around the z axis.
    inputs:
        source: PointCloud object (open3d), point cloud that rotates around the z axis
        target: PointCloud object (open3d), point cloud that remains fixed
        neighbors_distance: float, radius that define the area in which each point of the
                            source point cloud can find the closest point of the target
                            point cloud
        step: float, angle step rotation over the z axis
    return:
        best_icp: RegistrationResult object (open3d), is the best ipc result obtained
                  according to the maximun fitness criteria
    """
    steps_number = int(2 * np.pi // step)
    highest_icp_fitness = 0
    best_icp = None
    cld = source
    for z in range(steps_number):
        source_cp = copy.deepcopy(source)
        coords = (0, 0, z * step)
        R_aux = source_cp.get_rotation_matrix_from_xyz(coords)

        source_cp.rotate(R_aux, center=(0, 0, 0))
        # o3d.visualization.draw_geometries([source, source_cp])

        # source_cp.paint_uniform_color([1, 0, 1])
        # mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
        # o3d.visualization.draw_geometries([target, source, mesh])
        # o3d.visualization.draw_geometries([target, source_cp])

        icp = o3d.pipelines.registration.registration_icp(source_cp, target, neighbors_distance)
        fitness = icp.fitness
        # print(icp)
        # print(icp.transformation)

        # itn = z + 1
        # point_cloud_viewer([target, source_cp.transform(icp.transformation).paint_uniform_color([0, 1, 1])])
        # mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
        # o3d.visualization.draw_geometries([target, source_cp, mesh])
        if icp is not None:
            if fitness > highest_icp_fitness:
                highest_icp_fitness = fitness
                best_icp = icp
                cld = source_cp
                print(icp)
    return best_icp, cld


def custom_roto_translate(pc, rot, translate):
    """
    Roto translate a point cloud
    inputs:
        pc: PointCloud object (open3d), point cloud to roto translate
        rot: numpy float matrix, rotation matrix
        translate: numpy float array, translation vector
    """
    points = np.asarray(pc.points)
    points = points - translate
    points = np.dot(points, rot)
    pc.points = o3d.utility.Vector3dVector(points)

def icp_from_neighbors(source, target, threshold, n_neighbors, angle_step, distances_tolerance):
    """
    Compute icp algorithm to a set of alignments between the source and target clouds. The set of alignments is the
    result of align each point and his n_neighbors nearest neighbors from the source cloud with each point and his nearest neighbor
    from the target point cloud. Once each alignment is done, icp is compute for different rotations around the
    alignment axis. The number of alignments is give by the 2*pi / angle_step.
    inputs:
        source: PointCloud object (open3d), point cloud to be compare with target point cloud
        target: PointCloud object (open3d), point cloud to be compare with source point cloud
        threshold: float, radius that define the area in which each point of the
                   source point cloud can find the closest point of the target point cloud
        n_neighbors: int, number of neighbors for each point of each cloud, in which the alignment will be carried out
        angle_step: float, the angle unit that the source point cloud will be rotated over the aligned axis as a
        different initializations to aplly icp for each pairwise alignment.
        distances_tolerance: float, tolerance. If the distance between a pair of the target cloud doesn't match inside
                             the range [dist pair source * tolerance, dist pair source * (1-tolerance)], the alignment
                             is not carried out and icp is not calculated for that pair of pairs.
    return:
        tuple that contains:
            - int,  number of matching points
            - n_points_source: int, source's number of points
            - n_points_target: int, target' number of points
            - rmse: float, root mean square error
            - numpy matrix with shape (n, 2), Is the correspondence set. n is the number of matching points, each row
             correspond with a match pair, first number correspond with an index of the source cloud and the second
             number is the index of the target point.
    """

    highest_fitness = 0
    best_icp = None
    n_points_source = len(np.asarray(source.points))
    n_points_target = len(np.asarray(target.points))
    ca = 0

    for point1, nn_points1 in get_neighbors_generator(target, n_neighbors):
        cb = 0
        for point2, nn_points2 in get_neighbors_generator(source, n_neighbors):
            for i in range(n_neighbors):
                dist1 = np.linalg.norm(point1 - nn_points1[i])
                for j in range(n_neighbors):
                    dist2 = np.linalg.norm(point2 - nn_points2[j])
                    if compare_distances(dist1, dist2, distances_tolerance):

                        rot1_to_z, transl1_to_z = get_RT_in_z_direction(point1, nn_points1[i], dist1)
                        rot2_to_z, transl2_to_z = get_RT_in_z_direction(point2, nn_points2[j], dist2)


                        source_copy = copy.deepcopy(source)
                        target_copy = copy.deepcopy(target)
                        source_copy.paint_uniform_color([0, 0, 1])
                        target_copy.paint_uniform_color([1, 0, 1])

                        # point_cloud_viewer([target_copy, source_copy])

                        custom_roto_translate(target_copy, rot1_to_z, transl1_to_z)
                        custom_roto_translate(source_copy, rot2_to_z, transl2_to_z)

                        # point_cloud_viewer([target_copy, source_copy])

                        source_copy.paint_uniform_color([0, 0, 1])
                        target_copy.paint_uniform_color([1, 0, 1])
                        # o3d.visualization.draw_geometries([target_copy, source_copy])
                        icp, cld = icp_search_arround_z(source_copy, target_copy, threshold, angle_step)
                        if icp is not None:
                            if icp.fitness > highest_fitness:
                                highest_fitness = icp.fitness
                                best_icp = icp
                                cld_select = copy.deepcopy(cld)
                                print(f"sel: {icp}")
                                point_cloud_viewer(
                                    [target_copy, cld_select.transform(icp.transformation).paint_uniform_color([0, 0, 1])])
                        # rot1_to_zb, transl1_to_zb = get_RT_in_z_direction(nn_points1[i], point1, dist1)
                        # rot2_to_zb, transl2_to_zb = get_RT_in_z_direction(nn_points2[j], point2, dist2)
                        # source_copy = copy.deepcopy(source)
                        # target_copy = copy.deepcopy(target)
                        # source_copy.paint_uniform_color([0, 0, 1])
                        # target_copy.paint_uniform_color([1, 0, 1])
                        #
                        # # point_cloud_viewer([target_copy, source_copy])
                        #
                        # custom_roto_translate(target_copy, rot1_to_zb, transl1_to_zb)
                        # custom_roto_translate(source_copy, rot2_to_zb, transl2_to_zb)
                        #
                        # # point_cloud_viewer([target_copy, source_copy])
                        #
                        # source_copy.paint_uniform_color([0, 0, 1])
                        # target_copy.paint_uniform_color([1, 0, 1])
                        # # o3d.visualization.draw_geometries([target_copy, source_copy])
                        # icp, cld = icp_search_arround_z(source_copy, target_copy, threshold, angle_step)
                        # if icp is not None:
                        #     if icp.fitness > highest_fitness:
                        #         highest_fitness = icp.fitness
                        #         best_icp = icp
                        #         cld_select = copy.deepcopy(cld)
                        #         print(f"sel: {icp}")
                        #         point_cloud_viewer(
                        #             [target_copy, cld_select.transform(icp.transformation).paint_uniform_color([0, 0, 1])])

            cb += 1
        ca += 1
                        # o3d.visualization.draw_geometries([target_copy, source_copy])
    if best_icp is not None:
        return int(best_icp.fitness * n_points_source), n_points_source, n_points_target, best_icp.inlier_rmse, np.asarray(best_icp.correspondence_set)
    else:
        return 0, n_points_source, n_points_target, np.infty, []


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def pick_points(pcd):
    print("")
    print(
        "1) Please pick at least three correspondences using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press 'Q' to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()

def demo_manual_registration(source, target, id_vecors, select, th):
    sl = {
        0: [0, 1],
        1: [0, 2],
        2: [1, 2]
    }
    # pick points from two point clouds and builds correspondences
    # pick_points(source)
    picked_id_source = id_vecors[sl[select][0]] #pick_points(source)
    picked_id_target = id_vecors[sl[select][1]]  #pick_points(target)
    # print(picked_id_source)

    assert (len(picked_id_source) >= 3 and len(picked_id_target) >= 3)
    assert (len(picked_id_source) == len(picked_id_target))
    corr = np.zeros((len(picked_id_source), 2))
    corr[:, 0] = picked_id_source
    corr[:, 1] = picked_id_target

    # estimate rough transformation using correspondences
    print("Compute a rough transform using the correspondences given by user")
    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    trans_init = p2p.compute_transformation(source, target,
                                            o3d.utility.Vector2iVector(corr))

    # point-to-point ICP for refinement
    print("Perform point-to-point ICP refinement")
    threshold = th
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    print(reg_p2p)
    # icp = o3d.pipelines.registration.registration_icp(source_cp, target, th)
    draw_registration_result(source, target, reg_p2p.transformation)
    print("")



if __name__ == "__main__":
    ##### hiper-parámetros ####
    n_neighbors = 1             # cantidad de vecinos por cada punto de una nube con los que va a intentar alinear
    distances_tolerance = 0.2   # Solo comparará puntos que en ambas nubes estén a distancias similares ) +/- 20%
    threshold_percentage_list = [0.3]  # porcentaje de la distancia mínima en la nube a usar como trheshold
    angle_step_list = [1/4]     # paso de rotación de la nube "source" alrededor del eje z
    point_n = 95
    select = 0

    ###########################################
    # 0: corrida 01_03
    # 1: corrida 01_13
    # 2: corrida 03_13

    ##### Obtenemos nubes de los racimos #######
    dir_target = []
    dir_source = []
    clouds_names_target = []
    clouds_names_source = []
    path_01 = "../input/nubes_completas/bonarda/frames01/paths.txt" # 0
    path_03 = "../input/nubes_completas/bonarda/frames03/paths.txt" # 1
    path_13 = "../input/nubes_completas/bonarda/frames13/paths.txt" # 2

    select_dict = {
        #   source   target
        0: [path_01, path_03],
        1: [path_01, path_13],
        2: [path_03, path_13]
    }
    # [[01], [03], [13]]
    point_dict = {
        87: [[8, 0, 12], [9, 15, 8]],
        95: [[7, 5, 9], [10, 0, 9]],
        86: [[15, 10, 13], [8, 11, 9]],
        90: [[4, 7, 6], [8, 2, 0]],
        91: [[20, 13, 12], [19, 4, 5]],
        92: [[19, 1, 18], [7, 5, 8]],
        9091:  [[2, 19, 5], [0, 0, 0], [1, 11, 5]],
        8081: [[4, 6, 14], [0, 0, 0], [2, 5, 11]]
    }
    sel = select_dict[select]

    f = open(sel[0])
    for bunch in f:
        dir_source.append(bunch[0:-1])
        cloud = o3d.io.read_point_cloud(sel[0][:42]+bunch[0:-1])
        clouds_names_source.append(cloud)
    f.close()

    f = open(sel[1])
    for bunch in f:
        dir_target.append(bunch[0:-1])
        cloud = o3d.io.read_point_cloud(sel[1][:42]+bunch[0:-1])
        clouds_names_target.append(cloud)
    f.close()

    n_source_clouds = len(clouds_names_source)
    n_target_clouds = len(clouds_names_target)

    count = 0
    start_time = time()
    for thresh in threshold_percentage_list:
        for step in angle_step_list:
            for i in range(n_source_clouds):
                if i != 95: continue
                for j in range(n_target_clouds):
                    if j!= 95: continue
                    print(dir_source[i])
                    print(dir_target[j])

                    source = clouds_names_source[i]
                    target = clouds_names_target[j]
                    minimun_distance = get_minimum_distance(source) # lo hice sobre source porque es la que rota sobre z
                    demo_manual_registration(source, target, point_dict[point_n], select, minimun_distance * thresh)

                    # source.translate((0, 0, 0), relative=False)
                    # target.translate((0, 0, 0), relative=False)
                    # point_cloud_viewer([target.paint_uniform_color([0, 1, 0]), source.paint_uniform_color([1, 0, 0])])
                    # o3d.visualization.draw_geometries([target.paint_uniform_color([0, 1, 0]), source.paint_uniform_color([1, 0, 0])])
                    start = time()
                    angle = np.pi * step
                    metric = icp_from_neighbors(source, target, minimun_distance * thresh, n_neighbors, angle,
                                                distances_tolerance)
                    print(metric)



