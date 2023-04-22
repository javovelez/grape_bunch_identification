import copy
import random
import open3d as o3d
import numpy as np
from viewer import point_cloud_viewer
from cloud_management import get_minimum_distance,conform_point_cloud, get_median_distance_to_second_neighbord
import statistics
import bisect

def get_neighbors_generator(pc, n_neighbors):
    """
    at each iteration return a tuple with 2 elements, 1. each point of the point cloud
    2. the n_neighbors of the given point in 1.
    inputs:
        pc: PointCloud object (open3d)
        n_neighbors: int, number of neighbors to find per point
    return:
        tuple:
            - numpy array with shape (1, 3),are the coordinates of the reference point
            - numpy matrix with shape (n_neighbors, 3), that contains the coordinates of the
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
    for z in range(steps_number):
        source_cp = copy.deepcopy(source)
        coords = (0, 0, z * step)
        R_aux = source_cp.get_rotation_matrix_from_xyz(coords)

        source_cp.rotate(R_aux, center=(0, 0, 0))
        # source_cp.paint_uniform_color([1, 0, 1])
        # mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
        # o3d.visualization.draw_geometries([target, source, mesh])
        # o3d.visualization.draw_geometries([target, source_cp, mesh])

        #print(neighbors_distance)
        icp = o3d.pipelines.registration.registration_icp(source_cp, target, neighbors_distance)
        fitness_error = icp.fitness
        fitness = icp.fitness
        # print(icp)
        # itn = z + 1
        # point_cloud_viewer([target, source_cp.transform(icp.transformation)])
        # mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
        # o3d.visualization.draw_geometries([target, source_cp, mesh])
        if icp is not None:
            if fitness > highest_icp_fitness:
                highest_icp_fitness = fitness
                best_icp = icp
    return best_icp


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

    for point1, nn_points1 in get_neighbors_generator(target, n_neighbors):
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
                        icp = icp_search_arround_z(source_copy, target_copy, threshold, angle_step)
                        if icp is not None:
                            if icp.fitness > highest_fitness:
                                highest_fitness = icp.fitness
                                best_icp = icp
                        # o3d.visualization.draw_geometries([target_copy, source_copy])
    if best_icp is not None:
        return int(best_icp.fitness * n_points_source), n_points_source, n_points_target, best_icp.inlier_rmse, np.asarray(best_icp.correspondence_set)
    else:
        return 0, n_points_source, n_points_target, np.infty, []


def icp_with_pre_alignment(source, target, threshold, n_neighbors, angle_step):
    """
    Is a new version of icp_from_neighbors function that unifies the criterion that allows omitting comparisons between
    clouds. The new criterion is based on the difference between the segments that define the alignment should be lower
    than the threshold used in ICP.
    Compute icp algorithm to a set of alignments between the source and target clouds. The set of alignments is the
    result of align each point and his n_neighbors nearest neighbors from the source cloud with each point and his
    nearest neighbor from the target point cloud. Once each alignment is done, icp is computed for different rotations
    around the alignment axis. The number of alignments is given by the 2*pi / angle_step.
    inputs:
        source: PointCloud object (open3d), point cloud to be compared with target point cloud
        target: PointCloud object (open3d), point cloud to be compared with source point cloud
        threshold: float, radius that define the area in which each point of the
                   source point cloud can find the closest point of the target point cloud
        n_neighbors: int, number of neighbors for each point of each cloud, in which the alignment will be carried out
        angle_step: float, the angle unit that the source point cloud will be rotated over the aligned axis as a
        different initializations to apply icp for each pairwise alignment.
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

    for point1, nn_points1 in get_neighbors_generator(target, n_neighbors):
        for point2, nn_points2 in get_neighbors_generator(source, n_neighbors):
            for i in range(n_neighbors):
                dist1 = np.linalg.norm(point1 - nn_points1[i])
                for j in range(n_neighbors):
                    dist2 = np.linalg.norm(point2 - nn_points2[j])
                    if compare_distances_v2(dist1, dist2, threshold):

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
                        icp = icp_search_arround_z(source_copy, target_copy, threshold, angle_step)
                        if icp is not None:
                            if icp.fitness > highest_fitness:
                                highest_fitness = icp.fitness
                                best_icp = icp
                        # o3d.visualization.draw_geometries([target_copy, source_copy])
    if best_icp is not None:
        return int(best_icp.fitness * n_points_source), n_points_source, n_points_target, best_icp.inlier_rmse, np.asarray(best_icp.correspondence_set)
    else:
        return 0, n_points_source, n_points_target, np.infty, []



def icp_scaled_and_aligned(source, target, threshold_percentage, n_neighbors, angle_step, distance_criterion='target'):
    """
    Compute icp algorithm to a set of alignments between the source and target clouds. The set of alignments is the
    result of align each point and his n_neighbors nearest neighbors from the source cloud with each point and his nearest neighbor
    from the target point cloud. Once each alignment is done, icp is computed for different rotations around the
    alignment axis. The number of alignments is given by the 2*pi / angle_step.
    inputs:
        source: PointCloud object (open3d), point cloud to be compared with target point cloud
        target: PointCloud object (open3d), point cloud to be compared with source point cloud
        threshold_percentage: float, percentage of the minimun distance that will define the area in which each point of
                              the source point cloud can find the closest point of the target point cloud
        n_neighbors: int, number of neighbors for each point of each cloud, in which the alignment will be carried out
        angle_step: float, the angle unit that the source point cloud will be rotated over the aligned axis as a
        different initializations to apply icp for each pairwise alignment.

    return:
        tuple that contains:
            - int,  number of matching points
            - n_points_source: int, source's number of points
            - n_points_target: int, target's number of points
            - rmse: float, root mean-square error
            - numpy matrix with shape (n, 2), Is the correspondence set. n is the number of matching points, each row
             correspond with a match pair, first number correspond with an index of the source cloud and the second
             number is the index of the target point.
    """

    highest_fitness = 0
    best_icp = None
    source_points = np.asarray(source.points)
    n_points_source = len(source_points)
    target_points = np.asarray(target.points)
    n_points_target = len(target_points)

    for point1, nn_points1 in get_neighbors_generator(target, n_neighbors):
        for point2, nn_points2 in get_neighbors_generator(source, n_neighbors):
            for i in range(n_neighbors):
                dist_target = np.linalg.norm(point1 - nn_points1[i])
                for j in range(n_neighbors):
                    dist_source = np.linalg.norm(point2 - nn_points2[j])
                    scale_factor = dist_target/dist_source
                    point2 = point2 * scale_factor
                    nn_points2[j] = nn_points2[j]*scale_factor
                    source_copy = copy.deepcopy(source)
                    # point_cloud_viewer([target, source_copy])
                    source_copy.scale(scale_factor, (0, 0, 0))
                    if distance_criterion == 'min':
                        reference_distance = get_minimum_distance(source_copy)
                    elif distance_criterion == 'median':
                        reference_distance = get_median_distance_to_second_neighbord(source_copy)
                    elif distance_criterion == 'target': #es la distancia del par target en esta iteración
                        reference_distance = dist_target
                    # Si la nube source se achica mucho respecto a la target
                    # significa que no son la misma nube, por ende tomar la mediana de la distancia de la nube
                    # para el radio de matcheo implicará que matcheen nubes que no deben matchear

                    rot1_to_z, transl1_to_z = get_RT_in_z_direction(point1, nn_points1[i], dist_target)
                    rot2_to_z, transl2_to_z = get_RT_in_z_direction(point2, nn_points2[j], dist_target)

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
                    icp = icp_search_arround_z(source_copy, target_copy, threshold_percentage*reference_distance, angle_step)
                    if icp is not None:
                        if icp.fitness > highest_fitness:
                            highest_fitness = icp.fitness
                            best_icp = icp
                            # o3d.visualization.draw_geometries([target_copy, source_copy])
    if best_icp is not None:
        return int(best_icp.fitness * n_points_source), n_points_source, n_points_target, best_icp.inlier_rmse, np.asarray(best_icp.correspondence_set)
    else:
        return 0, n_points_source, n_points_target, np.infty, []


