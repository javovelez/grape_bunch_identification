import copy
import open3d as o3d
import numpy as np

def compare_clouds_with_streching(cl1, cl2, neighbors_distance, error_threshold):
    cl1_points = np.asarray(cl1.points)
    cl2_points = np.asarray(cl2.points)

    cl1_len = len(cl1_points)
    cl2_len = len(cl2_points)

    if cl1_len >= cl2_len:
        target = cl1
        source = cl2
        target_points = cl1_points
        source_points = cl2_points
    else:
        target = cl2
        source = cl1
        target_points = cl2_points
        source_points = cl1_points
    # o3d.visualization.draw_geometries([source, target])
    err = 100
    n = 0
    for target_pair in get_pairs(target_points):
        tpc = target_points - target_pair[0]
        for source_pair in get_pairs(source_points):
            spc = source_points - source_pair[0]
            target_copy = copy.deepcopy(target)
            source_copy = copy.deepcopy(source)
            source_copy.points = o3d.utility.Vector3dVector(spc)
            target_copy.points = o3d.utility.Vector3dVector(tpc)

            target_norm = np.linalg.norm(target_pair[0] - target_pair[1])
            source_norm = np.linalg.norm(source_pair[0] - source_pair[1])
            scale_factor = target_norm / source_norm
            # print(f"""Scale factor: {scale_factor}
            #        target pair: {target_pair}
            #        source pair: {source_pair}""")

            source_copy.scale(scale_factor, (0, 0, 0))
            icp, R2, iterations_number = icp_search(source_copy, target_copy, neighbors_distance, np.pi / 6,
                                                    error_threshold)
            # print(icp)
            # o3d.visualization.draw_geometries([target_copy, source_copy])
            n += 1
            print(n)
            if icp is not None:
                if icp.inlier_rmse < err:
                    source_copy.rotate(R2)
                    source_copy.transform(icp.transformation)
                    icp_2 = o3d.pipelines.registration.registration_icp(source_copy, target_copy, 1)
                    # print(icp_2)
                    # target_volume = target_copy.get_oriented_bounding_box().volume()
                    source_volume = source_copy.get_oriented_bounding_box().volume()
                    # volume_ratio = target_volume/source_volume
                    volume_condition = True  # volume_ratio > 0.85 and volume_ratio < 1.15
                    if icp_2.inlier_rmse < err and icp_2.fitness > 0.7 and volume_condition:
                        err = icp_2.inlier_rmse
                        sf = scale_factor
                        rot = R2
                        icp_winner = icp
                        tp_w = target_pair
                        sp_w = source_pair
                        # source_copy.rotate(R)
                        # source_copy.transform(icp.transformation)
                        # o3d.visualization.draw_geometries([target_copy, source_copy])
                        if (err < 1):  # and (icp_winner.fitness > 0.7)

                            print(icp_2)
                            source_copy.transform(icp_2.transformation)
                            # o3d.visualization.draw_geometries([target_copy, source_copy])
    return err, tp_w, rot