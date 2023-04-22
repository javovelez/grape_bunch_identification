from cloud_management import get_minimum_distance
import open3d as o3d
import numpy as np
import copy
import pandas as pd


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


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def demo_manual_registration(source, target, th):
    sl = {
        0: [0, 1],
        1: [0, 2],
        2: [1, 2]
    }
    # pick points from two point clouds and builds correspondences
    # pick_points(source)
    picked_id_source = pick_points(source)
    picked_id_target = pick_points(target)
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

    ###########################################
    thresh = 0.3
    select = 1
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

    for i in range(n_source_clouds):
        if i != 81: continue
        for j in range(n_target_clouds):
            if j!= 80: continue
            print(dir_source[i])
            print(dir_target[j])

            source = clouds_names_source[i]
            target = clouds_names_target[j]
            o3d.visualization.draw_geometries([target.paint_uniform_color([0, 1, 0]), source.paint_uniform_color([1, 0, 0])])

            minimun_distance = get_minimum_distance(source) # lo hice sobre source porque es la que rota sobre z
            demo_manual_registration(source, target, minimun_distance * thresh)

            # source.translate((0, 0, 0), relative=False)
            # target.translate((0, 0, 0), relative=False)
            # point_cloud_viewer([target.paint_uniform_color([0, 1, 0]), source.paint_uniform_color([1, 0, 0])])



