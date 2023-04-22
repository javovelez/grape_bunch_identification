import open3d as o3d
import numpy as np

def point_cloud_viewer(pcs):
    clouds_list = []
    for i, pc in enumerate(pcs):
        clouds_list.append({
            "name": f"{i}",
            "geometry": pc
        })
    o3d.visualization.draw(clouds_list, show_ui=True, point_size=7)


def dataset_viewer(ds, pc):
    pcs = ds[0]
    clouds_list = []
    print(pcs.shape)
    for i in range(pcs.shape[2]):
        clouds_list.append({
            "name": f"{i}",
            "geometry": conform_point_cloud(np.squeeze(pcs[:, :, i]))
        })
    clouds_list.append({
        "name": "Nube",
        "geometry": pc
    })
    o3d.visualization.draw(clouds_list, show_ui=True, point_size=7)