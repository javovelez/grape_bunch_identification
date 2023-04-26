import open3d as o3d
import numpy as np
from viewer import point_cloud_viewer


n = 100
theta = np.linspace(0, 2*np.pi, n)

x = np.cos(theta)
y = np.sin(theta)
z = np.zeros_like(x)
circunferencia = np.column_stack([x, y, z])
circ_cloud = o3d.geometry.PointCloud()
p_cloud = o3d.geometry.PointCloud()
circ_cloud.points = o3d.utility.Vector3dVector(circunferencia)
p_cloud.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0]]))
# point_cloud_viewer([circ_cloud, p_cloud])

icp = o3d.pipelines.registration.registration_icp(circ_cloud, p_cloud, 2)
print(f'icp: {icp}')
print(f'cs: {np.asarray(icp.correspondence_set)}')