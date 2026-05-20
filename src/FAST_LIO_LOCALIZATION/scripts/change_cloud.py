#!/usr/bin/env python3
import numpy as np
import open3d as o3d

# 1) 读入点云
pcd = o3d.io.read_point_cloud("/home/lin/FAST_LIO/src/FAST_LIO_LOCALIZATION/scripts/scans.pcd")

# 2) 构造旋转：例如绕 X 轴旋转 -90°（把“竖起来”的东西放倒，常用）
angle_deg = -90
angle = np.deg2rad(angle_deg)
R = pcd.get_rotation_matrix_from_axis_angle([angle, 0, 0])  # [rx, ry, rz] 的轴角表示（这里等价于绕X）

# 3) 围绕点云中心旋转（更直观，不会“甩飞”）
center = pcd.get_center()
pcd.rotate(R, center=center)

# 4) 可选：再平移一下（如果你需要把它移动到原点）
# pcd.translate(-center)

# 5) 保存
o3d.io.write_point_cloud("/home/lin/output_rotated.pcd", pcd)

print("done -> output_rotated.pcd")

