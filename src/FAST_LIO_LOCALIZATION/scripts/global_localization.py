#!/usr/bin/python3
# coding=utf8
from __future__ import print_function, division, absolute_import

import copy
import time
import threading
import numpy as np
import open3d as o3d

import rospy
import ros_numpy
import tf
import tf.transformations

from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, Point, Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2

lock = threading.Lock()

global_map = None
initialized = False
T_map_to_odom = np.eye(4, dtype=np.float64)
cur_odom = None
cur_scan = None


# ---------------------------
# Utilities
# ---------------------------
def transform_xyz(T, xyz):
    """xyz: (N,3) -> (N,3)"""
    if xyz is None or len(xyz) == 0:
        return xyz
    pts_h = np.hstack([xyz, np.ones((xyz.shape[0], 1), dtype=xyz.dtype)])
    out = (T @ pts_h.T).T[:, :3]
    return out


def pose_to_mat(pose_msg):
    return np.matmul(
        tf.listener.xyz_to_mat44(pose_msg.pose.pose.position),
        tf.listener.xyzw_to_mat44(pose_msg.pose.pose.orientation),
    )


def msg_to_array(pc_msg):
    pc_array = ros_numpy.numpify(pc_msg)
    pc = np.zeros([len(pc_array), 3], dtype=np.float64)
    pc[:, 0] = pc_array['x']
    pc[:, 1] = pc_array['y']
    pc[:, 2] = pc_array['z']
    return pc


def inverse_se3(trans):
    trans_inverse = np.eye(4, dtype=np.float64)
    trans_inverse[:3, :3] = trans[:3, :3].T
    trans_inverse[:3, 3] = -np.matmul(trans[:3, :3].T, trans[:3, 3])
    return trans_inverse


def rotation_angle_deg(R):
    """Angle of rotation matrix R (3x3), in degrees."""
    # clamp trace numeric
    tr = np.trace(R)
    c = (tr - 1.0) / 2.0
    c = np.clip(c, -1.0, 1.0)
    ang = np.arccos(c)
    return float(np.degrees(ang))


def se3_delta(T1, T2):
    """Return (translation_diff_m, rotation_diff_deg) between two SE3."""
    dR = T1[:3, :3].T @ T2[:3, :3]
    dt = T2[:3, 3] - T1[:3, 3]
    return float(np.linalg.norm(dt)), rotation_angle_deg(dR)


def mat_to_xyyaw(T):
    x, y = float(T[0, 3]), float(T[1, 3])
    yaw = float(np.arctan2(T[1, 0], T[0, 0]))
    return x, y, yaw


def xyyaw_to_mat(x, y, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4, dtype=np.float64)
    T[0, 0] = c
    T[0, 1] = -s
    T[1, 0] = s
    T[1, 1] = c
    T[0, 3] = x
    T[1, 3] = y
    return T


def robust_pick_T(candidates, use_se2_robust=False):
    """
    candidates: list of dict {T, fitness, rmse, score}
    Default: choose best score (fitness - lambda*rmse).
    Optional: use SE2 robust aggregation (median x/y + circular mean yaw) then return that.
    """
    if not candidates:
        return None

    if not use_se2_robust:
        return max(candidates, key=lambda d: d["score"])["T"]

    xs, ys, yaws = [], [], []
    for d in candidates:
        x, y, yaw = mat_to_xyyaw(d["T"])
        xs.append(x); ys.append(y); yaws.append(yaw)

    x_med = float(np.median(xs))
    y_med = float(np.median(ys))
    # circular mean for yaw
    sinm = float(np.mean(np.sin(yaws)))
    cosm = float(np.mean(np.cos(yaws)))
    yaw_mean = float(np.arctan2(sinm, cosm))
    return xyyaw_to_mat(x_med, y_med, yaw_mean)


# ---------------------------
# Open3D / ICP
# ---------------------------
def voxel_down_sample(pcd, voxel_size):
    try:
        return pcd.voxel_down_sample(voxel_size)
    except Exception:
        # for open3d 0.7 or lower
        return o3d.geometry.voxel_down_sample(pcd, voxel_size)


def registration_at_scale(pc_scan, pc_map, initial, scale):
    result = o3d.pipelines.registration.registration_icp(
        voxel_down_sample(pc_scan, SCAN_VOXEL_SIZE * scale),
        voxel_down_sample(pc_map, MAP_VOXEL_SIZE * scale),
        1.0 * scale,
        initial,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20)
    )
    return result.transformation, float(result.fitness), float(result.inlier_rmse)


def publish_point_cloud(publisher, header, pc):
    data = np.zeros(len(pc), dtype=[
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('intensity', np.float32),
    ])
    data['x'] = pc[:, 0]
    data['y'] = pc[:, 1]
    data['z'] = pc[:, 2]
    if pc.shape[1] == 4:
        data['intensity'] = pc[:, 3]
    msg = ros_numpy.msgify(PointCloud2, data)
    msg.header = header
    publisher.publish(msg)


def crop_global_map_in_FOV(global_map_pcd, pose_estimation, odom_msg):
    # 当前scan原点的位姿
    T_odom_to_base_link = pose_to_mat(odom_msg)
    T_map_to_base_link = np.matmul(pose_estimation, T_odom_to_base_link)
    T_base_link_to_map = inverse_se3(T_map_to_base_link)

    global_map_in_map = np.asarray(global_map_pcd.points)
    global_map_in_map_h = np.column_stack([global_map_in_map, np.ones(len(global_map_in_map), dtype=np.float64)])
    global_map_in_base_link = (T_base_link_to_map @ global_map_in_map_h.T).T

    if FOV > 3.14:
        indices = np.where(
            (global_map_in_base_link[:, 0] < FOV_FAR) &
            (np.abs(np.arctan2(global_map_in_base_link[:, 1], global_map_in_base_link[:, 0])) < FOV / 2.0)
        )[0]
    else:
        indices = np.where(
            (global_map_in_base_link[:, 0] > 0) &
            (global_map_in_base_link[:, 0] < FOV_FAR) &
            (np.abs(np.arctan2(global_map_in_base_link[:, 1], global_map_in_base_link[:, 0])) < FOV / 2.0)
        )[0]

    submap = o3d.geometry.PointCloud()
    submap.points = o3d.utility.Vector3dVector(global_map_in_map[indices, :3])

    # 发布fov内点云
    header = odom_msg.header
    header.frame_id = 'map'
    publish_point_cloud(pub_submap, header, np.asarray(submap.points)[::10])

    return submap


# ---------------------------
# Core localization (thread-safe snapshot)
# ---------------------------
def global_localization(pose_estimation, allow_update=True):
    """
    pose_estimation: initial guess T_map_to_odom (map->odom) or an initialpose-derived matrix
    allow_update: whether to write T_map_to_odom on success
    return: (ok, T, fitness, rmse)
    """
    global global_map, cur_scan, cur_odom, T_map_to_odom

    rospy.loginfo('Global localization by scan-to-map matching......')

    # Snapshot shared states (min lock time)
    with lock:
        odom_snapshot = cur_odom
        scan_snapshot = cur_scan
        map_snapshot = global_map

    if map_snapshot is None or odom_snapshot is None or scan_snapshot is None:
        rospy.logwarn("global_localization: missing data (map/odom/scan).")
        return False, None, 0.0, 1e9

    # Copy scan to avoid concurrent modification hazards
    scan_tobe_mapped = copy.deepcopy(scan_snapshot)

    tic = time.time()
    submap = crop_global_map_in_FOV(map_snapshot, pose_estimation, odom_snapshot)

    # coarse
    T1, f1, r1 = registration_at_scale(scan_tobe_mapped, submap, initial=pose_estimation, scale=5)
    # fine
    T2, f2, r2 = registration_at_scale(scan_tobe_mapped, submap, initial=T1, scale=1)

    toc = time.time()
    rospy.loginfo('Time: {:.3f}s, fitness={:.4f}, rmse={:.4f}'.format(toc - tic, f2, r2))

    if f2 >= LOCALIZATION_TH and r2 <= RMSE_TH:
        if allow_update:
            with lock:
                T_map_to_odom = T2.copy()

        # publish map_to_odom
        map_to_odom = Odometry()
        xyz = tf.transformations.translation_from_matrix(T2)
        quat = tf.transformations.quaternion_from_matrix(T2)
        map_to_odom.pose.pose = Pose(Point(*xyz), Quaternion(*quat))
        map_to_odom.header.stamp = odom_snapshot.header.stamp
        map_to_odom.header.frame_id = 'map'
        pub_map_to_odom.publish(map_to_odom)
        return True, T2, f2, r2

    rospy.logwarn('Not match! fitness={:.4f}, rmse={:.4f}'.format(f2, r2))
    return False, T2, f2, r2


# ---------------------------
# ROS callbacks (thread-safe)
# ---------------------------
def initialize_global_map(pc_msg):
    global global_map
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(msg_to_array(pc_msg)[:, :3])
    pcd = voxel_down_sample(pcd, MAP_VOXEL_SIZE)
    with lock:
        global_map = pcd
    rospy.loginfo('Global map received.')


def cb_save_cur_odom(odom_msg):
    global cur_odom
    with lock:
        cur_odom = odom_msg


def cb_save_cur_scan(pc_msg):
    global cur_scan, T_map_to_odom

    stamp = pc_msg.header.stamp

    # fastlio field reorder (copy to avoid polluting original)
    pc_msg2 = copy.deepcopy(pc_msg)
    if len(pc_msg2.fields) >= 8:
        pc_msg2.fields = [pc_msg2.fields[0], pc_msg2.fields[1], pc_msg2.fields[2],
                          pc_msg2.fields[4], pc_msg2.fields[5], pc_msg2.fields[6],
                          pc_msg2.fields[3], pc_msg2.fields[7]]

    pc_xyz = msg_to_array(pc_msg2)[:, :3]

    scan_pcd = o3d.geometry.PointCloud()
    scan_pcd.points = o3d.utility.Vector3dVector(pc_xyz)

    # Snapshot T and store scan under lock
    with lock:
        cur_scan = scan_pcd
        T = T_map_to_odom.copy()

    pc_map = transform_xyz(T, pc_xyz)

    header = pc_msg.header
    header.stamp = stamp
    header.frame_id = 'map'
    publish_point_cloud(pub_pc_in_map, header, pc_map)


# ---------------------------
# Init logic: multi-try + stability gate + lock in
# ---------------------------
def init_with_stability(initial_pose):
    """
    Try multiple ICP matches, require consecutive stable solutions.
    Return (ok, T0).
    """
    candidates = []
    stable_cnt = 0
    T_last = None

    start_t = time.time()
    attempt = 0

    rospy.logwarn("Init: start multi-try localization...")

    while not rospy.is_shutdown():
        if time.time() - start_t > INIT_TIMEOUT_SEC:
            rospy.logwarn("Init: timeout. Please provide /initialpose again.")
            return False, None

        # Use current best guess: first use initial_pose, later use last accepted T
        guess = T_last if T_last is not None else initial_pose

        ok, T, fitness, rmse = global_localization(guess, allow_update=False)
        attempt += 1

        if not ok:
            stable_cnt = 0
            T_last = None
            rospy.sleep(INIT_RETRY_SLEEP)
            continue

        # Quality gates already checked in global_localization (fitness/rmse),
        # but we still compute score for selection.
        score = fitness - SCORE_LAMBDA * rmse
        candidates.append({"T": T, "fitness": fitness, "rmse": rmse, "score": score})

        if T_last is not None:
            dt, da = se3_delta(T_last, T)
            if dt < STABLE_TRANS_TH and da < STABLE_ANGLE_TH_DEG:
                stable_cnt += 1
            else:
                stable_cnt = 1
        else:
            stable_cnt = 1

        T_last = T

        rospy.logwarn("Init attempt {}: stable_cnt={}/{} (dt<{:.2f}m, da<{:.1f}deg)".format(
            attempt, stable_cnt, STABLE_REQUIRED, STABLE_TRANS_TH, STABLE_ANGLE_TH_DEG
        ))

        # Keep only recent window to avoid old outliers
        if len(candidates) > CANDIDATE_WINDOW:
            candidates = candidates[-CANDIDATE_WINDOW:]

        if stable_cnt >= STABLE_REQUIRED and len(candidates) >= MIN_CANDIDATES_FOR_PICK:
            T0 = robust_pick_T(candidates, use_se2_robust=USE_SE2_ROBUST_PICK)
            return True, T0

        rospy.sleep(INIT_RETRY_SLEEP)


# ---------------------------
# Optional relocalization thread
# ---------------------------
def thread_localization():
    global T_map_to_odom
    while not rospy.is_shutdown():
        rospy.sleep(1.0 / FREQ_LOCALIZATION)
        with lock:
            T_guess = T_map_to_odom.copy()
        # If enabled, allow_update=True (will update T_map_to_odom when confident)
        global_localization(T_guess, allow_update=True)


# ---------------------------
# Main
# ---------------------------
if __name__ == '__main__':
    # ------------ Parameters (tune here) ------------
    MAP_VOXEL_SIZE = 0.4
    SCAN_VOXEL_SIZE = 0.1

    # If you only want a stable initial alignment, set False.
    ENABLE_RELOCALIZATION = False

    # Relocalization frequency (Hz) if enabled
    FREQ_LOCALIZATION = 1

    # ICP acceptance thresholds
    LOCALIZATION_TH = 0.99
    RMSE_TH = 0.35  # meters, tune by environment density

    # Init multi-try / stability
    INIT_TIMEOUT_SEC = 30.0
    INIT_RETRY_SLEEP = 0.3
    STABLE_REQUIRED = 5              # recommended default (more stable than 3)
    STABLE_TRANS_TH = 0.15           # meters
    STABLE_ANGLE_TH_DEG = 3.0        # degrees
    CANDIDATE_WINDOW = 20
    MIN_CANDIDATES_FOR_PICK = 6

    # Candidate selection
    SCORE_LAMBDA = 1.0               # score = fitness - lambda*rmse
    USE_SE2_ROBUST_PICK = True       # ground robot: robust SE2 aggregation is usually better

    # FOV
    FOV = 6.28
    FOV_FAR = 20

    # ------------ ROS init ------------
    rospy.init_node('fast_lio_localization')
    rospy.loginfo('Localization Node Inited...')

    pub_pc_in_map = rospy.Publisher('/cur_scan_in_map', PointCloud2, queue_size=1)
    pub_submap = rospy.Publisher('/submap', PointCloud2, queue_size=1)
    pub_map_to_odom = rospy.Publisher('/map_to_odom', Odometry, queue_size=1)

    rospy.Subscriber('/cloud_registered', PointCloud2, cb_save_cur_scan, queue_size=1)
    rospy.Subscriber('/Odometry', Odometry, cb_save_cur_odom, queue_size=1)

    # Load global map once
    rospy.logwarn('Waiting for global map......')
    initialize_global_map(rospy.wait_for_message('/map', PointCloud2))

    # Init loop: wait /initialpose, then multi-try until stable
    while not rospy.is_shutdown() and not initialized:
        rospy.logwarn('Waiting for initial pose....')
        pose_msg = rospy.wait_for_message('/initialpose', PoseWithCovarianceStamped)
        initial_pose = pose_to_mat(pose_msg)

        with lock:
            has_scan = (cur_scan is not None)
            has_odom = (cur_odom is not None)

        if not has_scan:
            rospy.logwarn('First scan not received!!!!!')
            continue
        if not has_odom:
            rospy.logwarn('Odometry not received yet!!!!!')
            continue

        ok, T0 = init_with_stability(initial_pose)
        if ok:
            with lock:
                T_map_to_odom = T0.copy()
            initialized = True

            # publish once with current stamp if available
            with lock:
                odom_snapshot = cur_odom
            if odom_snapshot is not None:
                msg = Odometry()
                xyz = tf.transformations.translation_from_matrix(T0)
                quat = tf.transformations.quaternion_from_matrix(T0)
                msg.pose.pose = Pose(Point(*xyz), Quaternion(*quat))
                msg.header.stamp = odom_snapshot.header.stamp
                msg.header.frame_id = 'map'
                pub_map_to_odom.publish(msg)

    rospy.loginfo('')
    rospy.loginfo('Initialize successfully!!!!!!')
    rospy.loginfo('ENABLE_RELOCALIZATION={}'.format(ENABLE_RELOCALIZATION))
    rospy.loginfo('')

    # Optional: start relocalization updates
    if ENABLE_RELOCALIZATION:
        t = threading.Thread(target=thread_localization, daemon=True)
        t.start()

    rospy.spin()