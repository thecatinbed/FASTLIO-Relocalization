#!/usr/bin/python3
# coding=utf8
from __future__ import print_function, division, absolute_import

import copy
import threading
lock = threading.Lock()
import time

import numpy as np
import rospy
import tf
import tf.transformations
from geometry_msgs.msg import Pose, Point, Quaternion
from nav_msgs.msg import Odometry

cur_odom_to_baselink = None
cur_map_to_odom = None


def pose_to_mat(pose_msg):
    return np.matmul(
        tf.listener.xyz_to_mat44(pose_msg.pose.pose.position),
        tf.listener.xyzw_to_mat44(pose_msg.pose.pose.orientation),
    )


def transform_fusion():
    global cur_odom_to_baselink, cur_map_to_odom
    br = tf.TransformBroadcaster()

    rate = rospy.Rate(FREQ_PUB_LOCALIZATION)
    last_stamp = rospy.Time(0)

    while not rospy.is_shutdown():
        rate.sleep()

        with lock:
            odom_msg = cur_odom_to_baselink
            map_odom_msg = cur_map_to_odom

        if odom_msg is None:
            continue
        stamp = odom_msg.header.stamp

        # map->odom 的值可以是上一次的（低频）
        if map_odom_msg is not None:
            T_map_to_odom = pose_to_mat(map_odom_msg)
        else:
            T_map_to_odom = np.eye(4)

        if stamp <= last_stamp:
            continue
        last_stamp = stamp

        br.sendTransform(
            tf.transformations.translation_from_matrix(T_map_to_odom),
            tf.transformations.quaternion_from_matrix(T_map_to_odom),
            stamp,
            'camera_init', 'map'
        )

        if odom_msg is None:
            continue

        localization = Odometry()
        T_odom_to_base_link = pose_to_mat(odom_msg)
        T_map_to_base_link = np.matmul(T_map_to_odom, T_odom_to_base_link)

        xyz = tf.transformations.translation_from_matrix(T_map_to_base_link)
        quat = tf.transformations.quaternion_from_matrix(T_map_to_base_link)
        localization.pose.pose = Pose(Point(*xyz), Quaternion(*quat))
        localization.twist = odom_msg.twist
        localization.header.stamp = odom_msg.header.stamp
        localization.header.frame_id = 'map'
        localization.child_frame_id = 'body'
        pub_localization.publish(localization)

def cb_save_cur_odom(odom_msg):
    global cur_odom_to_baselink
    with lock:
        cur_odom_to_baselink = odom_msg

def cb_save_map_to_odom(odom_msg):
    global cur_map_to_odom
    with lock:
        cur_map_to_odom = odom_msg


if __name__ == '__main__':
    # tf and localization publishing frequency (HZ)
    FREQ_PUB_LOCALIZATION = 50

    rospy.init_node('transform_fusion')
    rospy.loginfo('Transform Fusion Node Inited...')

    rospy.Subscriber('/Odometry', Odometry, cb_save_cur_odom, queue_size=1)
    rospy.Subscriber('/map_to_odom', Odometry, cb_save_map_to_odom, queue_size=1)

    pub_localization = rospy.Publisher('/localization', Odometry, queue_size=1)

    # 发布定位消息
    t = threading.Thread(target=transform_fusion)
    t.daemon = True
    t.start()
    rospy.spin()
