"""
motion_regression.py
Created in : 3/21/19
Author     : Zhijie Wang
Email      : paul dot wangzhijie at outlook dot com
"""

import numpy as np
import os
import pandas
import time
import math


# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())


def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.

    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
        [                0.0,                 0.0,                 0.0, 1.0]])


def euler_from_quaternion(quaternion, axes='sxyz'):
    """Return Euler angles from quaternion for specified axis sequence.

    >>> angles = euler_from_quaternion([0.99810947, 0.06146124, 0, 0])
    >>> np.allclose(angles, [0.123, 0, 0])
    True

    """
    return euler_from_matrix(quaternion_matrix(quaternion), axes)


def euler_matrix(ai, aj, ak, axes='sxyz'):
    """Return homogeneous rotation matrix from Euler angles and axis sequence.

    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # noqa: validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak

    si, sj, sk = math.sin(ai), math.sin(aj), math.sin(ak)
    ci, cj, ck = math.cos(ai), math.cos(aj), math.cos(ak)
    cc, cs = ci*ck, ci*sk
    sc, ss = si*ck, si*sk

    M = np.identity(4)
    if repetition:
        M[i, i] = cj
        M[i, j] = sj*si
        M[i, k] = sj*ci
        M[j, i] = sj*sk
        M[j, j] = -cj*ss+cc
        M[j, k] = -cj*cs-sc
        M[k, i] = -sj*ck
        M[k, j] = cj*sc+cs
        M[k, k] = cj*cc-ss
    else:
        M[i, i] = cj*ck
        M[i, j] = sj*sc-cs
        M[i, k] = sj*cc+ss
        M[j, i] = cj*sk
        M[j, j] = sj*ss+cc
        M[j, k] = sj*cs-sc
        M[k, i] = -sj
        M[k, j] = cj*si
        M[k, k] = cj*ci
    return M


def euler_from_matrix(matrix, axes='sxyz'):
    """Return Euler angles from rotation matrix for specified axis sequence.

    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # noqa: validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j]*M[i, j] + M[i, k]*M[i, k])
        if sy > _EPS:
            ax = math.atan2( M[i, j],  M[i, k])
            ay = math.atan2( sy,       M[i, i])
            az = math.atan2( M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2( sy,       M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])
        if cy > _EPS:
            ax = math.atan2( M[k, j],  M[k, k])
            ay = math.atan2(-M[k, i],  cy)
            az = math.atan2( M[j, i],  M[i, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2(-M[k, i],  cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az


def from_rotation_matrix(matrix):
    size = matrix.shape[0]
    result = np.zeros(shape=(size, 4), dtype=np.float)
    for i in range(matrix.shape[0]):
        rot_matrix = [[matrix[i, 0], matrix[i, 1], matrix[i, 2]], [matrix[i, 3], matrix[i, 4], matrix[i, 5]],
                      [matrix[i, 6], matrix[i, 7], matrix[i, 8]]]
        result[i] = [quaternion_from_matrix(rot_matrix)[3], quaternion_from_matrix(rot_matrix)[0],
                     quaternion_from_matrix(rot_matrix)[1], quaternion_from_matrix(rot_matrix)[2]]
    return result


def quaternion_from_matrix(matrix, isprecise=False):
    """Return quaternion from rotation matrix.

    """
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4, ))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 0, 1, 2
            if M[1, 1] > M[0, 0]:
                i, j, k = 1, 2, 0
            if M[2, 2] > M[i, i]:
                i, j, k = 2, 0, 1
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
            q = q[[3, 0, 1, 2]]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array([[m00-m11-m22, 0.0,         0.0,         0.0],
                         [m01+m10,     m11-m00-m22, 0.0,         0.0],
                         [m02+m20,     m12+m21,     m22-m00-m11, 0.0],
                         [m21-m12,     m02-m20,     m10-m01,     m00+m11+m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    return q


def from_euler_angles(alpha_beta_gamma, beta=None, gamma=None):
    """Return quaternion from euler angles.

    """
    if gamma is None:
        alpha_beta_gamma = np.asarray(alpha_beta_gamma, dtype=np.double)
        alpha = alpha_beta_gamma[..., 0]
        beta  = alpha_beta_gamma[..., 1]
        gamma = alpha_beta_gamma[..., 2]
    else:
        alpha = np.asarray(alpha_beta_gamma, dtype=np.double)
        beta  = np.asarray(beta, dtype=np.double)
        gamma = np.asarray(gamma, dtype=np.double)

    # Set up the output array
    R = np.empty(np.broadcast(alpha, beta, gamma).shape + (4,), dtype=np.double)

    # Compute the actual values of the quaternion components
    R[..., 0] =  np.cos(beta/2)*np.cos((alpha+gamma)/2)  # scalar quaternion components
    R[..., 1] = -np.sin(beta/2)*np.sin((alpha-gamma)/2)  # x quaternion components
    R[..., 2] =  np.sin(beta/2)*np.cos((alpha-gamma)/2)  # y quaternion components
    R[..., 3] =  np.cos(beta/2)*np.sin((alpha+gamma)/2)  # z quaternion components

    return R


def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([-x1*x0 - y1*y0 - z1*z0 + w1*w0,
                     x1*w0 + y1*z0 - z1*y0 + w1*x0,
                    -x1*z0 + y1*w0 + z1*x0 + w1*y0,
                     x1*y0 - y1*x0 + z1*w0 + w1*z0], dtype=np.float)


def quaternion_conjugate(quaternion0):
    w, x, y, z = quaternion0
    return np.array([w, -x, -y, -z], dtype=np.float)


def quaternion_normalize(quaternion0):
    w = quaternion0[:, 0]
    x = quaternion0[:, 1]
    y = quaternion0[:, 2]
    z = quaternion0[:, 3]
    norm = np.sqrt(w**2+x**2+y**2+z**2)
    result = w/norm
    result = np.c_[result, (x / norm)]
    result = np.c_[result, (y / norm)]
    result = np.c_[result, (z / norm)]
    return result


def motion_regression_1d(pnts, t):
    """
    Computation of the velocity and acceleration for the target time t
    using a sequence of points with time stamps for one dimension. This
    is an implementation of the algorithm presented by [1].
    [1] Sittel, Florian, Joerg Mueller, and Wolfram Burgard. Computing
        velocities and accelerations from a pose time sequence in
        three-dimensional space. Technical Report 272, University of
        Freiburg, Department of Computer Science, 2013.
    """

    nano_to_sec = 1000000000.0
    micro_to_sec = 1000000.0

    sx = 0.0
    stx = 0.0
    st2x = 0.0
    st = 0.0
    st2 = 0.0
    st3 = 0.0
    st4 = 0.0
    for pnt in pnts:
        ti = (pnt[1] - t) / nano_to_sec
        sx += pnt[0]
        stx += pnt[0] * ti
        st2x += pnt[0] * ti ** 2
        st += ti
        st2 += ti ** 2
        st3 += ti ** 3
        st4 += ti ** 4

    n = len(pnts)
    A = n * (st3 * st3 - st2 * st4) + \
        st * (st * st4 - st2 * st3) + \
        st2 * (st2 * st2 - st * st3)

    if A == 0.0:
        return 0.0, 0.0

    v = (1.0 / A) * (sx * (st * st4 - st2 * st3) +
                     stx * (st2 * st2 - n * st4) +
                     st2x * (n * st3 - st * st2))

    a = (2.0 / A) * (sx * (st2 * st2 - st * st3) +
                     stx * (n * st3 - st * st2) +
                     st2x * (st * st - n * st2))
    return v, a


def motion_regression_6d(pnts, qt, t):
    """
    Compute translational and rotational velocities and accelerations in
    the inertial frame at the target time t.
    [1] Sittel, Florian, Joerg Mueller, and Wolfram Burgard. Computing
        velocities and accelerations from a pose time sequence in
        three-dimensional space. Technical Report 272, University of
        Freiburg, Department of Computer Science, 2013.
    """

    pnts = pnts.T

    lin_vel = np.zeros(3)
    lin_acc = np.zeros(3)

    q_d = np.zeros(4)
    q_dd = np.zeros(4)

    for i in range(1, 4):
        # v, a = motion_regression_1d(
        #     [(pnt[i], pnt[0]) for pnt in pnts], t)
        # for pnt in pnts:
        tmp = np.c_[pnts[i], pnts[0]]
        v, a = motion_regression_1d(tmp, t)
        lin_vel[i-1] = v
        lin_acc[i-1] = a

    for i in range(4, 8):
        # v, a = motion_regression_1d(
        #     [(pnt[i], pnt[0]) for pnt in pnts], t)
        # for pnt in pnts:
        tmp = np.c_[pnts[i], pnts[0]]
        v, a = motion_regression_1d(tmp, t)
        q_d[i-4] = v
        q_dd[i-4] = a

    # Keeping all velocities and accelerations in the inertial frame
    ang_vel = 2 * quaternion_multiply(quaternion_conjugate(qt), q_d)
    ang_acc = 2 * quaternion_multiply(quaternion_conjugate(qt), q_dd)

    omega = [ang_vel[1], ang_vel[2], ang_vel[3]]

    # account for fictitious forces
    lin_acc = lin_acc - np.cross(omega, lin_vel)

    result = [t, lin_vel[0], lin_vel[1], lin_vel[2], lin_acc[0], lin_acc[1], lin_acc[2], omega[0], omega[1], omega[2]]

    return result


def motion_regression_pid(pid_id, pose_num, window_size, pose_seq, timestamp, q_t, return_dict):
    print("Run pid %s ..." % pid_id)
    result = np.zeros(shape=(pose_num, 10))
    for j in range(pid_id*pose_num, (pid_id+1)*pose_num):
        if (j >= window_size) and (j < (pose_seq.shape[0] - window_size)):
            result[j-pid_id*pose_num] = motion_regression_6d(pose_seq[(j-window_size):(j+window_size), :], q_t, timestamp[j])

    return_dict[pid_id] = result


if __name__ == '__main__':
    import argparse
    import multiprocessing

    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--pose_path', default=None)
    parser.add_argument('--output_path', default=None)
    parser.add_argument('--rotation_format', default=0, type=int, help=('0:quat,1:rot,2:eul'))
    parser.add_argument('--skip_front', type=int, default=1, help='Number of discarded records at beginning.')
    parser.add_argument('--samples_num', type=int, default=10, help='Number of samples for estimation')
    parser.add_argument('--pid', type=int, default=8, help="Number of pids")

    args = parser.parse_args()
    root_path = os.path.abspath(args.pose_path)
    root_folder_path = os.path.dirname(args.pose_path)

    if not args.output_path is None:
        output_folder = args.output_path
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
    else:
        output_folder = root_folder_path

    samples_num = int(args.samples_num / 2)

    pose_data = np.loadtxt(root_path, dtype=np.float, delimiter=',')
    pose_data = pose_data[args.skip_front:, ]
    pose_timestamp = pose_data[:, 0]

    qt = np.array([1, 0, 0, 0], dtype=np.float)

    if args.rotation_format == 0:
        pose = pose_data
        print('Rotation data input type: quaternion')

    elif args.rotation_format == 1:
        pose = pose_data[:, [0]]
        pose = np.c_[pose, pose_data[:, [4]] - pose_data[0, 4]]
        pose = np.c_[pose, pose_data[:, [8]] - pose_data[0, 8]]
        pose = np.c_[pose, pose_data[:, [12]] - pose_data[0, 12]]
        rotation_matrix = pose_data[:, [1, 2, 3, 5, 6, 7, 9, 10, 11]]
        quat_data = from_rotation_matrix(rotation_matrix)
        quat_data = quaternion_normalize(quat_data)
        pose = np.c_[pose, quat_data]
        print('Rotation data input type: rotation matrix')

    elif args.rotation_format == 2:
        pose = pose_data[:, [0, 1, 2, 3]]
        quat_data = from_euler_angles(pose_data[:, [4, 5, 6]])
        quat_data = quaternion_normalize(quat_data)
        pose = np.c_[pose, quat_data]
        print('Rotation data input type: euler angle')

    else:
        print('Wrong format!!')

    print("Number of poses: %i" %(pose_timestamp.shape[0]))
    print("Number of samples: %i. Skip first and last %i poses." %(samples_num, samples_num))
    # bar = progressbar.ProgressBar(max_value=(pose_timestamp.shape[0] - samples_num - 1))
    # # bar = progressbar.ProgressBar(max_value=1000)
    #
    # result = np.zeros(shape=(pose_timestamp.shape[0], 10))
    # # result = np.zeros(shape=(1000, 10))
    # for i in range(samples_num, (pose_timestamp.shape[0]-samples_num-1)):
    # # for i in range(samples_num, (1000-samples_num-1)):
    #     result[i-samples_num] = motion_regression_6d(pose[(i-samples_num):(i+samples_num), :], qt, pose_timestamp[i])
    #     time.sleep(0.1)
    #     bar.update(i)

    # result = np.zeros(shape=(pose_timestamp.shape[0], 10))
    pid_num = args.pid
    print("Start %i pid(s) for processing data." % pid_num)
    # p = Pool()
    pose_num_0 = pose_timestamp.shape[0] // pid_num
    # result = np.zeros(shape=((pose_num_0*pid_num), 10))
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    for i in range(pid_num):
        # result[i*pose_num_0:(i+1)*pose_num_0, ] = p.apply_async(
        #         motion_regression_pid(i, pose_num_0, samples_num, pose, pose_timestamp, qt), [i])
        p = multiprocessing.Process(target=motion_regression_pid, args=(i, pose_num_0, samples_num, pose, pose_timestamp, qt, return_dict))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    result = return_dict.values()[0]
    for i in range(1, pid_num):
        result = np.r_[result, return_dict.values()[i]]

    result = result[samples_num:-samples_num, ]
    column_list = 'time,lin_vel_x,lin_vel_y,lin_vel_z,lin_acc_x,lin_acc_y,lin_acc_z,ang_vel_r,ang_vel_p,ang_vel_y'.split(',')

    data_mat = np.concatenate([result], axis=1)

    data_pandas = pandas.DataFrame(data_mat, columns=column_list)
    data_pandas.to_csv(output_folder + '/regression_data.csv')

    print('Data written to ' + output_folder + '/regression_data.csv')

    end_time = time.time()

    print("%f seconds used for processing." %(end_time - start_time))

