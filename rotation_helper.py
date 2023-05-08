import math
import numpy as np
import quaternion
from scipy.spatial.transform import Rotation

D2R = math.pi/180.0
R2D = 180.0/math.pi

PASSED_STR = "\033[1;32mPASSED\033[0m"   # with green color
FAILED_STR = "\033[1;31mFAILED\033[0m"   # with red color

PASSED_COUNT = 0
FAILED_COUNT = 0

ROTATION_SEQUENCES = (\
    # three axis rotation
    'zyx', 'zxy', 'yxz', 'yzx', 'xyz', 'xzy', \
    # two axis rotation
    'zyz', 'zxz', 'yxy', 'yzy', 'xyx', 'xzx'\
)

def get_result(is_passed):
    global PASSED_COUNT
    global FAILED_COUNT

    if is_passed:
        PASSED_COUNT = PASSED_COUNT + 1
        return PASSED_STR
    else:
        FAILED_COUNT = FAILED_COUNT + 1
        return FAILED_STR

'''
output all kinds of expression of a rotation expressed by euler
as well as tests on the following functions provided by attitude
    euler2dcm()
    euler2quat()
    dcm2euler() and quat2euler() by all rotation sequneces. (looks like euler to euler from input)
    dcm2quat() and quat2dcm()
'''
def euler_2_x(euler_d_input, rot_seq):
    print('~~~~~~~~~~euler2x~~~~~~~~~~')

    euler_r_input = euler_d_input * D2R
    print('euler input by %s sequence intrinsicly:'% rot_seq)
    print('%s (%s)\n'% (euler_d_input, euler_r_input))

    rot = Rotation.from_euler(rot_seq.upper(), euler_r_input) # in body frame (intrinsic)

    # output dcm, as well as test on euler2dcm()
    dcm_scipy = rot.as_matrix()
    print('scipy: \n%s'% dcm_scipy)

    # output quat, as well as test on euler2quat()
    quat_scipy = rot.as_quat()
    print('scipy: \n%s'% quat_scipy)

    # output euler for all kinds of rotation sequences intrinsicly, as well as test on dcm2euler() and quat2euler()
    for rot_seq in ROTATION_SEQUENCES:
        euler_r_scipy = rot.as_euler(rot_seq.upper())
        euler_d_scipy = euler_r_scipy * R2D
        print('scipy:         %s (%s)'% (euler_d_scipy, euler_r_scipy))

    # output rotvec and mrp by scipy and test on their consistence
    rotvec_scipy = rot.as_rotvec()
    mrp_scipy = rot.as_mrp()
    cross_rotvec_mrp = np.cross(rotvec_scipy, mrp_scipy)
    norm_rotvec = np.linalg.norm(rotvec_scipy)
    norm_mrp = np.linalg.norm(mrp_scipy)

    result = get_result(np.allclose(cross_rotvec_mrp, [0, 0, 0]) and    # the two are parallel
                        np.allclose(math.tan(norm_rotvec/4), norm_mrp)) # angle relation
    print('***scipy rotvec and mrp output: %s***'% result)
    print('rotvec: %s angle:        %s'% (rotvec_scipy, norm_rotvec*R2D))
    print('mrp:    %s tan(angle/4): %s\n'% (mrp_scipy, norm_mrp))

    # output eulers for all kinds of rotation sequences by scipy extrinsicly
    print('***scipy euler output extrinsicly***')
    for rot_seq in ROTATION_SEQUENCES:
        euler_r_scipy = rot.as_euler(rot_seq)
        euler_d_scipy = euler_r_scipy * R2D
        print('%s: %s (%s)'% (rot_seq, euler_d_scipy, euler_r_scipy))
    print()

'''
to rotate by:
    euler angles once
    euler angles 3 times one by one
they should get the same results
'''
def euler_one_by_one(euler_d_input, rot_seq):
    print('~~~~~~~~~~euler one by one~~~~~~~~~~')

    euler_r_input = euler_d_input * D2R
    print('euler input by %s sequence intrinsicly:'% rot_seq)
    print('%s (%s)'% (euler_d_input, euler_r_input))

    rot_once = Rotation.from_euler(rot_seq.upper(), euler_r_input)
    euler_d_once = rot_once.as_euler(rot_seq.upper(), True)
    euler_r_once = euler_d_once * D2R

    rot1 = Rotation.from_euler(rot_seq.upper(), [euler_r_input[0], 0, 0])
    rot2 = Rotation.from_euler(rot_seq.upper(), [0, euler_r_input[1], 0])
    rot3 = Rotation.from_euler(rot_seq.upper(), [0, 0, euler_r_input[2]])
    rot_one_by_one = rot1 * rot2 * rot3
    euler_d_one_by_one = rot_one_by_one.as_euler(rot_seq.upper(), True)
    euler_r_one_by_one = euler_d_one_by_one * D2R

    result = get_result(np.allclose(euler_d_once, euler_d_one_by_one))
    print('***euler between once and one by one: %s***'% result)
    print('euler once:\n%s (%s)'% (euler_d_once, euler_r_once))
    print('euler one by one:\n%s (%s)\n'% (euler_d_one_by_one, euler_r_one_by_one))

'''
to rotate by 4 ways:
    Rotation
    dcm by scipy.Rotation
    dcm by attitude
    quaternion
they should get the same results
'''
def vectors_rotation(vectors_input, euler_d_input, rot_seq, times = 1, is_extrinsic = False, on_frame = False):
    times_str = ''
    if times > 1:
        times_str = '%s times composed '% times
    way_str = 'intrinsicly'
    if is_extrinsic:
        way_str = 'extrinsicly'
    on_str = 'on vector'
    if on_frame:
        on_str = 'on frame'
    print('~~~~~~~~~~vectors rotation %s%s %s~~~~~~~~~~'% (times_str, way_str, on_str))

    print('vector input:\n%s'% vectors_input)
    euler_r_input = euler_d_input * D2R
    print('euler input by %s sequence: %s (%s)\n'% (rot_seq, euler_d_input, euler_r_input))

    rot_seq_scipy = rot_seq.upper() # intrinsic
    if is_extrinsic:
        rot_seq_scipy = rot_seq     # extrinsic
    rot_once = Rotation.from_euler(rot_seq_scipy, euler_r_input)

    dcm_scipy_once = rot_once.as_matrix()

    quat_vectors_input = quaternion.from_vector_part(vectors_input)
    quat_once = quaternion.from_float_array(rot_once.as_quat()[[3, 0, 1, 2]])

    rot = rot_once
    dcm_scipy = dcm_scipy_once
    quat = quat_once
    for i in range(1, times):
        rot = rot * rot_once
        dcm_scipy = dcm_scipy.dot(dcm_scipy_once)
        quat = quat * quat_once

    vectors_rotated_scipy = rot.apply(vectors_input, on_frame)

    if (not on_frame):  # to rotate vectors
        vectors_rotated_scipy_dcm = (dcm_scipy.dot(vectors_input.T)).T  # left multiplication to rotate vectors
        vectors_rotated_quat = quaternion.as_vector_part(quat * quat_vectors_input * quat.conjugate())
    else:  # to rotate frame
        vectors_rotated_scipy_dcm = vectors_input.dot(dcm_scipy)        # right multiplication to rotate frame
        vectors_rotated_quat = quaternion.as_vector_part(quat.conjugate() * quat_vectors_input * quat)

    is_passed = False
    if np.allclose(vectors_rotated_scipy, vectors_rotated_scipy_dcm) and\
       np.allclose(vectors_rotated_scipy, vectors_rotated_quat):
          is_passed = True
    result = get_result(is_passed)

    print('***vetors rotated: %s***'% result)
    print('Rotation:\n%s'% vectors_rotated_scipy)
    print('scipy dcm:\n%s'% vectors_rotated_scipy_dcm)
    print('quaternion:\n%s'% vectors_rotated_quat)
    print()

    return vectors_rotated_scipy

# the key is the rotvec in space has no change. Just to get rotvec in the new frame
def euler_in_new_frame(euler_d_old_frame, euler_frame_old_2_new, rot_seq):
    print('~~~~~~~~~~euler in new frame~~~~~~~~~~')

    rot_frame_old_2_new = Rotation.from_euler(rot_seq, euler_frame_old_2_new, True)
    euler_frame_old_2_new = rot_frame_old_2_new.as_euler(rot_seq, True) # to get normalized eulers
    print('***frame from old to new:***')
    print('euler: %s\n'% euler_frame_old_2_new)

    rot_old_frame = Rotation.from_euler(rot_seq, euler_d_old_frame, True)
    euler_d_old_frame = rot_old_frame.as_euler(rot_seq, True) # to get normalized eulers
    rotvec_old_frame = rot_old_frame.as_rotvec()
    print('***rotation in old frame***:')
    print('euler:  %s'% euler_d_old_frame)
    print('rotvec: %s\n'% rotvec_old_frame)

    rotvec_new_frame = rot_frame_old_2_new.inv().apply(rotvec_old_frame)
    rot_new_frame = Rotation.from_rotvec(rotvec_new_frame)
    euler_d_new_frame = rot_new_frame.as_euler(rot_seq, True)
    print('***rotation in new frame***:')
    print('euler:  %s'% euler_d_new_frame)
    print('rotvec: %s\n'% rotvec_new_frame)

    return euler_d_new_frame

EULER_D_FRAME_NED_2_ENU = np.array([-90, 180, 0])

def att_ned_2_enu(att_d_ned_2_frd, expected_att_d_enu_2_rfu):
    att_d_enu_2_rfu = euler_in_new_frame(att_d_ned_2_frd, EULER_D_FRAME_NED_2_ENU, 'ZYX')
    result = get_result(np.allclose(att_d_enu_2_rfu, expected_att_d_enu_2_rfu))
    print('***attitude from ned to enu: %s***'% result)
    print('body from ned to frd:\n%s'% att_d_ned_2_frd)
    print('body from enu to rfu:\n%s\n'% att_d_enu_2_rfu)

'''
    rotation from enu (navigation frame) to rfu (body frame), enu / rfu is xyz of coordinates
    heading vector starts from direction along with north and ends to direction along with its front orient
    delta_x, delta_y, delta_z indicate tangent direction, which is end direction of heading
    roll_y indicates self rotation of the heading vector
'''
'''
    way 1: by euler
'''
def att_enu_2_rfu_by_euler(delta_x, delta_y, delta_z, roll_y):
    print('~~~~~~~~~~att from enu to rfu by delta xyz and roll, way 1: euler~~~~~~~~~~')

    yaw_z = math.atan2(-delta_x, delta_y)
    pitch_x = math.atan2(delta_z, math.sqrt(delta_x * delta_x + delta_y * delta_y))

    euler_r_delta_xyz = np.array([yaw_z, pitch_x, roll_y])
    euler_d_delta_xyz = euler_r_delta_xyz * R2D
    print('euler calculated by delta xyz by zxy sequence intrinsicly:')
    print('%s (%s)'% (euler_d_delta_xyz, euler_r_delta_xyz))

    rot = Rotation.from_euler("ZXY", euler_r_delta_xyz)

    euler_r_rotated = rot.as_euler('ZYX')
    euler_d_rotated = euler_r_rotated * R2D
    print('euler rotated by zyx sequence:')
    print('%s (%s)\n'% (euler_d_rotated, euler_r_rotated))

    # test by heading vector
    heading_start = np.array([0, 1, 0])
    heading_end = np.array([delta_x, delta_y, delta_z])
    heading_end = heading_end / np.linalg.norm(heading_end)

    heading_end_rotated = rot.apply(heading_start)
    result = get_result(np.allclose(heading_end, heading_end_rotated))
    print('***heading rotated by euler: %s***'% result)
    print('end heading expected: %s'% heading_end)
    print('end heading rotated: %s\n'% heading_end_rotated)

def get_vertical_rot_vec(v1, v2):
    rot_vec = np.cross(v1, v2)
    rot_angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    rot_vec = rot_vec / np.linalg.norm(rot_vec) * rot_angle
    return rot_vec

'''
    way 2: by vertical rot vec
    [ToDo] how to use roll_y
'''
def att_enu_2_rfu_by_vertical_rot_vec(delta_x, delta_y, delta_z, roll_y):
    print('~~~~~~~~~~att from enu to rfu by delta xyz and roll, way 2: by vertical rot vec~~~~~~~~~~')

    heading_start = np.array([0, 1, 0])
    heading_end = np.array([delta_x, delta_y, delta_z])
    heading_end = heading_end / np.linalg.norm(heading_end)

    vertical_rot_vec = get_vertical_rot_vec(heading_start, heading_end)

    rot = Rotation.from_rotvec(vertical_rot_vec)
    euler_r_roated = rot.as_euler('ZYX')
    euler_d_rotated = euler_r_roated * R2D
    print('euler rotated by zyx sequence:')
    print('%s (%s)\n'% (euler_d_rotated, euler_r_roated))

    heading_end_rotated = rot.apply(heading_start)
    result = get_result(np.allclose(heading_end, heading_end_rotated))
    print('***heading rotated by vertical vec: %s***'% result)
    print('end heading expected: %s'% heading_end)
    print('end heading rotated: %s\n'% heading_end_rotated)

def delta_att(att_d_input_1, att_d_input_2, rot_seq):
    print('~~~~~~~~~~delta att~~~~~~~~~~')

    att_r_input_1 = att_d_input_1 * D2R
    att_r_input_2 = att_d_input_2 * D2R
    print('att input by %s sequence:'% rot_seq)
    print('%s (%s)'% (att_d_input_1, att_r_input_1))
    print('%s (%s)\n'% (att_d_input_2, att_r_input_2))

    rot1 = Rotation.from_euler(rot_seq, att_r_input_1)
    rot2 = Rotation.from_euler(rot_seq, att_r_input_2)

    delta_rot = rot1.inv() * rot2

    delta_att_d = delta_rot.as_euler(rot_seq, True)
    delta_att_r = delta_rot.as_euler(rot_seq)
    print('delta att by %s sequence :'% rot_seq)
    print('%s (%s)'% (delta_att_d, delta_att_r))

    #euler_in_new_frame(delta_rot.as_euler(rot_seq, True), att_d_input_1, rot_seq)

    # test
    rot2_cal = rot1 * delta_rot
    att_d_2_cal = rot2_cal.as_euler(rot_seq, True)
    att_r_2_cal = rot2_cal.as_euler(rot_seq)
    print('att2 calculated by %s sequence :'% rot_seq)
    print('%s (%s)'% (att_d_2_cal, att_r_2_cal))
    result = get_result(np.allclose(att_d_2_cal, att_d_input_2))
    print('***att_d_input_2 vs att_d_cal_2: %s***\n'% result)
