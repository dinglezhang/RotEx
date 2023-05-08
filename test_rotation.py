import numpy as np

import rotation_helper as rh

def test_euler_2_x():
    print('============================test euler2x============================')

    euler_d_input = np.array([20, 1, 5])
    for seq in rh.ROTATION_SEQUENCES:
        rh.euler_2_x(euler_d_input, seq)
        break # remove it to test all rotation sequences

def test_euler_one_by_one():
    print('============================test euler one by one============================')

    euler_d_input = np.array([20, 10, 15])
    for seq in rh.ROTATION_SEQUENCES:
        rh.euler_one_by_one(euler_d_input, seq)
        break # remove it to test all rotation sequences

def test_vectors_rotation_once(is_extrinsic):
    way_str = 'intrinsicly'
    if is_extrinsic:
        way_str = 'extrinsicly'
    print('============================test vectors rotation %s once============================'% way_str)

    vectors_input = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1]
    ])
    euler_d_input = np.array([45, 90, 45])
    rh.vectors_rotation(vectors_input, euler_d_input, 'zyx', 1, is_extrinsic, False)
    rh.vectors_rotation(vectors_input, euler_d_input, 'zyx', 1, is_extrinsic, True)

def test_vectors_rotation_multple_times(is_extrinsic):
    way_str = 'intrinsicly'
    if is_extrinsic:
        way_str = 'extrinsicly'
    print('============================test vectors rotation %s 5 times============================'% way_str)

    vectors_input = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1]
    ])
    euler_d_input = np.array([3, 3, 3])
    times = 5

    vectors_rotated_one_by_one = vectors_input
    for i in range(1, times + 1):
        print('@' + str(i))
        vectors_rotated_one_by_one = rh.vectors_rotation(vectors_rotated_one_by_one, euler_d_input, 'zyx', 1, is_extrinsic, False)

    vectors_rotated_composed = rh.vectors_rotation(vectors_input, euler_d_input, 'zyx', times, is_extrinsic, False)

    result = rh.get_result(np.allclose(vectors_rotated_one_by_one, vectors_rotated_composed))
    print('***vectors rotated results are SAME between one by one and composed: %s***'% result)
    print('one by one:\n%s'% vectors_rotated_one_by_one)
    print('composed:\n%s\n'% vectors_rotated_composed)

    euler_d_input = euler_d_input * times
    vectors_rotated_multiple_angles = rh.vectors_rotation(vectors_input, euler_d_input, 'zyx', 1, is_extrinsic, False)

    # [ToDo] it maybe SAME if rotation on only one axis
    result = rh.get_result(not np.allclose(vectors_rotated_one_by_one, vectors_rotated_multiple_angles))
    print('***vectors rotated results are DIFFERENT between one by one and by multiple angles: %s***'% result)
    print('one by one:\n%s'% vectors_rotated_one_by_one)
    print('multiple angles:\n%s\n'% vectors_rotated_multiple_angles)

def test_att_ned_2_enu():
    rh.att_ned_2_enu(np.array([45, 0, 0]), np.array([-45, 0, 0]))
    rh.att_ned_2_enu(np.array([0, 45, 0]), np.array([0, 0, 45]))
    rh.att_ned_2_enu(np.array([0, 0, 45]), np.array([0, 45, 0]))
    rh.att_ned_2_enu(np.array([90, 45, 90]), np.array([0, 45, 90]))

def test_att_enu_2_rfu_by_delta_xyz():
    print('============================test att from enu to rfu by delta xyz and roll============================')

    delta_x = -1
    delta_y = 1
    delta_z = 0.01#math.sqrt(delta_x * delta_x + delta_y * delta_y)
    roll_y = 0.1
    print('delta_x: %s delta_y: %s delta_z: %s roll_y: %s\n'% (delta_x, delta_y, delta_z, roll_y))
    rh.att_enu_2_rfu_by_euler(delta_x, delta_y, delta_z, roll_y)
    rh.att_enu_2_rfu_by_vertical_rot_vec(delta_x, delta_y, delta_z, roll_y)

def test_delta_att():
    print('============================test delta att============================')

    att_d_input_1 = np.array([10, 1, 4])
    att_d_input_2 = np.array([11, 10, 5])

    rh.delta_att(att_d_input_1, att_d_input_2, 'ZYX')
    rh.delta_att(att_d_input_1, att_d_input_2, 'zyx')

if __name__ == '__main__':
    np.set_printoptions(precision = 8, suppress = True)

    test_euler_2_x()
    test_euler_one_by_one()

    test_vectors_rotation_once(False)
    test_vectors_rotation_multple_times(False)

    test_vectors_rotation_once(True)
    test_vectors_rotation_multple_times(True)

    test_att_ned_2_enu()

    test_att_enu_2_rfu_by_delta_xyz()
    test_delta_att()

    print('\n%s: %s\n%s: %s'% (rh.PASSED_STR, rh.PASSED_COUNT, rh.FAILED_STR, rh.FAILED_COUNT))
