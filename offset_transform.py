import numpy as np

# non-parameterized in seconds


def nonparam_offset_second(start, end, start_gt, end_gt):
    return [start_gt - start, end_gt - end]


def inv_nonparam_offset_second(start, end, off_s, off_e):
    return start + off_s, end + off_e


# non-parameterized in unit


def nonparam_offset_unit(start, end, start_gt, end_gt, unit_length):
    return [(start_gt - start) / unit_length, (end_gt - end) / unit_length]


def inv_nonparam_offset_unit(start, end, off_s, off_e, unit_length):
    return start + off_s * unit_length, end + off_e * unit_length


# parameterized offset


def parameterize_offset(start, end, start_gt, end_gt):
    if start >= end:
        print(start, end)
        raise ValueError

    center = (start + end)/2.0
    length = end - start
    center_gt = (start_gt + end_gt) / 2.0
    length_gt = end_gt - start_gt
    return [(center_gt-center)/length, np.log(length_gt) - np.log(length)]

def inv_parameterize_offset(start, end, o_ctr, o_len):
    center = (start + end)/2.0
    length = end - start
    new_center = center + o_ctr * length
    new_length = np.exp(o_len) * length
    return new_center - new_length / 2.0, new_center + new_length / 2.0
