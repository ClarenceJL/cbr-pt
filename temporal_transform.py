from scipy.interpolate import interp1d
import random
import numpy as np
import pdb


def temporal_padding(frame_list, size, method='loop'):
    if len(frame_list) >= size:
        return frame_list

    k = size - len(frame_list)
    if method == 'loop':
        for index in frame_list:
            if len(frame_list) >= size:
                break
            frame_list.append(index)
    elif method == 'rep_both':
        k_start = (k+1)//2
        k_end = k - k_start
        frame_list = [frame_list[0]] * k_start + frame_list + [frame_list[-1]] * k_end
    elif method == 'rep_start':
        frame_list = [frame_list[0]] * k + frame_list
    elif method == 'rep_end':
        frame_list = frame_list + [frame_list[-1]] * k
    elif method == 'zero':
        k_start = (k+1)//2
        k_end = k - k_start
        frame_list = [len(frame_list)] * k_start + frame_list + [len(frame_list)] * k_end
    else:
        # e.g mirror
        raise NotImplementedError
    return frame_list


def temporal_avgpool_sample(feat, start_idx, end_idx, sample_len):
    nclips = feat.shape[0]
    if nclips == 1:
        return np.tile(feat, [sample_len, 1])
    start_idx = min(max(start_idx, 0), nclips-0.01)
    end_idx = min(max(end_idx, 0), nclips)
    bounds = np.linspace(start_idx, end_idx, sample_len+1)
    result = np.zeros([sample_len, feat.shape[1]])
    for i in range(sample_len):
        lb = bounds[i]
        lbin = int(lb)
        rb = bounds[i+1]
        rbin = int(rb) if rb > int(rb) else int(rb-1)
        if lb == rb or lbin == rbin:
            result[i] = feat[lbin]
        else:
            assert lbin < rbin
            weight = np.ones([rbin-lbin+1])
            weight[0] -= lb-lbin
            weight[-1] -= (rbin+1) - rb
            result[i] = np.average(feat[lbin:rbin+1], axis=0, weights=weight)
    return result


def temporal_even_sample(feat, start_idx, end_idx, sample_len):
    nclips = feat.shape[0]
    start_idx = min(max(round(start_idx), 0), nclips-1)
    end_idx = min(max(round(end_idx), 0), nclips-1)
    inds = list(range(start_idx, end_idx+1))
    if end_idx - start_idx + 1 < sample_len:
        inds = temporal_padding(inds, sample_len)
        #feat = np.concatenate((feat, np.zeros([1, feat.shape[1]]), 0))

    interval = (len(inds) - 1) // (sample_len - 1)
    start_range = len(inds) - interval * (sample_len - 1) - 1
    sid = random.randint(0, start_range)
    inds = inds[sid:sid+interval*(sample_len-1)+1:interval]
    assert len(inds) == sample_len

    return feat[inds]


def temporal_random_sample(feat, start_idx, end_idx, sample_len):
    nclips = feat.shape[0]
    start_idx = min(max(round(start_idx), 0), nclips-1)
    end_idx = min(max(round(end_idx), 0), nclips-1)
    inds = list(range(start_idx, end_idx+1))
    if end_idx - start_idx + 1 < sample_len:
        inds = temporal_padding(inds, sample_len)
    sids = random.sample(range(len(inds)), sample_len)
    sids = sids.sort()
    inds = [inds[i] for i in sids]
    return feat[inds]


def temporal_interpolation(feat, start_idx, end_idx, sample_len):
    nclips = feat.shape[0]
    start_idx = min(max(start_idx, 0.5), nclips - 0.5)
    end_idx = min(max(end_idx, 0.5), nclips - 0.5)
    if nclips > 1:
        f_action = interp1d(np.arange(nclips) + 0.5, feat, axis=0, fill_value='extrapolate')
        new_t = np.linspace(start_idx, end_idx, sample_len)
        return f_action(new_t)
    else:
        return np.tile(feat, [sample_len, 1])
