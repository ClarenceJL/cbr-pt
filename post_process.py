import numpy as np
import pandas as pd
import json
import os
import multiprocessing as mp


post_process_top_K = 100
soft_nms_alpha = 0.75
soft_nms_low_thres = 0.65
soft_nms_high_thres = 0.9


def iou_with_anchors(anchors_min,anchors_max,len_anchors,box_min,box_max):
    """Compute jaccard score between a box and the anchors.
    """
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    union_len = len_anchors - inter_len + box_max - box_min
    jaccard = np.divide(inter_len, union_len)
    return jaccard


def Soft_NMS(df, duration):
    df = df.sort_values(by="score", ascending=False)

    tstart = list(df.xmin.values[:])
    tend = list(df.xmax.values[:])
    tscore = list(df.score.values[:])
    rstart = []
    rend = []
    rscore = []

    if 'label' in df.columns:
        tlabel = list(df.label.values[:])
        rlabel = []

    while len(tscore) > 0 and len(rscore) <= post_process_top_K:
        max_index = np.argmax(tscore)
        tmp_width = tend[max_index] - tstart[max_index]
        iou_list = iou_with_anchors(tstart[max_index], tend[max_index], tmp_width, np.array(tstart), np.array(tend))
        iou_exp_list = np.exp(-np.square(iou_list) / soft_nms_alpha)
        for idx in range(0, len(tscore)):
            if idx != max_index:
                tmp_iou = iou_list[idx]
                if tmp_iou > soft_nms_low_thres + (soft_nms_high_thres - soft_nms_low_thres) * tmp_width / duration:
                    tscore[idx] = tscore[idx] * iou_exp_list[idx]

        rstart.append(tstart[max_index])
        rend.append(tend[max_index])
        rscore.append(tscore[max_index])
        tstart.pop(max_index)
        tend.pop(max_index)
        tscore.pop(max_index)

        if 'label' in df.columns:
            rlabel.append(tlabel[max_index])
            tlabel.pop(max_index)

    newDf = pd.DataFrame()
    newDf['score'] = rscore
    newDf['xmin'] = rstart
    newDf['xmax'] = rend
    if 'label' in df.columns:
        newDf['label'] = rlabel

    return newDf


def video_post_process(opt, video_list, video_dict):
    for video_name in video_list:
        df = pd.read_csv(opt['checkpoint_path']+'/output/'+video_name+'.csv')
        df = df[['score', 'xmin', 'xmax']]

        if len(df) > 1:
            df = Soft_NMS(df, video_dict[video_name]['duration'])

        df = df.sort_values(by="score", ascending=False)
        df = df.head(post_process_top_K)
        df['segment'] = df[['xmin', 'xmax']].values.tolist()
        df = df[['score', 'segment']]

        result_dict[video_name] = df.to_dict(orient='records')


def video_post_process_detection(opt, video_list, video_dict):
    class_names = pd.read_csv('data/class_index.csv')
    class_names = class_names['class_name'].values

    for video_name in video_list:
        df = pd.read_csv(opt['checkpoint_path']+'/output/'+video_name+'.csv')
        assert 'label' in df.columns

        # per-class-nms
        df_new = pd.DataFrame(columns=['score', 'xmin', 'xmax', 'label'])
        for c in range(opt['num_classes']):
            df_c = df[df.label == class_names[c]]
            if len(df_c) > 1:
                df_c = Soft_NMS(df_c, video_dict[video_name]['duration'])
                df_new = pd.concat([df_new, df_c], sort=True)

        df = df.sort_values(by="score", ascending=False)
        df = df.head(post_process_top_K)
        df['segment'] = df[['xmin', 'xmax']].values.tolist()
        df = df[['score', 'segment', 'label']]

        result_dict[video_name] = df.to_dict(orient='records')


def get_dataset_dict(anno_path, subset):
    with open(anno_path, 'r') as f:
        anno = json.load(f)
        anno = anno['database']

    video_dict = {}
    for vid, info in anno.items():
        if info['subset'] != subset:
            continue
        # if not os.path.exists(prop_path+'/'+vid+'.csv'):
        #     continue
        video_dict[vid] = info
    return video_dict


def post_processing_wrapper(opt, subset='validation', num_threads=8):
    # print('Playground version: skip PEM and evaluate the PGM proposals + average action score')

    video_dict = get_dataset_dict(opt['video_anno'], subset)
    video_list = list(video_dict.keys())  # [:100]
    global result_dict
    result_dict = mp.Manager().dict()

    num_videos = len(video_list)
    num_videos_per_thread = num_videos // num_threads
    processes = []
    for tid in range(num_threads - 1):
        tmp_video_list = video_list[tid * num_videos_per_thread:(tid + 1) * num_videos_per_thread]
        p = mp.Process(target=video_post_process_detection, args=(opt, tmp_video_list,video_dict))
        p.start()
        processes.append(p)
    tmp_video_list = video_list[(num_threads - 1) * num_videos_per_thread:]
    p = mp.Process(target=video_post_process_detection, args=(opt, tmp_video_list,video_dict))
    p.start()
    processes.append(p)
    for p in processes:
        p.join()

    result_dict = dict(result_dict)
    output_dict = {"version": "VERSION 1.3", "results": result_dict, "external_data": {}}
    outfile = open(opt["result_file"], "w")
    json.dump(output_dict, outfile)
    outfile.close()
