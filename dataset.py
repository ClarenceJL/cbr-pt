import os
import torch
import torch.utils.data as data
import pandas as pd
import numpy as np
import json
import random
from Evaluation.eval_proposal import wrapper_segment_iou
#from offset_transform import *
from temporal_transform import temporal_avgpool_sample


def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data


class ANetDatasetCBR(data.Dataset):
    def __init__(self, opt, mode, subset, balance_fg_bg=True):
        super(ANetDatasetCBR, self).__init__()
        print('Create dataset for {}'.format(mode))
        self.mode = mode
        self.subset = subset
        self.frames_per_unit = opt['frames_per_unit']
        self.balance_fg_bg = balance_fg_bg
        self.bf_ratio = 1./opt['num_classes']
        self.pos_thresh = opt['pos_thresh']
        self.neg_thresh = opt['neg_thresh']
        self.ctx_num_unit = opt['ctx_num']
        self.feat_path = opt['feat_path']
        self._load_annotation(opt['anno_path'])
        self._load_class_index()
        self._load_proposals(opt['proposal_path'])
        print('    Number of {} proposals {}'.format(mode, self.__len__()))

    def _load_annotation(self, anno_file):
        anno = load_json(anno_file)
        anno = anno['database']
        self.ground_truth_anno = {}
        for vid, info in anno.items():
            if info['subset'] in self.subset:
                self.ground_truth_anno[vid] = info

    def _load_class_index(self):
        class_info = pd.read_csv('data/anet/class_index.csv')
        self.class_index = dict(zip(class_info.class_name.values, class_info.class_index.values + 1.))
        self.class_index['background'] = 0

    def _load_proposals(self, prop_path):
        self.proposals = []
        neg_proposals = []

        for vid, info in self.ground_truth_anno.items():
            corrected_second = info['duration'] * info['feature_frames'] / info['duration_frames']

            # load proposals
            pdf = pd.read_csv(prop_path + '/' + vid + '.csv')
            xmins = pdf.xmin.values * corrected_second
            xmaxs = pdf.xmax.values * corrected_second
            cands = np.stack((xmins, xmaxs), -1)
            if 'score' in pdf.columns:
                scores = pdf.score.values
            else:
                scores = np.ones([cands.shape[0]])

            # match proposals to ground-truth
            if self.mode != 'infer':
                assert len(self.ground_truth_anno[vid]['annotations']) > 0
                gts = []
                for gt in self.ground_truth_anno[vid]['annotations']:
                    gts.append(gt['segment'])
                gts = np.array(gts)

                ious = wrapper_segment_iou(gts, cands)  # (# of cands, # of gts)
                is_pos = np.max(ious, 1) >= self.pos_thresh
                match_idx = np.argmax(ious, 1)
                is_neg = np.max(ious, 1) < self.neg_thresh

                for i, (segment, score) in enumerate(zip(cands, scores)):
                    if is_pos[i]:
                        self.proposals.append({
                            'video_id': vid,
                            'segment': segment.tolist(),
                            'score': score,
                            'label': self.ground_truth_anno[vid]['annotations'][match_idx[i]]['label'],
                            'gt_segment': self.ground_truth_anno[vid]['annotations'][match_idx[i]]['segment']
                        })
                    elif is_neg[i]:
                        neg_proposals.append({
                            'video_id': vid,
                            'segment': segment.tolist(),
                            'score': 1.-score,
                            'label': 'background',
                            'gt_segment': [-1, -1]
                        })
            else:
                for i, (segment, score) in enumerate(zip(cands, scores)):
                    self.proposals.append({
                        'video_id': vid,
                        'score': score,
                        'segment': segment
                    })

        # sample from background proposals
        if self.mode != 'infer' and self.balance_fg_bg:
            num_neg_proposals = int(len(self.proposals) * self.bf_ratio)
            print('    Sample {} out of {} background samples'.format(num_neg_proposals, len(neg_proposals)))
            neg_proposals = random.sample(neg_proposals, num_neg_proposals)
            self.proposals += neg_proposals

    def generate_input(self, feat, start_unit, end_unit):
        start_feat = temporal_avgpool_sample(feat, start_unit-self.ctx_num_unit, start_unit, 1)
        mid_feat = temporal_avgpool_sample(feat, start_unit, end_unit, 1)
        end_feat = temporal_avgpool_sample(feat, end_unit, end_unit + self.ctx_num_unit, 1)
        data = np.concatenate((start_feat, mid_feat, end_feat), 1).squeeze(0)
        return torch.Tensor(data)

    def generate_target(self, label, start, end, gt_start, gt_end):
        target = [label, gt_start-start, gt_end - end]
        return torch.Tensor(target)

    def __getitem__(self, index):
        entry = self.proposals[index]
        vid = entry['video_id']
        feat = np.load(os.path.join(self.feat_path, entry['video_id'] + '.npy'))
        fps = self.ground_truth_anno[vid]['duration_frames'] / self.ground_truth_anno[vid]['duration']
        frames_per_unit = self.ground_truth_anno[vid]['feature_frames'] // feat.shape[0]
        assert frames_per_unit == self.frames_per_unit

        start_unit = entry['segment'][0] * fps / frames_per_unit
        end_unit = entry['segment'][1] * fps / frames_per_unit
        gt_start_unit = entry['gt_segment'][0] * fps / frames_per_unit
        gt_end_unit = entry['gt_segment'][1] * fps / frames_per_unit

        label = self.class_index[entry['label']]

        if self.mode == 'train':
            data = self.generate_input(feat, start_unit, end_unit)
            target = self.generate_target(label, start_unit, end_unit, gt_start_unit, gt_end_unit)
            return data, target
        elif self.mode == 'test':
            return feat, start_unit, end_unit, gt_start_unit, gt_end_unit, label
        else:
            return feat, start_unit, end_unit, entry['video_id'], entry['score'], frames_per_unit / fps

    def __len__(self):
        return len(self.proposals)


class THUMOSDatasetCBR(data.Dataset):
    # todo
    NotImplementedError
