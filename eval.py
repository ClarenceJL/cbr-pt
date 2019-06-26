from Evaluation.eval_proposal import ANETproposal
from Evaluation.eval_detection import ANETdetection
import numpy as np


def evaluation_proposal(opt):
    anet_proposal = ANETproposal(opt["anno_path"], opt["result_path"],
                                 tiou_thresholds=np.linspace(0.5, 0.95, 10),
                                 max_avg_nr_proposals=100,
                                 subset=opt.evaluate_subset, verbose=True, check_status=False)
    anet_proposal.evaluate()

    uniform_recall_valid = anet_proposal.recall

    print("AR@1 is \t", np.mean(uniform_recall_valid[:, 0]))
    print("AR@5 is \t", np.mean(uniform_recall_valid[:, 4]))
    print("AR@10 is \t", np.mean(uniform_recall_valid[:, 9]))
    print("AR@100 is \t", np.mean(uniform_recall_valid[:, -1]))


def evaluation_detection(opt):
    anet_detection = ANETdetection(opt["anno_path"], opt["result_path"],
                                 tiou_thresholds=np.linspace(0.5, 0.95, 10),
                                 subset='validation', verbose=True, check_status=False)
    anet_detection.evaluate()

