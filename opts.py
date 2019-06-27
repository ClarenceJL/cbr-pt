import argparse


def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        help='(train | infer | post_process | eval )')

    # file paths
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='/vclaire/result/detection_with_feat')
    parser.add_argument(
        '--anno_path',
        type=str,
        default='/mydata/ActivityNet/anno_19250_clip16_0610.json')
    parser.add_argument(
        '--result_path',
        type=str,
        default='/vclaire/result/detection_with_feat/result_detection_retnetprop.json')
    parser.add_argument(
        '--proposal_path',
        type=str,
        default='/vclaire/result/retinanet/retinanet_38/PGM_proposals'
    )
    parser.add_argument(
        '--feat_path',
        type=str,
        default='/mydata/ActivityNet/original_clip16_avgpool256_0610')

    # dataset
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='ActivityNet'
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        default=200
    )
    parser.add_argument(
        '--feature_dim',
        type=int,
        default=256
    )
    parser.add_argument(
        '--frames_per_unit',
        type=int,
        default=16
    )
    parser.add_argument(
        '--pos_thresh',
        type=float,
        default=0.5
    )
    parser.add_argument(
        '--neg_thresh',
        type=float,
        default=0.5
    )
    parser.add_argument(
        '--infer_subset',
        type=str,
        default='validation'
    )

    # model setting
    parser.add_argument(
        '--cas_step',
        type=int,
        default=3,
        help='Cascade levels. Notice that cascaded regression is only used for validation / inference, not training.'
    )
    parser.add_argument(
        '--ctx_num',
        type=int,
        default=2
    )
    parser.add_argument(
        '--lambda_reg',
        type=float,
        default=1.0
    )


    # optimization setting
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128
    )
    parser.add_argument(
        '--test_batch_size',
        type=int,
        default=32,
        help="The cascade operation in test/inference keeps a copy of the original features in memory, "
             "therefore the test/inference batch size should be smaller than training batch size."
    )
    parser.add_argument(
        '--lr_base',
        type=float,
        default=0.001
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.0001)
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=20
    )
    parser.add_argument(
        '--step_size',
        type=int,
        default=4
    )
    parser.add_argument(
        '--step_gamma',
        type=float,
        default=0.1
    )
    parser.add_argument(
        '--gpu_ids',
        default='0',
        type=str,
        help='gpu_idxs: e.g. 0, 0,1,2 0,2'
    )

    args = parser.parse_args()
    return args

