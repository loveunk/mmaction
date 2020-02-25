import mmcv
from mmaction import datasets
from mmcv.runner import obj_from_dict
from mmaction.core.evaluation.accuracy import (softmax, top_k_accuracy,
                                               mean_class_accuracy)
import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('score_files', nargs='+', type=str)
    parser.add_argument('--score_weights', nargs='+', type=float, default=None)
    parser.add_argument('--crop_agg', type=str, choices=['max', 'mean'], default='mean')
    return parser.parse_args()

def main():
    args = parse_args()
    assert len(args.score_files) == 2

    scores = [mmcv.load(x) for x in args.score_files]

    if args.score_weights is None:
        score_weights = [1, 1.5]
    else:
        score_weights = args.score_weights
        if len(score_weights) != len(scores):
            raise ValueError("Only {} weight specifed for a total of {} score files"
                            .format(len(score_weights), len(scores)))
    # refer to https://github.com/yjxiong/temporal-segment-networks#video-level-testing

    scores_softmax = []
    for scores_ in scores:
        scores_softmax.append([softmax(res, dim=1).mean(axis=0) for res in scores_])

    scores_final = [s1 * score_weights[0] + s2 * score_weights[1] \
                    for s1, s2 in zip(scores_softmax[0], scores_softmax[1])]

    cfg = mmcv.Config.fromfile(args.config)
    dataset = obj_from_dict(cfg.data.test, datasets, dict(test_mode=True))

    gt_labels = []
    for i in range(len(dataset)):
        ann = dataset.get_ann_info(i)
        gt_labels.append(ann['label'])

    top1, top5 = top_k_accuracy(scores_final, gt_labels, k=(1,5))
    print('Top1/Top5 accuracy {:02f}%, {:02f}%'.format(top1 * 100, top5 * 100))

    mean_acc = mean_class_accuracy(scores_final, gt_labels)
    print('Mean accuracy {:02f}%'.format(mean_acc * 100))

if __name__ == "__main__":
    '''
    Usage: python tools/test_recognizer_modality_fusion.py ${config} \
               RGB_SCORE_FILE FLOW_SCORE_FILE --score_weights 1 1.5
    e.g.,
        python tools/test_recognizer_modality_fusion.py \
            configs/ucf101/tsn_rgb_bninception.py \
            ./work_dirs/ucf101/tsn_2d_rgb_bninception_seg_3_f1s1_b32_g8/result.pkl \
            ./work_dirs/ucf101/tsn_2d_flow_bninception_seg_3_f1s1_b32_g8_lr_0.005/result.pkl \
            --score_weights 1 1.5
    '''
    main()
    
