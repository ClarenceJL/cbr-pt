import opts
import os
import json
import torch.nn.parallel
import torch.optim as optim
import numpy as np
import pandas as pd
from dataset import ANetDatasetCBR
from model import CBR, CBRLoss
from post_process import post_processing_wrapper
from eval import evaluation_detection
from utils import Logger, calculate_accuracy, calculate_offset


prob_weights = np.array([0.8, 0.1, 0.1])


def train_epoch(epoch, model, data_loader, optimizer, criterion, logger):
    model.train()
    epoch_losses = np.zeros([3])
    epoch_acc = 0
    epoch_off = 0

    for n_iter, (input, target) in enumerate(data_loader):
        output = model(input)
        target = target.cuda()
        losses = criterion(output, target)
        acc = calculate_accuracy(output, target)
        off = calculate_offset(output, target)

        optimizer.zero_grad()
        losses['loss'].backward()
        optimizer.step()

        epoch_losses[0] += losses['loss'].cpu().detach().numpy()
        epoch_losses[1] += losses['cls_loss'].cpu().detach().numpy()
        epoch_losses[2] += losses['reg_loss'].cpu().detach().numpy()
        epoch_acc += acc
        epoch_off += off

    epoch_losses /= (n_iter + 1)
    epoch_acc /= (n_iter + 1)
    epoch_off /= (n_iter + 1)

    logger.log({
        'epoch': epoch,
        'loss': epoch_losses[0],
        'cls_loss': epoch_losses[1],
        'reg_loss': epoch_losses[2],
        'acc': epoch_acc,
        'off': epoch_off
    })

    print('Epoch {} training loss: {} cls_loss: {} reg_loss: {} acc: {} off: {}'.format(epoch, epoch_losses[0],
                                                       epoch_losses[1], epoch_losses[2], epoch_acc, epoch_off))


def test_epoch(epoch, model, data_loader, criterion, logger, num_classes=200, cas_num=3):
    model.eval()
    epoch_losses_cas = np.zeros([cas_num, 3])
    epoch_acc_cas = np.zeros([cas_num])
    epoch_off_cas = np.zeros([cas_num])

    for n_iter, (feat, start_unit, end_unit, gt_start_unit, gt_end_unit, label) in enumerate(data_loader):
        mini_batch_size = len(feat)
        # In BMVC paper, we used prob multiplication to calculate final prob,
        # but later experiments showed that weighted average gives more stable results.
        # final_action_prob = torch.ones([mini_batch_size, num_classes])
        final_action_prob = torch.zeros([mini_batch_size, num_classes])
        for k in range(cas_num):
            # compute input and target
            inputs = []
            targets = []
            for b in range(mini_batch_size):
                inputs.append(data_loader.dataset.generate_input(feat[b], start_unit[b], end_unit[b]))
                targets.append(data_loader.dataset.generate_target(label[b], start_unit[b], end_unit[b],
                                                                  gt_start_unit[b], gt_end_unit[b]))

            inputs = torch.Tensor(inputs)
            targets = torch.Tensor(targets)

            # run model for current cascade
            outputs = model(inputs)
            losses = criterion(outputs, targets)
            epoch_losses_cas[k, 0] += losses['loss'].cpu().detach().numpy()
            epoch_losses_cas[k, 1] += losses['cls_loss'].cpu().detach().numpy()
            epoch_losses_cas[k, 2] += losses['reg_loss'].cpu().detach().numpy()

            epoch_acc_cas[k] += calculate_accuracy(outputs, targets)
            epoch_off_cas[k] += calculate_offset(outputs, targets)

            if k == cas_num - 1:
                break

            # update the boundaries of current proposals
            outputs = outputs.cpu().detach()
            action_score = outputs[:, 1:1+num_classes]
            action_prob = torch.softmax(action_score, 1)
            # In BMVC paper, we used prob multiplication to calculate final prob,
            # but later experiments showed that weighted average gives more stable results.
            # final_action_prob *= action_prob
            final_action_prob = final_action_prob + prob_weights[k] * action_prob
            pred_action = torch.argmax(final_action_prob, 1) + 1  # (b,), 1 ~ 200
            start_unit = start_unit + outputs[list(range(mini_batch_size)), (1+num_classes+pred_action.int()).tolist()]
            end_unit = end_unit + outputs[list(range(mini_batch_size)), (2*(1+num_classes)+pred_action.int()).tolist()]

    epoch_losses_cas /= (n_iter + 1)
    epoch_acc_cas /= (n_iter + 1)
    epoch_off_cas /= (n_iter + 1)

    for k in range(cas_num):
        logger.log({
            'epoch': epoch,
            'cas': k,
            'loss': epoch_losses_cas[k, 0],
            'cls_loss': epoch_losses_cas[k, 1],
            'reg_loss': epoch_losses_cas[k, 2],
            'acc': epoch_acc_cas[k],
            'off': epoch_off_cas[k]
        })

    print('Epoch {} validation (cascade 0) loss: {} cls_loss: {}, reg_loss: {} acc: {} acc3: {}'.format(
     epoch, epoch_losses_cas[0, 0], epoch_losses_cas[0, 1], epoch_losses_cas[0, 2], epoch_acc_cas[0], epoch_off_cas[0]))
    for k in range(1, cas_num):
        print('                    (cascade {}) loss: {} cls_loss: {}, reg_loss: {} acc: {} acc3: {}'.format(
         k, epoch_losses_cas[k, 0], epoch_losses_cas[k, 1], epoch_losses_cas[k, 2], epoch_acc_cas[k], epoch_off_cas[k]))

    return epoch_losses_cas[cas_num-1, 0], epoch_acc_cas[cas_num-1]


def train_wrapper(opt):
    # build model
    model = CBR(opt)
    model = torch.nn.DataParallel(model, device_ids=opt['gpu_ids']).cuda()

    # load pretrained weights (optional)

    # create optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt["lr_base"], weight_decay=opt["weight_decay"])
    criterion = CBRLoss()
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt["step_size"], gamma=opt["step_gamma"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt['num_epochs'], eta_min=1e-7)

    # create logger(s)
    train_logger = Logger(os.path.join(opt['checkpoint_path'], 'train_pac.log'),
                          ['epoch', 'loss', 'cls_loss', 'reg_loss', 'acc', 'off'])
    test_logger = Logger(os.path.join(opt['checkpoint_path'], 'test_pac.log'),
                          ['epoch', 'cas', 'loss', 'cls_loss', 'reg_loss', 'acc', 'off'])

    # make dataloader(s)
    train_loader = torch.utils.data.DataLoader(ANetDatasetCBR(opt, subset=["training"]),
                                               batch_size=opt["batch_size"], shuffle=True,
                                               num_workers=8, pin_memory=True, drop_last=True)

    test_loader = torch.utils.data.DataLoader(ANetDatasetCBR(opt, subset=["validation"]),  # todo: customize collate_fn
                                              batch_size=opt["test_batch_size"], shuffle=False,
                                              num_workers=8, pin_memory=True, drop_last=True)

    min_val_loss = 1e8
    max_val_acc = 0

    for epoch in range(opt['num_epochs']):
        scheduler.step()
        train_epoch(epoch, model, train_loader, optimizer, criterion, train_logger)
        loss, acc = test_epoch(epoch, model, test_loader, criterion, test_logger, opt['cas_step'], opt['num_classes'])
        state = {'epoch': epoch + 1, 'state_dict': model.state_dict()}
        torch.save(state, opt['checkpoint_path'] + '/model_checkpoint.pth.tar')
        if min_val_loss > loss:
            min_val_loss = loss
            torch.save(state, opt['checkpoint_path'] + '/model_best_loss.pth.tar')
        if max_val_acc < acc:
            max_val_acc = acc
            torch.save(state, opt['checkpoint_path'] + '/model_best_acc.pth.tar')


def inference_wrapper(opt):
    # build model
    model = CBR(opt)
    # load checkpoint
    checkpoint = torch.load(opt["checkpoint_path"] + "/model_best_loss.pth.tar")
    base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
    model.load_state_dict(base_dict)
    model = torch.nn.DataParallel(model, device_ids=opt['gpu_ids']).cuda()
    model.eval()

    # make dataloader
    data_loader = torch.utils.data.DataLoader(ANetDatasetCBR(opt, mode='infer', subset=opt['infer_subset']),
                                              batch_size=opt["test_batch_size"], shuffle=False,
                                              num_workers=8, pin_memory=True, drop_last=False)

    class_info = pd.read_csv('data/class_index.csv')
    class_names = class_info.class_name.values

    result_dict = {}
    for feat, start_unit, end_unit, video_id, prop_score, unit_sec_ratio in data_loader:
        mini_batch_size = len(feat)
        final_action_prob = torch.zeros([mini_batch_size, opt['num_classes']])
        for k in range(opt['cas_step']):
            # compute input and target
            inputs = []
            targets = []
            for b in range(mini_batch_size):
                inputs.append(data_loader.dataset.generate_input(feat[b], start_unit[b], end_unit[b]))

            inputs = torch.Tensor(inputs)

            # run model for current cascade
            outputs = model(inputs)

            if k == opt['cas_step'] - 1:
                break

            # update the boundaries of current proposals
            outputs = outputs.cpu().detach()
            action_score = outputs[:, 1:1 + opt['num_classes']]
            action_prob = torch.softmax(action_score, 1)
            final_action_prob = final_action_prob + prob_weights[k] * action_prob
            pred_action = torch.argmax(final_action_prob, 1) + 1  # (b,), 1 ~ 200
            start_unit = start_unit + outputs[list(range(mini_batch_size)), (1 + opt['num_classes'] + pred_action.int()).tolist()]
            end_unit = end_unit + outputs[list(range(mini_batch_size)), (2 * (1 + opt['num_classes']) + pred_action.int()).tolist()]

        cls_score = torch.max(final_action_prob, 1)
        label = torch.argmax(final_action_prob, 1)  # 0~199
        score = prop_score * cls_score

        for vid, lab, scr, su, eu, ratio in zip(video_id, label, score, start_unit, end_unit, unit_sec_ratio):
            if vid not in result_dict.keys():
                result_dict[vid] = []
            result_dict[vid].append({
                'label': class_names[lab],
                'score': scr,
                'segment': [su * ratio, eu * ratio]
            })

    # split results
    if not os.path.exists(opt['checkpoint_path']+'/output'):
        os.mkdir(opt['checkpoint_path']+'/output')

    for vid, props in result_dict.items():
        df = pd.DataFrame(props)
        df['xmin'] = df['segment'].apply(lambda seg: seg[0])
        df['xmax'] = df['segment'].apply(lambda seg: seg[1])
        df = df[['xmin', 'xmax', 'score', 'label']]
        df.to_csv(opt['checkpoint_path']+'/output/'+vid+'.csv')


def main(opt):
    if opt["mode"] == "train":
        print('Start training classifier...')
        train_wrapper(opt)
        print('Training done')
    elif opt["mode"] == "infer":
        print("Start inference...")
        inference_wrapper(opt)
        print('Inference done')
    elif opt["mode"] == "post_process":
        print("Start post-processing...")
        post_processing_wrapper(opt, opt['infer_subset'])
        print("Post-processing done")
    elif opt["mode"] == "evaluation":
        evaluation_detection(opt)
    else:
        raise NotImplementedError('Mode {} not supported'.format(opt["mode"]))


if __name__ == "__main__":
    # parse arguments
    opt = opts.parse_opt()
    opt = vars(opt)
    if not os.path.exists(opt["checkpoint_path"]):
        os.makedirs(opt["checkpoint_path"])

    os.environ['CUDA_VISIBLE_DEVICES'] = opt['gpu_ids']
    str_ids = opt['gpu_ids'].split(',')
    gpu_ids = [int(str_id) for str_id in str_ids if int(str_id) >= 0]
    opt['gpu_ids'] = gpu_ids

    with open(opt["checkpoint_path"]+"/opts.json", "w") as f:
        json.dump(opt, f)

    main(opt)



