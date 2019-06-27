import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Model
"""


class MultiLayer(nn.Module):
    def __init__(self, inplane, hiddenplane=512, outplane=201*3, dropout=True):
        super(MultiLayer, self).__init__()
        self.fc1 = nn.Linear(inplane, hiddenplane)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = dropout
        self.fc2 = nn.Linear(hiddenplane, outplane)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias'):
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        if self.dropout:
            x = F.dropout(x, p=0.5)
        x = self.fc2(x)
        return x


class CBR(nn.Module):
    def __init__(self, opt):
        super(CBR, self).__init__()
        self.num_class = opt['num_classes']
        inplane = 3 * opt['feature_dim']  # ctx-before | in-action | ctx-after
        hidplane = 512
        outplane = 3 * (1+opt['num_classes'])

        self.cls_reg = MultiLayer(inplane, hidplane, outplane)

    def forward(self, x):
        return self.cls_reg(x)


class ProposalClassifier(nn.Module):
    def __init__(self, opt):
        super(ProposalClassifier, self).__init__()
        inplane = 3 * opt['feature_dim']
        hidplane = 512
        outplane = 1 + opt['num_classes']
        self.fc1 = nn.Linear(inplane, hidplane)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidplane, outplane)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias'):
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


"""
Loss function
"""


class CBRLoss(nn.Module):
    def __init__(self, lambda_reg):
        super(CBRLoss, self).__init__()
        self.cls_loss = nn.CrossEntropyLoss()
        self.reg_loss = nn.L1Loss(reduction='none')
        self.lambda_reg = lambda_reg

    def forward(self, output, target):
        """
        :param output: (b, (1+num_class)*3)
        :param target: (b, 3)
        :return:
        """
        if output.is_cuda:
            target = target.cuda()

        batch_size = output.shape[0]
        num_class = output.shape[1] // 3 - 1

        cls_score_pred = output[:, :1+num_class]
        start_offset_pred = output[:, 1+num_class:2*(1+num_class)]
        end_offset_pred = output[:, :2*(1+num_class):]

        labels = target[:, 0].long()
        offsets = target[:, 1:]

        # classification loss
        cls_loss = self.cls_loss(cls_score_pred, labels)

        # regression loss
        # pick_start_offset_pred = []
        # pick_end_offset_pred = []
        # for i in range(batch_size):
        #     pick_start_offset_pred.append(start_offset_pred[i, labels[i]])
        #     pick_end_offset_pred.append(end_offset_pred[i, labels[i]])
        # pick_start_offset_pred = torch.Tensor(pick_start_offset_pred)
        # pick_end_offset_pred = torch.Tensor(pick_end_offset_pred)

        pick_start_offset_pred = start_offset_pred[list(range(batch_size)), labels.tolist()]
        pick_end_offset_pred = end_offset_pred[list(range(batch_size)), labels.tolist()]
        offsets_pred = torch.stack((pick_start_offset_pred, pick_end_offset_pred), -1)  # (b, 2)
        fg_mask = labels.ne(0).float().unsqueeze(-1)  # (b, 1)
        reg_loss = torch.mean(fg_mask * self.reg_loss(offsets_pred, offsets))

        loss = cls_loss + self.lambda_reg * reg_loss

        return {'loss': loss, 'cls_loss': cls_loss, 'reg_loss': reg_loss}





