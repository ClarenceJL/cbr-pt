import csv
import numpy as np


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def calculate_accuracy(outputs, targets, tolerance=1):
    batch_size = targets.size(0)
    num_class = outputs.shape[1] // 3 - 1
    outputs = outputs[:, :1+num_class]
    targets = targets[:, 0].long()
    _, pred = outputs.topk(tolerance, dim=1, largest=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().item()

    return n_correct_elems / batch_size


def calculate_offset(outputs, targets):
    batch_size = targets.size(0)
    num_class = outputs.shape[1] // 3 - 1

    outputs = outputs.cpu().detach().numpy()
    targets = targets.cpu().numpy()

    start_offset_pred = outputs[:, 1 + num_class:2 * (1 + num_class)]
    end_offset_pred = outputs[:, :2 * (1 + num_class):]
    labels = targets[:, 0]
    offsets = targets[:, 1:]
    # get the absolute offset difference at the CORRECT class
    pick_start_offset_pred = start_offset_pred[list(range(batch_size)), labels.astype(int).tolist()]
    pick_end_offset_pred = end_offset_pred[list(range(batch_size)), labels.astype(int).tolist()]
    offsets_pred = np.stack((pick_start_offset_pred, pick_end_offset_pred), -1)
    fg_mask = np.expand_dims((labels > 0).astype(float), -1)  # (b, 1)
    return np.sum(fg_mask * np.abs(offsets_pred-offsets)) / batch_size