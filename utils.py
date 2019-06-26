import csv


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
    targets = targets[:, 0]
    _, pred = outputs.topk(tolerance, dim=1, largest=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().item()

    return n_correct_elems / batch_size


def calculate_offset(outputs, targets):
    batch_size = targets.size(0)
    num_class = outputs.shape[1] // 3 - 1
    pred_start_offset = outputs[:, 1+num_class:2*(1+num_class)]
    pred_end_offset = outputs[:, 2*(1+num_class):]
