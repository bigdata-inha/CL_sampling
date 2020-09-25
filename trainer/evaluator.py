import logging

import numpy as np
import torch
import torch.nn.functional as F
from numpy.linalg import inv
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


# 3개 함수 모두 experiment에 넣자.
def make_pred(out, start, end, step_size):
    pred = {}
    pred_5 = {}
    pred['all'] = out.data.max(1, keepdim=True)[1]
    pred_5['all'] = torch.topk(out, 5, dim=1)[1]

    prev_out = out[:, start:end - step_size]
    curr_out = out[:, end - step_size:end]

    prev_soft = F.softmax(prev_out, dim=1)
    curr_soft = F.softmax(curr_out, dim=1)

    output = torch.cat((prev_soft, curr_soft), dim=1)

    pred['prev_new'] = output.data.max(1, keepdim=True)[1]
    pred_5['prev_new'] = torch.topk(output, 5, dim=1)[1]

    soft_arr = []
    for t in range(start, end, step_size):
        temp_out = out[:, t:t + step_size]
        temp_soft = F.softmax(temp_out, dim=1)
        soft_arr += [temp_soft]

    output = torch.cat(soft_arr, dim=1)

    pred['task'] = output.data.max(1, keepdim=True)[1]
    pred_5['task'] = torch.topk(output, 5, dim=1)[1]

    return pred, pred_5


def cnt_stat(target, start, end, step_size, mode, head, pred, pred_5, correct, correct_5, stat, batch_size):
    correct[head] += pred[head].eq(target.data.view_as(pred[head])).sum().item()
    correct_5[head] += pred_5[head].eq(target.data.unsqueeze(1).expand(pred_5[head].shape)).sum().item()

    if mode == 'prev':
        cp_ = pred[head].eq(target.data.view_as(pred[head])).sum()
        epn_ = (pred[head] >= end - step_size).int().sum()
        epp_ = (batch_size - (cp_ + epn_))
        stat[head][0] += cp_.item()
        stat[head][1] += epp_.item()
        stat[head][2] += epn_.item()
    else:
        cn_ = pred[head].eq(target.data.view_as(pred[head])).sum()
        enp_ = (pred[head] < end - step_size).int().sum()
        enn_ = (batch_size - (cn_ + enp_))
        stat[head][3] += cn_.item()
        stat[head][4] += enn_.item()
        stat[head][5] += enp_.item()
    return


def cheat(out, target, start, end, mod, correct, correct_5):
    output = out[:, start:end]
    target = target % (mod)

    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
    ans = pred.eq(target.data.view_as(pred)).sum()
    correct['cheat'] += ans.item()

    pred_5 = torch.topk(output, 5, dim=1)[1]
    ans = pred_5.eq(target.data.unsqueeze(1).expand(pred_5.shape)).sum()
    correct_5['cheat'] += ans.item()


class EvaluatorFactory():
    '''
    This class is used to get different versions of evaluators
    '''

    def __init__(self):
        pass

    # evaluator를 나누지 말고 trainer 정보만 넣어주자.
    @staticmethod
    def get_evaluator(testType="trainedClassifier", classes=1000, option='euclidean'):
        if testType == "trainedClassifier":
            return softmax_evaluator()
        if testType == "il2m":
            return IL2M_evaluator(classes)
        if testType == "bic":
            return BiC_evaluator(classes)
        if testType == "generativeClassifier":
            return GDA(classes, option=option)


class GDA():

    def __init__(self, classes, option='euclidean'):
        self.classes = classes
        self.option = option

    def update_moment(self, model, loader, step_size, tasknum):

        model.eval()
        with torch.no_grad():
            # compute means
            classes = step_size * (tasknum + 1)
            class_means = torch.zeros((classes, 512)).cuda()
            totalFeatures = torch.zeros((classes, 1)).cuda()
            total = 0
            # Iterate over all train Dataset
            for data, target in tqdm(loader):
                data, target = data.cuda(), target.cuda()
                if data.shape[0] < 4:
                    continue
                total += data.shape[0]
                try:
                    _, features = model.forward(data, feature_return=True)
                except:
                    continue

                class_means.index_add_(0, target, features.data)
                totalFeatures.index_add_(0, target, torch.ones_like(target.unsqueeze(1)).float().cuda())

            class_means = class_means / totalFeatures

            # compute precision
            covariance = torch.zeros(512, 512).cuda()
            euclidean = torch.eye(512).cuda()

            if self.option == 'Mahalanobis':
                for data, target in tqdm(loader):
                    data, target = data.cuda(), target.cuda()
                    _, features = model.forward(data, feature_return=True)

                    vec = (features.data - class_means[target])

                    np.expand_dims(vec, axis=2)
                    cov = torch.matmul(vec.unsqueeze(2), vec.unsqueeze(1)).sum(dim=0)
                    covariance += cov

                # avoid singular matrix
                covariance = covariance / totalFeatures.sum() + torch.eye(512).cuda() * 1e-9
                precision = covariance.inverse()

            self.class_means = class_means
            if self.option == 'Mahalanobis':
                self.precision = precision
            else:
                self.precision = euclidean

            return

    def evaluate(self, model, loader, start, end, mode='train', step_size=100):

        with torch.no_grad():
            model.eval()
            correct_cnt = 0
            correct_5_cnt = 0
            total = 0
            stat = {}
            correct = {}
            correct_5 = {}
            correct['cheat'] = 0
            correct_5['cheat'] = 0
            head_arr = ['all', 'prev_new', 'task']
            for head in head_arr:
                # cp, epp, epn, cn, enn, enp, total
                stat[head] = [0, 0, 0, 0, 0, 0, 0]
                correct[head] = 0
                correct_5[head] = 0

            for data, target in tqdm(loader):
                data, target = data.cuda(), target.cuda()
                if data.shape[0] < 4:
                    continue
                try:
                    _, features = model.forward(data, feature_return=True)
                except:
                    continue

                batch_size = data.shape[0]
                total += data.shape[0]

                batch_vec = (features.data.unsqueeze(1) - self.class_means.unsqueeze(0))
                temp = torch.matmul(batch_vec, self.precision)
                out = -torch.matmul(temp.unsqueeze(2), batch_vec.unsqueeze(3)).squeeze()

                # experiment에 out만 넘겨주고 끝내자.

                if mode == 'test' and end > step_size:
                    pred, pred_5 = make_pred(out, start, end, step_size)

                    if target[0] < end - step_size:  # prev

                        cnt_stat(target, start, end, step_size, 'prev', 'all', pred, pred_5, correct, correct_5, stat,
                                 batch_size)
                        cnt_stat(target, start, end, step_size, 'prev', 'prev_new', pred, pred_5, correct, correct_5,
                                 stat, batch_size)
                        cnt_stat(target, start, end, step_size, 'prev', 'task', pred, pred_5, correct, correct_5, stat,
                                 batch_size)

                        cheat(out, target, start, end - step_size, end - start - step_size, correct, correct_5)

                    else:  # new

                        cnt_stat(target, start, end, step_size, 'new', 'all', pred, pred_5, correct, correct_5, stat,
                                 batch_size)
                        cnt_stat(target, start, end, step_size, 'new', 'prev_new', pred, pred_5, correct, correct_5,
                                 stat, batch_size)
                        cnt_stat(target, start, end, step_size, 'new', 'task', pred, pred_5, correct, correct_5, stat,
                                 batch_size)

                        cheat(out, target, end - step_size, end, step_size, correct, correct_5)

                else:
                    output = out[:, start:end]
                    target = target % (end - start)

                    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                    correct_cnt += pred.eq(target.data.view_as(pred)).sum().item()

                    pred_5 = torch.topk(output, 5, dim=1)[1]
                    correct_5_cnt += pred_5.eq(target.data.unsqueeze(1).expand(pred_5.shape)).sum().item()

            if mode == 'test' and end > step_size:

                for head in ['all', 'prev_new', 'task', 'cheat']:
                    correct[head] = 100. * correct[head] / total
                    correct_5[head] = 100. * correct_5[head] / total
                stat['all'][6] = stat['prev_new'][6] = stat['task'][6] = total
                return correct, correct_5, stat

            return 100. * correct_cnt / total, 100. * correct_5_cnt / total,


class IL2M_evaluator():
    '''
    Evaluator class for softmax classification
    '''

    def __init__(self, classes):

        self.classes = classes

        self.init_class_means = torch.zeros(classes).cuda()
        self.current_class_means = torch.zeros(classes).cuda()

        self.model_confidence = torch.zeros(classes).cuda()

    def update_mean(self, model, loader, end, step_size):
        model.eval()
        with torch.no_grad():
            class_means = torch.zeros(self.classes).cuda()
            class_count = torch.zeros(self.classes).cuda()
            current_count = 0

            for data, target in tqdm(loader):
                data, target = data.cuda(), target.cuda()
                out = model(data)
                prob = F.softmax(out[:, :end], dim=1)
                confidence = prob.max(dim=1)[0] * (target >= (end - step_size)).float()
                class_means.index_add_(0, target, prob[torch.arange(data.shape[0]), target])
                class_count.index_add_(0, target, torch.ones_like(target).float().cuda())

                self.model_confidence[end - step_size:end] += confidence.sum()
                current_count += (target >= (end - step_size)).float().sum()

            # current task
            self.init_class_means[end - step_size:end] = class_means[end - step_size:end] / class_count[
                                                                                            end - step_size:end]
            #
            self.current_class_means[:end] = class_means[:end] / class_count[:end]

            self.model_confidence[end - step_size:end] /= current_count

    def evaluate(self, model, loader, start, end, mode='train', step_size=100):

        with torch.no_grad():
            model.eval()
            correct_cnt = 0
            correct_5_cnt = 0
            total = 0
            step_size = step_size
            stat = {}
            correct = {}
            correct_5 = {}
            correct['cheat'] = 0
            correct_5['cheat'] = 0
            head_arr = ['all', 'prev_new', 'task']
            for head in head_arr:
                # cp, epp, epn, cn, enn, enp, total
                stat[head] = [0, 0, 0, 0, 0, 0, 0]
                correct[head] = 0
                correct_5[head] = 0

            for data, target in tqdm(loader):
                data, target = data.cuda(), target.cuda()

                batch_size = data.shape[0]
                total += data.shape[0]

                out = model(data)[:, :end]

                if mode == 'test' and end > step_size:

                    pred = out.data.max(1, keepdim=True)[1]
                    mask = (pred >= end - step_size).int()
                    prob = F.softmax(out, dim=1)
                    rect_prob = prob * (self.init_class_means[:end] / self.current_class_means[:end]) \
                                * (self.model_confidence[end - 1] / self.model_confidence[:end])

                    out = (1 - mask).float() * prob + mask.float() * rect_prob

                    pred, pred_5 = make_pred(out, start, end, step_size)

                    if target[0] < end - step_size:  # prev

                        cnt_stat(target, start, end, step_size, 'prev', 'all', pred, pred_5, correct, correct_5, stat,
                                 batch_size)
                        cnt_stat(target, start, end, step_size, 'prev', 'prev_new', pred, pred_5, correct, correct_5,
                                 stat, batch_size)
                        cnt_stat(target, start, end, step_size, 'prev', 'task', pred, pred_5, correct, correct_5, stat,
                                 batch_size)

                        cheat(out, target, start, end - step_size, end - start - step_size, correct, correct_5)

                    else:  # new

                        cnt_stat(target, start, end, step_size, 'new', 'all', pred, pred_5, correct, correct_5, stat,
                                 batch_size)
                        cnt_stat(target, start, end, step_size, 'new', 'prev_new', pred, pred_5, correct, correct_5,
                                 stat, batch_size)
                        cnt_stat(target, start, end, step_size, 'new', 'task', pred, pred_5, correct, correct_5, stat,
                                 batch_size)

                        cheat(out, target, end - step_size, end, step_size, correct, correct_5)
                else:
                    output = out[:, start:end]
                    target = target % (end - start)

                    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                    correct_cnt += pred.eq(target.data.view_as(pred)).sum().item()

                    pred_5 = torch.topk(output, 5, dim=1)[1]
                    correct_5_cnt += pred_5.eq(target.data.unsqueeze(1).expand(pred_5.shape)).sum().item()

            if mode == 'test' and end > step_size:

                for head in ['all', 'prev_new', 'task', 'cheat']:
                    correct[head] = 100. * correct[head] / total
                    correct_5[head] = 100. * correct_5[head] / total
                stat['all'][6] = stat['prev_new'][6] = stat['task'][6] = total
                return correct, correct_5, stat

            return 100. * correct_cnt / total, 100. * correct_5_cnt / total,


class BiC_evaluator():
    '''
    Evaluator class for softmax classification
    '''

    def __init__(self, classes):

        self.classes = classes

    def evaluate(self, model, loader, start, end, bias_correction_layer, mode='train', step_size=100):

        with torch.no_grad():
            model.eval()
            correct_cnt = 0
            correct_5_cnt = 0
            total = 0
            step_size = step_size
            stat = {}
            correct = {}
            correct_5 = {}
            correct['cheat'] = 0
            correct_5['cheat'] = 0
            head_arr = ['all', 'prev_new', 'task']
            for head in head_arr:
                # cp, epp, epn, cn, enn, enp, total
                stat[head] = [0, 0, 0, 0, 0, 0, 0]
                correct[head] = 0
                correct_5[head] = 0

            for data, target in tqdm(loader):
                data, target = data.cuda(), target.cuda()

                batch_size = data.shape[0]
                total += data.shape[0]

                out = model(data)[:, :end]
                if end > step_size:
                    out_new = bias_correction_layer(out[:, end - step_size:end])
                    out = torch.cat((out[:, :end - step_size], out_new), dim=1)

                if mode == 'test' and end > step_size:

                    pred, pred_5 = make_pred(out, start, end, step_size)

                    if target[0] < end - step_size:  # prev

                        cnt_stat(target, start, end, step_size, 'prev', 'all', pred, pred_5, correct, correct_5, stat,
                                 batch_size)
                        cnt_stat(target, start, end, step_size, 'prev', 'prev_new', pred, pred_5, correct, correct_5,
                                 stat, batch_size)
                        cnt_stat(target, start, end, step_size, 'prev', 'task', pred, pred_5, correct, correct_5, stat,
                                 batch_size)

                        cheat(out, target, start, end - step_size, end - start - step_size, correct, correct_5)

                    else:  # new

                        cnt_stat(target, start, end, step_size, 'new', 'all', pred, pred_5, correct, correct_5, stat,
                                 batch_size)
                        cnt_stat(target, start, end, step_size, 'new', 'prev_new', pred, pred_5, correct, correct_5,
                                 stat, batch_size)
                        cnt_stat(target, start, end, step_size, 'new', 'task', pred, pred_5, correct, correct_5, stat,
                                 batch_size)

                        cheat(out, target, end - step_size, end, step_size, correct, correct_5)
                else:
                    output = out[:, start:end]
                    target = target % (end - start)

                    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                    correct_cnt += pred.eq(target.data.view_as(pred)).sum().item()

                    pred_5 = torch.topk(output, 5, dim=1)[1]
                    correct_5_cnt += pred_5.eq(target.data.unsqueeze(1).expand(pred_5.shape)).sum().item()

            if mode == 'test' and end > step_size:

                for head in ['all', 'prev_new', 'task', 'cheat']:
                    correct[head] = 100. * correct[head] / total
                    correct_5[head] = 100. * correct_5[head] / total
                stat['all'][6] = stat['prev_new'][6] = stat['task'][6] = total
                return correct, correct_5, stat

            return 100. * correct_cnt / total, 100. * correct_5_cnt / total,


class softmax_evaluator():
    '''
    Evaluator class for softmax classification
    '''

    def __init__(self):
        pass

    def evaluate(self, model, loader, start, end, mode='train', step_size=100):
        with torch.no_grad():
            model.eval()
            correct = 0
            correct_5 = 0
            total = 0
            self.start = start
            self.end = end
            self.step_size = step_size
            self.stat = {}
            self.correct = {}
            self.correct_5 = {}
            self.correct['cheat'] = 0
            self.correct_5['cheat'] = 0
            head_arr = ['all', 'prev_new', 'task']
            for head in head_arr:
                # cp, epp, epn, cn, enn, enp, total
                self.stat[head] = [0, 0, 0, 0, 0, 0, 0]
                self.correct[head] = 0
                self.correct_5[head] = 0

            for data, target, test_idx in loader:
                data, target = data.cuda(), target.cuda()

                self.batch_size = data.shape[0]
                total += data.shape[0]

                if mode == 'test' and end > step_size:

                    out = model(data)
                    self.make_pred(out)

                    if target[0] < end - step_size:  # prev

                        self.cnt_stat(target, 'prev', 'all')
                        self.cnt_stat(target, 'prev', 'prev_new')
                        self.cnt_stat(target, 'prev', 'task')

                        output = out[:, start:end - step_size]
                        target = target % (end - start - step_size)

                        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                        ans = pred.eq(target.data.view_as(pred)).sum().item()
                        self.correct['cheat'] += ans

                        pred_5 = torch.topk(output, 5, dim=1)[1]
                        ans = pred_5.eq(target.data.unsqueeze(1).expand(pred_5.shape)).sum().item()
                        self.correct_5['cheat'] += ans

                    else:  # new

                        self.cnt_stat(target, 'new', 'all')
                        self.cnt_stat(target, 'new', 'prev_new')
                        self.cnt_stat(target, 'new', 'task')

                        output = out[:, end - step_size:end]
                        target = target % (step_size)

                        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                        ans = pred.eq(target.data.view_as(pred)).sum().item()
                        self.correct['cheat'] += ans

                        pred_5 = torch.topk(output, 5, dim=1)[1]
                        ans = pred_5.eq(target.data.unsqueeze(1).expand(pred_5.shape)).sum().item()
                        self.correct_5['cheat'] += ans
                else:
                    output = model(data)[:, start:end]
                    target = target % (end - start)

                    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                    correct += pred.eq(target.data.view_as(pred)).sum().item()


                    pred_5 = torch.topk(output, 5, dim=1)[1]
                    correct_5 += pred_5.eq(target.data.unsqueeze(1).expand(pred_5.shape)).sum().item()

            if mode == 'test' and end > step_size:

                for head in ['all', 'prev_new', 'task', 'cheat']:
                    self.correct[head] = 100. * self.correct[head] / total
                    self.correct_5[head] = 100. * self.correct_5[head] / total
                self.stat['all'][6] = self.stat['prev_new'][6] = self.stat['task'][6] = total
                return self.correct, self.correct_5, self.stat

            return 100. * correct / total, 100. * correct_5 / total


    def evaluate_top1(self, model, loader, start, end, mode='train', step_size=100):
        with torch.no_grad():
            model.eval()
            correct = 0
            correct_5 = 0
            total = 0
            self.start = start
            self.end = end
            self.step_size = step_size
            self.stat = {}
            self.correct = {}
            self.correct_5 = {}
            self.correct['cheat'] = 0
            self.correct_5['cheat'] = 0

            head_arr = ['all', 'prev_new', 'task']
            for head in head_arr:
                # cp, epp, epn, cn, enn, enp, total
                self.stat[head] = [0, 0, 0, 0, 0, 0, 0]
                self.correct[head] = 0
                #self.correct_5[head] = 0

            for data, target, test_idx in loader:
                data, target = data.cuda(), target.cuda()

                self.batch_size = data.shape[0]
                total += data.shape[0]

                if mode == 'test' and end > step_size:

                    out = model(data)
                    self.make_pred(out)

                    if target[0] < end - step_size:  # prev

                        self.cnt_stat(target, 'prev', 'all')
                        self.cnt_stat(target, 'prev', 'prev_new')
                        self.cnt_stat(target, 'prev', 'task')

                        output = out[:, start:end - step_size]
                        target = target % (end - start - step_size)

                        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                        ans = pred.eq(target.data.view_as(pred)).sum().item()
                        self.correct['cheat'] += ans

                        #pred_5 = torch.topk(output, 5, dim=1)[1]
                        #ans = pred_5.eq(target.data.unsqueeze(1).expand(pred_5.shape)).sum().item()
                        #self.correct_5['cheat'] += ans

                    else:  # new

                        self.cnt_stat(target, 'new', 'all')
                        self.cnt_stat(target, 'new', 'prev_new')
                        self.cnt_stat(target, 'new', 'task')

                        output = out[:, end - step_size:end]
                        target = target % (step_size)

                        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                        ans = pred.eq(target.data.view_as(pred)).sum().item()
                        self.correct['cheat'] += ans

                        #pred_5 = torch.topk(output, 5, dim=1)[1]
                        #ans = pred_5.eq(target.data.unsqueeze(1).expand(pred_5.shape)).sum().item()
                        #self.correct_5['cheat'] += ans
                else:
                    output = model(data)[:, start:end]
                    target = target % (end - start)

                    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                    correct += pred.eq(target.data.view_as(pred)).sum().item()

                    #pred_5 = torch.topk(output, 5, dim=1)[1]
                    #correct_5 += pred_5.eq(target.data.unsqueeze(1).expand(pred_5.shape)).sum().item()

            if mode == 'test' and end > step_size:

                for head in ['all', 'prev_new', 'task', 'cheat']:
                    self.correct[head] = 100. * self.correct[head] / total
                    #self.correct_5[head] = 100. * self.correct_5[head] / total
                self.stat['all'][6] = self.stat['prev_new'][6] = self.stat['task'][6] = total
                return self.correct, self.stat

            return 100. * correct / total

    def make_pred(self, out):
        start, end, step_size = self.start, self.end, self.step_size
        self.pred = {}
        self.pred_5 = {}
        self.pred['all'] = out.data.max(1, keepdim=True)[1]
        self.pred_5['all'] = torch.topk(out, 5, dim=1)[1]

        prev_out = out[:, start:end - step_size]
        curr_out = out[:, end - step_size:end]

        prev_soft = F.softmax(prev_out, dim=1)
        curr_soft = F.softmax(curr_out, dim=1)

        output = torch.cat((prev_soft, curr_soft), dim=1)

        self.pred['prev_new'] = output.data.max(1, keepdim=True)[1]
        if step_size >= 5 :
            self.pred_5['prev_new'] = torch.topk(output, 5, dim=1)[1]

        soft_arr = []
        for t in range(start, end, step_size):
            temp_out = out[:, t:t + step_size]
            temp_soft = F.softmax(temp_out, dim=1)
            soft_arr += [temp_soft]

        output = torch.cat(soft_arr, dim=1)

        self.pred['task'] = output.data.max(1, keepdim=True)[1]
        if step_size >= 5:
            self.pred_5['task'] = torch.topk(output, 5, dim=1)[1]

        return

    def cnt_stat(self, target, mode, head):
        start, end, step_size = self.start, self.end, self.step_size
        pred = self.pred[head]
        if step_size >= 5:
            pred_5 = self.pred_5[head]

        self.correct[head] += pred.eq(target.data.view_as(pred)).sum().item()
        if step_size >= 5 :
            self.correct_5[head] += pred_5.eq(target.data.unsqueeze(1).expand(pred_5.shape)).sum().item()

        if mode == 'prev':
            cp_ = pred.eq(target.data.view_as(pred)).sum()
            epn_ = (pred >= end - step_size).int().sum()
            epp_ = (self.batch_size - (cp_ + epn_))
            self.stat[head][0] += cp_.item()
            self.stat[head][1] += epp_.item()
            self.stat[head][2] += epn_.item()
        else:
            cn_ = pred.eq(target.data.view_as(pred)).cpu().sum()
            enp_ = (pred.cpu().numpy() < end - step_size).sum()
            enn_ = (self.batch_size - (cn_ + enp_))
            self.stat[head][3] += cn_.item()
            self.stat[head][4] += enn_.item()
            self.stat[head][5] += enp_.item()
        return