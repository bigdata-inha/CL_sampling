from __future__ import print_function

import copy
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import models.model_resnet
import trainer
import trainer.trainer_warehouse as trainer_warehouse

class Trainer(trainer_warehouse.GenericTrainer):
    def __init__(self, trainDataIterator, testDataIterator, dataset, model, args, optimizer):
        super().__init__(trainDataIterator, testDataIterator, dataset, model, args, optimizer)
        self.loss = torch.nn.CrossEntropyLoss(reduction='mean')

        self.training_output = torch.tensor([])
        self.training_target = torch.tensor([])

        self.old_weight = []
        self.new_weight = []

        self.cumulative_training_acc = torch.tensor([]).cuda()
        self.cumulative_training_target = torch.tensor([]).cuda()
        '''
        self.count_forgetting_list = []

        self.prev_forgetting = np.zeros(self.train_data_iterator.dataset.current_len)
        self.new_forgetting = np.zeros(self.train_data_iterator.dataset.current_len)
        self.count_forgetting = np.zeros(self.train_data_iterator.dataset.current_len)
        '''

        self.class_weight = np.array([]) # array for the sampling the class

    def update_lr(self, epoch, schedule):
        for temp in range(0, len(schedule)):
            if schedule[temp] == epoch:
                for param_group in self.optimizer.param_groups:
                    self.current_lr = param_group['lr']
                    param_group['lr'] = self.current_lr * self.args.gammas[temp]
                    print("Changing learning rate from %0.4f to %0.4f" % (self.current_lr,
                                                                          self.current_lr * self.args.gammas[temp]))
                    self.current_lr *= self.args.gammas[temp]

    def increment_classes(self, mode):
        if mode == "imbal" :
            #class weight 구하는 부분 수정 필요
            self.class_weight = self.get_class_weight(target=self.cumulative_training_target, outputs=self.cumulative_training_acc, epochs=100)
            self.train_data_iterator.dataset.imbal_update_exemplar(class_weight=self.class_weight)
        elif mode == "forgetting_bal" :
            self.train_data_iterator.dataset.update_exemplar_by_forgetting()

        elif mode == "forgetting_imbal" :
            self.class_weight = self.get_class_weight(target=self.cumulative_training_target,
                                                      outputs=self.cumulative_training_acc, epochs=80)
            self.train_data_iterator.dataset.imbal_update_exemplar_by_forgetting(class_weight = self.class_weight)
        else :
            self.train_data_iterator.dataset.update_exemplar()

        self.train_data_iterator.dataset.task_change()
        self.test_data_iterator.dataset.task_change()


    def setup_training(self, lr):
        for param_group in self.optimizer.param_groups:
            print("Setting LR to %0.4f" % lr)
            param_group['lr'] = lr
            self.current_lr = lr

    def update_frozen_model(self):
        self.model.eval()
        self.model_fixed = copy.deepcopy(self.model)
        self.model_fixed.eval()

        for param in self.model_fixed.parameters():
            param.requires_grad = False

    def weight_align(self):
        end = self.train_data_iterator.dataset.end
        start = end - self.args.step_size
        #weight = self.model.module.fc.weight.data
        weight = self.model.fc.weight.data

        prev = weight[:start, :]
        new = weight[start:end, :]

        self.old_weight.append(prev)
        self.new_weight.append(new)

        print(prev.shape, new.shape)
        mean_prev = torch.mean(torch.norm(prev, dim=1)).item()
        mean_new = torch.mean(torch.norm(new, dim=1)).item()

        gamma = mean_prev / mean_new
        print(mean_prev, mean_new, gamma)
        new = new * gamma
        result = torch.cat((prev, new), dim=0)
        weight[:end, :] = result

        print(torch.mean(torch.norm(self.model.fc.weight.data[:start], dim=1)).item())

        print(torch.mean(torch.norm(self.model.fc.weight.data[start:end], dim=1)).item())

    def train(self, epoch):

        T = 2
        self.model.train()
        print("Epochs %d" % epoch)

        tasknum = self.train_data_iterator.dataset.t
        end = self.train_data_iterator.dataset.end
        start = end - self.args.step_size

        lamb = start / end

        self.training_target = torch.tensor([]).cuda()
        self.training_output = torch.tensor([]).cuda()

        for data, target, traindata_idx in tqdm(self.train_data_iterator):
            target = target.type(dtype=torch.long)
            data, target = data.cuda(), target.cuda()

            output = self.model(data)[:, :end]
            loss_CE = self.loss(output, target)

            _,predicted = output.max(1)
            loss_KD = 0

            #***count forgetting rate**** very important
            correct_idx = np.array(torch.where(predicted.eq(target) == True)[0].cpu())

            wrong_idx = np.array(torch.where(predicted.eq(target) == False)[0].cpu())

            correct_idx = traindata_idx[correct_idx]
            wrong_idx = traindata_idx[wrong_idx]

            if epoch == 1 :
                self.train_data_iterator.dataset.prev_forgetting[correct_idx] = 1

            elif epoch > 1 :
                self.train_data_iterator.dataset.new_forgetting[correct_idx] = 1
                self.train_data_iterator.dataset.new_forgetting[wrong_idx] = 0


            if tasknum > 0:
                end_KD = start
                start_KD = end_KD - self.args.step_size
                prev_KD = start_KD - self.args.step_size
                #score = self.model_fixed(data)[:, :end_KD].data
                score = self.model_fixed(data)[:, prev_KD:end_KD].data

                soft_target = F.softmax(score / T, dim=1)
                #output_log = F.log_softmax(output[:, :end_KD] / T, dim=1)
                output_log = F.log_softmax(output[:, prev_KD:end_KD] / T, dim=1)
                loss_KD = F.kl_div(output_log, soft_target, reduction='batchmean')


            if epoch < 100 :
                self.training_output = torch.cat(
                    (self.training_output.type(dtype=torch.long), predicted)
                    , dim=0)

                self.training_target = torch.cat(
                    (self.training_target.type(dtype=torch.long), target)
                    , dim=0)

            self.optimizer.zero_grad()

            #loss_CE.backward() #without distillation loss

            (lamb * loss_KD + (1 - lamb) * loss_CE).backward()

            self.optimizer.step()

            #self.model.module.fc.bias.data[:] = 0
            #self.model.fc.bias.data[:] = 0

            # weight cliping 0인걸 없애기
            #weight = self.model.module.fc.weight.data
            #weight = self.model.fc.weight.data

            # print(weight.shape)
            #weight[weight < 0] = 0

        ###################################### follow up the epoch##############################################
        if epoch > 1 :
            for j in range(len(self.train_data_iterator.dataset.prev_forgetting)):
                if self.train_data_iterator.dataset.prev_forgetting[j] == 1 and self.train_data_iterator.dataset.new_forgetting[j] == 0:
                    self.train_data_iterator.dataset.count_forgetting[j] = self.train_data_iterator.dataset.count_forgetting[j] + 1

            self.train_data_iterator.dataset.prev_forgetting = self.train_data_iterator.dataset.new_forgetting.copy()


        if epoch < 50 :
            self.cumulative_training_acc = torch.cat((self.cumulative_training_acc.type(dtype=torch.long),
                                                 self.training_output), dim=0)
            self.cumulative_training_target = torch.cat((self.cumulative_training_target.type(dtype=torch.long),
                                                    self.training_target), dim=0)


    def get_class_weight(self, target, outputs, epochs):
        self.cumulative_training_acc = torch.tensor([]).cuda()
        self.cumulative_training_target = torch.tensor([]).cuda()

        stacked = torch.stack(
            (target, outputs), dim=1
        )
        cmt = torch.zeros(self.train_data_iterator.dataset.end, self.train_data_iterator.dataset.end, dtype=torch.int64)

        for p in stacked:
            tl, pl = p.tolist()
            cmt[tl, pl] = cmt[tl, pl] + 1

        cmt = cmt / epochs

        old_class_weight = []
        new_class_weight = []
        tasknum = self.train_data_iterator.dataset.t


        for i in range(self.train_data_iterator.dataset.start) :
            old_class_weight.append(cmt[i,i])

        for i in range(self.train_data_iterator.dataset.start, self.train_data_iterator.dataset.end) :
            new_class_weight.append(cmt[i,i])

        old_class_weight = np.asarray(old_class_weight)
        new_class_weight = np.asarray(new_class_weight)

        old_class_weight = self.normalize(old_class_weight)
        new_class_weight = self.normalize(new_class_weight)



        if tasknum == 0 :
            total_class_weight = new_class_weight
        else :
            old_class_weight = old_class_weight * (tasknum / (tasknum+1))
            new_class_weight = new_class_weight * (1 / (tasknum+1))

            total_class_weight = np.concatenate((old_class_weight, new_class_weight))
            #total_class_weight = total_class_weight / 2

        print(total_class_weight)
        return total_class_weight



    def normalize(self, input):
        output = 1 / (input + 5e-8)
        output = output/np.sum(output)
        return output

    def forgetting_count_class(self, forgetting_count):
        class_index_list = []
        class_forgetting_count = []
        cls_sb_hard_idx = []
        end = self.train_data_iterator.dataset.end
        temp = 0
        for i in (self.train_loader.labels_arr) :

            class_index_list.append(np.where(self.train_loader.labels==i))
            print(class_index_list[temp])
            class_forgetting_count.append(forgetting_count[class_index_list[temp]])
            rm_class_forget = np.where(class_forgetting_count[temp] < 4)[0]
            temp_argsort = np.argsort(class_forgetting_count[temp][rm_class_forget])  # ascending order

            temp_index = np.random.choice(np.arange(0, int(len(temp_argsort))), int(2000/end),
                                          replace=False)
            # print(temp_index, int(len(temp_argsort) * 0.7), int(class_weight[k]))
            temp_argsort = rm_class_forget[temp_index]

            # temp_value = int(5000-int(class_weight[k])) #sb hard index
            # temp_argsort = temp_argsort[temp_value:]   ##sb hard index

            temp_class_index = class_index_list[temp][temp_argsort]

            cls_sb_hard_idx.append(temp_class_index)
            temp += 1
        print(cls_sb_hard_idx)

        return cls_sb_hard_idx





