import copy
import logging
import time
import math

import numpy as np
import torch
import torch.utils.data as td
from sklearn.utils import shuffle
from PIL import Image
from torch.autograd import Variable
import torchvision.transforms.functional as trnF


class FullLoader(td.Dataset):
    def __init__(self, data, labels, classes, step_size, mem_sz, mode, transform=None, loader=None, shuffle_idx=None,
                 base_classes=50, approach='bic', model=None):
        if shuffle_idx is not None:
            # label shuffle
            print("Label shuffled")
            labels = shuffle_idx[labels]

        sort_index = np.argsort(labels)
        self.data = data[sort_index]

        self.model = model

        labels = np.array(labels)
        self.labels = labels[sort_index]
        self.labelsNormal = np.copy(self.labels)
        self.transform = transform
        self.loader = loader
        self.total_classes = classes

        # Imagenet에서는 class shuffle 후 label < current_class 에서 argmin을 찾으면 length 출력 가능하다.

        self.step_size = step_size
        self.base_classes = base_classes
        self.t = 0

        self.start = 0
        self.end = base_classes

        self.start_idx = 0
        self.end_idx = np.argmax(self.labelsNormal > (self.end - 1))  # end data index

        if self.end == classes:
            self.end_idx = len(labels) - 1

        self.tr_idx = range(self.end_idx)
        self.len = len(self.tr_idx)


    def task_change(self, t=0):
        self.t = t

        if t == 0:
            self.end = self.base_classes
        else:
            self.end = self.base_classes + self.step_size * t

        self.end_idx = np.argmax(self.labelsNormal > (self.end - 1))  # end data index
        if self.end_idx == 0:
            self.end_idx = self.labels.shape[0]

        self.tr_idx = range(self.start_idx, self.end_idx)

        self.len = len(self.tr_idx)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        #         time.sleep(0.1)
        index = self.tr_idx[index]

        img = self.data[index]

        try:
            img = Image.fromarray(img)
        except:
            img = self.loader(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, self.labelsNormal[index]


class IncrementalLoader(td.Dataset):
    def __init__(self, data, labels, classes, step_size, mem_sz, mode, transform=None, loader=None, shuffle_idx=None,
                 base_classes=50, approach='bic', model=None):
        if shuffle_idx is not None:
            # label shuffle
            print("Label shuffled")
            print(labels)
            labels = shuffle_idx[labels]
            print(labels)
            #labels = shuffle_idx

        sort_index = np.argsort(labels)
        self.temp_sort_index = sort_index
        self.data = data[sort_index]

        self.model = model

        labels = np.array(labels)

        self.labels_arr = np.arange(0,100)  # to cite for the labels list
        self.labels = labels[sort_index]

        self.labelsNormal = np.copy(self.labels)

        self.transform = transform
        self.loader = loader
        self.total_classes = classes


        # Imagenet에서는 class shuffle 후 label < current_class 에서 argmin을 찾으면 length 출력 가능하다.

        self.step_size = step_size
        self.base_classes = base_classes
        self.t = 0

        self.mem_sz = mem_sz
        self.validation_buffer_size = int(mem_sz / 10) * 2
        self.mode = mode

        self.start = 0
        self.end = base_classes

        self.start_idx = 0
        self.end_idx = np.argmax(self.labelsNormal > (self.end - 1))  # end data index

        if self.end == classes:
            self.end_idx = len(labels) - 1

        self.tr_idx = range(self.end_idx)
        self.len = len(self.tr_idx)
        self.current_len = self.len

        self.approach = approach
        self.memory_buffer = []
        self.exemplar = []
        self.validation_buffer = []
        self.bft_buffer = []
        self.start_point = []
        self.end_point = []

        #variables to record the forgetting event
        self.count_forgetting = np.zeros(len(self.data))
        self.count_forgetting_list = []

        self.prev_forgetting = np.zeros(len(self.data))
        self.new_forgetting = np.zeros(len(self.data))

        self.class_index_list = []
        self.class_forgetting_count = []
        self.cls_sb_hard_idx = []

        for i in range(classes):
            self.start_point.append(np.argmin(self.labelsNormal < i))
            self.end_point.append(np.argmax(self.labelsNormal > (i)))
            self.memory_buffer.append([])

        self.end_point[-1] = len(labels)

    def task_change(self):
        self.t += 1

        self.start = self.end
        self.end += self.step_size
        print(self.end)
        self.start_idx = np.argmin(self.labelsNormal < self.start)  # start data index
        self.end_idx = np.argmax(self.labelsNormal > (self.end - 1))  # end data index
        if self.end_idx == 0:
            self.end_idx = self.labels.shape[0]

        self.tr_idx = range(self.start_idx, self.end_idx)

        # validation set for bic
        if 'bic' in self.approach and self.start < self.total_classes and self.mode != 'test':
            val_per_class = (self.validation_buffer_size // 2) // self.step_size
            self.tr_idx = []
            for i in range(self.step_size):
                end = self.end_point[self.start + i]
                start = self.start_point[self.start + i]
                self.validation_buffer += range(end - val_per_class, end)
                self.tr_idx += range(start, end - val_per_class)

        self.len = len(self.tr_idx)
        self.current_len = self.len

        if self.approach != 'ssil' and self.approach != 'lwf':
            self.len += len(self.exemplar)


        self.prev_forgetting = np.zeros(len(self.data))
        self.new_forgetting = np.zeros(len(self.data))
        self.count_forgetting = np.zeros(len(self.data))

    def update_bft_buffer(self):
        self.bft_buffer = copy.deepcopy(self.memory_buffer)
        min_len = 1e8
        for arr in self.bft_buffer:
            min_len = min(min_len, len(arr))

        # buffer_per_class = math.ceil(self.mem_sz / (self.end-self.step_size))
        buffer_per_class = min_len

        for i in range(self.start, self.end):
            start_idx = self.start_point[i]
            end_idx = self.end_point[i]
            idx = shuffle(np.arange(end_idx - start_idx), random_state=self.t)[:buffer_per_class]
            #             self.bft_buffer[i] += range(start_idx, start_idx+buffer_per_class)
            self.bft_buffer[i] += list(idx)
        for arr in self.bft_buffer:
            if len(arr) > buffer_per_class:
                arr.pop()

        self.bft_exemplar = []
        for arr in self.bft_buffer:
            self.bft_exemplar += arr

    def update_exemplar(self, forgetting_count=None):
        buffer_per_class = math.ceil(self.mem_sz / self.end)

        buffer_per_class = 20
        # first, add new exemples

        for i in range(self.start, self.end):
            start_idx = self.start_point[i]
            self.memory_buffer[i] += range(start_idx, start_idx + buffer_per_class)

        # second, throw away the previous samples
        if buffer_per_class > 0:
            for i in range(self.start):
                if len(self.memory_buffer[i]) > buffer_per_class:
                    del self.memory_buffer[i][buffer_per_class:]

        #verify it
        for i in range(self.end) :
            print(self.memory_buffer[i].__len__())

        # third, select classes from previous classes, and throw away only 1 samples per class
        # randomly select classes. **random seed = self.t or start** <-- IMPORTANT!

        length = sum([len(i) for i in self.memory_buffer])
        remain = length - self.mem_sz
        if remain > 0:
            imgs_per_class = [len(i) for i in self.memory_buffer]
            selected_classes = np.argsort(imgs_per_class)[-remain:]
            for c in selected_classes:
                self.memory_buffer[c].pop()

        self.exemplar = []
        for arr in self.memory_buffer:
            self.exemplar += arr

        # validation set for bic
        if 'bic' in self.approach:
            self.bic_memory_buffer = copy.deepcopy(self.memory_buffer)
            self.validation_buffer = []
            validation_per_class = (self.validation_buffer_size // 2) // self.end
            if validation_per_class > 0:
                for i in range(self.end):
                    self.validation_buffer += self.bic_memory_buffer[i][-validation_per_class:]
                    del self.bic_memory_buffer[i][-validation_per_class:]

            remain = self.validation_buffer_size // 2 - validation_per_class * self.end

            if remain > 0:
                imgs_per_class = [len(i) for i in self.bic_memory_buffer]
                selected_classes = np.argsort(imgs_per_class)[-remain:]
                for c in selected_classes:
                    self.validation_buffer.append(self.bic_memory_buffer[c].pop())
            self.exemplar = []
            for arr in self.bic_memory_buffer:
                self.exemplar += arr

    def imbal_update_exemplar(self, class_weight=None, forgetting_count=None):

        #class_weight = np.arra

        buffer_per_class = math.ceil(self.mem_sz / self.end)

        #class_min = int(buffer_per_class/5)
        class_min = int(buffer_per_class / 5)

        shared_memory = (buffer_per_class - class_min) * self.end


        #for the unbalanced but reasonable data set
        class_weight = class_min + np.array(class_weight * shared_memory)

        # first, add new exemples
        buffer_per_class = class_weight.astype(np.int)
        print("start : {}, end : {}".format(self.start, self.end))
        for i in range(self.end) :
            print(self.labels_arr[i])

        for i in range(self.start, self.end):
            start_idx = self.start_point[i]

            self.memory_buffer[i] += range(start_idx, start_idx + buffer_per_class[i])


        # second, throw away the previous samples
        if buffer_per_class.all() > 0:
            for i in range(self.start):
                if len(self.memory_buffer[i]) > buffer_per_class[i]:
                    del self.memory_buffer[i][buffer_per_class[i]:]
        # third, select classes from previous classes, and throw away only 1 samples per class
        # randomly select classes. **random seed = self.t or start** <-- IMPORTANT!
        length = sum([len(i) for i in self.memory_buffer])
        remain = length - self.mem_sz

        total_mem = 0
        for i in range(self.end) :
            total_mem = total_mem + buffer_per_class[i]

        print(total_mem)

        if remain > 0:
            imgs_per_class = [len(i) for i in self.memory_buffer]
            selected_classes = np.argsort(imgs_per_class)[-remain:]
            for c in selected_classes:
                self.memory_buffer[c].pop()

        self.exemplar = []
        for arr in self.memory_buffer:
            self.exemplar += arr

        # validation set for bic
        if 'bic' in self.approach:
            self.bic_memory_buffer = copy.deepcopy(self.memory_buffer)
            self.validation_buffer = []
            validation_per_class = (self.validation_buffer_size // 2) // self.end
            if validation_per_class > 0:
                for i in range(self.end):
                    self.validation_buffer += self.bic_memory_buffer[i][-validation_per_class:]
                    del self.bic_memory_buffer[i][-validation_per_class:]

            remain = self.validation_buffer_size // 2 - validation_per_class * self.end

            if remain > 0:
                imgs_per_class = [len(i) for i in self.bic_memory_buffer]
                selected_classes = np.argsort(imgs_per_class)[-remain:]
                for c in selected_classes:
                    self.validation_buffer.append(self.bic_memory_buffer[c].pop())
            self.exemplar = []
            for arr in self.bic_memory_buffer:
                self.exemplar += arr

    def imbal_update_exemplar_by_forgetting(self, class_weight=None):

        self.count_forgetting_list.append(self.count_forgetting)

        buffer_per_class = math.ceil(self.mem_sz / self.end)

        print(buffer_per_class)
        class_min = int(buffer_per_class / 5)

        shared_memory = (buffer_per_class - class_min) * self.end

        #for the unbalanced but reasonable data set
        class_weight = class_min + np.array(class_weight * shared_memory)


        buffer_per_class = class_weight.astype(np.int)
        print("start : {}, end : {}".format(self.start, self.end))


        temp = self.start

        #easy forgetting
        for i in (self.labels_arr[self.start:self.end]) :

            self.class_index_list.append(np.where(self.labelsNormal==i)[0])

            self.class_forgetting_count.append(self.count_forgetting[self.class_index_list[temp]])
            ######################################################################################################################################################
            '''
            #selected by random selection
            rm_class_forget = np.where(self.class_forgetting_count[temp] < self.class_forgetting_count[temp].mean())[0]  #easy random sample
            print(self.class_forgetting_count[temp].mean())

            #rm_class_forget = np.where(self.class_forgetting_count[temp] > np.median(self.class_forgetting_count[temp]))[0]  # hard random sample
            #print(np.median(self.class_forgetting_count[temp]))

            temp_argsort = np.argsort(self.class_forgetting_count[temp][rm_class_forget])  # ascending order

            temp_index = np.random.choice(np.arange(0, int(len(temp_argsort))), int(buffer_per_class),
                                          replace=False)

            temp_argsort = rm_class_forget[temp_index]
            '''
            #####################################################################################################################################################
            #selected by following order
            rm_class_forget = self.class_forgetting_count[temp]
            #temp_argsort = np.argsort(rm_class_forget)# ascending order // select the hard one
            temp_argsort = np.argsort(rm_class_forget)[::-1] # descending order // select the easy one

            temp_value = int(500 - int(buffer_per_class[temp]))
            temp_argsort = temp_argsort[temp_value:]

            temp_class_index = self.class_index_list[temp][temp_argsort]

            self.cls_sb_hard_idx.append(temp_class_index)
            #print("forgetting count : ", cls_sb_hard_idx[temp])
            temp += 1

        # first, add new exemples
        for i in range(self.start, self.end):
            start_idx = self.start_point[i]

            self.memory_buffer[i] = self.cls_sb_hard_idx[i].copy()

        # second, throw away the previous samples
        if buffer_per_class.all() > 0:
            for i in range(self.start):
                if len(self.memory_buffer[i]) > buffer_per_class[i]:
                    self.memory_buffer[i] = self.memory_buffer[i][:buffer_per_class[i]]
        # third, select classes from previous classes, and throw away only 1 samples per class
        # randomly select classes. **random seed = self.t or start** <-- IMPORTANT!
        length = sum([len(i) for i in self.memory_buffer])
        remain = length - self.mem_sz

        total_mem = 0
        for i in range(self.end) :
            total_mem = total_mem + buffer_per_class[i]

        print(total_mem)

        if remain > 0:
            imgs_per_class = [len(i) for i in self.memory_buffer]
            selected_classes = np.argsort(imgs_per_class)[-remain:]
            for c in selected_classes:
                self.memory_buffer[c] = np.delete(self.memory_buffer[c], (self.memory_buffer[c]-1))
                #self.memory_buffer[c] = self.memory_buffer[c].pop()

        self.exemplar = []
        for arr in self.memory_buffer[:self.end]:
            print(arr)
            arr = arr.tolist()
            self.exemplar += arr

        print(self.exemplar)
        #self.exemplar = np.concatenate(self.exemplar)
        #print(self.exemplar)


        # validation set for bic
        if 'bic' in self.approach:
            self.bic_memory_buffer = copy.deepcopy(self.memory_buffer)
            self.validation_buffer = []
            validation_per_class = (self.validation_buffer_size // 2) // self.end
            if validation_per_class > 0:
                for i in range(self.end):
                    self.validation_buffer += self.bic_memory_buffer[i][-validation_per_class:]
                    del self.bic_memory_buffer[i][-validation_per_class:]

            remain = self.validation_buffer_size // 2 - validation_per_class * self.end

            if remain > 0:
                imgs_per_class = [len(i) for i in self.bic_memory_buffer]
                selected_classes = np.argsort(imgs_per_class)[-remain:]
                for c in selected_classes:
                    self.validation_buffer.append(self.bic_memory_buffer[c].pop())
            self.exemplar = []
            for arr in self.bic_memory_buffer:
                self.exemplar += arr

    def update_exemplar_by_forgetting(self, class_weight=None, forgetting_count=None):
        self.count_forgetting_list.append(self.count_forgetting)

        buffer_per_class = math.ceil(self.mem_sz / self.end)
        buffer_per_class = 20
        print(buffer_per_class)
        #class_min = int(buffer_per_class / 5)

        #shared_memory = (buffer_per_class - class_min) * self.end
        #for the unbalanced but reasonable data set
        #class_weight = class_min + np.array(class_weight * shared_memory)


        #buffer_per_class = class_weight.astype(np.int)
        print("start : {}, end : {}".format(self.start, self.end))


        temp = self.start

        #easy forgetting
        for i in (self.labels_arr[self.start:self.end]) :

            self.class_index_list.append(np.where(self.labelsNormal==i)[0])

            self.class_forgetting_count.append(self.count_forgetting[self.class_index_list[temp]])

            ######################################################################################################################################################
            '''
            #selected by random selection
            rm_class_forget = np.where(self.class_forgetting_count[temp] < self.class_forgetting_count[temp].mean())[0]  #easy random sample
            print(self.class_forgetting_count[temp].mean())

            #rm_class_forget = np.where(self.class_forgetting_count[temp] > np.median(self.class_forgetting_count[temp]))[0]  # hard random sample
            #print(np.median(self.class_forgetting_count[temp]))

            temp_argsort = np.argsort(self.class_forgetting_count[temp][rm_class_forget])  # ascending order

            temp_index = np.random.choice(np.arange(0, int(len(temp_argsort))), int(buffer_per_class),
                                          replace=False)

            temp_argsort = rm_class_forget[temp_index]
            '''
            #####################################################################################################################################################

            #selected by following order
            rm_class_forget = self.class_forgetting_count[temp]
            #temp_argsort = np.argsort(rm_class_forget)# ascending order // select the hard one
            temp_argsort = np.argsort(rm_class_forget)[::-1] # descending order // select the easy one

            temp_value = int(500 - int(buffer_per_class))
            temp_argsort = temp_argsort[temp_value:]

            temp_class_index = self.class_index_list[temp][temp_argsort]

            self.cls_sb_hard_idx.append(temp_class_index)
            #print("forgetting count : ", cls_sb_hard_idx[temp])
            temp += 1

        # first, add new exemples
        for i in range(self.start, self.end):
            start_idx = self.start_point[i]

            self.memory_buffer[i] = self.cls_sb_hard_idx[i].copy()

        # second, throw away the previous samples
        if buffer_per_class > 0:
            for i in range(self.start):
                if len(self.memory_buffer[i]) > buffer_per_class:
                    self.memory_buffer[i] = self.memory_buffer[i][:buffer_per_class]
        # third, select classes from previous classes, and throw away only 1 samples per class
        # randomly select classes. **random seed = self.t or start** <-- IMPORTANT!
        length = sum([len(i) for i in self.memory_buffer])
        remain = length - self.mem_sz

        total_mem = 0
        for i in range(self.end) :
            total_mem = total_mem + buffer_per_class

        print(total_mem)

        if remain > 0:
            imgs_per_class = [len(i) for i in self.memory_buffer]
            selected_classes = np.argsort(imgs_per_class)[-remain:]
            for c in selected_classes:
                self.memory_buffer[c] = np.delete(self.memory_buffer[c], (self.memory_buffer[c]-1))
                #self.memory_buffer[c] = self.memory_buffer[c].pop()

        self.exemplar = []

        for arr in self.memory_buffer[:self.end]:
            print(arr)
            arr = arr.tolist()
            self.exemplar += arr

        print(self.exemplar)
        #self.exemplar = np.concatenate(self.exemplar)
        #print(self.exemplar)


        # validation set for bic
        if 'bic' in self.approach:
            self.bic_memory_buffer = copy.deepcopy(self.memory_buffer)
            self.validation_buffer = []
            validation_per_class = (self.validation_buffer_size // 2) // self.end
            if validation_per_class > 0:
                for i in range(self.end):
                    self.validation_buffer += self.bic_memory_buffer[i][-validation_per_class:]
                    del self.bic_memory_buffer[i][-validation_per_class:]

            remain = self.validation_buffer_size // 2 - validation_per_class * self.end

            if remain > 0:
                imgs_per_class = [len(i) for i in self.bic_memory_buffer]
                selected_classes = np.argsort(imgs_per_class)[-remain:]
                for c in selected_classes:
                    self.validation_buffer.append(self.bic_memory_buffer[c].pop())
            self.exemplar = []
            for arr in self.bic_memory_buffer:
                self.exemplar += arr

    def __len__(self):
        if self.mode == 'train':
            return self.len
        elif self.mode == 'bias':
            return len(self.validation_buffer)
        elif self.mode == 'b-ft':
            return len(self.bft_exemplar)
        else:
            return self.end_idx

    def __getitem__(self, index):
        #         time.sleep(0.1)
        temp_index = 0
        if self.mode == 'train':
            if index >= self.current_len:  # for bic, ft, icarl, il2m
                index = self.exemplar[index - self.current_len]
                temp_index = index
            else:
                temp_index = index
                index = self.tr_idx[index]

        elif self.mode == 'bias':  # for bic bias correction
            index = self.validation_buffer[index]
        elif self.mode == 'b-ft':
            index = self.bft_exemplar[index]

        img = self.data[index]

        try:
            img = Image.fromarray(img)
        except:
            img = self.loader(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, self.labelsNormal[index], index


class myIncrementalLoader(td.Dataset):
    def __init__(self, data, labels, classes, step_size, mem_sz, mode, transform=None, loader=None, shuffle_idx=None,
                 base_classes=50, approach='bic', model=None):
        if shuffle_idx is not None:
            # label shuffle
            print("Label shuffled")
            labels = shuffle_idx[labels]
            #labels = shuffle_idx

        sort_index = np.argsort(labels)
        self.data = data[sort_index]

        self.model = model

        labels = np.array(labels)

        self.labels_arr = labels  # to cite for the labels list
        self.labels = labels[sort_index]
        self.labelsNormal = np.copy(self.labels)
        self.transform = transform
        self.loader = loader
        self.total_classes = classes


        # Imagenet에서는 class shuffle 후 label < current_class 에서 argmin을 찾으면 length 출력 가능하다.

        self.step_size = step_size
        self.base_classes = base_classes
        self.t = 0   #task number

        self.mem_sz = mem_sz
        self.validation_buffer_size = int(mem_sz / 10) * 2
        self.mode = mode

        self.start = 0
        self.end = base_classes

        self.start_idx = 0
        self.end_idx = np.argmax(self.labelsNormal > (self.end - 1))  # end data index

        if self.end == classes:
            self.end_idx = len(labels) - 1

        self.tr_idx = range(self.end_idx)
        self.len = len(self.tr_idx)
        self.current_len = self.len

        self.approach = approach
        self.memory_buffer = []
        self.exemplar = []
        self.validation_buffer = []
        self.bft_buffer = []
        self.start_point = []
        self.end_point = []

        self.cur_newData_X = []
        self.cur_newData_Y = []



        for i in range(classes):
            self.start_point.append(np.argmin(self.labelsNormal < i))
            self.end_point.append(np.argmax(self.labelsNormal > (i)))
            self.memory_buffer.append([])
        self.end_point[-1] = len(labels)

    def task_change(self):
        self.t += 1

        self.start = self.end
        self.end += self.step_size

        self.start_idx = np.argmin(self.labelsNormal < self.start)  # start data index
        self.end_idx = np.argmax(self.labelsNormal > (self.end - 1))  # end data index
        if self.end_idx == 0:
            self.end_idx = self.labels.shape[0]

        self.tr_idx = range(self.start_idx, self.end_idx)

        # validation set for bic
        if 'bic' in self.approach and self.start < self.total_classes and self.mode != 'test':
            val_per_class = (self.validation_buffer_size // 2) // self.step_size
            self.tr_idx = []
            for i in range(self.step_size):
                end = self.end_point[self.start + i]
                start = self.start_point[self.start + i]
                self.validation_buffer += range(end - val_per_class, end)
                self.tr_idx += range(start, end - val_per_class)

        self.len = len(self.tr_idx)
        self.current_len = self.len

        if self.approach != 'ssil' and self.approach != 'lwf':
            self.len += len(self.exemplar)

    def construct_dataset(self):


        for i in self.labels_arr : #new dataset construction
            newDataidx = np.where(self.labelsNormal==i)[0]
            self.cur_newData_X.append(self.data[newDataidx])
            self.cur_newData_Y.append(self.labelsNormal[newDataidx])


    '''
    def update_bft_buffer(self):
        self.bft_buffer = copy.deepcopy(self.memory_buffer)
        min_len = 1e8
        for arr in self.bft_buffer:
            min_len = min(min_len, len(arr))

        # buffer_per_class = math.ceil(self.mem_sz / (self.end-self.step_size))
        buffer_per_class = min_len

        for i in range(self.start, self.end):
            start_idx = self.start_point[i]
            end_idx = self.end_point[i]
            idx = shuffle(np.arange(end_idx - start_idx), random_state=self.t)[:buffer_per_class]
            #             self.bft_buffer[i] += range(start_idx, start_idx+buffer_per_class)
            self.bft_buffer[i] += list(idx)
        for arr in self.bft_buffer:
            if len(arr) > buffer_per_class:
                arr.pop()

        self.bft_exemplar = []
        for arr in self.bft_buffer:
            self.bft_exemplar += arr
    '''
    def update_exemplar(self, forgetting_count):

        buffer_per_class = math.ceil(self.mem_sz / self.end)
        # first, add new exemples

        for i in range(self.start, self.end):
            start_idx = self.start_point[i]
            self.memory_buffer[i] += range(start_idx, start_idx + buffer_per_class)

        # second, throw away the previous samples
        if buffer_per_class > 0:
            for i in range(self.start):
                if len(self.memory_buffer[i]) > buffer_per_class:
                    del self.memory_buffer[i][buffer_per_class:]


        #verify it
        for i in range(self.end) :
            print(self.memory_buffer[i].__len__())

        # third, select classes from previous classes, and throw away only 1 samples per class
        # randomly select classes. **random seed = self.t or start** <-- IMPORTANT!

        length = sum([len(i) for i in self.memory_buffer])
        remain = length - self.mem_sz
        if remain > 0:
            imgs_per_class = [len(i) for i in self.memory_buffer]
            selected_classes = np.argsort(imgs_per_class)[-remain:]
            for c in selected_classes:
                self.memory_buffer[c].pop()

        self.exemplar = []
        for arr in self.memory_buffer:
            self.exemplar += arr

        # validation set for bic
        if 'bic' in self.approach:
            self.bic_memory_buffer = copy.deepcopy(self.memory_buffer)
            self.validation_buffer = []
            validation_per_class = (self.validation_buffer_size // 2) // self.end
            if validation_per_class > 0:
                for i in range(self.end):
                    self.validation_buffer += self.bic_memory_buffer[i][-validation_per_class:]
                    del self.bic_memory_buffer[i][-validation_per_class:]

            remain = self.validation_buffer_size // 2 - validation_per_class * self.end

            if remain > 0:
                imgs_per_class = [len(i) for i in self.bic_memory_buffer]
                selected_classes = np.argsort(imgs_per_class)[-remain:]
                for c in selected_classes:
                    self.validation_buffer.append(self.bic_memory_buffer[c].pop())
            self.exemplar = []
            for arr in self.bic_memory_buffer:
                self.exemplar += arr

    def imbal_update_exemplar(self, class_weight=None, forgetting_count=None):

        #class_weight = np.arra

        buffer_per_class = math.ceil(self.mem_sz / self.end)

        #class_min = int(buffer_per_class/5)
        class_min = int(buffer_per_class / 5)

        shared_memory = (buffer_per_class - class_min) * self.end


        #for the unbalanced but reasonable data set
        class_weight = class_min + np.array(class_weight * shared_memory)

        # first, add new exemples
        buffer_per_class = class_weight.astype(np.int)
        print("start : {}, end : {}".format(self.start, self.end))
        for i in range(self.end) :
            print(self.labels_arr[i])

        for i in range(self.start, self.end):
            start_idx = self.start_point[i]

            self.memory_buffer[i] += range(start_idx, start_idx + buffer_per_class[i])


        # second, throw away the previous samples
        if buffer_per_class.all() > 0:
            for i in range(self.start):
                if len(self.memory_buffer[i]) > buffer_per_class[i]:
                    del self.memory_buffer[i][buffer_per_class[i]:]
        # third, select classes from previous classes, and throw away only 1 samples per class
        # randomly select classes. **random seed = self.t or start** <-- IMPORTANT!
        length = sum([len(i) for i in self.memory_buffer])
        remain = length - self.mem_sz

        total_mem = 0
        for i in range(self.end) :
            total_mem = total_mem + buffer_per_class[i]

        print(total_mem)

        if remain > 0:
            imgs_per_class = [len(i) for i in self.memory_buffer]
            selected_classes = np.argsort(imgs_per_class)[-remain:]
            for c in selected_classes:
                self.memory_buffer[c].pop()

        self.exemplar = []
        for arr in self.memory_buffer:
            self.exemplar += arr

        # validation set for bic
        if 'bic' in self.approach:
            self.bic_memory_buffer = copy.deepcopy(self.memory_buffer)
            self.validation_buffer = []
            validation_per_class = (self.validation_buffer_size // 2) // self.end
            if validation_per_class > 0:
                for i in range(self.end):
                    self.validation_buffer += self.bic_memory_buffer[i][-validation_per_class:]
                    del self.bic_memory_buffer[i][-validation_per_class:]

            remain = self.validation_buffer_size // 2 - validation_per_class * self.end

            if remain > 0:
                imgs_per_class = [len(i) for i in self.bic_memory_buffer]
                selected_classes = np.argsort(imgs_per_class)[-remain:]
                for c in selected_classes:
                    self.validation_buffer.append(self.bic_memory_buffer[c].pop())
            self.exemplar = []
            for arr in self.bic_memory_buffer:
                self.exemplar += arr

    def update_exemplar_by_forgetting(self, class_weight=None, forgetting_count=None):


        buffer_per_class = math.ceil(self.mem_sz / self.end)

        #class_min = int(buffer_per_class / 5)

        #shared_memory = (buffer_per_class - class_min) * self.end
        #for the unbalanced but reasonable data set
        #class_weight = class_min + np.array(class_weight * shared_memory)

        # first, add new exemples
        #buffer_per_class = class_weight.astype(np.int)
        print("start : {}, end : {}".format(self.start, self.end))

        class_index_list = []
        class_forgetting_count = []
        cls_sb_hard_idx = []

        #easy forgetting
        for i in (self.labels) :
            class_index_list.append(torch.where(self.train_loader==i))
            class_forgetting_count.append(forgetting_count[class_index_list[i]])
            rm_class_forget = torch.where(class_forgetting_count[i] < 4)[0]
            temp_argsort = np.argsort(class_forgetting_count[i][rm_class_forget])  # ascending order

            temp_index = np.random.choice(np.arange(0, int(len(temp_argsort))), int(buffer_per_class),
                                          replace=False)

            temp_argsort = rm_class_forget[temp_index]


            temp_class_index = class_index_list[i][temp_argsort]

            cls_sb_hard_idx.append(temp_class_index)


        for i in range(self.start, self.end):
            start_idx = self.start_point[i]
            print(start_idx)
            self.memory_buffer[i] += range(start_idx, start_idx + buffer_per_class)


        # second, throw away the previous samples
        if buffer_per_class.all() > 0:
            for i in range(self.start):
                if len(self.memory_buffer[i]) > buffer_per_class:
                    del self.memory_buffer[i][buffer_per_class:]
        # third, select classes from previous classes, and throw away only 1 samples per class
        # randomly select classes. **random seed = self.t or start** <-- IMPORTANT!
        length = sum([len(i) for i in self.memory_buffer])
        remain = length - self.mem_sz

        total_mem = 0
        for i in range(self.end) :
            total_mem = total_mem + buffer_per_class

        print(total_mem)

        if remain > 0:
            imgs_per_class = [len(i) for i in self.memory_buffer]
            selected_classes = np.argsort(imgs_per_class)[-remain:]
            for c in selected_classes:
                self.memory_buffer[c].pop()

        self.exemplar = []
        for arr in self.memory_buffer:
            self.exemplar += arr

        # validation set for bic
        if 'bic' in self.approach:
            self.bic_memory_buffer = copy.deepcopy(self.memory_buffer)
            self.validation_buffer = []
            validation_per_class = (self.validation_buffer_size // 2) // self.end
            if validation_per_class > 0:
                for i in range(self.end):
                    self.validation_buffer += self.bic_memory_buffer[i][-validation_per_class:]
                    del self.bic_memory_buffer[i][-validation_per_class:]

            remain = self.validation_buffer_size // 2 - validation_per_class * self.end

            if remain > 0:
                imgs_per_class = [len(i) for i in self.bic_memory_buffer]
                selected_classes = np.argsort(imgs_per_class)[-remain:]
                for c in selected_classes:
                    self.validation_buffer.append(self.bic_memory_buffer[c].pop())
            self.exemplar = []
            for arr in self.bic_memory_buffer:
                self.exemplar += arr

    def __len__(self):
        if self.mode == 'train':
            return self.len
        elif self.mode == 'bias':
            return len(self.validation_buffer)
        elif self.mode == 'b-ft':
            return len(self.bft_exemplar)
        else:
            return self.end_idx

    def __getitem__(self, index):
        #         time.sleep(0.1)
        temp_index = 0
        if self.mode == 'train':
            if index >= self.current_len:  # for bic, ft, icarl, il2m
                index = self.exemplar[index - self.current_len]
                temp_index = index
            else:
                temp_index = index
                index = self.tr_idx[index]

        elif self.mode == 'bias':  # for bic bias correction
            index = self.validation_buffer[index]
        elif self.mode == 'b-ft':
            index = self.bft_exemplar[index]

        img = self.data[index]

        try:
            img = Image.fromarray(img)
        except:
            img = self.loader(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, self.labelsNormal[index], temp_index

class ResultLoader(td.Dataset):
    def __init__(self, data, labels, transform=None, loader=None):

        self.data = data
        self.labels = labels
        self.labelsNormal = np.copy(self.labels)
        self.transform = transform
        self.loader = loader
        self.transformLabels()

    def transformLabels(self):
        '''Change labels to one hot coded vectors'''
        b = np.zeros((self.labels.size, self.labels.max() + 1))
        b[np.arange(self.labels.size), self.labels] = 1
        self.labels = b

    def __len__(self):
        return self.labels.shape[0]


    def __getitem__(self, index):
        #         time.sleep(0.1)
        img = self.data[index]
        try:
            img = Image.fromarray(img)
        except:
            img = self.loader(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, self.labelsNormal[index], index


def make_ResultLoaders(data, labels, classes, step_size, transform=None, loader=None, shuffle_idx=None,
                       base_classes=50):
    if shuffle_idx is not None:
        labels = shuffle_idx[labels]
    sort_index = np.argsort(labels)
    data = data[sort_index]
    labels = np.array(labels)
    labels = labels[sort_index]

    start = 0
    end = base_classes

    loaders = []

    while (end <= classes):

        start_idx = np.argmin(labels < start)  # start data index
        end_idx = np.argmax(labels > (end - 1))  # end data index
        if end_idx == 0:
            end_idx = data.shape[0]

        loaders.append(
            ResultLoader(data[start_idx:end_idx], labels[start_idx:end_idx], transform=transform, loader=loader))

        start = end
        end += step_size

    return loaders