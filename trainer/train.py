import torch
import torch.utils.data.dataset as Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from models.myNetwork import *
from myData.iDataset import*
import os
import copy
from torch.nn import functional as F
from PIL import Image
from tqdm import tqdm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class naive_CL() :
    def __init__(self, numclass, feature_extractor, batch_size, task_size, memory_size, epochs, learning_rate):

        super(naive_CL, self).__init__()
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model = network(numclass, feature_extractor)
        self.exemplar_set = []
        self.class_mean_set = []

        self.numclass = numclass    #현재 task 클래스 수

        self.transform = transforms.Compose([  # transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

        self.old_model = None

        self.train_transform = transforms.Compose([  # transforms.Resize(img_size),
            transforms.RandomCrop((32, 32), padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.24705882352941178),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

        self.test_transform = transforms.Compose([  # transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

        self.classify_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=1.),
                                                      # transforms.Resize(img_size),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize((0.5071, 0.4867, 0.4408),
                                                                           (0.2675, 0.2565, 0.2761))])

        self.train_dataset = iCIFAR100('myData', transform=self.train_transform, download=True)
        self.test_dataset = iCIFAR100('myData', test_transform=self.test_transform, train=False, download=True)

        self.batchsize = batch_size
        self.memory_size = memory_size
        self.task_size = task_size
        self.best_acc = 0

        self.train_loader = None
        self.test_loader = None

    def beforeTrain(self):
        #training 하기전 data 정리 및 fc layer 구멍 뚫기
        self.model.eval()
        classes = [self.numclass - self.task_size, self.numclass]

        self.train_loader, self.test_loader = self._get_train_and_test_dataloader(classes)
        if self.numclass > self.task_size:
            self.model.Incremental_learning(self.numclass)
        self.model.train()
        self.model.to(device)


    def _get_train_and_test_dataloader(self, classes):
        self.train_dataset.getTrainData(classes, self.exemplar_set)
        self.test_dataset.getTestData(classes)
        train_loader = DataLoader(dataset=self.train_dataset,
                                  shuffle=True,
                                  batch_size=self.batchsize)

        test_loader = DataLoader(dataset=self.test_dataset,
                                 shuffle=False,  #shuffle = False for cacluate the class accuracy
                                 batch_size=100)

        return train_loader, test_loader

    def train(self, task_id):
        '''
        Training the current step model
        :param task_id: step number
        :return: model accuracy, but
        '''
        accuracy = 0
        opt = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=.00001)
        for epoch in range(self.epochs):
            if epoch == 48:
                if self.numclass == self.task_size:
                    opt = optim.SGD(self.model.parameters(), lr=0.1 / 10, momentum= 0.9, weight_decay=.00001)
                else:
                    for p in opt.param_groups:
                        p['lr'] = self.learning_rate / 10
                    # opt = optim.SGD(self.model.parameters(), lr=self.learning_rate/ 5,weight_decay=0.00001,momentum=0.9,nesterov=True,)
                print("change learning rate:%.3f" % (self.learning_rate / 10))
            elif epoch == 65:
                if self.numclass > self.task_size:
                    for p in opt.param_groups:
                        p['lr'] = self.learning_rate / 100
                    # opt = optim.SGD(self.model.parameters(), lr=self.learning_rate/ 25,weight_decay=0.00001,momentum=0.9,nesterov=True,)
                else:
                    opt = optim.SGD(self.model.parameters(), lr=0.1 / 100, momentum=0.9, weight_decay=0.00001)
                print("change learning rate:%.3f" % (self.learning_rate / 100))
            elif epoch == 85:
                if self.numclass == self.task_size:
                    opt = optim.SGD(self.model.parameters(), lr=0.1 / 1000, momentum=0.9, weight_decay=0.00001)
                else:
                    for p in opt.param_groups:
                        p['lr'] = self.learning_rate / 1000
                    # opt = optim.SGD(self.model.parameters(), lr=self.learning_rate / 125,weight_decay=0.00001,momentum=0.9,nesterov=True,)
                print("change learning rate:%.3f" % (self.learning_rate / 1000))

            # save old model logit to calculate the distillation loss
            if self.old_model != None:
                self.q = torch.zeros(len(self.train_dataset), self.numclass - self.task_size).cuda()
                for indexs, images, labels in self.train_loader:
                    images = images.cuda()
                    indexs = indexs.cuda()
                    #g = F.sigmoid(self.old_model(images))
                    g = torch.sigmoid(self.old_model(images))
                    self.q[indexs] = g.data
                self.q = self.q.cuda()

            for step, (indexs, images, target) in enumerate(self.train_loader):
                images, target = images.cuda(), target.cuda()
                # output = self.model(images)
                loss_value = self._compute_loss(indexs, images, target)
                opt.zero_grad()
                loss_value.backward()
                opt.step()
                #print('epoch:%d,step:%d,loss:%.3f' % (epoch, step, loss_value.item()))
            #print('epoch:%d,step:%d,loss:%.3f' % (epoch, step, loss_value.item()))

            accuracy = self._test(self.test_loader, 1)  # test using test data

            # if best acc save model
            if accuracy > self.best_acc:
                self.best_acc = accuracy
                print("Best accuracy：" + str(accuracy.item()) + "Saving..")
                state = {
                    'net': self.model.state_dict(),
                    'acc': self.best_acc,
                    'epoch': epoch,
                }
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')

                filename = 'model_{}_net.pth'.format(task_id)
                torch.save(state, "./checkpoint/{}".format(filename))
            print('epoch:%d,accuracy:%.3f||best accuracy : %.3f' % (epoch, accuracy, self.best_acc))

        return accuracy

    def _test(self, testloader, mode):

        self.model.eval()
        correct, total = 0, 0
        for setp, (indexs, imgs, labels) in enumerate(testloader):
            imgs, labels = imgs.cuda(), labels.cuda()
            with torch.no_grad():
                outputs = self.model(imgs)
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
        accuracy = 100 * correct / total
        self.model.train()
        return accuracy

    def _compute_loss(self, indexs, imgs, target):
        output = self.model(imgs)
        output, target = output.cuda(), target.long().cuda()
        if self.old_model == None:
            return nn.CrossEntropyLoss()(output, target)
        else:
            # old_target = torch.tensor(np.array([self.old_model_output[index.item()] for index in indexs]))
            cls = nn.CrossEntropyLoss()(output, target)
            old_target = torch.sigmoid(output)
            q_i = self.q[indexs]
            dist_loss = sum(nn.BCELoss()(old_target[:,y], q_i[:,y]) for y in range(self.numclass - self.task_size))
            #dist_loss = dist_loss / (self.numclass - self.task_size)

            loss = cls + dist_loss

            return loss

    def afterTrain(self):
        self.best_acc = 0
        self.model.eval()
        m = int(self.memory_size / self.numclass)
        self._reduce_exemplar_sets(m)
        for i in range(self.numclass - self.task_size, self.numclass):
            print('construct class %s examplar:' % (i), end='')
            images = self.train_dataset.get_image_class(i)
            self._construct_exemplar_set(images, m, i)
        self.numclass += self.task_size

        # copy the previous model to calculate the distillation loss
        self.old_model = copy.deepcopy(self.model)
        self.old_model.cuda()
        self.old_model.eval()

    def _construct_exemplar_set(self, images, m, class_index):
        exemplar = []
        print(images.shape)
        r = np.arange(images.shape[0])
        np.random.shuffle(r)
        r = torch.LongTensor(r)
        b = r[0:m]
        self.exemplar_set.append(images[b])
        print("the size of exemplar :%s" % (str(len(self.exemplar_set[class_index]))))
        # self.exemplar_set.append(images)

    def _reduce_exemplar_sets(self, m):
        '''
        m : number of exemplar set for class
        '''
        for index in range(len(self.exemplar_set)):
            self.exemplar_set[index] = self.exemplar_set[index][:m]
            print('Size of class %d examplar: %s' % (index, str(len(self.exemplar_set[index]))))

    def Image_transform(self, images, transform):
        data = transform(Image.fromarray(images[0])).unsqueeze(0)
        for index in range(1, len(images)):
            data = torch.cat((data, self.transform(Image.fromarray(images[index])).unsqueeze(0)), dim=0)

        return data