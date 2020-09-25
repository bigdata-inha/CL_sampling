import argparse
from trainer.train import *
from models.model_resnet import *
import myData.iDataset
import myData.iDataLoader
from utils import *
from sklearn.utils import shuffle
import trainer.trainer_warehouse
import trainer.evaluator

########basic argument##################
parser = argparse.ArgumentParser()
parser.add_argument("--lr", default = 0.1, type = float,\
                    help = "optimizer learning rate")

parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1, 0.1],
                    help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')

parser.add_argument("--memory_size", default = 2000, type = int,\
                    help = "exemplar set memory size")
parser.add_argument("--epochs", default = 160, type = int,\
                    help = "number of Epochs")

parser.add_argument("--step_size", default = 5, type = int,\
                    help = "number of continual learning steps")

parser.add_argument("--nb_proto", default = 20, type = int,\
                    help = "number of fixed exemplar per class")


#parser.add_argument("--numclass", default = 5, type=int,
#                       help = "initial number of class for continual learning")

parser.add_argument("--batch_size", default = 128, type=int,
                    help = "batch size for training")

#parser.add_argument("--task_size", default = 10, type = int,
#                   help = "number of class in one step")

parser.add_argument("--start_classes", default=5, type=int,
                    help = "number of class in one step")

parser.add_argument('--schedule', type=int, nargs='+', default=[80,120],
                    help='learning rate decay epoch')

parser.add_argument("--trainer", default="wa", choices=["wa",'icarl','eeil'], type=str, help="trainer name")

parser.add_argument("--myData", default="CIFAR100", choices=['CIFAR10', 'CIFAR100', "Imagenet"], type=str, help='myData name')

#parser.add_argument("")
args = parser.parse_args()

#seed
set_seed(1994)

dataset = myData.iDataset.CIFAR100()

shuffle_idx = shuffle(np.arange(dataset.classes), random_state = 1994)

myNet = resnet32().cuda()

train_dataset_loader = myData.iDataLoader.IncrementalLoader(dataset.train_data,
                                                            dataset.train_labels,
                                                            dataset.classes,
                                                            args.step_size,
                                                            args.memory_size,
                                                            'train',
                                                            transform=dataset.train_transform,
                                                            loader=None,
                                                            shuffle_idx=None,
                                                            base_classes=args.start_classes,
                                                            approach= "wa",
                                                            model = myNet)

test_dataset_loader = myData.iDataLoader.IncrementalLoader(dataset.test_data,
                                                            dataset.test_labels,
                                                            dataset.classes,
                                                            args.step_size,
                                                            args.memory_size,
                                                            'test',
                                                            transform=dataset.test_transform,
                                                            loader=None,
                                                            shuffle_idx=None,
                                                            base_classes=args.start_classes,
                                                            approach= "wa",
                                                            model = myNet)

result_dataset_loaders = myData.iDataLoader.make_ResultLoaders(dataset.test_data,
                                                         dataset.test_labels,
                                                         dataset.classes,
                                                         args.step_size,
                                                         transform=dataset.test_transform,
                                                         loader=None,
                                                         shuffle_idx = None,
                                                         base_classes = args.start_classes
                                                        )


train_iterator = torch.utils.data.DataLoader(train_dataset_loader, batch_size=128, shuffle=True)
test_iterator = torch.utils.data.DataLoader(test_dataset_loader, batch_size=100, shuffle=False)

optimizer = optim.SGD(myNet.parameters(), args.lr, momentum=0.9,
                        weight_decay=5e-4, nesterov=True)

myTrainer = trainer.trainer_warehouse.TrainerFactory.get_trainer(train_iterator, test_iterator, dataset, myNet, args, optimizer)
myEvaluator = trainer.evaluator.EvaluatorFactory.get_evaluator("trainedClassifier", classes=dataset.classes)

train_start = 0
train_end = args.start_classes
test_start = 0
test_end = args.start_classes

tasknum = (dataset.classes - args.start_classes) // args.step_size + 1

results = {}
for head in ['all', 'prev_new', 'task', 'cheat']:
    results[head] = {}
    results[head]['correct'] = []
    results[head]['correct_5'] = []
    results[head]['stat'] = []

results['task_soft_1'] = np.zeros((tasknum, tasknum))
results['task_soft_5'] = np.zeros((tasknum, tasknum))

train_cumulative_output_list = []
train_cumulative_target_list = []
print(tasknum)

for t in range(tasknum) :
    lr = args.lr

    myTrainer.update_frozen_model()
    myTrainer.setup_training(lr)

    mem_base ={}
    mem_base['CIFAR100'] = 2000
    mem_base['ImageNet'] = 20000

    for epoch in range(args.epochs) :
        myTrainer.update_lr(epoch, args.schedule)
        myTrainer.train(epoch)

    if t > 0 and (args.trainer == 'ft_wa' or args.trainer == 'wa'):
        myTrainer.weight_align()


    if args.step_size < 5  :
        if t > 0:
            train_1 = myEvaluator.evaluate_top1(myTrainer.model, test_iterator, 0, train_end)
            print("*********CURRENT EPOCH********** : %d" % epoch)
            print("Train Classifier top-1 (Softmax): %0.2f" % train_1)
            #print("Train Classifier top-5 (Softmax): %0.2f" % train_5)

            correct, stat = myEvaluator.evaluate_top1(myTrainer.model, test_iterator,
                                                         test_start, test_end,
                                                         mode='test', step_size=args.step_size)

            print("Test Classifier top-1 (Softmax, all): %0.2f" % correct['all'])

            print("Test Classifier top-1 (Softmax, prev_new): %0.2f" % correct['prev_new'])


            for head in ['all', 'prev_new', 'task']:
                results[head]['correct'].append(correct[head])
                #results[head]['correct_5'].append(correct_5[head])
                results[head]['stat'].append(stat[head])

        else:
            ###################### 폐기처분 대상 ######################
            train_1 = myEvaluator.evaluate_top1(myTrainer.model, train_iterator, 0, train_end)
            print("*********CURRENT EPOCH********** : %d" % epoch)
            print("Train Classifier top-1 (Softmax): %0.2f" % train_1)

            test_1 = myEvaluator.evaluate_top1(myTrainer.model, test_iterator, test_start, test_end,
                                                   mode='test', step_size=args.step_size)
            print("Test Classifier top-1 (Softmax): %0.2f" % test_1)


            for head in ['all', 'prev_new', 'task', 'cheat']:
                results[head]['correct'].append(test_1)

    else :
        if t > 0:
            train_1, train_5 = myEvaluator.evaluate(myTrainer.model, test_iterator, 0, train_end)
            print("*********CURRENT EPOCH********** : %d" % epoch)
            print("Train Classifier top-1 (Softmax): %0.2f" % train_1)
            print("Train Classifier top-5 (Softmax): %0.2f" % train_5)

            correct, correct_5, stat = myEvaluator.evaluate(myTrainer.model, test_iterator,
                                                            test_start, test_end,
                                                            mode='test', step_size=args.step_size)

            print("Test Classifier top-1 (Softmax, all): %0.2f" % correct['all'])
            print("Test Classifier top-5 (Softmax, all): %0.2f" % correct_5['all'])
            print("Test Classifier top-1 (Softmax, prev_new): %0.2f" % correct['prev_new'])
            print("Test Classifier top-5 (Softmax, prev_new): %0.2f" % correct_5['prev_new'])

            for head in ['all', 'prev_new', 'task']:
                results[head]['correct'].append(correct[head])
                results[head]['correct_5'].append(correct_5[head])
                results[head]['stat'].append(stat[head])

        else:
            ###################### 폐기처분 대상 ######################
            train_1, train_5 = myEvaluator.evaluate(myTrainer.model, train_iterator, 0, train_end)
            print("*********CURRENT EPOCH********** : %d" % epoch)
            print("Train Classifier top-1 (Softmax): %0.2f" % train_1)
            print("Train Classifier top-5 (Softmax): %0.2f" % train_5)

            test_1, test_5 = myEvaluator.evaluate(myTrainer.model, test_iterator, test_start, test_end,
                                                  mode='test', step_size=args.step_size)
            print("Test Classifier top-1 (Softmax): %0.2f" % test_1)
            print("Test Classifier top-5 (Softmax): %0.2f" % test_5)

            for head in ['all', 'prev_new', 'task', 'cheat']:
                results[head]['correct'].append(test_1)
                results[head]['correct_5'].append(test_5)

    start = 0
    end = args.start_classes

    for i in range(t + 1):
        dataset_loader = result_dataset_loaders[i]
        iterator = torch.utils.data.DataLoader(dataset_loader,
                                               batch_size=args.batch_size)

        if 'bic' in args.trainer:
            results['task_soft_1'][t][i], results['task_soft_5'][t][i] = myEvaluator.evaluate(myTrainer.model,
                                                                                               iterator, start, end,
                                                                                               myTrainer.bias_correction_layer)
        else:
            results['task_soft_1'][t][i], results['task_soft_5'][t][i] = myEvaluator.evaluate(myTrainer.model,
                                                                                               iterator, start, end)
        start = end
        end += args.step_size

    torch.save(myNet.state_dict(), './checkpoint/comparasion/' + 'prevKD_naive_{}_{}.pt'.format(tasknum, t))
    #myTrainer.forgetting_count_class(myTrainer.count_forgetting)
    myTrainer.increment_classes(mode="bal")

    test_dataset_loader.update_exemplar()
    test_dataset_loader.task_change()

    #bias_dataset_loader.update_exemplar()
    #bias_dataset_loader.task_change()

    train_end = train_end + args.step_size
    test_end = test_end + args.step_size

    train_cumulative_output_list.append(myTrainer.cumulative_training_acc)
    train_cumulative_target_list.append(myTrainer.cumulative_training_target)

'''
for i in range(args.step_size):
    trainer.beforeTrain()
    accuracy = trainer.train(i)
    trainer.afterTrain()
'''