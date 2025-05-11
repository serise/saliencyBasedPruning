import torch
import torch.nn as nn
from torch import optim
from utils.trainingUtils import *
import torch_pruning as tp
import numpy as np
import fnmatch

class Pruner:
    def __init__(self,config,model,training_loader,test_loader,prune_list):
        self.config = config
        self.model = model
        self.training_loader = training_loader
        self.test_loader = test_loader
        self.prune_list = prune_list

    def pruneOneLayer(self, layerName, removeList):
        assert type(removeList) == list or type(removeList) == np.ndarray, 'removeList should be a list or numpy array'
        self.model = self.model.cpu()
        DG = tp.DependencyGraph().build_dependency(self.model, example_inputs=torch.randn(1, 3, 32, 32))
        group = DG.get_pruning_group(dict(self.model.named_modules()).get(layerName), tp.prune_conv_out_channels,
                                     idxs=removeList)
        successPrune = False
        if DG.check_pruning_group(group):
            group.prune()
            successPrune = True
        return successPrune

    def printFlopsAndParams(self):
        toPruneDict = {}
        for name, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d):
                toPruneDict[name] = m.weight.shape[0]

        tempModel = load_model(self.config['prune_setting']['model_name'], numClasses=self.config['model']['num_classes'])
        ori_ops, ori_size = tp.utils.count_ops_and_params(tempModel, example_inputs=torch.randn(1, 3, 32, 32))

        for name, m in tempModel.named_modules():
            if name in toPruneDict.keys():
                pruneNum = m.weight.shape[0] - toPruneDict[name]
                removeList = list(range(pruneNum))
                DG = tp.DependencyGraph().build_dependency(tempModel, example_inputs=torch.randn(1, 3, 32, 32))
                group = DG.get_pruning_group(m, tp.prune_conv_out_channels, idxs=removeList)
                if DG.check_pruning_group(group):
                    group.prune()

        pruned_ops, pruned_size = tp.utils.count_ops_and_params(tempModel, example_inputs=torch.randn(1, 3, 32, 32))

        flops = "FLOPs: {:.2f} M => {:.2f} M ({:.2f}%, {:.2f}X )".format(
            ori_ops / 1e6,
            pruned_ops / 1e6,
            pruned_ops / ori_ops * 100,
            ori_ops / pruned_ops,
        )
        params = "Params: {:.2f} M => {:.2f} M ({:.2f}%)".format(
            ori_size / 1e6, pruned_size / 1e6, pruned_size / ori_size * 100
        )

        print(flops)
        print(params)
        del tempModel

        return pruned_ops / ori_ops

    def retrain(self,trainingType):
        def train(epoch):

            self.model.train()
            for batch_index, (images, labels) in enumerate(self.training_loader):

                if self.config[trainingType]['use_gpu']:
                    labels = labels.cuda()
                    images = images.cuda()

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

                if epoch <= self.config[trainingType]['warmup']:
                    warmup_scheduler.step()

            print('epoch:{} , loss: {:0.4f}'.format(epoch, loss))

        @torch.no_grad()
        def eval_training():

            self.model.eval()

            test_loss = 0.0  # cost function error
            correct = 0.0

            for (images, labels) in self.test_loader:

                if self.config[trainingType]['use_gpu']:
                    images = images.cuda()
                    labels = labels.cuda()

                outputs = self.model(images)
                loss = loss_function(outputs, labels)

                test_loss += loss.item()
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum()

            return correct.float() / len(self.test_loader.dataset)

        if self.config[trainingType]['use_gpu']:
            self.model = self.model.cuda()

        loss_function = nn.CrossEntropyLoss()
        if self.config[trainingType]['optimizerName'] == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.config[trainingType]['learning_rate'], momentum=0.9, weight_decay=5e-4)
        elif self.config[trainingType]['optimizerName'] == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.config[trainingType]['learning_rate'], weight_decay=5e-4)
        else:
            raise ValueError('optimizer choices： sgd, adam')
        train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.config[trainingType]['milestones'],gamma=0.2)
        iter_per_epoch = len(self.training_loader)
        warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * self.config[trainingType]['warmup'])

        best_acc = 0.0
        for epoch in range(1, self.config[trainingType]['epochs'] + 1):
            if epoch > self.config[trainingType]['warmup']:
                train_scheduler.step(epoch)

            train(epoch)
            acc = eval_training()
            if acc > best_acc:
                best_acc = acc
                if trainingType == 'fine_tuning':
                    torch.save(self.model, '{}_{}_pruned.pt'.format(self.config['prune_setting']['model_name'],self.config['prune_setting']['dataset_name']))

            print('epoch:{} , acc: {:0.4f}'.format(epoch, acc))
        print('Best acc: {:0.4f}'.format(best_acc))

    def prune(self):
        for sName in self.config['pruning']['step_names']:
            for name, m in self.model.named_modules():
                if (isinstance(m, torch.nn.Conv2d)) and (fnmatch.fnmatch(name, sName)) and (
                not fnmatch.fnmatch(name, self.config['pruning']['skip_layers'])):
                    removeList = self.prune_list[name]
                    self.pruneOneLayer(name, removeList)

            self.retrain('retrain')

        self.retrain('fine_tuning')