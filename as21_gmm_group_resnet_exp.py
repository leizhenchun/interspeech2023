########################################
#        Zhenchun Lei
#  zhenchun.lei@hotmail.com
########################################

import logging

import numpy
import torch
import torchinfo
import torchsummary
from torch import nn, Tensor
from torch.nn import BatchNorm1d, ModuleList
from torch.utils.data import DataLoader, TensorDataset

from as21_experiment import AS21GMMExperiment
from asvspoof19.as19_experiment import get_parameter
from model.gmm import GMMLayer
from util.util import show_model


def exp_parameters():
    exp_param = get_parameter()
    exp_param['asvspoof_root_path'] = '/home/lzc/lzc/ASVspoof/'

    exp_param['batch_size'] = 32
    exp_param['batch_size_test'] = 128

    exp_param['feature_size'] = 60
    exp_param['feature_num'] = 400
    exp_param['feature_num_test'] = 400
    # exp_param['feature_ufm_length'] = 400 * 160 + 240  # 400
    # exp_param['feature_ufm_hop'] = 200
    exp_param['feature_file_extension'] = '.h5'

    exp_param['feature_keep_in_memory'] = True
    exp_param['feature_keep_in_memory_debug'] = False

    exp_param['gmm_size'] = 512
    exp_param['groups'] = 1
    exp_param['weight_decay'] = 0.0

    exp_param['num_epochs'] = 100

    exp_param['lr'] = 0.0001
    exp_param['min_lr'] = 1e-8
    exp_param['use_regularization_loss'] = False
    exp_param['use_scheduler'] = True

    exp_param['test_train2019'] = False
    exp_param['test_dev2019'] = True
    exp_param['test_eval2019'] = True
    exp_param['evaluate_asvspoof2021'] = True
    exp_param['evaluate_asvspoof2021_df'] = True

    exp_param['test_data_basic'] = True
    exp_param['test_data_ufm'] = False
    exp_param['test_data_adaptive'] = False

    return exp_param


class ResNetBlock(nn.Module):
    def __init__(self, in_channels=512, groups=1, kernel_size=3) -> None:
        super(ResNetBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=1,
                               padding=kernel_size // 2, groups=groups, dilation=1, bias=False)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=1,
                               padding=kernel_size // 2, groups=groups, dilation=1, bias=False)
        self.bn2 = nn.BatchNorm1d(in_channels)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet_Path(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1, layer_num=6) -> None:
        super(ResNet_Path, self).__init__()

        self.conv0 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,
                               padding=0, dilation=1, groups=groups, bias=False)
        self.bn0 = nn.BatchNorm1d(out_channels)
        self.relu0 = nn.ReLU(inplace=True)

        self.layer_num = layer_num
        self.blocks = ModuleList()
        for _ in range(self.layer_num):
            self.blocks.append(ResNetBlock(in_channels=out_channels, groups=groups, kernel_size=3))
            # self.blocks.append(BottleneckBlock(in_channels=out_channels, groups=groups, kernel_size=5))
            # self.blocks.append(
            #     Res2NetBlock(in_channels=out_channels, out_channels=out_channels, scale=4, kernel_size=3, dilation=1))

        self.bn = BatchNorm1d(out_channels * self.layer_num)
        self.maxpool = nn.AdaptiveMaxPool1d(1)

        self.out_size = out_channels * layer_num

    def forward(self, x: Tensor) -> Tensor:

        x = self.relu0(self.bn0(self.conv0(x)))

        y = []
        for idx in range(self.layer_num):
            x = self.blocks[idx](x)
            y.append(x)
        x = torch.cat(y, dim=1)

        x = self.bn(x)
        x = self.maxpool(x)
        x = x.squeeze(2)

        return x


class Group_GMM_ResNet(nn.Module):
    def __init__(self, gmm, group_num, group_dim, regroup=False, layer_num=6) -> None:
        super(Group_GMM_ResNet, self).__init__()

        self.gmm_size = gmm.size()

        self.group_num = group_num
        self.group_dim = group_dim
        self.layer_num = layer_num

        if regroup and group_num > 1:
            self.gmm_layer = GMMLayer(gmm, requires_grad=False, regroup_num=group_num)
        else:
            self.gmm_layer = GMMLayer(gmm, requires_grad=False, regroup_num=1)

        self.paths = ModuleList()
        for _ in range(group_num):
            self.paths.append(
                ResNet_Path(in_channels=self.gmm_size // group_num, out_channels=group_dim, groups=1,
                            layer_num=self.layer_num))

        self.classifier = nn.Linear(in_features=group_num * group_dim * layer_num, out_features=2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.gmm_layer(x)

        y = []
        for i, x_i in enumerate(torch.chunk(x, self.group_num, dim=1)):
            y.append(self.paths[i](x_i))
        x = torch.cat(y, dim=1)

        x = self.classifier(x)

        return x


class AS21GMMGroupResNetExperiment(AS21GMMExperiment):
    def __init__(self, model_type, feature_type, access_type, parm):
        super(AS21GMMGroupResNetExperiment, self).__init__(model_type, feature_type, access_type, parm=parm)
        self.group_num = parm['group_num']
        self.group_dim = parm['group_dim']

    def get_net(self, num_classes=2):

        if self.model_type == 'Group_GMM_ResNet':
            model = Group_GMM_ResNet(gmm=self.gmm_ubm,
                                     group_num=self.parm['group_num'],
                                     group_dim=self.parm['group_dim'],
                                     regroup=self.parm['regroup'],
                                     layer_num=self.parm['layer_num'],
                                     )

        # elif self.model_type == 'GMM_XVECTOR':
        #     model = GMM_XVECTOR(self.gmm_ubm)
        #     # model = nn.Sequential(GMMLayer(self.gmm_ubm),
        #     #                               Xvector(in_channels=512),
        #     #                               nn.Linear(3000, 2)
        #     #                               )
        #
        # elif self.model_type == 'GMM_ECAPA':
        #     model = GMM_TDNN(self.gmm_ubm)

        # classifier = nn.Sequential(nn.Linear(resnet_blocks.output_size, 256),
        #                            nn.ReLU(),
        #                            nn.Dropout(p=0.5),
        #                            nn.Linear(256, num_classes),
        #                            )
        # classifier = nn.Linear(resnet_blocks.output_size * 1, num_classes)
        # model = nn.Sequential(resnet_blocks, classifier)

        model = model.cuda()
        # summary(model, (self.parm['feature_size'], self.parm['feature_num']))
        torchinfo.summary(model, (2, self.parm['feature_size'], self.parm['feature_num']), depth=5)
        torchsummary.summary(model, (self.parm['feature_size'], self.parm['feature_num']))

        return model

    def train_model(self, model, train_loader):
        if self.group_num == 1:
            return super().train_model(model, train_loader)

        self.train_groups(model, train_loader)
        self.train_classifier(model, train_loader)

    def train_groups(self, model, train_loader):
        logging.info('=======Training Groups ......')

        class_num = 2  # get_num_classes(self.label_type)

        group_models = []
        group_optimizers = []
        group_scheduler = []
        group_criterion = []
        for idx in range(self.group_num):
            group_model = nn.Sequential(model.paths[idx], nn.Linear(model.paths[idx].out_size, class_num))
            if self.use_gpu:
                group_model = group_model.cuda()

            optimizer = self.get_optimizer(group_model)
            scheduler = self.get_scheduler(optimizer)
            criterion = self.get_criterion()
            if self.use_gpu:
                criterion = criterion.cuda()
            group_model.train()

            group_models.append(group_model)
            group_optimizers.append(optimizer)
            group_scheduler.append(scheduler)
            group_criterion.append(criterion)
        show_model(group_models[0])

        for epoch in range(self.num_epochs):

            group_correct = numpy.zeros((self.group_num,), dtype=int)
            group_total = numpy.zeros((self.group_num,), dtype=int)
            group_total_classify_loss = numpy.zeros((self.group_num,), dtype=numpy.float64)

            # group_size = model.group_size

            for batch_idx, (data, target, data_idx) in enumerate(train_loader):
                if self.use_gpu:
                    data, target = data.cuda(), target.cuda()
                data = model.gmm_layer(data)
                # data = data.permute(0, 2, 1)

                for idx, x_i in enumerate(torch.chunk(data, self.group_num, dim=1)):
                    output = group_models[idx](x_i)
                    # output = output.squeeze(2)

                    # for idx in range(self.gmm_group_num):
                    #     # group_data = data[:, :, idx * group_size:(idx + 1) * group_size]
                    #     group_data = data[:, :, idx::self.gmm_group_num]
                    #
                    #     output = group_models[idx](group_data)
                    #     output = output.squeeze(1)

                    loss = group_criterion[idx](output, target)
                    group_total_classify_loss[idx] += loss.item()

                    group_optimizers[idx].zero_grad()
                    loss.backward()
                    group_optimizers[idx].step()

                    _, predict_label = torch.max(output, 1)
                    group_correct[idx] += (predict_label.cpu() == target.cpu()).sum().numpy()
                    group_total[idx] += target.size(0)

            if self.parm['use_scheduler']:
                for idx in range(self.group_num):
                    group_scheduler[idx].step(group_total_classify_loss[idx])

            if self.verbose >= 1:
                for idx in range(self.group_num):
                    logging.info(
                        "Train Epoch: {}/{} group {} classify loss = {:.8f} accuracy={:.8f}%".format(epoch + 1,
                                                                                                     self.num_epochs,
                                                                                                     idx,
                                                                                                     group_total_classify_loss[
                                                                                                         idx],
                                                                                                     100.0 *
                                                                                                     group_correct[
                                                                                                         idx] /
                                                                                                     group_total[idx]))

    def train_classifier(self, model, train_loader):
        logging.info('=======Training Classifier ......')

        for idx in range(self.group_num):
            model.paths[idx].eval()

        # group_size = model.group_size

        group_output = None
        label_id = torch.zeros((len(train_loader.dataset),), dtype=torch.long)

        with torch.no_grad():
            for batch_idx, (data, target, data_idx) in enumerate(train_loader):
                if self.use_gpu:
                    data, target = data.cuda(), target.cuda()
                data = model.gmm_layer(data)
                # data = data.permute(0, 2, 1)

                # output = []
                # for idx in range(self.gmm_group_num):
                #     # group_data = data[:, :, idx * group_size:(idx + 1) * group_size]
                #     group_data = data[:, :, idx::self.gmm_group_num]
                #     output.append(model.model_groups[idx](group_data))

                output = []
                for idx, x_i in enumerate(torch.chunk(data, self.group_num, dim=1)):
                    output.append(model.paths[idx](x_i))

                data = torch.cat(output, dim=1)
                # data = data.squeeze(1)

                if group_output is None:
                    group_output = torch.zeros((len(train_loader.dataset), data.shape[1]))
                group_output[data_idx, :] = data.cpu()
                label_id[data_idx] = target.cpu()

        group_out_dataset = TensorDataset(group_output, label_id)
        train_loader = DataLoader(group_out_dataset, batch_size=self.parm['batch_size'], shuffle=True)

        logging.info(model.classifier)
        optimizer = self.get_optimizer(model.classifier)
        scheduler = self.get_scheduler(optimizer)
        criterion = self.get_criterion()
        if self.use_gpu:
            criterion = criterion.cuda()

        model.classifier.train()
        for epoch in range(self.num_epochs):
            correct = 0
            total = 0
            total_classify_loss = 0.0

            for batch_idx, (data, target) in enumerate(train_loader):
                if self.use_gpu:
                    data, target = data.cuda(), target.cuda()

                output = model.classifier(data)

                loss = criterion(output, target)
                total_classify_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _, predict_label = torch.max(output, 1)
                correct += (predict_label.cpu() == target.cpu()).sum().numpy()
                total += target.size(0)

            if self.parm['use_scheduler']:
                scheduler.step(total_classify_loss)

            if self.verbose >= 1:
                logging.info(
                    "Train Epoch: {}/{} Total classify loss = {:.8f} accuracy={:.8f}%".format(epoch + 1,
                                                                                              self.num_epochs,
                                                                                              total_classify_loss,
                                                                                              100.0 * correct / total))


if __name__ == '__main__':
    exp_param = exp_parameters()

    access_type = 'LA'
    feature_type = 'LFCC21NN'
    model_type = 'Group_GMM_ResNet'

    # exp_param['num_planes'] = [256, ]
    # exp_param['num_layers'] = [6, ]  # [3, 4, 6, 3]

    exp_param['feature_size'] = 60
    exp_param['feature_num'] = 400
    exp_param['feature_num_test'] = 400

    exp_param['lr'] = 0.0001
    exp_param['weight_decay'] = 0.0

    # exp_param['data_augmentation'] = ["Original", "ALAW", "ULAW", "RIR", "NOISE", "MUSIC", "SPEECH", "RB1", "RB2",
    #                                   "RB3", "RB4", "RB5", "RB6", "RB7", "RB8"]

    # exp_param['gmm_size'] = 1024
    exp_param[
        'gmm_file'] = r'/home/lzc/lzc/ASVspoof/ASVspoof2021exp/GMM_aug2_rb4_{}/ASVspoof2019_GMM_{}_{}_1024.h5'.format(
        feature_type, feature_type, access_type)

    exp_param['data_augmentation'] = ['Original', 'RB4']
    exp_param['regroup'] = True

    exp_param['layer_num'] = 6

    # exp_param['group_num'] = 1
    # exp_param['group_dim'] = 512
    # for _ in range(5):
    #     AS21GMMGroupResNetExperiment(model_type, feature_type, access_type, parm=exp_param).run()

    # exp_param['group_num'] = 2
    # exp_param['group_dim'] = 256
    # for _ in range(3):
    #     AS21GMMGroupResNetExperiment(model_type, feature_type, access_type, parm=exp_param).run()
    #
    # exp_param['group_num'] = 4
    # exp_param['group_dim'] = 256
    # for _ in range(3):
    #     AS21GMMGroupResNetExperiment(model_type, feature_type, access_type, parm=exp_param).run()

    exp_param['group_num'] = 8
    exp_param['group_dim'] = 256
    for _ in range(3):
        AS21GMMGroupResNetExperiment(model_type, feature_type, access_type, parm=exp_param).run()

    # exp_param['group_num'] = 16
    # exp_param['group_dim'] = 256
    # for _ in range(5):
    #     AS21GMMGroupResNetExperiment(model_type, feature_type, access_type, parm=exp_param).run()
