import warnings

import torch
from torch import nn
import torchvision
from torchvision import transforms
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.models import mnasnet1_3, MNASNet1_3_Weights
from torchvision.models import regnet_y_1_6gf, RegNet_Y_1_6GF_Weights
from torchvision.models import regnet_y_3_2gf, RegNet_Y_3_2GF_Weights
from torchvision.models import maxvit_t, MaxVit_T_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import inception_v3, Inception_V3_Weights

import torch.nn.functional as F
import numpy as np
import math

'''

Some parts of the code in this file are re-implemented based on the official AAA implementation:
link: https://github.com/Sizhe-Chen/AAA/blob/main/victim.py

'''

my_device = torch.device('cuda:0')


class Model(nn.Module):
    def __init__(self, defense='None', args=None):
        super(Model, self).__init__()
        self.arch = args.arch
        self.device = torch.device(args.device)

        self.get_cnn()

        self.batch_size = args.batch_size
        self.defense = defense
        self.mean = np.reshape([0.485, 0.456, 0.406], [1, 3, 1, 1])
        self.std = np.reshape([0.229, 0.224, 0.225], [1, 3, 1, 1])

        # AAA parameters
        self.dev = 0.5
        self.tau = args.tau
        self.reverse_step = 0.7 #0.7

        # RND parameters
        self.n_in = args.rnd_nu
        self.n_out = 0.3

        self.high_ratio = args.high_ratio
        self.s_step = args.s_step

        #for plotting
        self.high_cnt = [0]
        self.low_cnt = [0]
        self.high_low_bits = []




    def get_cnn(self):
        if self.arch == 'wide_resnet50_2':
            self.cnn = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2).to(self.device).eval()
        elif self.arch == 'resnet101':
            self.cnn = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2).to(self.device).eval()
        elif self.arch == 'resnet34':
            self.cnn = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).to(self.device).eval()
        elif self.arch == 'mnasnet1_3':
            self.cnn = mnasnet1_3(weights=MNASNet1_3_Weights.IMAGENET1K_V1).to(self.device).eval()
        elif self.arch == 'regnet_y_1_6gf':
            self.cnn = regnet_y_1_6gf(weights=RegNet_Y_1_6GF_Weights.IMAGENET1K_V2).to(self.device).eval()
        elif self.arch == 'regnet_y_3_2gf':
            self.cnn = regnet_y_3_2gf(weights=RegNet_Y_3_2GF_Weights.IMAGENET1K_V2).to(self.device).eval()
        elif self.arch == 'maxvit_t':
            self.cnn = maxvit_t(weights=MaxVit_T_Weights.IMAGENET1K_V1).to(self.device).eval()
        elif self.arch == 'resnet50':
            self.cnn = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(self.device).eval()
        elif self.arch == 'inception_v3':
            self.cnn = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1).to(self.device).eval()
        else:
            warnings.warn('cnn model not imported')


    def predict(self, x, model, batch_size, device):
        if isinstance(x, np.ndarray):
            if self.arch == 'inception_v3':
                x = torch.from_numpy(x).float()
                x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
                x = x.numpy()

            x = (x - self.mean) / self.std
            x = x.astype(np.float32)

            batch_amount = math.ceil(x.shape[0] / batch_size)
            batch_logits = []
            with torch.no_grad():
                for i in range(batch_amount):
                    x_now = torch.as_tensor(x[i * batch_size: (i + 1) * batch_size], device=device, dtype=torch.float32)
                    batch_logits.append(model(x_now).detach().cpu().numpy())
            logits = np.vstack(batch_logits)
            return logits
        else:
            if self.arch == 'inception_v3':
                x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
            x = (x - torch.as_tensor(self.mean, device=device)) / torch.as_tensor(self.std, device=device)
            return model(x)


    def forward(self, x):
        if self.defense == 'None':
            return self.predict(x=x, model=self.cnn, batch_size=self.batch_size, device=self.device)
        elif self.defense =='AAASine':
            return self.aaa_sine_forward(x)
        elif self.defense == 'AAALinear':
            return self.aaa_linear_forward(x)
        elif self.defense == 'inRND' or self.defense == 'outRND' or self.defense == 'inoutRND':
            return self.rnd_forward(x)
        elif self.defense =='DLD':
            return self.dashed_line_single_forward(x)
        else:
            warnings.warn('no such defense method')


    def aaa_sine_forward(self, x):
        with torch.no_grad():
            logits = self.predict(x=x, model=self.cnn, batch_size=self.batch_size, device=self.device)
            if isinstance(logits, np.ndarray):
                logits = torch.as_tensor(logits, device=self.device)
        logits_ori = logits.detach()

        value, index_ori = torch.topk(logits_ori, k=2, dim=1)

        margin_ori = value[:, 0] - value[:, 1]
        attractor = ((margin_ori / self.tau + self.dev).round() - self.dev) * self.tau
        target = margin_ori - self.reverse_step * self.tau * torch.sin(
            (1 - 2 / self.tau * (margin_ori - attractor)) * torch.pi)
        gap_to_target = target - margin_ori
        logits_ori[torch.arange(logits_ori.shape[0]), index_ori[:, 0]] += gap_to_target

        logits_ret = logits_ori.detach().cpu()
        if isinstance(x, np.ndarray):
            logits_ret = logits_ret.numpy()
        return logits_ret


    def aaa_linear_forward(self, x):

        reverse_step = 1

        with torch.no_grad():
            logits = self.predict(x=x, model=self.cnn, batch_size=self.batch_size, device=self.device)
            if isinstance(logits, np.ndarray):
                logits = torch.as_tensor(logits, device=self.device)
        logits_ori = logits.detach()

        value, index_ori = torch.topk(logits_ori, k=2, dim=1)

        margin_ori = value[:, 0] - value[:, 1]
        attractor = ((margin_ori / self.tau + self.dev).round() - self.dev) * self.tau
        target = attractor - reverse_step * (margin_ori - attractor)
        gap_to_target = target - margin_ori
        logits_ori[torch.arange(logits_ori.shape[0]), index_ori[:, 0]] += gap_to_target

        logits_ret = logits_ori.detach().cpu()
        if isinstance(x, np.ndarray):
            logits_ret = logits_ret.numpy()
        return logits_ret


    def randbool(self, size, exp=0.5):
        randvec = torch.rand(size, device=self.device)
        return randvec < exp


    def dashed_line_single_forward(self, x):
        h = 0.3
        tau = self.tau
        high_ratio = self.high_ratio
        s_step = self.s_step


        with torch.no_grad():
            logits = self.predict(x=x, model=self.cnn, batch_size=self.batch_size, device=self.device)
            if isinstance(logits, np.ndarray):
                logits = torch.as_tensor(logits, device=self.device)
        logits_ori = logits.detach()

        value, index_ori = torch.topk(logits_ori, k=2, dim=1)

        margin_ori = value[:, 0] - value[:, 1]

        L_bias = torch.floor(margin_ori / tau) * tau
        target_low = L_bias + h*tau - h*(margin_ori - L_bias)
        target_high = L_bias + h*tau + (1-h) * (margin_ori - L_bias)

        # for random-DLD
        #int_size = int(margin_ori.shape[0])
        #decision1 = self.randbool(size=int_size, exp=high_ratio)

        condition = (margin_ori - L_bias) / tau

        #decision2 = (condition < 0.1) | ((condition > 0.2) & (condition < 0.3)) | ((condition > 0.4) & (condition < 0.5)) | ((condition > 0.6) & (condition < 0.7)) | ((condition > 0.8) & (condition < 0.9))

        decision3 = condition > 1
        temp = 0
        while temp <= 1:
            temp += s_step
            decision3 |= ((condition > temp - s_step * high_ratio) & (condition < temp))

        decision = decision3
        self.high_low_bits = decision

        true_cnt = int(decision.sum().item())
        false_cnt = int((~decision).sum().item())
        self.high_cnt.append(true_cnt)
        self.low_cnt.append(false_cnt)

        target = target_high * decision + target_low * ~decision
        gap_to_target = target - margin_ori
        logits_ori[torch.arange(logits_ori.shape[0]), index_ori[:, 0]] += gap_to_target

        logits_ret = logits_ori.detach().cpu()
        if isinstance(x, np.ndarray):
            logits_ret = logits_ret.numpy()
        return logits_ret



    def rnd_forward(self, x):
        if self.defense=='inRND' or self.defense=='inoutRND':
            noise_in = np.random.normal(scale=self.n_in, size=x.shape)
        else:
            noise_in = np.zeros(shape=x.shape)

        logits = self.predict(x=np.clip(x + noise_in, 0, 1), model=self.cnn, batch_size=self.batch_size, device=self.device)

        if self.defense=='outRND' or self.defense=='inoutRND':
            noise_out = np.random.normal(scale=self.n_out, size=logits.shape)
        else:
            noise_out = np.zeros(shape=logits.shape)

        return logits + noise_out


    def randbool(self, size, exp=0.5):
        randvec = torch.rand(size, device=self.device)
        return randvec < exp











































