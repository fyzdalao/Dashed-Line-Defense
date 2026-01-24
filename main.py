import argparse

import torch
import time
import victim
import utils
from utils import *
from square import square_attack_linf
from square_L2 import square_attack_l2
import numpy as np
import torch.nn.functional as F


def load_undefended_model(args):
   return victim.Model(defense='None', args=args)

def load_model(args):
   return victim.Model(defense=args.defense, args=args)


def load_data(model, amount=50000, random_seed=0, need_right_prediction=True):
   #return utils.sample_imagenet(model, amount, random_seed, need_right_prediction=False)
   return utils.sample_imagenet_every_class(model, random_seed, need_right_prediction=need_right_prediction)


def try_the_model(model, x_test, y_test):
   logits = model(x_test)
   margin = margin_loss_n(y_test, logits)
   ifcorr = margin>0
   acc = (ifcorr).sum() / x_test.shape[0]
   print(acc)



def parse_args():
   parser = argparse.ArgumentParser()



   parser.add_argument('--arch', type=str, default='regnet_y_1_6gf', help='wide_resnet50_2/regnet_y_1_6gf/maxvit_t')
   parser.add_argument('--batch_size', default=300, type=int, help='up to 1000')

   parser.add_argument('--seed', default=19260817, type=int, help='random seed')
   parser.add_argument('--dataseed', default=19260817, type=int, help='random seed for loading data')

   parser.add_argument('--budget', default=2500, type=int, help='query budget')

   parser.add_argument('--attack', default='square', type=str, help='attack method: square/square2')
   parser.add_argument('--tactic', default='none', type=str, help='none/reverse/sa/explore')
   parser.add_argument('--sa_temp', default=25, type=int, help='beginning temperature')
   parser.add_argument('--sa_rate', default=0.997, type=float, help='cooling ratio')
   parser.add_argument('--sa_reheat', default=20, type=int, help='after sa_reheat iterations without change, reheat')
   parser.add_argument('--reverse_time_to_reverse', default=23, type=int, help='after reverse_time_to_reverse iterations without change, reverse')
   parser.add_argument('--explore_rate', default=0.5, type=float, help='accept worse by the chance of explore_rate')

   parser.add_argument('--defense', default='DLD', type=str, help='None/DLD/AAASine/AAALinear/inRND/outRND/inoutRND')
   parser.add_argument('--rnd_nu', default=0.02, type=float, help='noise level')
   parser.add_argument('--tau', default=6, type=float)
   parser.add_argument('--high_ratio', default=0.5, type=float)
   parser.add_argument('--s_step', default=0.04, type=float)

   parser.add_argument('--eps', default=0.05, type=float)

   parser.add_argument('--device', default='cuda:0', type=str)


   args = parser.parse_args()

   return args


if __name__ == '__main__':
   args = parse_args()

   result_path = 'results' + '/' + get_time() + '/log.log'
   log = Logger(result_path)
   log.print(str(args))

   undefended_model = load_undefended_model(args)

   x_test, y_test = load_data(model=undefended_model, random_seed=args.dataseed)

   np.random.seed(args.seed)
   torch.manual_seed(args.seed)

   model = load_model(args)

   try_the_model(undefended_model,x_test,y_test)
   try_the_model(model, x_test, y_test)

   logits_clean = model(x_test)
   prob_clean = F.softmax(torch.as_tensor(logits_clean),dim=1).numpy()

   margin = margin_loss(y_test, logits_clean)
   correct_idx = margin > 0
   correct_idx = correct_idx.reshape((-1,))


   if args.attack == 'square':
      square_attack_linf(model=model, x=x_test, y=y_test, correct=correct_idx, n_iters=args.budget, eps=args.eps, p_init=0.05, attack_tactic=args.tactic, log=log, args=args)
   elif args.attack == 'square2':
      square_attack_l2(model=model, x=x_test, y=y_test, correct=correct_idx, n_iters=args.budget, eps=args.eps,
                          p_init=0.3, attack_tactic=args.tactic, log=log, args=args)






















