from copy import deepcopy

import numpy as np
import time
from utils import margin_loss_n, margin_loss, Logger, get_time


'''

This file is adapted from the original implementation of the Square Attack paper:
link: https://github.com/max-andr/square-attack/blob/master/attack.py

@article{ACFH2020square,
  title={Square Attack: a query-efficient black-box adversarial attack via random search},
  author={Andriushchenko, Maksym and Croce, Francesco and Flammarion, Nicolas and Hein, Matthias},
  conference={ECCV},
  year={2020}
}

The original license is included at the end of this file.

'''


def pseudo_gaussian_pert_rectangles(x, y):
    delta = np.zeros([x, y])
    x_c, y_c = x // 2 + 1, y // 2 + 1

    counter2 = [x_c - 1, y_c - 1]
    for counter in range(0, max(x_c, y_c)):
        delta[max(counter2[0], 0):min(counter2[0] + (2 * counter + 1), x),
              max(0, counter2[1]):min(counter2[1] + (2 * counter + 1), y)] += 1.0 / (counter + 1) ** 2

        counter2[0] -= 1
        counter2[1] -= 1

    delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))

    return delta


def meta_pseudo_gaussian_pert(s):
    delta = np.zeros([s, s])
    n_subsquares = 2
    if n_subsquares == 2:
        delta[:s // 2] = pseudo_gaussian_pert_rectangles(s // 2, s)
        delta[s // 2:] = pseudo_gaussian_pert_rectangles(s - s // 2, s) * (-1)
        delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))
        if np.random.rand(1) > 0.5: delta = np.transpose(delta)

    elif n_subsquares == 4:
        delta[:s // 2, :s // 2] = pseudo_gaussian_pert_rectangles(s // 2, s // 2) * np.random.choice([-1, 1])
        delta[s // 2:, :s // 2] = pseudo_gaussian_pert_rectangles(s - s // 2, s // 2) * np.random.choice([-1, 1])
        delta[:s // 2, s // 2:] = pseudo_gaussian_pert_rectangles(s // 2, s - s // 2) * np.random.choice([-1, 1])
        delta[s // 2:, s // 2:] = pseudo_gaussian_pert_rectangles(s - s // 2, s - s // 2) * np.random.choice([-1, 1])
        delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))

    return delta


def p_selection(p_init, it, n_iters):
    """ Piece-wise constant schedule for p (the fraction of pixels changed on every iteration). """
    it = int(it / n_iters * 10000)

    if 10 < it <= 50:
        p = p_init / 2
    elif 50 < it <= 200:
        p = p_init / 4
    elif 200 < it <= 500:
        p = p_init / 8
    elif 500 < it <= 1000:
        p = p_init / 16
    elif 1000 < it <= 2000:
        p = p_init / 32
    elif 2000 < it <= 4000:
        p = p_init / 64
    elif 4000 < it <= 6000:
        p = p_init / 128
    elif 6000 < it <= 8000:
        p = p_init / 256
    elif 8000 < it <= 10000:
        p = p_init / 512
    else:
        p = p_init

    return p


def randbool(size, exp=0.1):
    randvec = np.random.rand(size)
    return randvec < exp


def square_attack_linf(model, x, y, correct, n_iters, eps, p_init, attack_tactic, logits_amt=3, log=None, args=None):

    y = np.array(y, dtype=bool)

    min_val, max_val = 0, 1
    c, h, w = x.shape[1:]
    n_features = c * h * w
    example_amount = x.shape[0]
    if correct is not None:
        y = y[correct]
        x = x[correct]

    # [c, 1, w], i.e. vertical stripes work best for untargeted attacks
    init_delta = np.random.choice([-eps, eps], size=[x.shape[0], c, 1, w])

    x_best = np.clip(x + init_delta, min_val, max_val)

    logits = model(x_best)
    margin_min = margin_loss(y, logits)
    n_queries = np.ones(x.shape[0])

    acc = (margin_min > 0.0).sum() / example_amount

    persuit = np.zeros(x.shape[0], dtype=bool) # persuit = 0 indicates descending toward the valley, persuit = 1 indicates ascending toward the peak
    iters_without_change = np.zeros(x.shape[0], dtype=int) 
    time_to_reverse = args.reverse_time_to_reverse


    #sa parameters
    sa_temp = args.sa_temp
    sa_rate = args.sa_rate
    tmp = np.ones(x.shape[0]) * sa_temp
    time_to_reheat = args.sa_reheat
    convergence_times = np.zeros(x.shape[0], dtype=int)


    explore_rate = args.explore_rate


    time_start = time.time()
    i_iter = -1
    while i_iter < n_iters:
        i_iter += 1
        idx_to_fool = margin_min > 0.0

        x_curr, x_best_curr = x[idx_to_fool], x_best[idx_to_fool]
        y_curr, margin_min_curr = y[idx_to_fool], margin_min[idx_to_fool]
        delta_curr = x_best_curr - x_curr

        #sa
        tmp_curr = tmp[idx_to_fool]
        convergence_times_curr = convergence_times[idx_to_fool]


        p = p_selection(p_init, i_iter, n_iters)

        for i_img in range(x_best_curr.shape[0]):
            s = int(round(np.sqrt(p * n_features / c)))
            s = min(max(s, 1), h-1)  # at least c x 1 x 1 window is taken and at most c x h-1 x h-1
            center_h = np.random.randint(0, h - s)
            center_w = np.random.randint(0, w - s)

            x_curr_window = x_curr[i_img, :, center_h:center_h+s, center_w:center_w+s]
            x_best_curr_window = x_best_curr[i_img, :, center_h:center_h+s, center_w:center_w+s]
            # prevent trying out a delta if it doesn't change x_curr (e.g. an overlapping patch)
            while np.sum(np.abs(np.clip(x_curr_window + delta_curr[i_img, :, center_h:center_h+s, center_w:center_w+s], min_val, max_val) - x_best_curr_window) < 10**-7) == c*s*s:
                delta_curr[i_img, :, center_h:center_h+s, center_w:center_w+s] = np.random.choice([-eps, eps], size=[c, 1, 1])

        x_new = np.clip(x_curr + delta_curr, min_val, max_val)

        logits = model(x_new)

        margin = margin_loss(y_curr, logits)

        idx_suc = margin <= 0

        #sa
        p = np.exp( -(margin - margin_min_curr) / tmp_curr )
        r = np.random.rand(p.size)
        idx_improved_sa = (margin < margin_min_curr) + (p > r)

        tmp_curr = tmp_curr * sa_rate

        idx_improved = margin < margin_min_curr
        if attack_tactic == 'none':
            pass
        elif attack_tactic =='sa':
            idx_improved = idx_improved_sa
        elif attack_tactic == 'explore':
            idx_explore = randbool(idx_improved.shape[0], explore_rate)
            idx_improved = idx_improved | idx_explore
        elif attack_tactic == 'reverse':
            persuit_curr, iters_without_change_curr = persuit[idx_to_fool], iters_without_change[idx_to_fool]

            idx_higher = margin > margin_min_curr
            idx_lower = margin < margin_min_curr
            idx_improved = idx_higher * persuit_curr + idx_lower * ~persuit_curr
            idx_improved = idx_improved | idx_suc
            iters_without_change_curr[~idx_improved] += 1
            idx_to_reverse = iters_without_change_curr > time_to_reverse
            idx_improved += idx_to_reverse
            iters_without_change_curr[idx_improved] = 0

            persuit_curr[idx_to_reverse] = ~persuit_curr[idx_to_reverse]

            # write back
            persuit[idx_to_fool] = persuit_curr
            iters_without_change[idx_to_fool] = iters_without_change_curr

        # write back
        margin_min[idx_to_fool] = idx_improved * margin + ~idx_improved * margin_min_curr

        #SA reheat
        zero_vector = np.zeros(convergence_times_curr.size, dtype=int)
        one_vector = np.ones(convergence_times_curr.size, dtype=int)
        convergence_times_curr = ~idx_improved * convergence_times_curr + ~idx_improved * one_vector + idx_improved * zero_vector
        idx_to_heat = convergence_times_curr > time_to_reheat

        if i_iter > 400:
            tmp_curr[idx_to_heat] = sa_temp
        else:
            tmp_curr[idx_to_heat] = sa_temp

        tmp[idx_to_fool] = tmp_curr
        convergence_times_curr[idx_to_heat] = 0
        convergence_times[idx_to_fool] =  convergence_times_curr


        idx_improved = np.reshape(idx_improved, [-1, *[1] * len(x.shape[:-1])])
        x_best[idx_to_fool] = idx_improved * x_new + ~idx_improved * x_best_curr
        n_queries[idx_to_fool] += 1


        # measures
        acc = (margin_min > 0.0).sum() / example_amount
        acc_corr = (margin_min > 0.0).mean()
        mean_nq, mean_nq_ae, median_nq, median_nq_ae = np.mean(n_queries), np.mean(
            n_queries[margin_min <= 0]), np.median(n_queries), np.median(n_queries[margin_min <= 0])


        # logs
        time_total = time.time() - time_start
        if log is not None:
            log.print(
                '{}: acc={:.2%} acc_corr={:.2%} avg#q_ae={:.1f} med#q_ae={:.1f}, n_ex={}, {:.0f}s'.
                format(i_iter + 1, acc, acc_corr, mean_nq_ae, median_nq_ae, x.shape[0], time_total))

        if acc == 0:
            break

    return n_queries, x_best


'''
The original license:

Copyright (c) 2019, Maksym Andriushchenko, Francesco Croce, Nicolas Flammarion, Matthias Hein
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the copyright holder nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''





