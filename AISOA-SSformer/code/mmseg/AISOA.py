import torch
from . import _functional as F
from .optimizer import Optimizer
import numpy as np

class AISOA(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False, sparrow_factor=0.01, anneal_rate=0.00001, update_interval=10):
        # 参数检查
        # 参数检查
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        sparrow_factor=sparrow_factor)
        super(AdamW, self).__init__(params, defaults)
        self.anneal_rate = anneal_rate
        self.update_interval = update_interval
        self.global_step = 0

    def anneal_sparrow_factor(self):
        """逐步减少 sparrow_factor 的值来实现退火机制"""
        for group in self.param_groups:
            group['sparrow_factor'] = group['sparrow_factor'] * (1 - self.anneal_rate * self.global_step)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.global_step += 1  # 更新全局步数
        self.anneal_sparrow_factor()  # 应用退火机制

        for group in self.param_groups:
            amsgrad = group['amsgrad']
            beta1, beta2 = group['betas']
            sparrow_factor = group['sparrow_factor']

            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                grads.append(p.grad.data)
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                if amsgrad:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])
                state['step'] += 1
                state_steps.append(state['step'])

            # AdamW update logic
            F.adamw(params_with_grad,
                    grads,
                    exp_avgs,
                    exp_avg_sqs,
                    max_exp_avg_sqs,
                    state_steps,
                    amsgrad=amsgrad,
                    beta1=beta1,
                    beta2=beta2,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    eps=group['eps'])

            # Sparrow update
        if self.global_step % self.update_interval == 0:
            for p in params_with_grad:
                sparrow_update = group['sparrow_factor'] * np.random.randn(*p.data.size())
                sparrow_update_tensor = torch.from_numpy(sparrow_update).to(p.data.device)
                p.data.add_(sparrow_update_tensor, alpha=group['lr'])

        return loss
