"""Adam parameter groups shared by training and the resource benchmark."""
import torch


def build_optimizer(model, config):
    cycn = getattr(model, 'cycn_branch', None)
    branch_parameters = list(cycn.parameters()) if cycn is not None else []
    branch_ids = {id(parameter) for parameter in branch_parameters}
    backbone = [parameter for parameter in model.parameters()
                if parameter.requires_grad and id(parameter) not in branch_ids]
    policy = config['cycn']
    lr = config['learning_rate']
    groups = []
    if cycn is not None and policy['train_policy'] == 'frozen':
        for parameter in backbone:
            parameter.requires_grad_(False)
    else:
        backbone_lr = lr * policy['backbone_lr_ratio'] if cycn is not None else lr
        groups.append({'params': backbone, 'lr': backbone_lr, 'name': 'backbone'})
    if cycn is not None:
        groups.append({'params': branch_parameters, 'lr': lr, 'name': 'cycn'})
    return torch.optim.Adam(groups, lr=lr, betas=tuple(config['betas']), weight_decay=config['weight_decay'])


def learning_rates(optimizer):
    return {group['name']:group['lr'] for group in optimizer.param_groups}
