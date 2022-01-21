import numpy as np
from copo.algo_copo.constants import *
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import get_activation_fn, try_import_tf, try_import_torch

torch, nn = try_import_torch()

NEI_REWARDS = "nei_rewards"
NEI_VALUES = "nei_values"
NEI_ADVANTAGE = "nei_advantage"
NEI_TARGET = "nei_target"
SVO_LR = "svo_lr"

GLOBAL_VALUES = "global_values"
GLOBAL_REWARDS = "global_rewards"
GLOBAL_ADVANTAGES = "global_advantages"
GLOBAL_TARGET = "global_target"


def _flatten(tensor):
    assert tensor is not None
    flat = tensor.reshape(-1, )
    return flat, tensor.shape, flat.shape


def build_meta_gradient_and_update_svo_by_torch(policy, input_dict, new_model, old_model):

    # == step1. Build the loss between new policy and ego advantage. ==
    # 既然不能两次反向传播，并且第一次反向传播的backward()被封装在rllib中不方便设置retain_graph=True,
    # 那咱们就再过一遍网络！而且现在过网络，网络参数就是最新的，就是我们要的θnew。
    term1_opt = torch.optim.Adam(new_model.parameters(), lr=policy.config["lr"])
    logits, state = new_model(input_dict)
    curr_dist = policy.dist_class(logits, new_model)
    logp_ratio = torch.exp(curr_dist.logp(input_dict[SampleBatch.ACTIONS]) - input_dict[SampleBatch.ACTION_LOGP])
    # 上面求的logp_ratio和下面这个就是不应该相等的啊，因为model又根据这里存的logp_ratio更新了一次！
    # logp_ratio = policy._logp_ratio
    if policy.config["use_global_value"]:
        adv = input_dict[GLOBAL_ADVANTAGES]
    else:
        adv = input_dict["normalized_ego_advantages"]
    adv = torch.tensor(adv).to(policy.device)
    surrogate_loss = torch.min(
        adv * logp_ratio,
        adv * torch.clamp(logp_ratio, 1 - policy.config["clip_param"], 1 + policy.config["clip_param"])
    )
    new_policy_ego_loss = torch.mean(surrogate_loss)  # 修改为不加负号，后面对svo手动做梯度上升
    # policy.optimizer()[0].zero_grad(set_to_none=True)
    term1_opt.zero_grad(set_to_none=True)  # 将参数的梯度归零
    new_policy_ego_loss.backward()

    # == (ok) step2. Build the loss between old policy and old log prob. ==
    term2_opt = torch.optim.Adam(old_model.parameters(), lr=policy.config["lr"])
    old_model.to(policy.device)  # 之前没找到好时机设置old_model的device，那就每次过网络前设置一下吧~
    old_logits, old_state = old_model(input_dict)  # old_logits.shape = (batchsize, 4)
    old_dist = policy.dist_class(old_logits, old_model)
    old_logp = old_dist.logp(input_dict[SampleBatch.ACTIONS])  # old_logp.shape = (batchsize, )
    old_policy_logp_loss = torch.mean(old_logp)
    term2_opt.zero_grad(set_to_none=True)
    old_policy_logp_loss.backward()

    # == step3. Build the loss between SVO and SVO advantage ==
    if policy.config[USE_DISTRIBUTIONAL_SVO]:
        svo_rad = policy._svo * np.pi / 2 + \
                  policy._svo_std * torch.randn(input_dict[NEI_ADVANTAGE].shape).to(policy.device)
    else:
        svo_rad = policy._svo * np.pi / 2
    advantages = torch.cos(svo_rad) * input_dict[Postprocessing.ADVANTAGES] + \
                 torch.sin(svo_rad) * input_dict[NEI_ADVANTAGE]
    svo_advantages = (advantages - policy._raw_svo_adv_mean) / policy._raw_svo_adv_std
    svo_svo_adv_loss = torch.mean(svo_advantages)
    svo_svo_adv_loss.backward()  # 反向传播后policy._svo_param.grad不再是None

    # == step4. Multiple gradients one by one ==
    new_policy_ego_grad_flatten = []
    shape_list = []  # For verification used.

    new_policy_ego_grad = [params.grad for name, params in
                           new_model.named_parameters() if params.grad is not None]
    for g in new_policy_ego_grad:
        fg, s, _ = _flatten(g)
        shape_list.append(s)
        new_policy_ego_grad_flatten.append(fg)
    new_policy_ego_grad_flatten = torch.concat(new_policy_ego_grad_flatten, axis=0)  # shape = (90628, )
    new_policy_ego_grad_flatten = torch.reshape(new_policy_ego_grad_flatten, (1, -1))  # 行向量

    old_policy_logp_grad_flatten = []
    old_policy_logp_grad = [params.grad for name, params in
                            old_model.named_parameters() if params.grad is not None]
    for g, verify_shape in zip(old_policy_logp_grad, shape_list):
        fg, s, _ = _flatten(g)
        assert verify_shape == s
        old_policy_logp_grad_flatten.append(fg)
    old_policy_logp_grad_flatten = torch.concat(old_policy_logp_grad_flatten, axis=0)
    old_policy_logp_grad_flatten = torch.reshape(old_policy_logp_grad_flatten, (-1, 1))  # 列向量

    grad_value = torch.matmul(new_policy_ego_grad_flatten, old_policy_logp_grad_flatten)  # scalar, 行向量 * 列向量
    final_loss = torch.reshape(grad_value, ()) * svo_svo_adv_loss  # Eqn. 11!

    # == step5. apply gradient ==
    if policy.config[USE_DISTRIBUTIONAL_SVO]:
        single_grad_0 = policy._svo_param.grad.to(policy.device)
        single_grad_1 = policy._svo_std_param.grad.to(policy.device)
        final_grad_0 = torch.matmul(grad_value, torch.reshape(single_grad_0, (1, 1)))
        final_grad_0 = torch.reshape(final_grad_0, ())
        final_grad_1 = torch.matmul(grad_value, torch.reshape(single_grad_1, (1, 1)))
        final_grad_1 = torch.reshape(final_grad_1, ())
        # apply gradient!
        policy._svo_param.grad = -final_grad_0  # 修改梯度，负号因为要做梯度上升
        policy._svo_std_param.grad = -final_grad_1
        policy.svo_opt.step()
        # policy._svo_param.data = policy._svo_param.data + policy.config['svo_lr'] * final_grad_0
        # policy._svo_std_param.data = policy._svo_std_param.data + policy.config['svo_lr'] * final_grad_1
        # assign
        policy._svo = torch.clamp(torch.tanh(policy._svo_param), -1 + 1e-6, 1 - 1e-6)
        policy._svo_std = torch.exp(torch.clamp(policy._svo_std_param, -20, 2))

    else:
        single_grad = policy._svo_param.grad.to(policy.device)
        final_grad = torch.matmul(grad_value, torch.reshape(single_grad, (1, 1)))
        final_grad = torch.reshape(final_grad, ())
        # apply gradient!
        policy._svo_param.grad = -final_grad  # 修改梯度，负号因为要做梯度上升
        policy.svo_opt.step()
        # policy._svo_param.data = policy._svo_param.data + policy.config['svo_lr'] * final_grad
        # assign
        policy._svo = torch.clamp(torch.tanh(policy._svo_param), -1 + 1e-6, 1 - 1e-6)

    # zero grad
    term1_opt.zero_grad(set_to_none=True)  # for new_policy_ego_grad
    term2_opt.zero_grad(set_to_none=True)  # for old_policy_logp_grad
    policy._svo_param.grad.zero_()
    if policy.config[USE_DISTRIBUTIONAL_SVO]:
        policy._svo_std_param.grad.zero_()


    ret = [None, policy._svo.item(), policy._svo_param.item(),
           new_policy_ego_loss.item(), old_policy_logp_loss.item(),
           svo_svo_adv_loss.item(), final_loss.item()]
    if policy.config[USE_DISTRIBUTIONAL_SVO]:
        ret.append(policy._svo_std_param.item())
        ret.append(policy._svo_std.item())
    else:
        ret.append(None)
        ret.append(None)

    return ret
