"""
CleanRL PPO implementation in TinyGrad [In Progress]

Pytorch stuff first 
"""
import math
from typing import List, Tuple, Union

import numpy as np
import torch
from tinygrad import Tensor, dtypes, nn


def entropy(logits: Tensor) -> Tensor:
    min_real = dtypes.min(logits.dtype)
    logits = logits.clamp(min_=min_real)
    p_log_p = logits * logits_to_probs(logits)
    return -p_log_p.sum(-1)

def log_prob(logits, value):
    value = value.long().unsqueeze(-1)
    value, log_pmf = torch.broadcast_tensors(value, logits)
    value = value[..., :1]
    return log_pmf.gather(-1, value).squeeze(-1)

def logits_to_probs(logits: Tensor, is_binary: bool = False):
    if is_binary:
        return logits.sigmoid()
    return logits.softmax()

# TODO: rewrite in tinygrad. Leaving this as a placeholder here since we are only using the cleanrl policy
def sample_logits(
    logits: Union[Tensor, List[Tensor]],
    action=None,
    is_continuous=False
) -> Tuple[Tensor, Tensor, Tensor]:
    is_discrete = isinstance(logits, Tensor)
    if is_continuous:
        batch = logits.loc.shape[0]
        if action is None:
            action = logits.sample().view(batch, -1)

        log_probs = logits.log_prob(action.view(batch, -1)).sum(1)
        logits_entropy = logits.entropy().view(batch, -1).sum(1)
        return action, log_probs, logits_entropy
    elif is_discrete:
        normalized_logits = [logits - logits.logsumexp(dim=-1, keepdim=True)]
        logits = [logits]
    else: # not sure what else it could be
        normalized_logits = [l - l.logsumexp(dim=-1, keepdim=True) for l in logits]
    
    if action is None:
        action = torch.stack([torch.multinomial(logits_to_probs(l), 1).squeeze() for l in logits])
    else:
        batch = logits[0].shape[0]
        action = action.view(batch, -1).T

    assert len(logits) == len(action)
    logprob = torch.stack([log_prob(l, a) for l, a in zip(normalized_logits, action)]).T.sum(1)
    logits_entropy = torch.stack([entropy(l) for l in normalized_logits]).T.sum(1)

    if is_discrete:
        return action.squeeze(0), logprob.squeeze(0), logits_entropy.squeeze(0)

    return action.T, logprob, logits_entropy


def layer_init(layer: nn.Linear, std=np.sqrt(2), bias_const=0.0):
    """CleanRL's default layer initialization"""
    layer.weight = tiny_orthogonal_(layer.weight, std)
    layer.bias = tiny_constant_(layer.bias, bias_const)
    return layer


def tiny_orthogonal_(tensor: Tensor, gain=1, generator=None):
    """
    NOTE: Since initialization occurs only once, we are being lazy and using numpy linear algebra to perform certain operations.
    TODO: try and convert these to native tinygrad ops
    """
    if tensor.ndim < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    if tensor.numel() == 0:
        return tensor # no-op for empty tensors

    rows, cols = tensor.shape[0], tensor.numel() // tensor.shape[0]
    flattened = Tensor.randn(rows, cols) # figure out if it has the same device configs as the input tensor

    if rows < cols:
        flattened = flattened.transpose()

    q, r = np.linalg.qr(flattened.numpy())
    d = np.diag(r, 0)
    ph = np.sign(d)
    q *= ph

    if rows < cols:
        q.transpose()

    return Tensor(q).mul(gain)


def tiny_constant_(tensor: Tensor, val: float):
    """
    """
    return Tensor.ones(tensor.shape) * val


class NormalDistribution:
    def __init__(self, mean: Tensor, std: Tensor):
        self.mean = mean
        self.std = std

    def sample(self):
        noise = Tensor.randn(*self.mean.shape)
        return self.mean + self.std * noise

    def log_prob(self, value: Tensor): # TODO: Double check log_prob formulation
        var = self.std * self.std
        log_scale = self.std.log()
        log_probs = - ((value - self.mean) ** 2) / (2 * var) - log_scale - 0.5 * math.log(2 * math.pi)
        return log_probs.sum(axis=-1)

    def entropy(self): # TODO: Double check entropy formulation
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + self.std.log()
        return entropy.sum(axis=-1)


class TinyPolicy:
    def __init__(self, policy):
        self.policy = policy
        self.is_continuous = hasattr(policy, 'is_continuous') and policy.is_continuous
    
    def __call__(self, x, action=None):
        return self.get_action_and_value(x, action)

    def get_value(self, x, state=None):
        _, value = self.policy(x)
        return value

    def get_action_and_value(self, x, action=None):
        logits, value = self.policy(x)
        action, logprob, entropy = sample_logits(logits, action, self.is_continuous)
        return action, logprob, entropy, value
    
class TinyCleanRLPolicy(TinyPolicy):
    def __init__(self, envs, hidden_size=64):
        super().__init__(policy=None)  # Just to get the right init
        self.is_continuous = True
        # self.obs_size = np.array(envs.single_observation_space.shape).prod()
        # action_size = np.prod(envs.single_action_space.shape) # normzlize observations TODO
        action_size = 1
        self.obs_size = 1
        self.critic = Critic(self.obs_size, hidden_size)
        self.actor_encoder = ActorEncoder(self.obs_size, hidden_size)
        self.actor_decoder_mean = layer_init(nn.Linear(hidden_size, action_size), std=0.01)
        self.actor_decoder_logstd = Tensor.zeros(1, action_size)

    def get_value(self, x):
        x = x.float()
        # x = self.obs_norm(x)
        return self.critic(x)

    def get_action_and_value(self, x: Tensor, action=None):
        x = x.float()
        # x = self.obs_norm(x) # commented out until observation normalization is implemented
        batch = x.shape[0]
        encoding = self.actor_encoder(x)
        action_mean = self.actor_decoder_mean(encoding)
        action_logstd = self.actor_decoder_logstd.expand_as(action_mean)
        action_std = action_logstd.exp()
        
        logits = NormalDistribution(action_mean, action_std)
        if action is None:
            action = logits.sample()
        
        log_probs = logits.log_prob(action.view(batch, -1)).sum(1)
        logits_entropy = logits.entropy().sum(1)

        # NOTE: entropy can go negative, when std is small (e.g. 0.1)
        return action, log_probs, logits_entropy, self.critic(x)
    
    #  def update_obs_stats(self, x):
    #      self.obs_norm.update(x)


class Critic:
    def __init__(self, obs_size, hidden_size):
        self.l1 = layer_init(nn.Linear(obs_size, hidden_size))
        self.l2 = layer_init(nn.Linear(hidden_size, hidden_size))
        self.l3 = layer_init(nn.Linear(hidden_size, 1))

    def __call__(self, x: Tensor):
        x = self.l1(x).tanh()
        x = self.l2(x).tanh()
        return self.l3(x)

class ActorEncoder:
    def __init__(self, obs_size, hidden_size):
        self.l1 = layer_init(nn.Linear(obs_size, hidden_size))
        self.l2 = layer_init(nn.Linear(hidden_size, hidden_size))

    def __call__(self, x: Tensor):
        x = self.l1(x).tanh()
        return self.l2(x).tanh()