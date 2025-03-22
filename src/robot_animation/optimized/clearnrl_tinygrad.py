"""
Pytorch stuff first 
"""
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tinygrad import Tensor


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """CleanRL's default layer initialization"""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# This replaces gymnasium's NormalizeObservation wrapper
# NOTE: Tried BatchNorm1d with momentum=None, but the policy did not learn. Check again later.
class RunningNorm(nn.Module):
    def __init__(self, shape: int, epsilon=1e-5, clip=10.0):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros((1, shape), dtype=torch.float32))
        self.register_buffer("running_var", torch.ones((1, shape), dtype=torch.float32))
        self.register_buffer("count", torch.ones(1, dtype=torch.float32))
        self.epsilon = epsilon
        self.clip = clip

    def forward(self, x):
        return torch.clamp(
            (x - self.running_mean.expand_as(x))
            / torch.sqrt(self.running_var.expand_as(x) + self.epsilon),
            -self.clip,
            self.clip,
        )

    @torch.jit.ignore
    def update(self, x):
        # NOTE: Separated update from forward to compile the policy
        # update() must be called to update the running mean and var
        if self.training:
            with torch.no_grad():
                x = x.float()
                assert x.dim() == 2, "x must be 2D"
                mean = x.mean(0, keepdim=True)
                var = x.var(0, unbiased=False, keepdim=True)
                weight = 1 / self.count
                self.running_mean = self.running_mean * (1 - weight) + mean * weight
                self.running_var = self.running_var * (1 - weight) + var * weight
                self.count += 1

    # NOTE: below are needed to torch.save() the model
    @torch.jit.ignore
    def __getstate__(self):
        return {
            "running_mean": self.running_mean,
            "running_var": self.running_var,
            "count": self.count,
            "epsilon": self.epsilon,
            "clip": self.clip,
        }

    @torch.jit.ignore
    def __setstate__(self, state):
        self.running_mean = state["running_mean"]
        self.running_var = state["running_var"]
        self.count = state["count"]
        self.epsilon = state["epsilon"]
        self.clip = state["clip"]


def entropy(logits):
    min_real = torch.finfo(logits.dtype).min
    logits = torch.clamp(logits, min=min_real)
    p_log_p = logits * logits_to_probs(logits)
    return -p_log_p.sum(-1)

def log_prob(logits, value):
    value = value.long().unsqueeze(-1)
    value, log_pmf = torch.broadcast_tensors(value, logits)
    value = value[..., :1]
    return log_pmf.gather(-1, value).squeeze(-1)

def logits_to_probs(logits, is_binary=False):
    r"""
    Converts a tensor of logits into probabilities. Note that for the
    binary case, each value denotes log odds, whereas for the
    multi-dimensional case, the values along the last dimension denote
    the log probabilities (possibly unnormalized) of the events.
    """
    if is_binary:
        return torch.sigmoid(logits)
    return F.softmax(logits, dim=-1)

def sample_logits(logits: Union[torch.Tensor, List[torch.Tensor]],
        action=None, is_continuous=False):
    is_discrete = isinstance(logits, torch.Tensor)
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

class Policy(torch.nn.Module):
    '''Wrap a non-recurrent PyTorch model for use with CleanRL'''
    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        self.is_continuous = hasattr(policy, 'is_continuous') and policy.is_continuous

    def get_value(self, x, state=None):
        _, value = self.policy(x)
        return value

    def get_action_and_value(self, x, action=None):
         logits, value = self.policy(x)
         action, logprob, entropy = sample_logits(logits, action, self.is_continuous)
         return action, logprob, entropy, value

    def forward(self, x, action=None):
        return self.get_action_and_value(x, action)
    
class CleanRLPolicy(Policy):
    def __init__(self, envs, hidden_size=64):
        super().__init__(policy=None)  # Just to get the right init
        self.is_continuous = True

        self.obs_size = np.array(envs.single_observation_space.shape).prod()
        action_size = np.prod(envs.single_action_space.shape)

        self.obs_norm = torch.jit.script(RunningNorm(self.obs_size))

        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.obs_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )

        self.actor_encoder = nn.Sequential(
            layer_init(nn.Linear(self.obs_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
        )
        self.actor_decoder_mean = layer_init(nn.Linear(hidden_size, action_size), std=0.01)
        self.actor_decoder_logstd = nn.Parameter(torch.zeros(1, action_size))

    def get_value(self, x):
        x = x.float()
        x = self.obs_norm(x)
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        x = x.float()
        x = self.obs_norm(x)
        batch = x.shape[0]

        encoding = self.actor_encoder(x)
        action_mean = self.actor_decoder_mean(encoding)
        action_logstd = self.actor_decoder_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)

        # NOTE: This produces nans, so disabling for now
        # Work around for CUDA graph capture
        # if torch.cuda.is_current_stream_capturing():
        #     if action is None:
        #         action = action_mean + action_std * torch.randn_like(action_mean)

        #     # Avoid using the torch.distributions.Normal
        #     log_probs = (-0.5 * (((action - action_mean) / action_std) ** 2 + 2 * action_std.log() + torch.log(torch.tensor(2 * torch.pi)))).sum(1)
        #     logits_entropy = (action_std.log() + 0.5 * torch.log(torch.tensor(2 * torch.pi * torch.e))).sum(1)

        # else:

        logits = torch.distributions.Normal(action_mean, action_std)
        if action is None:
            action = logits.sample()
        log_probs = logits.log_prob(action.view(batch, -1)).sum(1)
        logits_entropy = logits.entropy().sum(1)

        # NOTE: entropy can go negative, when std is small (e.g. 0.1)
        return action, log_probs, logits_entropy, self.critic(x)

    def update_obs_stats(self, x):
        self.obs_norm.update(x)

"""
Next, we will one by one translate the torch code to tinygrad
"""

def tiny_orthogonal_(tensor: Tensor, gain=1, generator=None):



