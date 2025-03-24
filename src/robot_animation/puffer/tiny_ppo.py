# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import random
import time
from types import SimpleNamespace

import gymnasium
import numpy as np
from tinygrad import Tensor, dtypes, nn

from robot_animation.puffer.clearnrl_tinygrad import TinyCleanRLPolicy
from robot_animation.puffer.puffer_environment import cleanrl_env_creator

if __name__ == "__main__":
    from robot_animation.puffer.train import parse_args
    from robot_animation.puffer.utils import init_wandb

    args_dict, env_name = parse_args()
    run_name = f"cleanrl_{env_name}_{args_dict['train']['seed']}_{int(time.time())}"

    # Translate puffer args to cleanrl args
    args = SimpleNamespace(**args_dict["train"])
    args.env_id = env_name
    args.wandb_project = args_dict["wandb_project"]
    args.wandb_group = args_dict["wandb_group"]
    args.track = args_dict["track"]
    args.capture_video = False
    
    # Match SB3 PPO parameters
    args.learning_rate = 3e-4  # From model_kwargs
    args.num_steps = 1024      # n_steps from model_kwargs
    args.batch_size = 8        # batch_size from model_kwargs
    args.update_epochs = 5     # n_epochs from model_kwargs
    args.num_envs = 1         # Single environment
    
    # These parameters match SB3 defaults: TODO: pass these args instead of hardcoding
    args.gamma = 0.99         # discount factor
    args.gae_lambda = 0.95    # GAE lambda parameter
    args.clip_coef = 0.2      # clip_range in SB3
    args.ent_coef = 0.0       # entropy coefficient
    args.vf_coef = 0.5        # value function coefficient
    args.max_grad_norm = 0.5  # max gradient norm
    args.target_kl = None     # target KL divergence
    args.norm_adv = True      # normalize advantage estimates
    
    # CleanRL specific calculations
    args.batch_size = int(args.num_envs * args.num_steps)
    args.num_minibatches = args.minibatch_size
    args.num_iterations = args.total_timesteps // args.batch_size

    wandb = None
    if args.track:
        wandb = init_wandb(args_dict, run_name)
    episode_stats = {
        "episode_return": [],
        "episode_length": [],
        "average_reward": [],
        "normalized_reward": [],
        "last30episode_return": 0,
    }

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    Tensor.manual_seed(args.seed)


    envs = gymnasium.vector.SyncVectorEnv(
        [
            cleanrl_env_creator(args.env_id, args.capture_video, args.gamma, i)
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(
        envs.single_action_space, gymnasium.spaces.Box
    ), "only continuous action space is supported"

    # agent = CleanRLPolicy(envs).to(device)
    agent = TinyCleanRLPolicy(envs)
    optimizer = nn.optim.Adam(nn.state.get_parameters(agent), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = Tensor.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).contiguous()
    actions = Tensor.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape)
    logprobs = Tensor.zeros((args.num_steps, args.num_envs))
    rewards = Tensor.zeros((args.num_steps, args.num_envs))
    dones = Tensor.zeros((args.num_steps, args.num_envs)).contiguous()
    values = Tensor.zeros((args.num_steps, args.num_envs))

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = Tensor(next_obs, dtype=dtypes.float32)
    next_done = Tensor.zeros(args.num_envs)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            Tensor.no_grad = True
            action, logprob, _, value = agent.get_action_and_value(next_obs)
            values[step] = value.flatten()
            Tensor.no_grad = False
            
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = Tensor(reward).view(-1)
            next_obs, next_done = (
                Tensor(next_obs),
                Tensor(next_done),
            )

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode_return" in info:
                        print(
                            f"global_step: {global_step}, episode_return: {int(info['episode_return'])}, average_reward: {info['average_reward']:.5f}, normalized_reward: {info['normalized_reward']:.5f}"
                        )
                        episode_stats["episode_return"].append(info["episode_return"])
                        episode_stats["episode_length"].append(info["episode_length"])
                        episode_stats["average_reward"].append(info["average_reward"])
                        episode_stats["normalized_reward"].append(info["normalized_reward"])
                        episode_stats["last30episode_return"] = info["last30episode_return"]

        # bootstrap value if not done
        Tensor.no_grad = True
        next_value = agent.get_value(next_obs).reshape(1, -1)
        advantages = Tensor.zeros_like(rewards)
        lastgaelam = 0
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = (
                delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            )
        returns = advantages + values
        Tensor.no_grad = False

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                Tensor.no_grad = True
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
                Tensor.no_grad = False

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = pg_loss1.maximum(pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    value_diff = newvalue - b_returns[mb_inds]
                    v_loss_unclipped = value_diff ** 2
                    
                    v_clipped = b_values[mb_inds] + value_diff.clamp(
                        -args.vf_clip_coef,
                        args.vf_clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = v_loss_unclipped.maximum(v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        print(f"Steps: {global_step}, SPS: {int(global_step / (time.time() - start_time))}")
        if args.track and wandb is not None:
            wandb.log(
                {
                    "0verview/agent_steps": global_step,
                    "0verview/SPS": int(global_step / (time.time() - start_time)),
                    "0verview/epoch": iteration,
                    "0verview/learning_rate": optimizer.param_groups[0]["lr"],
                    "losses/value_loss": v_loss.item(),
                    "losses/policy_loss": pg_loss.item(),
                    "losses/entropy": entropy_loss.item(),
                    "losses/old_approx_kl": old_approx_kl.item(),
                    "losses/approx_kl": approx_kl.item(),
                    "losses/clipfrac": np.mean(clipfracs),
                    "losses/explained_variance": explained_var,
                    "environment/episode_return": np.mean(episode_stats["episode_return"]),
                    "environment/episode_length": np.mean(episode_stats["episode_length"]),
                    "environment/average_reward": np.mean(episode_stats["average_reward"]),
                    "environment/normalized_reward": np.mean(episode_stats["normalized_reward"]),
                    "environment/last30episode_return": episode_stats["last30episode_return"],
                }
            )
        episode_stats["episode_return"].clear()
        episode_stats["episode_length"].clear()
        episode_stats["average_reward"].clear()
        episode_stats["normalized_reward"].clear()

        # TODO: save model

    envs.close()
    if args.track and wandb is not None:
        wandb.finish()
