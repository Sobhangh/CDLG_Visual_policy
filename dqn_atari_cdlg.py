# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_ataripy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

try:
    from torchlogix.layers import Binarization, FixedBinarization, LogicConv2d, LogicDense, OrPooling2d
except ImportError:
    Binarization = None
    FixedBinarization = None
    LogicConv2d = None
    LogicDense = None
    OrPooling2d = None

from cleanrl_utils.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from cleanrl_utils.buffers import ReplayBuffer


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    logic_learning_rate: float = 1e-2
    """the learning rate when using the CDLGNN Q-network"""
    agent_arch: str = "cdlgnn"
    """network architecture: 'cnn' or 'cdlgnn'"""
    use_cnn_target: bool = True
    """if True and agent_arch is cdlgnn, use standard CNN for target network instead of CDLGNN"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 1000000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 1000
    """the timesteps it takes to update the target network"""
    batch_size: int = 32
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.10
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 80000
    """timestep to start learning"""
    train_frequency: int = 4
    """the frequency of training"""

    # CDLGNN specific arguments
    logic_lut_rank: int = 2
    """rank of the lookup table for CDLGNN"""
    logic_num_bits: int = 2
    """thermometer bits per input channel for CDLGNN"""
    logic_tree_depth: int = 3
    """logic tree depth for LogicConv2d/LogicDense"""
    logic_k_num: int = 128
    """base kernel width multiplier for CDLGNN"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)

        env.action_space.seed(seed)
        return env

    return thunk


def ensure_nchw(x: torch.Tensor, expected_channels: int = 4) -> torch.Tensor:
    """Convert observations to NCHW if they arrive as NHWC."""
    if x.ndim != 4:
        raise ValueError(f"Expected a 4D tensor, got shape={tuple(x.shape)}")
    if x.shape[1] == expected_channels:
        return x
    if x.shape[-1] == expected_channels:
        return x.permute(0, 3, 1, 2)
    return x


def get_distributive_channel_thresholds(calibration_obs: torch.Tensor, num_bits: int) -> torch.Tensor:
    """Compute distributive thermometer thresholds for NCHW observations in [0, 1]."""
    if Binarization is None:
        raise ImportError("torchlogix is not available. Cannot compute distributive thresholds.")
    return Binarization.get_initial_thresholds(
        calibration_obs,
        num_bits=num_bits,
        one_per="global",
        method="distributive",
    )


def build_logic_q_backbone(
    *,
    h: int,
    k: int,
    c0: int,
    tree_depth: int,
    lut_rank: int,
    parametrization: str = "warp",
) -> tuple[nn.Sequential, int]:
    """Build a shared-style logic backbone for DQN Q-value prediction."""
    backbone = nn.Sequential(
        LogicConv2d(
            in_dim=h,
            channels=c0,
            num_kernels=k,
            tree_depth=tree_depth,
            receptive_field_size=6,
            stride=3,
            padding=0,
            lut_rank=lut_rank,
            connections_kwargs={"init_method": "random-unique"},
            parametrization=parametrization,
        ),
        OrPooling2d(kernel_size=2, stride=2, padding=0),
        LogicConv2d(
            in_dim=13,
            channels=k,
            num_kernels=2 * k,
            tree_depth=tree_depth,
            receptive_field_size=3,
            stride=2,
            padding=0,
            lut_rank=lut_rank,
            connections_kwargs={"init_method": "random-unique", "channel_group_size": 2},
            parametrization=parametrization,
        ),
        LogicConv2d(
            in_dim=6,
            channels=2 * k,
            num_kernels=4 * k,
            tree_depth=tree_depth,
            receptive_field_size=3,
            stride=1,
            padding=0,
            lut_rank=lut_rank,
            connections_kwargs={"init_method": "random-unique", "channel_group_size": 2},
            parametrization=parametrization,
        ),
        LogicConv2d(
            in_dim=4,
            channels=4 * k,
            num_kernels=8 * k,
            tree_depth=tree_depth,
            receptive_field_size=3,
            stride=1,
            padding=0,
            lut_rank=lut_rank,
            connections_kwargs={"init_method": "random-unique", "channel_group_size": 2},
            parametrization=parametrization,
        ),
        nn.Flatten(),
        LogicDense(
            in_dim=8 * k * 2 * 2,
            out_dim=8 * k,
            parametrization=parametrization,
            lut_rank=lut_rank,
            connections_kwargs={"init_method": "random-unique"},
        ),
    )
    return backbone, 8 * k


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.single_action_space.n),
        )

    def forward(self, x):
        x = ensure_nchw(x, expected_channels=4).float()
        return self.network(x / 255.0)


class LogicQNetwork(nn.Module):
    def __init__(self, env, args: Args, thresholds: torch.Tensor):
        super().__init__()
        if LogicConv2d is None:
            raise ImportError("torchlogix is not available. Install torchlogix to use --agent-arch cdlgnn")

        obs_shape = env.single_observation_space.shape
        expected_channels = 4
        if len(obs_shape) == 3:
            if obs_shape[0] == expected_channels:
                _, h, _ = obs_shape
            elif obs_shape[-1] == expected_channels:
                h, _, _ = obs_shape
            else:
                raise ValueError(f"Unsupported observation shape for Atari frame stack: {obs_shape}")
        else:
            raise ValueError(f"Expected 3D observation shape, got: {obs_shape}")

        self.expected_channels = expected_channels
        self.binarization = FixedBinarization(
            thresholds=thresholds,
            feature_dim=1,
        )

        c0 = expected_channels * args.logic_num_bits
        k = args.logic_k_num
        n_actions = env.single_action_space.n

        self.backbone, backbone_out_dim = build_logic_q_backbone(
            h=h,
            k=k,
            c0=c0,
            tree_depth=args.logic_tree_depth,
            lut_rank=args.logic_lut_rank,
            parametrization="warp",
        )
        self.q_head = LogicDense(
            in_dim=backbone_out_dim,
            out_dim=n_actions,
            parametrization="warp",
            lut_rank=args.logic_lut_rank,
            connections_kwargs={"init_method": "random-unique"},
        )

    def forward(self, x):
        x = ensure_nchw(x, expected_channels=self.expected_channels).float() / 255.0
        x = self.binarization(x)
        x = self.backbone(x)
        return self.q_head(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    if args.agent_arch.lower() == "cdlgnn":
        required_layers = {
            "Binarization": Binarization,
            "FixedBinarization": FixedBinarization,
            "LogicConv2d": LogicConv2d,
            "LogicDense": LogicDense,
            "OrPooling2d": OrPooling2d,
        }
        missing = [name for name, sym in required_layers.items() if sym is None]
        if missing:
            raise ImportError(
                "CDLGNN mode was requested, but torchlogix layers are unavailable: "
                + ", ".join(missing)
                + ". Install torchlogix, for example with 'pip install -e ./torchlogix'."
            )

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)

    thresholds = None
    if args.agent_arch.lower() == "cdlgnn":
        calibration_obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
        calibration_obs = ensure_nchw(calibration_obs, expected_channels=4) / 255.0
        thresholds = get_distributive_channel_thresholds(
            calibration_obs=calibration_obs,
            num_bits=args.logic_num_bits,
        )
        q_network = LogicQNetwork(envs, args=args, thresholds=thresholds).to(device)
        if args.use_cnn_target:
            # Use standard CNN for target network
            target_network = QNetwork(envs).to(device)
        else:
            # Use CDLGNN for target network
            target_network = LogicQNetwork(envs, args=args, thresholds=thresholds).to(device)
            target_network.load_state_dict(q_network.state_dict())
        
        optimizer = optim.Adam(q_network.parameters(), lr=args.logic_learning_rate)
    else:
        q_network = QNetwork(envs).to(device)
        target_network = QNetwork(envs).to(device)
        target_network.load_state_dict(q_network.state_dict())
        optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.as_tensor(obs, dtype=torch.float32, device=device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

    if args.save_model:
        os.makedirs(f"runs/{run_name}", exist_ok=True)
        if args.agent_arch.lower() == "cdlgnn":
            model_path = f"runs/{run_name}/{args.exp_name}_cdlgnn_checkpoint.pt"
            torch.save(
                {
                    "model_state_dict": q_network.state_dict(),
                    "target_model_state_dict": target_network.state_dict(),
                    "args": vars(args),
                    "thresholds": thresholds.detach().cpu() if thresholds is not None else None,
                },
                model_path,
            )
            print(f"CDLGNN checkpoint saved to {model_path}")
        else:
            model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
            torch.save(q_network.state_dict(), model_path)
            print(f"model saved to {model_path}")
            from cleanrl_utils.evals.dqn_eval import evaluate

            episodic_returns = evaluate(
                model_path,
                make_env,
                args.env_id,
                eval_episodes=10,
                run_name=f"{run_name}-eval",
                Model=QNetwork,
                device=device,
                epsilon=args.end_e,
            )
            for idx, episodic_return in enumerate(episodic_returns):
                writer.add_scalar("eval/episodic_return", episodic_return, idx)

            if args.upload_model:
                from cleanrl_utils.huggingface import push_to_hub

                repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
                repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
                push_to_hub(args, episodic_returns, repo_id, "DQN", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
