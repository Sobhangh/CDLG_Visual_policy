# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import os
import random
import time
import importlib
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
#from torchlogix.src.torchlogix.layers.binarization import FixedBinarization
from tqdm import tqdm
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

# drive = None
# try:
#     drive = importlib.import_module("google.colab.drive")
# except ImportError:
#     drive = None
try:
    import ale_py
except ImportError:
    ale_py = None


from torchlogix.layers import (
        Binarization,
        FixedBinarization,
        GroupSum,
        LogicConv2d,
        LogicDense,
        OrPooling2d,
        SoftBinarization,
    )
#from torchlogix.src.torchlogix.layers import Binarization, GroupSum, LogicConv2d, LogicDense, OrPooling2d, SoftBinarization, FixedBinarization



from atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)


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
    precision: str = "float32"
    """floating-point precision: float16 or float32"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    google_colab: bool = False
    """whether this is running on Google Colab (enables some Colab-specific features)"""
    load_model: bool = True
    """if True, load model/optimizer state from model_path before training"""
    model_path: str = ""
    """path to a checkpoint file (.pt) to load before training"""
    cnn_critic_load_model: bool = True
    """if True, load model/optimizer state for CNN critic from model_path before training"""
    cnn_critic_model_path: str = ""

    

    # Algorithm specific arguments
    env_id: str = "PongNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 10_000_000 #10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    logic_learning_rate: float = 3e-2
    """learning rate used when the CDLGNN backbone is selected (torchlogix-typical)"""
    agent_arch: str = "cdlgnn"
    """agent architecture: 'cnn' (original) or 'cdlgnn' (torchlogix logic conv net)"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5 #0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # CDLGNN specific arguments
    logic_lut_rank: int = 2
    """rank of the lookup table for CDLGNN"""
    logic_num_bits: int = 2
    """thermometer bits per input channel for CDLGNN"""
    logic_tree_depth: int = 3
    """logic tree depth for LogicConv2d/LogicDense"""
    logic_k_num: int = 300
    """base kernel width multiplier for CDLGNN"""
    logic_tau: float = 20.0
    """temperature scaling value for logic features (reported for reproducibility)"""
    logic_sampling_temperature: float = 0.1
    """soft binarization temperature during training"""
    logic_actor_group_size: int = 2000
    """number of logic neurons per action class; actor LogicDense outputs n_actions * this,
    then GroupSum sums each group into one logit per action"""
    logic_shared_network: bool = False
    """if True, the CDLGNN backbone is shared between actor and critic (like the original CNN Agent);
    if False (default), actor uses CDLGNN and critic uses a separate standard CNN"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name):
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
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env,4)
        return env

    return thunk


def get_runtime_dtype(device: torch.device, precision: str) -> torch.dtype:
    precision_norm = precision.lower()
    if precision_norm not in {"float16", "float32"}:
        raise ValueError(f"Unsupported precision: {precision}. Choose from: float16, float32.")
    if precision_norm == "float16" and device.type != "cuda":
        print("Requested float16 on non-CUDA device; falling back to float32.")
        return torch.float32
    return torch.float16 if precision_norm == "float16" else torch.float32


def get_module_dtype(module: nn.Module) -> torch.dtype:
    for param in module.parameters():
        return param.dtype
    return torch.float32


def get_module_size_bits(module: nn.Module) -> tuple[int, int]:
    """Return (parameter_count, storage_bits) for a module's parameters."""
    total_params = 0
    total_bits = 0
    for param in module.parameters():
        n = param.numel()
        total_params += n
        total_bits += n * param.element_size() * 8
    return total_params, total_bits

def evaluate(
    agent,
    make_env_fn,
    env_id: str,
    eval_episodes: int,
    run_name: str = "eval",
    device: torch.device = torch.device("cpu"),
    capture_video: bool = False,
    writer=None,
    global_step=0,
):
    envs = gym.vector.SyncVectorEnv([make_env_fn(env_id, 0, capture_video, run_name)])
    agent.eval()
    model_dtype = get_module_dtype(agent)

    obs_raw, _ = envs.reset()
    obs = torch.as_tensor(obs_raw, dtype=model_dtype, device=device)
    episodic_returns = []
    episodic_lengths = []
    while len(episodic_returns) < eval_episodes:
        with torch.no_grad():
            actions, _, _, _ = agent.get_action_and_value(torch.as_tensor(obs, dtype=model_dtype, device=device))
        next_obs_raw, _, _, _, infos = envs.step(actions.cpu().numpy())
        if "episode" in infos:
            ep_mask = infos.get("_episode", np.ones(len(infos["episode"]["r"]), dtype=bool))
            ep = infos["episode"]
            for i, ended in enumerate(ep_mask):
                if ended:
                    episodic_returns.append(float(ep["r"][i]))
                    episodic_lengths.append(int(ep["l"][i]))
        elif "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    episodic_returns.append(float(info["episode"]["r"]))
                    episodic_lengths.append(int(info["episode"]["l"]))
        obs = torch.as_tensor(next_obs_raw, dtype=model_dtype, device=device)

    ret_mean = float(np.mean(episodic_returns))
    ret_std = float(np.std(episodic_returns))
    len_mean = float(np.mean(episodic_lengths))
    if writer is not None:
        writer.add_scalar("eval/episodic_return_mean", ret_mean, global_step or 0)
        writer.add_scalar("eval/episodic_return_std", ret_std, global_step or 0)
        writer.add_scalar("eval/episodic_length_mean", len_mean, global_step or 0)
    agent.train()
    return episodic_returns, episodic_lengths

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    # orthogonal_ is not implemented for CPU float16; initialize in float32 and cast back.
    weight_dtype = layer.weight.dtype
    if layer.weight.device.type == "cpu" and weight_dtype == torch.float16:
        with torch.no_grad():
            w = layer.weight.data.float()
            torch.nn.init.orthogonal_(w, std)
            layer.weight.data.copy_(w.to(dtype=weight_dtype))
    else:
        torch.nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


def ensure_nchw(x: torch.Tensor, expected_channels: int = 4) -> torch.Tensor:
    """Convert observations to NCHW if they arrive as NHWC."""
    if x.ndim != 4:
        raise ValueError(f"Expected a 4D tensor, got shape={tuple(x.shape)}")
    if x.shape[1] == expected_channels:
        return x
    if x.shape[-1] == expected_channels:
        return x.permute(0, 3, 1, 2)
    return x


def save_one_grayscale_png(obs_batch: np.ndarray, run_name: str, env_index: int = 0) -> str | None:
    """Save one grayscale frame from a stacked Atari observation as PNG."""
    try:
        from PIL import Image
    except ImportError:
        return None

    sample = np.asarray(obs_batch[env_index])
    if sample.ndim == 3:
        # Stacked grayscale can arrive either as HWC (84,84,4) or CHW (4,84,84).
        if sample.shape[-1] == 4:
            frame = sample[..., -1]
        elif sample.shape[0] == 4:
            frame = sample[-1, ...]
        else:
            frame = sample[..., 0]
    elif sample.ndim == 2:
        frame = sample
    else:
        return None

    frame_uint8 = np.asarray(frame).clip(0, 255).astype(np.uint8)
    out_dir = os.path.join("runs", run_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sample_grayscale.png")
    Image.fromarray(frame_uint8, mode="L").save(out_path)
    return out_path


def build_logic_actor_backbone(
    *,
    h: int,
    k: int,
    c0: int,
    actor_out_dim: int,
    tree_depth: int,
    lut_rank: int,
    parametrization: str = "warp",
) -> tuple[nn.Sequential, int]:
    """Build the shared logic actor backbone used by both CDLG agent variants.

    Spatial transitions for 84x84 input:
    84x84 -> conv(6,s3)=27x27 -> pool(2,s2)=13x13 -> conv(3,s2)=6x6 -> conv(3,s1)=4x4 -> conv(3,s1)=2x2
    """
    channels_per_group = None # 2
    backbone = nn.Sequential(
        LogicConv2d(
            in_dim=h,
            channels=c0,
            num_kernels=k,
            tree_depth=5,
            receptive_field_size=6,
            stride=3,
            padding=0,
            lut_rank=lut_rank,
            grad_factor=2,
            connections_kwargs={"init_method": "random-unique"},
            parametrization=parametrization,
        ),
        OrPooling2d(kernel_size=2, stride=2, padding=0),
        LogicConv2d(
            in_dim=13,
            channels=k,
            num_kernels=4 * k,
            tree_depth=tree_depth,
            receptive_field_size=3,
            stride=2,
            padding=0,
            lut_rank=lut_rank,
            grad_factor=2,
            connections_kwargs={"init_method": "random-unique", "channel_group_size": channels_per_group},
            parametrization=parametrization,
        ),
        LogicConv2d(
            in_dim=6,
            channels=4 * k,
            num_kernels=16 * k,
            tree_depth=tree_depth,
            receptive_field_size=3,
            stride=1,
            padding=0,
            grad_factor=2,
            lut_rank=lut_rank,
            connections_kwargs={"init_method": "random-unique", "channel_group_size": channels_per_group},
            parametrization=parametrization,
        ),
        LogicConv2d(
            in_dim=4,
            channels=16 * k,
            num_kernels=64 * k,
            tree_depth=tree_depth,
            receptive_field_size=3,
            stride=1,
            padding=0,
            grad_factor=2,
            lut_rank=lut_rank,
            connections_kwargs={"init_method": "random-unique", "channel_group_size": channels_per_group},
            parametrization=parametrization,
        ),
        nn.Flatten(),
        LogicDense(
            in_dim=64 * k * 2 * 2,
            out_dim=actor_out_dim * 4,
            parametrization=parametrization,
            lut_rank=lut_rank,
            grad_factor=2,
            connections_kwargs={"init_method": "random-unique"},
        ),
        LogicDense(
            in_dim=actor_out_dim * 4,
            out_dim=actor_out_dim * 2,
            parametrization=parametrization,
            lut_rank=lut_rank,
            grad_factor=2,
            connections_kwargs={"init_method": "random-unique"},
        ),
    )
    return backbone, actor_out_dim * 2


class CDLGAagent(nn.Module):
    """PPO agent where only the actor uses a TorchLogix CDLGNN backbone."""

    def __init__(self, envs, args: Args, thresholds: torch.Tensor):
        super().__init__()
        if LogicConv2d is None:
            raise ImportError("torchlogix is not available. Install torchlogix to use --agent-arch cdlgnn")

        obs_shape = envs.single_observation_space.shape
        expected_channels = 4
        if len(obs_shape) == 3:
            if obs_shape[0] == expected_channels:
                _, h, w = obs_shape
            elif obs_shape[-1] == expected_channels:
                h, w, _ = obs_shape
            else:
                raise ValueError(f"Unsupported observation shape for Atari frame stack: {obs_shape}")
        else:
            raise ValueError(f"Expected 3D observation shape, got: {obs_shape}")

        self.expected_channels = expected_channels
        self.logic_tau = args.logic_tau
        self.binarization = FixedBinarization(
            thresholds=thresholds,
            #temperature=args.logic_sampling_temperature,
            feature_dim=1,
        )

        c0 = expected_channels * args.logic_num_bits
        k = args.logic_k_num
        n_actions = envs.single_action_space.n
        actor_out_dim = n_actions * args.logic_actor_group_size

        self.actor_logic_backbone, actor_backbone_out_dim = build_logic_actor_backbone(
            h=h,
            k=k,
            c0=c0,
            actor_out_dim=actor_out_dim,
            tree_depth=args.logic_tree_depth,
            lut_rank=args.logic_lut_rank,
            parametrization="warp",
        )

        # Keep a standard CNN critic path for PPO value estimation.
        self.critic_network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )

        # GroupSum: sums each group of logic_actor_group_size neurons into one action logit.
        self.actor = nn.Sequential(
            LogicDense(
                in_dim=actor_backbone_out_dim,
                out_dim=actor_out_dim,
                parametrization="warp",
                lut_rank=args.logic_lut_rank,
                connections_kwargs={"init_method": "random-unique"},
            ),
            GroupSum(k=n_actions, tau=args.logic_tau),
        )
        self.critic = nn.Linear(512, 1)

        backbone_params, backbone_bits = get_module_size_bits(self.actor_logic_backbone)
        actor_head_params, actor_head_bits = get_module_size_bits(self.actor)
        print(
            "[CDLGAagent] actor_logic_backbone: "
            f"params={backbone_params:,}, size={backbone_bits:,} bits ({backbone_bits / 8 / (1024 ** 2):.2f} MiB)"
        )
        print(
            "[CDLGAagent] actor_head: "
            f"params={actor_head_params:,}, size={actor_head_bits:,} bits ({actor_head_bits / 8 / (1024 ** 2):.2f} MiB)"
        )

    def _actor_features(self, x: torch.Tensor) -> torch.Tensor:
        #x = ensure_nchw(x, expected_channels=self.expected_channels).float() / 255.0
        x = self.binarization(x/255.0)
        #print(f"Actor backbone input shape: {x.shape}")
        return self.actor_logic_backbone(x)

    def _critic_features(self, x: torch.Tensor) -> torch.Tensor:
        #x = ensure_nchw(x, expected_channels=self.expected_channels).float() / 255.0
        return self.critic_network(x/255.0)

    def get_value(self, x):
        return self.critic(self._critic_features(x))

    def get_action_and_value(self, x, action=None):
        actor_hidden = self._actor_features(x)
        logits = self.actor(actor_hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        value = self.critic(self._critic_features(x))
        return action, probs.log_prob(action), probs.entropy(), value


class CDLGAgentShared(nn.Module):
    """PPO agent with a shared TorchLogix CDLGNN backbone for both actor and critic."""

    def __init__(self, envs, args: Args, thresholds: torch.Tensor):
        super().__init__()
        if LogicConv2d is None:
            raise ImportError("torchlogix is not available. Install torchlogix to use --agent-arch cdlgnn")

        obs_shape = envs.single_observation_space.shape
        expected_channels = 4
        if len(obs_shape) == 3:
            if obs_shape[0] == expected_channels:
                _, h, w = obs_shape
            elif obs_shape[-1] == expected_channels:
                h, w, _ = obs_shape
            else:
                raise ValueError(f"Unsupported observation shape for Atari frame stack: {obs_shape}")
        else:
            raise ValueError(f"Expected 3D observation shape, got: {obs_shape}")

        self.expected_channels = expected_channels
        self.binarization = FixedBinarization(
            thresholds=thresholds,
            #temperature=args.logic_sampling_temperature,
            feature_dim=1,
        )

        c0 = expected_channels * args.logic_num_bits
        k = args.logic_k_num
        n_actions = envs.single_action_space.n
        actor_out_dim = n_actions * args.logic_actor_group_size

        self.backbone, actor_backbone_out_dim = build_logic_actor_backbone(
            h=h,
            k=k,
            c0=c0,
            actor_out_dim=actor_out_dim,
            tree_depth=args.logic_tree_depth,
            lut_rank=args.logic_lut_rank,
            parametrization="warp",
        )
        
        # GroupSum: sums each group of logic_actor_group_size neurons into one action logit.
        self.actor = nn.Sequential(
            LogicDense(
                in_dim=actor_backbone_out_dim,
                out_dim=actor_out_dim,
                parametrization="warp",
                lut_rank=args.logic_lut_rank,
                connections_kwargs={"init_method": "random-unique"},
            ),
            GroupSum(k=n_actions, tau=args.logic_tau))
        # Critic takes the full backbone representation to predict state value.
        self.critic = layer_init(nn.Linear(actor_backbone_out_dim, 1))

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        #x = ensure_nchw(x, expected_channels=self.expected_channels).float() / 255.0
        x = self.binarization(x / 255.0)
        return self.backbone(x)

    def get_value(self, x):
        return self.critic(self._features(x))

    def get_action_and_value(self, x, action=None):
        hidden = self._features(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


def get_distributive_channel_thresholds(calibration_obs: torch.Tensor, num_bits: int) -> torch.Tensor:
    """Compute distributive thermometer thresholds per channel for NCHW observations in [0, 1]."""
    if Binarization is None:
        raise ImportError("torchlogix is not available. Cannot compute distributive thresholds.")
    return Binarization.get_initial_thresholds(
        calibration_obs,
        num_bits=num_bits,
        one_per="global",  #"channel",
        method="distributive",
    )


def maybe_enable_multi_gpu(agent: nn.Module, device: torch.device) -> nn.Module:
    if device.type != "cuda":
        return agent
    n_gpus = torch.cuda.device_count()
    if n_gpus <= 1:
        return agent

    if isinstance(agent, CDLGAagent):
        print(f"Using {n_gpus} GPUs for CDLG actor modules.")
        # Wrap actor branches only; keep top-level agent API unchanged.
        agent.actor_logic_backbone = nn.DataParallel(agent.actor_logic_backbone)
        agent.actor = nn.DataParallel(agent.actor)
    elif isinstance(agent, CDLGAgentShared):
        print(f"Using {n_gpus} GPUs for shared CDLG backbone + actor.")
        agent.backbone = nn.DataParallel(agent.backbone)
        agent.actor = nn.DataParallel(agent.actor)
    return agent


def _to_portable_cdlg_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    portable: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        key = key.replace("actor_logic_backbone.module.", "actor_logic_backbone.")
        key = key.replace("backbone.module.", "backbone.")
        key = key.replace("actor.module.", "actor.")
        portable[key] = value
    return portable


def _adapt_cdlg_state_dict_for_agent(agent: nn.Module, model_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    state = _to_portable_cdlg_state_dict(model_state_dict)
    target_state = agent.state_dict()

    needs_actor_logic_module = any(k.startswith("actor_logic_backbone.module.") for k in target_state.keys())
    needs_backbone_module = any(k.startswith("backbone.module.") for k in target_state.keys())
    needs_actor_module = any(k.startswith("actor.module.") for k in target_state.keys())

    adapted: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        new_key = key
        if needs_actor_logic_module and key.startswith("actor_logic_backbone.") and not key.startswith("actor_logic_backbone.module."):
            new_key = key.replace("actor_logic_backbone.", "actor_logic_backbone.module.", 1)
        elif needs_backbone_module and key.startswith("backbone.") and not key.startswith("backbone.module."):
            new_key = key.replace("backbone.", "backbone.module.", 1)
        elif needs_actor_module and key.startswith("actor.") and not key.startswith("actor.module."):
            new_key = key.replace("actor.", "actor.module.", 1)
        adapted[new_key] = value
    return adapted


def save_checkpoint(
    *,
    checkpoint_path: str,
    agent: nn.Module,
    optimizer: optim.Optimizer,
    args: Args,
    thresholds: torch.Tensor | None = None,
):
    checkpoint = {
        "model_state_dict": _to_portable_cdlg_state_dict(agent.state_dict()),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
    }

    if args.agent_arch.lower() == "cdlgnn":
        threshold_tensor = thresholds
        if threshold_tensor is None and hasattr(agent, "binarization") and hasattr(agent.binarization, "thresholds"):
            threshold_tensor = agent.binarization.thresholds
        if threshold_tensor is not None:
            checkpoint["thresholds"] = threshold_tensor.detach().cpu()

    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    *,
    checkpoint_path: str,
    agent: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    load_optimizer_state: bool = True,
):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state_dict = checkpoint["model_state_dict"]
    agent.load_state_dict(_adapt_cdlg_state_dict_for_agent(agent, model_state_dict))
    if load_optimizer_state and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


if __name__ == "__main__":
    args = tyro.cli(Args)

    if ale_py is not None:
        gym.register_envs(ale_py)

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    if args.agent_arch.lower() == "cdlgnn":
        required_layers = {
            "Binarization": Binarization,
            "FixedBinarization": FixedBinarization,
            "GroupSum": GroupSum,
            "LogicConv2d": LogicConv2d,
            "LogicDense": LogicDense,
            "OrPooling2d": OrPooling2d,
        }
        missing = [name for name, sym in required_layers.items() if sym is None]
        if missing:
            raise ImportError(
                "CDLGNN mode was requested, but torchlogix layers are unavailable: "
                + ", ".join(missing)
                + ". Activate your project virtual environment and install torchlogix, "
                + "for example with 'pip install -e ./torchlogix'."
            )

    run_name = f"{args.env_id}__{args.agent_arch}__{args.exp_name}__{args.seed}__{int(time.time())}"
    checkpoint_root = "runs"
    if args.google_colab:
        try:
            #Mounting drive from a python script in colab would probably not work
            #drive.mount("/content/drive", force_remount=False)
            checkpoint_root = "/content/drive/MyDrive/VisualPolicyDWN_checkpoints"
        except Exception:
            checkpoint_root = "runs"
    checkpoint_run_dir = os.path.join(checkpoint_root, run_name)
    os.makedirs(checkpoint_run_dir, exist_ok=True)
    print(f"Checkpoint directory: {checkpoint_run_dir}")

    if args.track:
        import wandb
        #TOD DO: remove hardcoded key and use environment variable imported from local file
        wandb.login(key="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")
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
    torch.set_default_dtype(get_runtime_dtype(device, args.precision))
    runtime_dtype = get_runtime_dtype(device, args.precision)

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs_raw, _ = envs.reset(seed=args.seed)
    #saved_png = save_one_grayscale_png(next_obs_raw, run_name=run_name)
    #if saved_png is not None:
    #    print(f"Saved grayscale frame to {saved_png}")
    next_obs = torch.as_tensor(next_obs_raw, dtype=runtime_dtype, device=device)
    next_done = torch.zeros(args.num_envs).to(device)
    thresholds = None

    
    if args.agent_arch.lower() == "cdlgnn":
        calibration_obs = ensure_nchw(next_obs, expected_channels=4) / 255.0
        thresholds = get_distributive_channel_thresholds(
            calibration_obs=calibration_obs,
            num_bits=args.logic_num_bits,
        )
        if args.logic_shared_network:
            print("Using shared CDLGNN backbone for actor and critic.")
            agent = CDLGAgentShared(envs, args=args, thresholds=thresholds).to(device=device, dtype=runtime_dtype)
            agent = maybe_enable_multi_gpu(agent, device)
            optimizer = optim.Adam(agent.parameters(), lr=args.logic_learning_rate, eps=1e-5)
            base_lrs = [args.logic_learning_rate]
        else:
            print("Using separate CDLGNN backbone for actor and standard CNN for critic.")
            agent = CDLGAagent(envs, args=args, thresholds=thresholds).to(device=device, dtype=runtime_dtype)
            agent = maybe_enable_multi_gpu(agent, device)
            optimizer = optim.Adam(
                [
                    {
                        "params": (
                            #list(agent.binarization.parameters()) +
                            list(agent.actor_logic_backbone.parameters())
                            + list(agent.actor.parameters())
                        ),
                        "lr": args.logic_learning_rate,
                    },
                    {
                        "params": list(agent.critic_network.parameters()) + list(agent.critic.parameters()),
                        "lr": args.learning_rate,
                    },
                ],
                eps=1e-5,
            )
            base_lrs = [args.logic_learning_rate, args.learning_rate]
    else:
        agent = Agent(envs).to(device=device, dtype=runtime_dtype)
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
        base_lrs = [args.learning_rate]

    if args.load_model:
        # if not args.model_path:
        #     raise ValueError("--load-model is enabled, but --model-path is empty.")
        # if not os.path.isfile(args.model_path):
        #     raise FileNotFoundError(f"Checkpoint not found: {args.model_path}")
        if args.cnn_critic_load_model:
            agnet_temp = Agent(envs).to(device=device, dtype=runtime_dtype)
            checkpoint = torch.load(args.cnn_critic_model_path, map_location=device)
            agnet_temp.load_state_dict(checkpoint["model_state_dict"])
            agent.critic_network.load_state_dict(agnet_temp.network.state_dict())
            agent.critic.load_state_dict(agnet_temp.critic.state_dict())
            print("Loaded CNN critic state from checkpoint for CDLGNN agent.")
            if args.model_path:
                agent_cdlg_temp = maybe_enable_multi_gpu(CDLGAagent(envs, args=args, thresholds=thresholds), device)
                checkpoint = torch.load(args.model_path, map_location=device)
                agent_cdlg_temp.load_state_dict(_adapt_cdlg_state_dict_for_agent(agent_cdlg_temp, checkpoint["student_state_dict"]))
                if torch.cuda.device_count() > 1:
                    agent.actor_logic_backbone.module.load_state_dict(agent_cdlg_temp.actor_logic_backbone.module.state_dict())
                    agent.actor.module.load_state_dict(agent_cdlg_temp.actor.module.state_dict())
                else:
                    agent.actor_logic_backbone.load_state_dict(agent_cdlg_temp.actor_logic_backbone.state_dict())
                    agent.actor.load_state_dict(agent_cdlg_temp.actor.state_dict())
                print("Loaded CDLGNN actor state from checkpoint for CDLGNN agent. Actor thresholds:")
                print(agent.binarization.thresholds)
            optimizer = optim.Adam([{"params": (list(agent.actor_logic_backbone.parameters())+ list(agent.actor.parameters())),"lr": args.logic_learning_rate,}],eps=1e-5)
        elif args.model_path:
            loaded_checkpoint = load_checkpoint(
                checkpoint_path=args.model_path,
                agent=agent,
                optimizer=optimizer,
                device=device,
                load_optimizer_state=True,
            )
            print("Loaded model and optimizer state from checkpoint. Agent thresholds:")
            print(agent.binarization.thresholds)
            optimizer.param_groups[0]["lr"] = args.logic_learning_rate
            optimizer.param_groups[1]["lr"] = args.learning_rate
            # if args.agent_arch.lower() == "cdlgnn" and "thresholds" in loaded_checkpoint:
            #     thresholds = loaded_checkpoint["thresholds"].to(device)
            #     print(f"Loaded thresholds from checkpoint: {thresholds}")
            #     if hasattr(agent, "binarization") and hasattr(agent.binarization, "thresholds"):
            #         with torch.no_grad():
            #             agent.binarization.thresholds.copy_(thresholds)
            #print(f"Loaded checkpoint from: {args.model_path}")

    # print("Agent state_dict entries:")
    # for name, tensor in agent.state_dict().items():
    #     print(f"{name}: {tuple(tensor.shape)}")
    

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, dtype=runtime_dtype).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    for iteration in tqdm(range(1, args.num_iterations + 1)):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
                param_group["lr"] = frac * base_lr

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.as_tensor(next_obs, dtype=runtime_dtype, device=device)
            next_done = torch.as_tensor(next_done, dtype=torch.float32, device=device)

            if "episode" in infos:
                ep_mask = infos.get("_episode", np.logical_or(terminations, truncations))
                ep = infos["episode"]
                for i, ended in enumerate(ep_mask):
                    if ended:
                        r = float(ep["r"][i])
                        l = int(ep["l"][i])
                        #print(f"global_step={global_step}, episodic_return={r}")
                        writer.add_scalar("charts/episodic_return", r, global_step)
                        writer.add_scalar("charts/episodic_length", l, global_step)
            elif "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        r = float(info["episode"]["r"])
                        l = int(info["episode"]["l"])
                        #print(f"global_step={global_step}, episodic_return={r}")
                        writer.add_scalar("charts/episodic_return", r, global_step)
                        writer.add_scalar("charts/episodic_length", l, global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

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

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
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
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        #print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        if iteration % (args.num_iterations // 10 ) == 0:
            eval_returns, eval_lengths = evaluate(
                agent,
                make_env_fn=make_env,
                env_id=args.env_id,
                eval_episodes=10,
                writer=writer,
                device=device,
            )
            rt_mean = float(np.mean(eval_returns))
            ret_std = float(np.std(eval_returns))
            print(f"mean return={rt_mean:.2f} +/- {ret_std:.2f}; mean length={np.mean(eval_lengths):.2f}")

        if iteration % (args.num_iterations // 4 ) == 0:
            checkpoint_path = os.path.join(checkpoint_run_dir, f"checkpoint_{global_step}.pt")
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                agent=agent,
                optimizer=optimizer,
                args=args,
                thresholds=thresholds,
            )
            print(f"Saved checkpoint: {checkpoint_path}")
            

    # Save final checkpoint and keep threshold initialization tensor for reproducibility.
    final_checkpoint_path = os.path.join(checkpoint_run_dir, "checkpoint.pt")
    save_checkpoint(
        checkpoint_path=final_checkpoint_path,
        agent=agent,
        optimizer=optimizer,
        args=args,
        thresholds=thresholds,
    )
    print(f"Saved final checkpoint: {final_checkpoint_path}")

    envs.close()
    writer.close()
