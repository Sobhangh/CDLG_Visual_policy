from __future__ import annotations

import json
import multiprocessing as mp
import os
import random
import time
from dataclasses import asdict, dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from ppo_atari_cdlg import (
	Agent,
	Args as PPOArgs,
	CDLGAagent,
	ensure_nchw,
	evaluate,
	get_distributive_channel_thresholds,
	load_checkpoint,
	make_env,
)


@dataclass
class Args:
	mode: str = "both"
	"""one of: collect, train, both, stream"""

	# Shared
	seed: int = 1
	cuda: bool = True
	env_id: str = "PongNoFrameskip-v4"
	num_envs: int = 2

	# Teacher / collection
	teacher_checkpoint_path: str = ""
	"""path to checkpoint of CNN teacher (Agent class)"""
	random_action_prob: float = 0.05
	"""epsilon for random actions during collection"""
	max_buffer_gb: float = 5.0
	"""hard cap for (observation, logits) buffer size on disk"""
	collect_max_steps: int = 0
	"""optional step cap; 0 means collect until storage cap"""

	# Distillation
	dataset_dir: str = "distill_dataset"
	output_dir: str = "distill_runs"
	batch_size: int = 1024
	num_workers: int = 0
	epochs: int = 50
	eval_episodes: int = 10
	"""number of episodes for final student evaluation"""
	student_lr: float = 3e-2
	temperature: float = 0.1
	"""distillation temperature used in KL loss"""
	validation_fraction: float = 0.1
	"""fraction of dataset held out for validation metrics"""

	# Streaming mode (collect + distill in parallel using shard files)
	stream_dir: str = "distill_stream"
	stream_shard_samples: int = 65536
	"""max samples per shard file"""
	stream_poll_seconds: float = 5.0
	"""how often trainer checks for new shards"""
	stream_min_samples_before_train: int = 100000
	"""wait for at least this many samples before starting training"""
	stream_collector_cuda: bool = False
	"""if True, run streaming collector on CUDA when available"""

	# Student CDLG config (mirrors PPO args)
	logic_lut_rank: int = 2
	logic_num_bits: int = 2
	logic_tree_depth: int = 3
	logic_k_num: int = 512
	logic_tau: float = 10.0
	logic_actor_group_size: int = 1024


class DistillDataset(Dataset):
	def __init__(self, dataset_dir: str):
		meta_path = os.path.join(dataset_dir, "metadata.json")
		if not os.path.isfile(meta_path):
			raise FileNotFoundError(f"Missing metadata file: {meta_path}")
		with open(meta_path, "r", encoding="utf-8") as f:
			self.meta = json.load(f)

		self.n_samples = int(self.meta["n_samples"])
		self.n_actions = int(self.meta["n_actions"])
		self.obs_path = os.path.join(dataset_dir, "observations.uint8.mmap")
		self.logits_path = os.path.join(dataset_dir, "teacher_logits.float16.mmap")

		# Open lazily so each DataLoader worker creates its own local mmap handles.
		self._obs = None
		self._logits = None

	def _ensure_open(self):
		if self._obs is None:
			self._obs = np.memmap(
				self.obs_path,
				mode="r",
				dtype=np.uint8,
				shape=(self.n_samples, 4, 84, 84),
			)
		if self._logits is None:
			self._logits = np.memmap(
				self.logits_path,
				mode="r",
				dtype=np.float16,
				shape=(self.n_samples, self.n_actions),
			)

	def __getstate__(self):
		state = self.__dict__.copy()
		state["_obs"] = None
		state["_logits"] = None
		return state

	def __len__(self) -> int:
		return self.n_samples

	def __getitem__(self, idx: int):
		self._ensure_open()
		obs = torch.from_numpy(np.asarray(self._obs[idx], dtype=np.uint8))
		logits = torch.from_numpy(np.asarray(self._logits[idx], dtype=np.float16))
		return obs, logits


class ShardDistillDataset(Dataset):
	def __init__(self, shard_pairs: list[tuple[str, str]]):
		if not shard_pairs:
			raise ValueError("No shard files were provided.")
		self.shard_pairs = shard_pairs
		self.obs_shapes = []
		self.n_actions = None
		self.cum_sizes = []
		total = 0
		for obs_path, logits_path in shard_pairs:
			obs_arr = np.load(obs_path, mmap_mode="r")
			logits_arr = np.load(logits_path, mmap_mode="r")
			if obs_arr.shape[0] != logits_arr.shape[0]:
				raise ValueError(f"Mismatched shard lengths: {obs_path} vs {logits_path}")
			if obs_arr.shape[1:] != (4, 84, 84):
				raise ValueError(f"Unexpected obs shape in {obs_path}: {obs_arr.shape}")
			if self.n_actions is None:
				self.n_actions = int(logits_arr.shape[1])
			elif int(logits_arr.shape[1]) != self.n_actions:
				raise ValueError(f"Inconsistent action dimension in {logits_path}: {logits_arr.shape}")
			self.obs_shapes.append(obs_arr.shape)
			total += int(obs_arr.shape[0])
			self.cum_sizes.append(total)
			del obs_arr
			del logits_arr
		self.n_samples = total

		# Open lazily so each DataLoader worker creates its own local mmap handles.
		self._obs_arrays = None
		self._logit_arrays = None

	def _ensure_open(self):
		if self._obs_arrays is None or self._logit_arrays is None:
			self._obs_arrays = []
			self._logit_arrays = []
			for obs_path, logits_path in self.shard_pairs:
				self._obs_arrays.append(np.load(obs_path, mmap_mode="r"))
				self._logit_arrays.append(np.load(logits_path, mmap_mode="r"))

	def __getstate__(self):
		state = self.__dict__.copy()
		state["_obs_arrays"] = None
		state["_logit_arrays"] = None
		return state

	def __len__(self) -> int:
		return self.n_samples

	def __getitem__(self, idx: int):
		if idx < 0 or idx >= self.n_samples:
			raise IndexError(idx)
		self._ensure_open()
		shard_idx = int(np.searchsorted(self.cum_sizes, idx, side="right"))
		start = 0 if shard_idx == 0 else self.cum_sizes[shard_idx - 1]
		local_idx = idx - start
		obs = torch.from_numpy(np.asarray(self._obs_arrays[shard_idx][local_idx], dtype=np.uint8))
		logits = torch.from_numpy(np.asarray(self._logit_arrays[shard_idx][local_idx], dtype=np.float16))
		return obs, logits


def set_seed(seed: int):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)


def get_device(cuda: bool) -> torch.device:
	return torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")


def get_runtime_dtype(device: torch.device) -> torch.dtype:
	return torch.float16 if device.type == "cuda" else torch.float32


def get_module_dtype(module: torch.nn.Module) -> torch.dtype:
	for param in module.parameters():
		return param.dtype
	return torch.float32


def build_teacher(envs: gym.vector.SyncVectorEnv, checkpoint_path: str, device: torch.device) -> Agent:
	if not checkpoint_path:
		raise ValueError("--teacher-checkpoint-path must be set for collection.")
	if not os.path.isfile(checkpoint_path):
		raise FileNotFoundError(f"Teacher checkpoint not found: {checkpoint_path}")

	teacher = Agent(envs).to(device)
	optimizer_stub = optim.Adam(teacher.parameters(), lr=2.5e-4, eps=1e-5)
	load_checkpoint(
		checkpoint_path=checkpoint_path,
		agent=teacher,
		optimizer=optimizer_stub,
		device=device,
		load_optimizer_state=False,
	)
	teacher = teacher.to(device=device, dtype=get_runtime_dtype(device))
	teacher.eval()
	return teacher


def get_actor_logits(agent: torch.nn.Module, obs_nchw_float: torch.Tensor) -> torch.Tensor:
	if isinstance(agent, Agent):
		hidden = agent.network(obs_nchw_float / 255.0)
		return agent.actor(hidden)

	if isinstance(agent, CDLGAagent):
		actor_hidden = agent._actor_features(obs_nchw_float)
		return agent.actor(actor_hidden)

	raise TypeError(f"Unsupported agent type for logits extraction: {type(agent)!r}")


def _collection_budget(args: Args, n_actions: int) -> tuple[int, int]:
	obs_shape = (4, 84, 84)
	obs_dtype = np.uint8
	logits_dtype = np.float16
	obs_bytes = int(np.prod(obs_shape) * np.dtype(obs_dtype).itemsize)
	logits_bytes = n_actions * np.dtype(logits_dtype).itemsize
	per_sample_bytes = obs_bytes + logits_bytes
	max_buffer_bytes = int(args.max_buffer_gb * (1024**3))
	max_samples = max_buffer_bytes // per_sample_bytes
	if max_samples <= 0:
		raise ValueError("max_buffer_gb is too small to store even one sample.")
	return max_samples, per_sample_bytes

AVG_EPISODE_LENGTH = 10000  
AVG_SAMPLE_PER_EPISODE = 100
OBS_STEP = AVG_EPISODE_LENGTH // AVG_SAMPLE_PER_EPISODE
def collect_dataset(args: Args):
	set_seed(args.seed)
	device = get_device(args.cuda)
	runtime_dtype = get_runtime_dtype(device)

	os.makedirs(args.dataset_dir, exist_ok=True)
	run_name = f"distill_collect__{args.env_id}__{args.seed}__{int(time.time())}"
	envs = gym.vector.SyncVectorEnv(
		[make_env(args.env_id, i, capture_video=False, run_name=run_name) for i in range(args.num_envs)]
	)
	assert isinstance(envs.single_action_space, gym.spaces.Discrete)
	n_actions = int(envs.single_action_space.n)

	teacher = build_teacher(envs, args.teacher_checkpoint_path, device)
	max_samples, per_sample_bytes = _collection_budget(args, n_actions)

	obs_shape = (4, 84, 84)
	obs_mmap_path = os.path.join(args.dataset_dir, "observations.uint8.mmap")
	logits_mmap_path = os.path.join(args.dataset_dir, "teacher_logits.float16.mmap")
	obs_mm = np.memmap(obs_mmap_path, mode="w+", dtype=np.uint8, shape=(max_samples, *obs_shape))
	logits_mm = np.memmap(logits_mmap_path, mode="w+", dtype=np.float16, shape=(max_samples, n_actions))

	next_obs_raw, _ = envs.reset(seed=args.seed)
	written = 0
	total_steps = 0

	pbar = tqdm(total=max_samples, desc="Collecting distill samples", unit="sample")
	with torch.inference_mode():
		while written < max_samples:
			if args.collect_max_steps > 0 and total_steps >= args.collect_max_steps:
				break
			stepnb = 1 #random.randint(1, OBS_STEP) if total_steps == 0 else OBS_STEP
			for _ in range(stepnb):
				obs_t = torch.as_tensor(next_obs_raw, dtype=runtime_dtype, device=device)
				teacher_logits = get_actor_logits(teacher, obs_t)
				sampled_actions = Categorical(logits=teacher_logits).sample()
				random_mask = torch.rand(args.num_envs, device=device) < args.random_action_prob
				random_actions = torch.randint(0, n_actions, (args.num_envs,), device=device)
				actions = torch.where(random_mask, random_actions, sampled_actions)
				next_obs_raw, _, _, _, _ = envs.step(actions.detach().cpu().numpy())
				total_steps += 1
			obs_t = torch.as_tensor(next_obs_raw, dtype=runtime_dtype, device=device)

			# Keep only samples where the executed action came from the teacher policy.
			policy_idx = (~random_mask).nonzero(as_tuple=False).squeeze(-1)
			if policy_idx.numel() > 0:
				to_write = min(int(policy_idx.numel()), max_samples - written)
				obs_np = obs_t.index_select(0, policy_idx).detach().cpu().to(torch.uint8).numpy()
				logits_np = teacher_logits.index_select(0, policy_idx).detach().cpu().to(torch.float16).numpy()
				obs_mm[written : written + to_write] = obs_np[:to_write]
				logits_mm[written : written + to_write] = logits_np[:to_write]
				written += to_write
				pbar.update(to_write)

			# next_obs_raw, _, _, _, _ = envs.step(actions.detach().cpu().numpy())
			# total_steps += 1

	pbar.close()
	obs_mm.flush()
	logits_mm.flush()

	if written < max_samples:
		trimmed_obs = np.memmap(obs_mmap_path, mode="r", dtype=np.uint8, shape=(max_samples, *obs_shape))
		trimmed_logits = np.memmap(logits_mmap_path, mode="r", dtype=np.float16, shape=(max_samples, n_actions))
		obs_arr = np.asarray(trimmed_obs[:written]).copy()
		logits_arr = np.asarray(trimmed_logits[:written]).copy()

		del trimmed_obs
		del trimmed_logits
		del obs_mm
		del logits_mm

		os.remove(obs_mmap_path)
		os.remove(logits_mmap_path)

		obs_mm = np.memmap(obs_mmap_path, mode="w+", dtype=np.uint8, shape=(written, *obs_shape))
		logits_mm = np.memmap(logits_mmap_path, mode="w+", dtype=np.float16, shape=(written, n_actions))
		obs_mm[:] = obs_arr
		logits_mm[:] = logits_arr
		obs_mm.flush()
		logits_mm.flush()

	meta = {
		"env_id": args.env_id,
		"seed": args.seed,
		"n_samples": int(written),
		"n_actions": int(n_actions),
		"observation_shape": [4, 84, 84],
		"observation_dtype": "uint8",
		"logits_dtype": "float16",
		"random_action_prob": float(args.random_action_prob),
		"teacher_checkpoint_path": args.teacher_checkpoint_path,
		"max_buffer_gb": float(args.max_buffer_gb),
		"bytes_used": int(written * per_sample_bytes),
		"run_name": run_name,
	}
	with open(os.path.join(args.dataset_dir, "metadata.json"), "w", encoding="utf-8") as f:
		json.dump(meta, f, indent=2)

	envs.close()
	print(f"Collected {written} samples into {args.dataset_dir}")
	print(f"Approx bytes used: {meta['bytes_used'] / (1024**3):.2f} GB")


def _collect_stream_shards(args: Args):
	set_seed(args.seed)
	device = get_device(args.stream_collector_cuda and args.cuda)
	runtime_dtype = get_runtime_dtype(device)
	os.makedirs(args.stream_dir, exist_ok=True)

	run_name = f"distill_stream_collect__{args.env_id}__{args.seed}__{int(time.time())}"
	envs = gym.vector.SyncVectorEnv(
		[make_env(args.env_id, i, capture_video=False, run_name=run_name) for i in range(args.num_envs)]
	)
	n_actions = int(envs.single_action_space.n)
	teacher = build_teacher(envs, args.teacher_checkpoint_path, device)
	max_samples, per_sample_bytes = _collection_budget(args, n_actions)

	next_obs_raw, _ = envs.reset(seed=args.seed)
	written_total = 0
	total_steps = 0
	shard_idx = 0

	while written_total < max_samples:
		if args.collect_max_steps > 0 and total_steps >= args.collect_max_steps:
			break

		current_capacity = min(args.stream_shard_samples, max_samples - written_total)
		obs_buf = np.empty((current_capacity, 4, 84, 84), dtype=np.uint8)
		logits_buf = np.empty((current_capacity, n_actions), dtype=np.float16)
		written = 0

		with torch.inference_mode():
			while written < current_capacity:
				if args.collect_max_steps > 0 and total_steps >= args.collect_max_steps:
					break
				obs_t = torch.as_tensor(next_obs_raw, dtype=runtime_dtype, device=device)
				obs_t = ensure_nchw(obs_t, expected_channels=4)

				teacher_logits = get_actor_logits(teacher, obs_t)
				sampled_actions = Categorical(logits=teacher_logits).sample()
				random_mask = torch.rand(args.num_envs, device=device) < args.random_action_prob
				random_actions = torch.randint(0, n_actions, (args.num_envs,), device=device)
				actions = torch.where(random_mask, random_actions, sampled_actions)

				policy_idx = (~random_mask).nonzero(as_tuple=False).squeeze(-1)
				if policy_idx.numel() > 0:
					to_write = min(int(policy_idx.numel()), current_capacity - written)
					obs_buf[written : written + to_write] = (
						obs_t.index_select(0, policy_idx).detach().cpu().to(torch.uint8).numpy()[:to_write]
					)
					logits_buf[written : written + to_write] = (
						teacher_logits.index_select(0, policy_idx).detach().cpu().to(torch.float16).numpy()[:to_write]
					)
					written += to_write

				next_obs_raw, _, _, _, _ = envs.step(actions.detach().cpu().numpy())
				total_steps += 1

		if written == 0:
			break

		obs_path = os.path.join(args.stream_dir, f"shard_{shard_idx:06d}_obs.npy")
		logits_path = os.path.join(args.stream_dir, f"shard_{shard_idx:06d}_logits.npy")
		np.save(obs_path, obs_buf[:written])
		np.save(logits_path, logits_buf[:written])

		written_total += written
		shard_idx += 1

	status = {
		"done": True,
		"n_samples": int(written_total),
		"bytes_used": int(written_total * per_sample_bytes),
		"n_actions": int(n_actions),
		"run_name": run_name,
	}
	with open(os.path.join(args.stream_dir, "DONE.json"), "w", encoding="utf-8") as f:
		json.dump(status, f, indent=2)
	envs.close()


def build_student(envs: gym.vector.SyncVectorEnv, args: Args, device: torch.device) -> CDLGAagent:
	ppo_args = PPOArgs()
	ppo_args.logic_lut_rank = args.logic_lut_rank
	ppo_args.logic_num_bits = args.logic_num_bits
	ppo_args.logic_tree_depth = args.logic_tree_depth
	ppo_args.logic_k_num = args.logic_k_num
	ppo_args.logic_tau = args.logic_tau
	ppo_args.logic_actor_group_size = args.logic_actor_group_size

	calib_obs_raw, _ = envs.reset(seed=args.seed)
	runtime_dtype = get_runtime_dtype(device)
	calib_obs = torch.as_tensor(calib_obs_raw, dtype=runtime_dtype, device=device)
	calib_obs = ensure_nchw(calib_obs, expected_channels=4) / 255.0
	thresholds = get_distributive_channel_thresholds(calib_obs, num_bits=args.logic_num_bits)

	student = CDLGAagent(envs, args=ppo_args, thresholds=thresholds).to(device=device, dtype=runtime_dtype)
	return student


def build_split_loaders(dataset: Dataset, args: Args) -> tuple[DataLoader, DataLoader | None]:
	n = len(dataset)
	if n <= 0:
		raise ValueError("Dataset is empty.")

	n_val = int(n * args.validation_fraction)
	n_val = min(max(n_val, 0), n - 1) if n > 1 else 0

	generator = torch.Generator().manual_seed(args.seed)
	perm = torch.randperm(n, generator=generator).tolist()

	if n_val > 0:
		val_idx = perm[:n_val]
		train_idx = perm[n_val:]
		train_ds = Subset(dataset, train_idx)
		val_ds = Subset(dataset, val_idx)
	else:
		train_ds = dataset
		val_ds = None

	train_loader = DataLoader(
		train_ds,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=args.num_workers,
		pin_memory=(get_device(args.cuda).type == "cuda"),
		drop_last=False,
	)
	val_loader = None
	if val_ds is not None:
		val_loader = DataLoader(
			val_ds,
			batch_size=args.batch_size,
			shuffle=False,
			num_workers=args.num_workers,
			pin_memory=(get_device(args.cuda).type == "cuda"),
			drop_last=False,
		)
	return train_loader, val_loader


def run_epoch(
	*,
	student: CDLGAagent,
	loader: DataLoader,
	optimizer: optim.Optimizer | None,
	device: torch.device,
	temperature: float,
	desc: str,
) -> tuple[float, float]:
	is_train = optimizer is not None
	if is_train:
		student.train()
	else:
		student.eval()

	total_loss = 0.0
	total_match = 0.0
	total_batches = 0
	ctx = torch.enable_grad() if is_train else torch.inference_mode()
	model_dtype = get_module_dtype(student)
	with ctx:
		for obs_u8, teacher_logits in tqdm(loader, desc=desc):
			obs = obs_u8.to(device=device, dtype=model_dtype, non_blocking=True)
			teacher_logits = teacher_logits.to(device=device, dtype=model_dtype, non_blocking=True)
			student_logits = get_actor_logits(student, obs)

			loss = F.kl_div(
				F.log_softmax(student_logits, dim=-1), #F.log_softmax(student_logits / temperature, dim=-1),
				F.softmax(teacher_logits / temperature, dim=-1), 
				reduction="batchmean",
			) #* (temperature * temperature)

			if is_train:
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

			match = (student_logits.argmax(dim=-1) == teacher_logits.argmax(dim=-1)).float().mean().item()
			total_loss += float(loss.item())
			total_match += float(match)
			total_batches += 1

	avg_loss = total_loss / max(1, total_batches)
	avg_match = total_match / max(1, total_batches)
	return avg_loss, avg_match


def _save_student_checkpoint(
	*,
	args: Args,
	student: CDLGAagent,
	optimizer: optim.Optimizer,
	epoch: int,
	train_kl: float,
	train_top1: float,
	val_kl: float | None,
	val_top1: float | None,
	ckpt_name: str,
):
	os.makedirs(args.output_dir, exist_ok=True)
	ckpt = {
		"student_state_dict": student.state_dict(),
		"optimizer_state_dict": optimizer.state_dict(),
		"epoch": epoch,
		"train_kl": train_kl,
		"train_top1": train_top1,
		"val_kl": val_kl,
		"val_top1": val_top1,
		"args": asdict(args),
	}
	torch.save(ckpt, os.path.join(args.output_dir, ckpt_name))


def train_distillation(args: Args):
	set_seed(args.seed)
	device = get_device(args.cuda)
	dataset = DistillDataset(args.dataset_dir)
	train_loader, val_loader = build_split_loaders(dataset, args)

	run_name = f"distill_train__{args.env_id}__{args.seed}__{int(time.time())}"
	envs = gym.vector.SyncVectorEnv([make_env(args.env_id, 0, capture_video=False, run_name=run_name)])
	student = build_student(envs, args, device)
	trainable_params = list(student.actor_logic_backbone.parameters()) + list(student.actor.parameters())
	optimizer = optim.Adam(trainable_params, lr=args.student_lr, eps=1e-5)

	for epoch in range(1, args.epochs + 1):
		train_kl, train_top1 = run_epoch(
			student=student,
			loader=train_loader,
			optimizer=optimizer,
			device=device,
			temperature=args.temperature,
			desc=f"Train {epoch}/{args.epochs}",
		)
		val_kl = None
		val_top1 = None
		if val_loader is not None:
			if epoch % (args.epochs // 10) == 0:
				val_kl, val_top1 = run_epoch(
                    student=student,
                    loader=val_loader,
                    optimizer=None,
                    device=device,
                    temperature=args.temperature,
                    desc=f"Val {epoch}/{args.epochs}",
                )
				print(
                    f"Epoch {epoch}: train_kl={train_kl:.6f}, train_top1={train_top1:.4f}"
                    + ("" if val_kl is None else f", val_kl={val_kl:.6f}, val_top1={val_top1:.4f}")
                )
		
		if epoch % (args.epochs // 4) == 0:
			_save_student_checkpoint(
                args=args,
                student=student,
                optimizer=optimizer,
                epoch=epoch,
                train_kl=train_kl,
                train_top1=train_top1,
                val_kl=val_kl,
                val_top1=val_top1,
                ckpt_name=f"student_epoch_{epoch}.pt",
            )
		if epoch % (args.epochs // 2) == 0:
			eval_returns, eval_lengths = evaluate(
                agent=student,
                make_env_fn=make_env,
                env_id=args.env_id,
                eval_episodes=args.eval_episodes,
                device=device,
                capture_video=False,
                writer=None,
                global_step=0,
            )
			print(
                f"Final student eval: return_mean={np.mean(eval_returns):.2f} +/- {np.std(eval_returns):.2f}, "
                f"length_mean={np.mean(eval_lengths):.2f}"
            )

	_save_student_checkpoint(
		args=args,
		student=student,
		optimizer=optimizer,
		epoch=args.epochs,
		train_kl=train_kl,
		train_top1=train_top1,
		val_kl=val_kl,
		val_top1=val_top1,
		ckpt_name="student_distilled.pt",
	)
	print(f"Saved final distilled student to: {os.path.join(args.output_dir, 'student_distilled.pt')}")
	
	
	envs.close()


def _discover_shard_pairs(stream_dir: str) -> list[tuple[str, str]]:
	pairs: list[tuple[str, str]] = []
	for name in sorted(os.listdir(stream_dir)):
		if not name.startswith("shard_") or not name.endswith("_obs.npy"):
			continue
		obs_path = os.path.join(stream_dir, name)
		logits_path = os.path.join(stream_dir, name.replace("_obs.npy", "_logits.npy"))
		if os.path.isfile(logits_path):
			pairs.append((obs_path, logits_path))
	return pairs


def _count_samples_in_pairs(pairs: list[tuple[str, str]]) -> int:
	total = 0
	for obs_path, _ in pairs:
		total += int(np.load(obs_path, mmap_mode="r").shape[0])
	return total


def train_distillation_stream(args: Args):
	set_seed(args.seed)
	device = get_device(args.cuda)
	os.makedirs(args.stream_dir, exist_ok=True)

	collector = mp.Process(target=_collect_stream_shards, args=(args,), daemon=False)
	collector.start()

	run_name = f"distill_stream_train__{args.env_id}__{args.seed}__{int(time.time())}"
	envs = gym.vector.SyncVectorEnv([make_env(args.env_id, 0, capture_video=False, run_name=run_name)])
	student = build_student(envs, args, device)
	trainable_params = list(student.actor_logic_backbone.parameters()) + list(student.actor.parameters())
	optimizer = optim.Adam(trainable_params, lr=args.student_lr, eps=1e-5)

	seen_pairs: set[tuple[str, str]] = set()
	epoch = 0

	while True:
		pairs = _discover_shard_pairs(args.stream_dir)
		total_samples = _count_samples_in_pairs(pairs) if pairs else 0
		new_available = any(pair not in seen_pairs for pair in pairs)
		done_path = os.path.join(args.stream_dir, "DONE.json")
		collector_done = os.path.isfile(done_path)

		if total_samples >= args.stream_min_samples_before_train and (new_available or (collector_done and pairs)):
			dataset = ShardDistillDataset(pairs)
			train_loader, val_loader = build_split_loaders(dataset, args)
			epoch += 1

			train_kl, train_top1 = run_epoch(
				student=student,
				loader=train_loader,
				optimizer=optimizer,
				device=device,
				temperature=args.temperature,
				desc=f"Stream Train Epoch {epoch}",
			)
			val_kl = None
			val_top1 = None
			if val_loader is not None:
				val_kl, val_top1 = run_epoch(
					student=student,
					loader=val_loader,
					optimizer=None,
					device=device,
					temperature=args.temperature,
					desc=f"Stream Val Epoch {epoch}",
				)

			print(
				f"Stream epoch {epoch}: train_kl={train_kl:.6f}, train_top1={train_top1:.4f}"
				+ ("" if val_kl is None else f", val_kl={val_kl:.6f}, val_top1={val_top1:.4f}")
			)
			_save_student_checkpoint(
				args=args,
				student=student,
				optimizer=optimizer,
				epoch=epoch,
				train_kl=train_kl,
				train_top1=train_top1,
				val_kl=val_kl,
				val_top1=val_top1,
				ckpt_name=f"student_stream_epoch_{epoch}.pt",
			)
			seen_pairs = set(pairs)

		if collector_done and not new_available:
			break

		time.sleep(args.stream_poll_seconds)

	collector.join(timeout=10)
	_save_student_checkpoint(
		args=args,
		student=student,
		optimizer=optimizer,
		epoch=epoch,
		train_kl=0.0,
		train_top1=0.0,
		val_kl=None,
		val_top1=None,
		ckpt_name="student_stream_final.pt",
	)
	print(f"Saved streaming distilled student to: {os.path.join(args.output_dir, 'student_stream_final.pt')}")
	eval_returns, eval_lengths = evaluate(
		agent=student,
		make_env_fn=make_env,
		env_id=args.env_id,
		eval_episodes=args.eval_episodes,
		device=device,
		capture_video=False,
		writer=None,
		global_step=0,
	)
	print(
		f"Final student eval: return_mean={np.mean(eval_returns):.2f} +/- {np.std(eval_returns):.2f}, "
		f"length_mean={np.mean(eval_lengths):.2f}"
	)
	envs.close()


def main():
	args = tyro.cli(Args)
	torch.set_default_dtype(get_runtime_dtype(get_device(args.cuda)))

	if args.mode not in {"collect", "train", "both", "stream"}:
		raise ValueError("--mode must be one of: collect, train, both, stream")

	if args.mode in {"collect", "both"}:
		collect_dataset(args)
	if args.mode in {"train", "both"}:
		train_distillation(args)
	if args.mode == "stream":
		train_distillation_stream(args)
	


if __name__ == "__main__":
	main()
