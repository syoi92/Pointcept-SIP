"""
Trainer

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import sys
import weakref
import wandb
import torch
import torch.nn as nn
import torch.utils.data
from packaging import version
from functools import partial
from pathlib import Path

if sys.version_info >= (3, 10):
    from collections.abc import Iterator
else:
    from collections import Iterator
from tensorboardX import SummaryWriter

from .defaults import create_ddp_model, worker_init_fn
from .hooks import HookBase, build_hooks
import pointcept.utils.comm as comm
from pointcept.datasets import build_dataset, point_collate_fn, collate_fn
from pointcept.models import build_model
from pointcept.utils.logger import get_root_logger
from pointcept.utils.optimizer import build_optimizer
from pointcept.utils.scheduler import build_scheduler
from pointcept.utils.events import EventStorage, ExceptionWriter
from pointcept.utils.registry import Registry


TRAINERS = Registry("trainers")
AMP_DTYPE = dict(
    float16=torch.float16,
    bfloat16=torch.bfloat16,
)


class TrainerBase:
    def __init__(self) -> None:
        self.hooks = []
        self.model = None
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = 0
        self.max_iter = 0
        self.comm_info = dict()
        self.data_iterator: Iterator = enumerate([])
        self.storage: EventStorage
        self.writer: SummaryWriter

    def register_hooks(self, hooks) -> None:
        hooks = build_hooks(hooks)
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self.hooks.extend(hooks)

    def train(self):
        with EventStorage() as self.storage:
            # => before train
            self.before_train()
            for self.epoch in range(self.start_epoch, self.max_epoch):
                # => before epoch
                self.before_epoch()
                # => run_epoch
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    # => before_step
                    self.before_step()
                    # => run_step
                    self.run_step()
                    # => after_step
                    self.after_step()
                # => after epoch
                self.after_epoch()
            # => after train
            self.after_train()

    def before_train(self):
        for h in self.hooks:
            h.before_train()

    def before_epoch(self):
        for h in self.hooks:
            h.before_epoch()

    def before_step(self):
        for h in self.hooks:
            h.before_step()

    def run_step(self):
        raise NotImplementedError

    def after_step(self):
        for h in self.hooks:
            h.after_step()

    def after_epoch(self):
        for h in self.hooks:
            h.after_epoch()
        self.storage.reset_histories()

    def after_train(self):
        # Sync GPU before running train hooks
        comm.synchronize()
        for h in self.hooks:
            h.after_train()
        if comm.is_main_process():
            self.writer.close()


@TRAINERS.register_module("DefaultTrainer")
class Trainer(TrainerBase):
    def __init__(self, cfg):
        super(Trainer, self).__init__()
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = cfg.eval_epoch
        self.best_metric_value = -torch.inf
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "train.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.logger.info(f"Save path: {cfg.save_path}")
        self.logger.info(f"Config:\n{cfg.pretty_text}")
        self.logger.info("=> Building model ...")
        self.model = self.build_model()
        self.logger.info("=> Building writer ...")
        self.writer = self.build_writer()
        self.logger.info("=> Building train dataset & dataloader ...")
        self.train_loader = self.build_train_loader()
        self.logger.info("=> Building val dataset & dataloader ...")
        self.val_loader = self.build_val_loader()
        self.logger.info("=> Building optimize, scheduler, scaler(amp) ...")
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        self.scaler = self.build_scaler()
        self.logger.info("=> Building hooks ...")
        self.register_hooks(self.cfg.hooks)

    def train(self):
        with EventStorage() as self.storage, ExceptionWriter():
            # => before train
            self.before_train()
            self.logger.info(">>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>")
            for self.epoch in range(self.start_epoch, self.max_epoch):
                # => before epoch
                if comm.get_world_size() > 1:
                    self.train_loader.sampler.set_epoch(self.epoch)
                self.model.train()
                self.data_iterator = enumerate(self.train_loader)
                self.before_epoch()
                # => run_epoch
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    # => before_step
                    self.before_step()
                    # => run_step
                    self.run_step()
                    # => after_step
                    self.after_step()
                # => after epoch
                self.after_epoch()
            # => after train
            self.after_train()

    def run_step(self):
        if version.parse(torch.__version__) >= version.parse("2.4"):
            auto_cast = partial(torch.amp.autocast, device_type="cuda")
        else:
            # deprecated warning
            auto_cast = torch.cuda.amp.autocast

        input_dict = self.comm_info["input_dict"]
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].cuda(non_blocking=True)

        with auto_cast(
            enabled=self.cfg.enable_amp, dtype=AMP_DTYPE[self.cfg.amp_dtype]
        ):
            output_dict = self.model(input_dict)
            loss = output_dict["loss"]
        self.optimizer.zero_grad()
        if self.cfg.enable_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            if self.cfg.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.clip_grad
                )
            self.scaler.step(self.optimizer)

            # When enable amp, optimizer.step call are skipped if the loss scaling factor is too large.
            # Fix torch warning scheduler step before optimizer step.
            scaler = self.scaler.get_scale()
            self.scaler.update()
            if scaler <= self.scaler.get_scale():
                self.scheduler.step()
        else:
            loss.backward()
            if self.cfg.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.clip_grad
                )
            self.optimizer.step()
            self.scheduler.step()
        if self.cfg.empty_cache:
            torch.cuda.empty_cache()
        self.comm_info["model_output_dict"] = output_dict

    def after_epoch(self):
        for h in self.hooks:
            h.after_epoch()
        self.storage.reset_histories()
        if self.cfg.empty_cache_per_epoch:
            torch.cuda.empty_cache()

    def build_model(self):
        model = build_model(self.cfg.model)
        if self.cfg.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # logger.info(f"Model: \n{self.model}")
        self.logger.info(f"Num params: {n_parameters}")
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        return model

    def build_writer(self):
        writer = SummaryWriter(self.cfg.save_path) if comm.is_main_process() else None
        self.logger.info(f"Tensorboard writer logging dir: {self.cfg.save_path}")
        if self.cfg.enable_wandb and comm.is_main_process():
            tag, name = Path(self.cfg.save_path).parts[-2:]
            wandb.init(
                project=self.cfg.wandb_project,
                name=f"{tag}/{name}",
                tags=[tag],
                dir=self.cfg.save_path,
                settings=wandb.Settings(api_key=self.cfg.wandb_key),
                config=self.cfg,
            )
        return writer

    def build_train_loader(self):
        train_data = build_dataset(self.cfg.data.train)

        if comm.get_world_size() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        else:
            train_sampler = None

        init_fn = (
            partial(
                worker_init_fn,
                num_workers=self.cfg.num_worker_per_gpu,
                rank=comm.get_rank(),
                seed=self.cfg.seed,
            )
            if self.cfg.seed is not None
            else None
        )

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.cfg.batch_size_per_gpu,
            shuffle=(train_sampler is None),
            num_workers=self.cfg.num_worker_per_gpu,
            sampler=train_sampler,
            collate_fn=partial(point_collate_fn, mix_prob=self.cfg.mix_prob),
            pin_memory=True,
            worker_init_fn=init_fn,
            drop_last=len(train_data) > self.cfg.batch_size,
            persistent_workers=True,
        )
        return train_loader

    def build_val_loader(self):
        val_loader = None
        if self.cfg.evaluate:
            val_data = build_dataset(self.cfg.data.val)
            if comm.get_world_size() > 1:
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
            else:
                val_sampler = None
            val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=self.cfg.batch_size_val_per_gpu,
                shuffle=False,
                num_workers=self.cfg.num_worker_per_gpu,
                pin_memory=True,
                sampler=val_sampler,
                collate_fn=collate_fn,
            )
        return val_loader

    def build_optimizer(self):
        return build_optimizer(self.cfg.optimizer, self.model, self.cfg.param_dicts)

    def build_scheduler(self):
        assert hasattr(self, "optimizer")
        assert hasattr(self, "train_loader")
        self.cfg.scheduler.total_steps = len(self.train_loader) * self.cfg.eval_epoch
        return build_scheduler(self.cfg.scheduler, self.optimizer)

    def build_scaler(self):
        if version.parse(torch.__version__) >= version.parse("2.4"):
            grad_scaler = partial(torch.amp.GradScaler, device="cuda")
        else:
            # deprecated warning
            grad_scaler = torch.cuda.amp.GradScaler
        scaler = grad_scaler() if self.cfg.enable_amp else None
        return scaler


@TRAINERS.register_module("MultiDatasetTrainer")
class MultiDatasetTrainer(Trainer):
    def build_train_loader(self):
        from pointcept.datasets import MultiDatasetDataloader

        train_data = build_dataset(self.cfg.data.train)
        train_loader = MultiDatasetDataloader(
            train_data,
            self.cfg.batch_size_per_gpu,
            self.cfg.num_worker_per_gpu,
            self.cfg.mix_prob,
            self.cfg.seed,
        )
        self.comm_info["iter_per_epoch"] = len(train_loader)
        return train_loader
    

class _SIPTrainerMixin:
    """
    Shared utilities for SIP scene/fragment trainers.

    fragment_batch_size: int
        Number of fragments collated per forward.
    min_last_fragment_batch: int, default=1
        If the last fragment batch is smaller than this threshold,
        merge it into the previous fragment batch.
    """

    def _get_autocast(self):
        if version.parse(torch.__version__) >= version.parse("2.4"):
            return partial(torch.amp.autocast, device_type="cuda")
        return torch.cuda.amp.autocast

    def _get_fragment_ranges(self, num_fragments, frag_bs):
        """
        Split [0, num_fragments) into fragment mini-batches.
        Optionally merge a tiny last batch into the previous one.
        """
        if num_fragments <= 0:
            return []

        ranges = []
        s = 0
        while s < num_fragments:
            e = min(s + frag_bs, num_fragments)
            ranges.append((s, e))
            s = e

        # default=1 -> no behavior change from current version
        min_last = int(getattr(self.cfg, "min_last_fragment_batch", 1))

        if (
            min_last > 1
            and len(ranges) >= 2
            and (ranges[-1][1] - ranges[-1][0]) < min_last
        ):
            prev_s, _ = ranges[-2]
            _, last_e = ranges[-1]
            ranges[-2] = (prev_s, last_e)
            ranges.pop(-1)

        return ranges

    def _move_to_cuda(self, input_dict):
        for k, v in input_dict.items():
            if isinstance(v, torch.Tensor):
                input_dict[k] = v.cuda(non_blocking=True)
        return input_dict

    def _forward_loss(self, input_dict, auto_cast):
        with auto_cast(enabled=self.cfg.enable_amp, dtype=AMP_DTYPE[self.cfg.amp_dtype]):
            output_dict = self.model(input_dict)
            loss = output_dict["loss"]
        return output_dict, loss

    def _backward(self, loss):
        if self.cfg.enable_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def _optimizer_step(self):
        if self.cfg.enable_amp:
            self.scaler.unscale_(self.optimizer)
            if self.cfg.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_grad)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            if self.cfg.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_grad)
            self.optimizer.step()

    def _scheduler_step(self):
        self.scheduler.step()

    def build_train_loader(self):
        train_data = build_dataset(self.cfg.data.train)

        if comm.get_world_size() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        else:
            train_sampler = None

        init_fn = (
            partial(
                worker_init_fn,
                num_workers=self.cfg.num_worker_per_gpu,
                rank=comm.get_rank(),
                seed=self.cfg.seed,
            )
            if self.cfg.seed is not None
            else None
        )

        return torch.utils.data.DataLoader(
            train_data,
            batch_size=self.cfg.batch_size_per_gpu,
            shuffle=(train_sampler is None),
            num_workers=self.cfg.num_worker_per_gpu,
            sampler=train_sampler,
            collate_fn=self._sip_collate_fn,
            pin_memory=True,
            worker_init_fn=init_fn,
            drop_last=len(train_data) > self.cfg.batch_size,
            persistent_workers=True,
        )

    def build_val_loader(self):
        if not self.cfg.evaluate:
            return None

        val_data = build_dataset(self.cfg.data.val)
        if comm.get_world_size() > 1:
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
        else:
            val_sampler = None

        return torch.utils.data.DataLoader(
            val_data,
            batch_size=self.cfg.batch_size_val_per_gpu,
            shuffle=False,
            num_workers=self.cfg.num_worker_per_gpu,
            pin_memory=True,
            sampler=val_sampler,
            collate_fn=self._sip_collate_fn,
        )

    @staticmethod
    def _sip_collate_fn(batch):
        out = {}
        for k in batch[0].keys():
            if k == "fragment_list":
                out[k] = [b[k] for b in batch]  # list of fragment_lists (one per scene)
            else:
                out[k] = collate_fn([b[k] for b in batch])
        return out


@TRAINERS.register_module("SIPSceneTrainer")
class SIPSceneTrainer(_SIPTrainerMixin, Trainer):
    """
    Current behavior:
    - forward/backward over fragment mini-batches
    - one optimizer update per scene-batch
    """

    def train(self):
        return super().train()

    def run_step(self):
        auto_cast = self._get_autocast()

        batch = self.comm_info["input_dict"]
        assert "fragment_list" in batch
        scene_fragment_lists = batch["fragment_list"]

        frag_bs = int(getattr(self.cfg, "fragment_batch_size", 1))

        self.optimizer.zero_grad(set_to_none=True)
        last_output = None

        for fragment_list in scene_fragment_lists:
            frag_ranges = self._get_fragment_ranges(len(fragment_list), frag_bs)
            num_frag_batches = max(1, len(frag_ranges))

            for s, e in frag_ranges:
                input_dict = collate_fn(fragment_list[s:e])
                input_dict = self._move_to_cuda(input_dict)

                output_dict, loss = self._forward_loss(input_dict, auto_cast)
                loss = loss #/ num_frag_batches

                self._backward(loss)
                last_output = output_dict

        self._optimizer_step()
        self._scheduler_step()

        if self.cfg.empty_cache:
            torch.cuda.empty_cache()

        if last_output is not None:
            self.comm_info["model_output_dict"] = last_output


@TRAINERS.register_module("SIPFragmentTrainer")
class SIPFragmentTrainer(_SIPTrainerMixin, Trainer):
    """
    Fragment-batch update behavior:
    - forward/backward/optimizer-step per fragment mini-batch
    - BN stats, gradient, optimizer update all aligned at fragment-batch scale
    """

    def build_scheduler(self):
        assert hasattr(self, "optimizer")
        max_update = int(getattr(self.cfg, "max_update", 0))
        assert max_update > 0, "SIPFragmentTrainer requires cfg.max_update > 0"
        self.cfg.scheduler.total_steps = max_update
        return build_scheduler(self.cfg.scheduler, self.optimizer)

    def train(self):
        max_update = int(getattr(self.cfg, "max_update", 0))
        assert max_update > 0, "SIPFragmentTrainer requires cfg.max_update > 0"

        self.max_iter = max_update
        self.comm_info["global_update_step"] = 0

        with EventStorage() as self.storage, ExceptionWriter():
            self.before_train()
            self.logger.info(">>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>")

            stop_training = False
            while not stop_training:
                if comm.get_world_size() > 1:
                    self.train_loader.sampler.set_epoch(self.epoch)
                self.model.train()
                self.data_iterator = enumerate(self.train_loader)
                self.before_epoch()

                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    self.before_step()
                    self.run_step()
                    self.after_step()

                    if self.comm_info["global_update_step"] >= max_update:
                        stop_training = True
                        break

                self.after_epoch()
                self.epoch += 1

            self.after_train()

    def run_step(self):
        auto_cast = self._get_autocast()

        batch = self.comm_info["input_dict"]
        assert "fragment_list" in batch
        scene_fragment_lists = batch["fragment_list"]

        frag_bs = int(getattr(self.cfg, "fragment_batch_size", 1))
        max_update = int(getattr(self.cfg, "max_update", 0))
        last_output = None

        update_in_this_loader_step = 0

        for fragment_list in scene_fragment_lists:
            frag_ranges = self._get_fragment_ranges(len(fragment_list), frag_bs)

            for s, e in frag_ranges:
                if self.comm_info["global_update_step"] >= max_update:
                    break

                self.optimizer.zero_grad(set_to_none=True)

                input_dict = collate_fn(fragment_list[s:e])
                input_dict = self._move_to_cuda(input_dict)

                output_dict, loss = self._forward_loss(input_dict, auto_cast)
                self._backward(loss)
                self._optimizer_step()
                self._scheduler_step()

                self.comm_info["global_update_step"] += 1
                update_in_this_loader_step += 1
                last_output = output_dict

            if self.comm_info["global_update_step"] >= max_update:
                break

        if self.cfg.empty_cache:
            torch.cuda.empty_cache()

        if last_output is not None:
            self.comm_info["model_output_dict"] = last_output

        self.comm_info["update_in_step"] = update_in_this_loader_step
    
    # def train(self):
    #     return super().train()

    # def run_step(self):
    #     auto_cast = self._get_autocast()

    #     batch = self.comm_info["input_dict"]
    #     assert "fragment_list" in batch
    #     scene_fragment_lists = batch["fragment_list"]

    #     frag_bs = int(getattr(self.cfg, "fragment_batch_size", 1))
    #     last_output = None

    #     for fragment_list in scene_fragment_lists:
    #         frag_ranges = self._get_fragment_ranges(len(fragment_list), frag_bs)

    #         for s, e in frag_ranges:
    #             self.optimizer.zero_grad(set_to_none=True)

    #             input_dict = collate_fn(fragment_list[s:e])
    #             input_dict = self._move_to_cuda(input_dict)

    #             output_dict, loss = self._forward_loss(input_dict, auto_cast)
    #             self._backward(loss)
    #             self._optimizer_step()

    #             last_output = output_dict
            
    #         self._scheduler_step()

    #     if self.cfg.empty_cache:
    #         torch.cuda.empty_cache()

    #     if last_output is not None:
    #         self.comm_info["model_output_dict"] = last_output