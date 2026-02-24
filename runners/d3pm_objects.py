import os
import math
from copy import deepcopy

import numpy as np
import torch as th
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP

import utils as U
import runners.denovo_objects as denovo_objects
from models.mdlm.train_backends import create_mdlm_train_wrapper

BaseDenovo = denovo_objects.BaseDenovo


class DenovoD3PMObj(BaseDenovo):
    """D3PM runner object.

    This class is intentionally copy-adapted from DenovoMDLMObj to preserve
    all existing BaseDenovo training/eval behavior while swapping only the
    diffusion backend/model path.
    """

    def __init__(self, config, svdir="./save/", rddir=None):
        super().__init__(
            config=config,
            svdir=svdir,
            rddir=rddir,
        )
        self.training_loss_keys.extend(["loss"])
        self.eval_kwargs = {}

        from models.seq2seq import Seq2SeqD3PM

        diff_config = config["decoder_d3pm"]["diffusion_config"]
        self.diff_config = diff_config
        self.max_length = diff_config["model"]["length"]
        self.steps = diff_config["T"]

        config["decoder_diff"]["diffusion_config"]["pad_tok_id"] = self.data.amod_dic[
            "X"
        ]
        config["decoder_diff"]["diffusion_config"]["resume_checkpoint"] = False
        config["decoder_diff"]["diffusion_config"]["sequence_len"] = (
            self.config["pep_length"][1] + 1
        )

        # Keep shared decoder model config in sync with diffusion config.
        config["decoder_diff"]["model_config"]["self_condition"] = diff_config["model"][
            "self_condition"
        ]

        self.model = Seq2SeqD3PM(
            encoder_config=config["encoder_dict"],
            decoder_config=config["decoder_diff"]["model_config"],
            diff_config=diff_config,
            top_peaks=config["top_peaks"],
            max_peptide_length=config["pep_length"][1],
            token_dict=self.data.amod_dic,
            ensemble_config=config["decoder_diff"]["ensemble"],
            masses_path=config["loader"]["masses_path"],
            use_precomputed_encoder=config["loader"].get(
                "use_precomputed_encoder", False
            ),
            precomputed_encoder_dim=config["loader"].get("precomputed_encoder_dim"),
            precomputed_kv_indim=config["loader"].get("precomputed_kv_indim"),
        )

        print(f"<DSCOMMENT> Total model parameters: {self.model.total_params():,}")
        self.opt = th.optim.Adam(self.model.parameters(), self.starting_lr)

        if config["prev_wts"] is not None:
            retain = False if config["load_last"] else True
            self.load_saved_weights(
                self.model, "model", config["load_last"], retain=retain
            )
            self.load_saved_weights(self.opt, "opt", config["load_last"])
            U.optimizer_to(self.opt, denovo_objects.device)

        self.model.to(denovo_objects.device)
        self.model.diff_obj.device = denovo_objects.device

        self.train_wrapper = create_mdlm_train_wrapper(
            self.model,
            backend="d3pm",
        ).to(denovo_objects.device)

        self.ddp_train_wrapper = None
        if self.distributed:
            if denovo_objects.device.type == "cuda":
                self.ddp_train_wrapper = DDP(
                    self.train_wrapper,
                    device_ids=[denovo_objects.device.index],
                    output_device=denovo_objects.device.index,
                )
            else:
                self.ddp_train_wrapper = DDP(self.train_wrapper)

    def inptarg(self, batch):
        target = deepcopy(batch["intseq"])
        model = self.get_model()
        target = model.decoder.append_null_token(target)
        target = model.decoder.replace_with_eos_token(target, batch["peplen"])
        loss_mask = model.decoder.sequence_mask(target)
        return None, target, loss_mask

    def train_step(self, batch):
        model = self.get_model()
        block_decoding = False
        batch = U.Dict2dev(batch, denovo_objects.device)
        _, target, loss_mask = self.inptarg(batch)
        training_mask = None

        model.to(denovo_objects.device)
        if self.ddp_train_wrapper is not None:
            self.ddp_train_wrapper.train()
        else:
            model.train()
        model.zero_grad()

        if self.ddp_train_wrapper is not None:
            loss = self.ddp_train_wrapper(
                batch,
                target,
                training_mask,
                block_decoding,
                self.diff_config.get("custom_loss", False),
            )
        else:
            loss = self.train_wrapper(
                batch,
                target,
                training_mask,
                block_decoding,
                self.diff_config.get("custom_loss", False),
            )
        token_nll = loss.mean()
        losses = {"loss": token_nll}

        token_nll.backward()
        self.update_lr()
        self.opt.step()

        return losses

    def log_wandb(self, losses, grad_norm):
        wandb.log(
            {
                "Total loss": losses["loss"],
                "Global step": self.global_step,
                "Global grad norm": grad_norm,
            }
        )

    def on_train_epoch_end(self):
        if self.log:
            try:
                avg_loss = self.token_loss / (self.token_count + 1e-5)
                avg_loss = avg_loss.cpu().detach().numpy()
                if self.is_main:
                    np.savetxt(
                        os.path.join(self.svdir, "token_loss.tsv"),
                        avg_loss,
                        delimiter="\t",
                    )
                self.initialize_token_loss()
            except Exception:
                pass
