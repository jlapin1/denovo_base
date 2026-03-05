from copy import deepcopy

import torch as th
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP

import utils as U
import runners.denovo_objects as denovo_objects
from models.mdlm.train_backends import create_mdlm_train_wrapper

BaseDenovo = denovo_objects.BaseDenovo


class DenovoInsertDeleteObj(BaseDenovo):
    """Insert/delete runner object."""

    def __init__(self, config, svdir="./save/", rddir=None):
        super().__init__(
            config=config,
            svdir=svdir,
            rddir=rddir,
        )
        self.training_loss_keys.extend(["loss"])
        self.eval_kwargs = {}

        from models.seq2seq import Seq2SeqInsertDelete

        diff_config = config["decoder_insertdelete"]["diffusion_config"]
        self.diff_config = diff_config
        self.max_length = diff_config["model"]["length"]

        config["decoder_diff"]["model_config"]["self_condition"] = diff_config["model"][
            "self_condition"
        ]

        self.model = Seq2SeqInsertDelete(
            encoder_config=config["encoder_dict"],
            decoder_config=config["decoder_diff"]["model_config"],
            diff_config=diff_config,
            top_peaks=config["top_peaks"],
            max_peptide_length=config["pep_length"][1],
            token_dict=self.data.amod_dic,
            ensemble_config=config["decoder_diff"]["ensemble"],
            masses_path=config["loader"]["masses_path"],
            use_precomputed_encoder=config["loader"].get("use_precomputed_encoder", False),
            precomputed_encoder_dim=config["loader"].get("precomputed_encoder_dim"),
            precomputed_kv_indim=config["loader"].get("precomputed_kv_indim"),
        )

        print(f"<DSCOMMENT> Total model parameters: {self.model.total_params():,}")
        self.opt = th.optim.Adam(self.model.parameters(), self.starting_lr)

        if config["prev_wts"] is not None:
            retain = False if config["load_last"] else True
            self.load_saved_weights(self.model, "model", config["load_last"], retain=retain)
            self.load_saved_weights(self.opt, "opt", config["load_last"])
            U.optimizer_to(self.opt, denovo_objects.device)

        self.model.to(denovo_objects.device)
        self.model.diff_obj.device = denovo_objects.device

        self.train_wrapper = create_mdlm_train_wrapper(
            self.model,
            backend="insertdelete",
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
        """Build model target sequence and loss mask from a raw batch.

        Args:
            batch: Training batch dictionary from dataloader.

        Returns:
            Tuple `(None, target, loss_mask)` matching BaseDenovo runner API.
        """
        target = deepcopy(batch["intseq"])
        model = self.get_model()
        # Insert/delete diffusion should not model <SOS>; strip it if present
        # in column 0 while preserving tensor shape.
        sos_token_id = getattr(model.decoder, "SOS", None)
        if sos_token_id is not None and target.shape[1] > 0:
            has_sos = target[:, 0].eq(int(sos_token_id))
            if has_sos.any():
                target[has_sos, :-1] = target[has_sos, 1:]
                target[has_sos, -1] = model.decoder.NT
        target = model.decoder.append_null_token(target)
        loss_mask = model.decoder.sequence_mask(target)
        return None, target, loss_mask

    def train_step(self, batch):
        """Run one optimization step for insert/delete training.

        Args:
            batch: Training batch dictionary.

        Returns:
            Dict containing scalar training loss for logging/aggregation.
        """
        model = self.get_model()
        batch = U.Dict2dev(batch, denovo_objects.device)
        _, target, _ = self.inptarg(batch)

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
                None,
                False,
                False,
            )
        else:
            loss = self.train_wrapper(
                batch,
                target,
                None,
                False,
                False,
            )
        token_nll = loss.mean()
        losses = {"loss": token_nll}

        token_nll.backward()
        self.update_lr()
        self.opt.step()

        return losses

    def log_wandb(self, losses, grad_norm):
        """Log training scalars for the current step to Weights & Biases."""
        wandb.log(
            {
                "Total loss": losses["loss"],
                "Global step": self.global_step,
                "Global grad norm": grad_norm,
            }
        )
