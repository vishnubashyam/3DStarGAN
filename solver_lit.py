import os
import time
from munch import Munch
import numpy as np
import pandas as pd
import logging
from argparse import ArgumentParser
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.utils as vutils
from torchmetrics import StructuralSimilarityIndexMeasure

from network import build_model
from utils import he_init, debug_image

global iter
iter = 0

logging.getLogger("lightning").setLevel(logging.ERROR)


class LitBrainMRI(pl.LightningModule):
    def __init__(
        self,
        args: ArgumentParser,
        loaders: Munch,
    ):
        super().__init__()
        self.automatic_optimization = False
        # self.name = args.model_name
        self.learning_rate = args.lr
        self.batch_size = args.batch_size
        self.max_steps = args.max_steps
        # self.pretrained_weights = args.pretrained_weights
        # self.mixed_precision = args.mixed_precision
        # self.job_id = args.job_id
        self.args = args
        self.latent_dim = args.latent_dim
        self.initial_lambda_ds = self.args.lambda_ds
        self.loaders = loaders
        self.nets, self.nets_ema = build_model(args)
        self.generator = self.nets.generator
        self.mapping_network = self.nets.mapping_network
        self.style_encoder = self.nets.style_encoder
        self.discriminator = self.nets.discriminator
        self.generator_ema = self.nets_ema.generator
        self.mapping_network_ema = self.nets_ema.mapping_network
        self.style_encoder_ema = self.nets_ema.style_encoder
        self.nets = None
        self.nets_ema = None
        self.save_hyperparameters(ignore=["net"])
        self.val_batch_src = self.loaders.src.dataset[0]
        self.val_batch_ref = self.loaders.ref.dataset[0]
        self.ssim = StructuralSimilarityIndexMeasure()

    def train_dataloader(self):
        return [self.loaders.src, self.loaders.ref]

    def training_step(self, batch, batch_idx):
        global iter
        iter += 1

        wandb_logger = self.logger.experiment
        x_real, y_org = batch[0]
        x_ref, x_ref2, y_trg = batch[1]
        z_trg = torch.randn(x_real.size(0), self.latent_dim, device=self.device)
        z_trg2 = torch.randn(x_real.size(0), self.latent_dim, device=self.device)
        if iter == 1:
            if self.args.log_initial_debug_images == True:
                grid = vutils.make_grid(x_real.cpu(), nrow=4, padding=0)
                wandb_logger.log({f"Sample Real ": [wandb.Image(grid)]})
                grid = vutils.make_grid(x_ref.cpu(), nrow=4, padding=0)
                wandb_logger.log({f"Sample Ref ": [wandb.Image(grid)]})
                grid = vutils.make_grid(x_ref2.cpu(), nrow=4, padding=0)
                wandb_logger.log({f"Sample Ref2 ": [wandb.Image(grid)]})
        (
            optims_discriminator,
            optims_generator,
            optims_style_encoder,
            optims_mapping_network,
        ) = self.optimizers()

        # train the discriminator
        d_loss, d_losses_latent = compute_d_loss(
            self.discriminator,
            self.generator,
            self.mapping_network,
            self.style_encoder,
            self.args,
            x_real,
            y_org,
            y_trg,
            z_trg=z_trg,
            masks=None,
        )
        self._reset_grad(
            optims_discriminator,
            optims_generator,
            optims_style_encoder,
            optims_mapping_network,
        )
        self.manual_backward(d_loss)
        optims_discriminator.step()

        d_loss, d_losses_ref = compute_d_loss(
            self.discriminator,
            self.generator,
            self.mapping_network,
            self.style_encoder,
            self.args,
            x_real,
            y_org,
            y_trg,
            x_ref=x_ref,
            masks=None,
        )
        self._reset_grad(
            optims_discriminator,
            optims_generator,
            optims_style_encoder,
            optims_mapping_network,
        )
        self.manual_backward(d_loss)
        optims_discriminator.step()

        # train the generator
        g_loss, x_fake, g_losses_latent = compute_g_loss(
            self.discriminator,
            self.generator,
            self.mapping_network,
            self.style_encoder,
            self.args,
            x_real,
            y_org,
            y_trg,
            z_trgs=[z_trg, z_trg2],
            masks=None,
        )
        self._reset_grad(
            optims_discriminator,
            optims_generator,
            optims_style_encoder,
            optims_mapping_network,
        )
        self.manual_backward(g_loss)
        optims_generator.step()
        optims_mapping_network.step()
        optims_style_encoder.step()

        g_loss, x_fake, g_losses_ref = compute_g_loss(
            self.discriminator,
            self.generator,
            self.mapping_network,
            self.style_encoder,
            self.args,
            x_real,
            y_org,
            y_trg,
            x_refs=[x_ref, x_ref2],
            masks=None,
        )
        self._reset_grad(
            optims_discriminator,
            optims_generator,
            optims_style_encoder,
            optims_mapping_network,
        )
        self.manual_backward(g_loss)
        optims_generator.step()
        wandb_logger.log({"train/SSIM": self.ssim(x_real, x_fake)})
        if iter % 100 == 0:
            self.ssim.reset()

        wandb_logger.log({"train/G Loss": g_loss, "train/D loss": d_loss})
        # compute moving average of network parameters
        moving_average(self.generator, self.generator_ema, beta=0.999)
        moving_average(self.mapping_network, self.mapping_network_ema, beta=0.999)
        moving_average(self.style_encoder, self.style_encoder_ema, beta=0.999)

        # decay weight for diversity sensitive loss
        if self.args.lambda_ds > 0:
            self.args.lambda_ds -= self.initial_lambda_ds / self.args.ds_iter
        if iter % self.args.sample_every == 0:
            self.validation(x_real[:6], y_org[:6], x_ref[:6], y_trg[:6], wandb_logger)

    def configure_optimizers(self):
        optims_discriminator = torch.optim.Adam(
            params=self.discriminator.parameters(),
            lr=self.args.lr,
            betas=[self.args.beta1, self.args.beta2],
            weight_decay=self.args.weight_decay,
        )
        optims_generator = torch.optim.Adam(
            params=self.generator.parameters(),
            lr=self.args.lr,
            betas=[self.args.beta1, self.args.beta2],
            weight_decay=self.args.weight_decay,
        )
        optims_style_encoder = torch.optim.Adam(
            params=self.style_encoder.parameters(),
            lr=self.args.lr,
            betas=[self.args.beta1, self.args.beta2],
            weight_decay=self.args.weight_decay,
        )
        optims_mapping_network = torch.optim.Adam(
            params=self.mapping_network.parameters(),
            lr=self.args.f_lr,
            betas=[self.args.beta1, self.args.beta2],
            weight_decay=self.args.weight_decay,
        )
        return (
            optims_discriminator,
            optims_generator,
            optims_style_encoder,
            optims_mapping_network,
        )

    def _reset_grad(
        self,
        optims_discriminator,
        optims_generator,
        optims_style_encoder,
        optims_mapping_network,
    ):
        optims_discriminator.zero_grad()
        optims_generator.zero_grad()
        optims_style_encoder.zero_grad()
        optims_mapping_network.zero_grad()

    def validation(
        self, x_real=None, y_org=None, x_ref=None, y_trg=None, wandb_logger=None
    ):
        x_real = x_real.to(self.device)
        x_ref = x_ref.to(self.device)
        # x_real = torch.unsqueeze(x_real, 0)
        # x_ref = torch.unsqueeze(x_ref, 0)
        # y_org = torch.unsqueeze(y_org, 0)
        # y_trg = torch.unsqueeze(y_trg, 0)
        debug_image(
            self.generator_ema,
            self.style_encoder_ema,
            self.mapping_network_ema,
            self.args,
            x_real,
            y_org,
            x_ref,
            y_trg,
            step=iter + 1,
            logger=wandb_logger,
        )


def compute_d_loss(
    discriminator,
    generator,
    mapping_network,
    style_encoder,
    args,
    x_real,
    y_org,
    y_trg,
    z_trg=None,
    x_ref=None,
    masks=None,
):
    assert (z_trg is None) != (x_ref is None)
    # with real images
    x_real.requires_grad_()
    out = discriminator(x_real, y_org)
    loss_real = adv_loss(out, 1)
    loss_reg = r1_reg(out, x_real)

    # with fake images
    with torch.no_grad():
        if z_trg is not None:
            s_trg = mapping_network(z_trg, y_trg)
        else:  # x_ref is not None
            s_trg = style_encoder(x_ref, y_trg)

        x_fake = generator(x_real, s_trg, masks=masks)
    out = discriminator(x_fake, y_trg)
    loss_fake = adv_loss(out, 0)

    loss = loss_real + loss_fake + args.lambda_reg * loss_reg
    return loss, Munch(
        real=loss_real.item(), fake=loss_fake.item(), reg=loss_reg.item()
    )


def compute_g_loss(
    discriminator,
    generator,
    mapping_network,
    style_encoder,
    args,
    x_real,
    y_org,
    y_trg,
    z_trgs=None,
    x_refs=None,
    masks=None,
):
    assert (z_trgs is None) != (x_refs is None)
    if z_trgs is not None:
        z_trg, z_trg2 = z_trgs
    if x_refs is not None:
        x_ref, x_ref2 = x_refs

    # adversarial loss
    if z_trgs is not None:
        s_trg = mapping_network(z_trg, y_trg)
    else:
        s_trg = style_encoder(x_ref, y_trg)

    x_fake = generator(x_real, s_trg, masks=None)
    out = discriminator(x_fake, y_trg)
    loss_adv = adv_loss(out, 1)

    # style reconstruction loss
    s_pred = style_encoder(x_fake, y_trg)
    loss_sty = torch.mean(torch.abs(s_pred - s_trg))

    # diversity sensitive loss
    if z_trgs is not None:
        s_trg2 = mapping_network(z_trg2, y_trg)
    else:
        s_trg2 = style_encoder(x_ref2, y_trg)
    x_fake2 = generator(x_real, s_trg2, masks=None)
    x_fake2 = x_fake2.detach()
    loss_ds = torch.mean(torch.abs(x_fake - x_fake2))

    # cycle-consistency loss
    s_org = style_encoder(x_real, y_org)
    x_rec = generator(x_fake, s_org, masks=None)
    loss_cyc = torch.mean(torch.abs(x_rec - x_real))

    loss = (
        loss_adv
        + args.lambda_sty * loss_sty
        - args.lambda_ds * loss_ds
        + args.lambda_cyc * loss_cyc
    )
    return (
        loss,
        x_fake,
        Munch(
            adv=loss_adv.item(),
            sty=loss_sty.item(),
            ds=loss_ds.item(),
            cyc=loss_cyc.item(),
        ),
    )


def moving_average(model, model_test, beta=0.999):
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)


def adv_loss(logits, target):
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss


def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(),
        inputs=x_in,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert grad_dout2.size() == x_in.size()
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg
