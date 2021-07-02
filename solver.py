import os
import time
from munch import Munch
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from network import build_model
from data import InputFetcher
from utils import he_init
import nibabel as nib
import imageio
global tag_i
tag_i=0

class Solver(nn.Module):
    def __init__(self, args):
        super().__init__()        
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        self.nets, self.nets_ema = build_model(args)
        # below setattrs are to make networks be children of Solver, e.g., for self.to(self.device)
        for name, module in self.nets.items():
            setattr(self, name, module)
        for name, module in self.nets_ema.items():
            setattr(self, name + '_ema', module)

        if args.mode == 'train':
            self.optims = Munch()
            for net in self.nets.keys():
                self.optims[net] = torch.optim.Adam(
                    params=self.nets[net].parameters(),
                    lr=args.f_lr if net == 'mapping_network' else args.lr,
                    betas=[args.beta1, args.beta2],
                    weight_decay=args.weight_decay)
        # self = nn.DataParallel(self, device_ids=[0,1])
        # self.to(self.device)

        for name, network in self.named_children():
            network.apply(he_init)
            network = nn.DataParallel(network, device_ids=[0,1])
            network.to(self.device)
        # self.to(self.device)


    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()


    def train(self, loaders):
        args = self.args
        nets = self.nets
        nets_ema = self.nets_ema
        optims = self.optims

        # fetch random validation images for debugging
        fetcher = InputFetcher(loaders.src, loaders.ref, args.latent_dim, 'train')
        fetcher_val = InputFetcher(loaders.val, None, args.latent_dim, 'val')
        inputs_val = next(fetcher_val)

        # resume training if necessary
        # if args.resume_iter > 0:
        #     self._load_checkpoint(args.resume_iter)

        # remember the initial value of ds weight
        initial_lambda_ds = args.lambda_ds

        print('Start training...')
        start_time = time.time()
        for i in range(args.resume_iter, args.total_iters):
            # fetch images and labels
            inputs = next(fetcher)
            x_real, y_org = inputs.x_src, inputs.y_src
            x_ref, x_ref2, y_trg = inputs.x_ref, inputs.x_ref2, inputs.y_ref
            z_trg, z_trg2 = inputs.z_trg, inputs.z_trg2

            masks = nets.fan.get_heatmap(x_real) if args.w_hpf > 0 else None

            # train the discriminator
            d_loss, d_losses_latent = compute_d_loss(
                nets, args, x_real, y_org, y_trg, z_trg=z_trg, masks=masks)
            self._reset_grad()
            d_loss.backward()
            optims.discriminator.step()

            d_loss, d_losses_ref = compute_d_loss(
                nets, args, x_real, y_org, y_trg, x_ref=x_ref, masks=masks)
            self._reset_grad()
            d_loss.backward()
            optims.discriminator.step()

            # train the generator
            g_loss, g_losses_latent = compute_g_loss(
                nets, args, x_real, y_org, y_trg, z_trgs=[z_trg, z_trg2], masks=masks)
            self._reset_grad()
            g_loss.backward()
            optims.generator.step()
            optims.mapping_network.step()
            optims.style_encoder.step()

            g_loss, g_losses_ref = compute_g_loss(
                nets, args, x_real, y_org, y_trg, x_refs=[x_ref, x_ref2], masks=masks)
            self._reset_grad()
            g_loss.backward()
            optims.generator.step()
            print(f'G Loss: {g_loss}, D loss {d_loss}')
            # compute moving average of network parameters
            moving_average(nets.generator, nets_ema.generator, beta=0.999)
            moving_average(nets.mapping_network, nets_ema.mapping_network, beta=0.999)
            moving_average(nets.style_encoder, nets_ema.style_encoder, beta=0.999)

            # decay weight for diversity sensitive loss
            if args.lambda_ds > 0:
                args.lambda_ds -= (initial_lambda_ds / args.ds_iter)


def compute_d_loss(nets, args, x_real, y_org, y_trg, z_trg=None, x_ref=None, masks=None):
    assert (z_trg is None) != (x_ref is None)
    # with real images
    x_real.requires_grad_()
    out = nets.discriminator(x_real, y_org)
    loss_real = adv_loss(out, 1)
    loss_reg = r1_reg(out, x_real)

    # with fake images
    with torch.no_grad():
        if z_trg is not None:
            s_trg = nets.mapping_network(z_trg, y_trg)
        else:  # x_ref is not None
            s_trg = nets.style_encoder(x_ref, y_trg)

        x_fake = nets.generator(x_real, s_trg, masks=masks)
    out = nets.discriminator(x_fake, y_trg)
    loss_fake = adv_loss(out, 0)

    loss = loss_real + loss_fake + args.lambda_reg * loss_reg
    return loss, Munch(real=loss_real.item(),
                        fake=loss_fake.item(),
                        reg=loss_reg.item())


def compute_g_loss(nets, args, x_real, y_org, y_trg, z_trgs=None, x_refs=None, masks=None):
    assert (z_trgs is None) != (x_refs is None)
    if z_trgs is not None:
        z_trg, z_trg2 = z_trgs
    if x_refs is not None:
        x_ref, x_ref2 = x_refs

    # adversarial loss
    if z_trgs is not None:
        s_trg = nets.mapping_network(z_trg, y_trg)
    else:
        s_trg = nets.style_encoder(x_ref, y_trg)

    x_fake = nets.generator(x_real, s_trg, masks=masks)
    out = nets.discriminator(x_fake, y_trg)
    loss_adv = adv_loss(out, 1)

    # style reconstruction loss
    s_pred = nets.style_encoder(x_fake, y_trg)
    loss_sty = torch.mean(torch.abs(s_pred - s_trg))

    # diversity sensitive loss
    if z_trgs is not None:
        s_trg2 = nets.mapping_network(z_trg2, y_trg)
    else:
        s_trg2 = nets.style_encoder(x_ref2, y_trg)
    x_fake2 = nets.generator(x_real, s_trg2, masks=masks)
    x_fake2 = x_fake2.detach()
    loss_ds = torch.mean(torch.abs(x_fake - x_fake2))

    # cycle-consistency loss
    masks = nets.fan.get_heatmap(x_fake) if args.w_hpf > 0 else None
    s_org = nets.style_encoder(x_real, y_org)
    x_rec = nets.generator(x_fake, s_org, masks=masks)
    loss_cyc = torch.mean(torch.abs(x_rec - x_real))

    global tag_i
    # empty_header = nib.Nifti1Header()
    if tag_i%50==0:
        new_image = (x_rec.cpu().detach().numpy()[0]*255).astype(np.uint8).transpose(1,2,3,0)
        imageio.mimwrite(f'sample_videos/sample{tag_i}.mp4', new_image, fps = 24)
    tag_i+=1
    # nib.save(new_image, f'output.nii.gz')
    

    loss = loss_adv + args.lambda_sty * loss_sty \
        - args.lambda_ds * loss_ds + args.lambda_cyc * loss_cyc
    return loss, Munch(adv=loss_adv.item(),
                        sty=loss_sty.item(),
                        ds=loss_ds.item(),
                        cyc=loss_cyc.item())


# def compute_g_loss_latent(nets, args, x_real, y_org, y_trg, covar, z_trgs=None, x_refs=None, masks=None):
#     assert (z_trgs is None) != (x_refs is None)
#     if z_trgs is not None:
#         z_trg, z_trg2 = z_trgs
#     if x_refs is not None:
#         x_ref, x_ref2 = x_refs

#     # adversarial loss
#     if z_trgs is not None:
#         s_trg = nets.mapping_network(z_trg, y_trg)
#     else:
#         s_trg = nets.style_encoder(x_ref, y_trg)

#     x_fake = nets.generator(x_real, s_trg, masks=masks)
#     out = nets.discriminator(x_fake, y_trg)
#     loss_adv = adv_loss(out, 1)

#     # style reconstruction loss
#     s_pred = nets.style_encoder(x_fake, y_trg)
#     loss_sty = torch.mean(torch.abs(s_pred - s_trg))

#     # diversity sensitive loss
#     if z_trgs is not None:
#         s_trg2 = nets.mapping_network(z_trg2, y_trg)
#     else:
#         s_trg2 = nets.style_encoder(x_ref2, y_trg)
#     x_fake2 = nets.generator(x_real, s_trg2, masks=masks)
#     x_fake2 = x_fake2.detach()
#     loss_ds = torch.mean(torch.abs(x_fake - x_fake2))

#     # cycle-consistency loss
#     masks = nets.fan.get_heatmap(x_fake) if args.w_hpf > 0 else None
#     s_org = nets.style_encoder(x_real, y_org)

#     ###Covariate loss ####

#     # loss_covar = torch.mean(torch.square(s_org[:,0] - covar))



    x_rec = nets.generator(x_fake, s_org, masks=masks)
    loss_cyc = torch.mean(torch.abs(x_rec - x_real))
    print(x_rec)
    print(loss_ds, loss_cyc, loss_sty)

    loss = loss_adv + args.lambda_sty * loss_sty \
        - args.lambda_ds * loss_ds + args.lambda_cyc * loss_cyc
    return loss, Munch(adv=loss_adv.item(),
                        sty=loss_sty.item(),
                        ds=loss_ds.item(),
                        cyc=loss_cyc.item())



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
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg