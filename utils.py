import torchvision.utils as vutils
import torch.nn as nn
import numpy as np
import torch
import wandb

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)


def he_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def denormalize(x):
    out = (x + 0.5) / 2
    return out


@torch.no_grad()
def calculate_latent_vector(nets, args, x_src, y_src):
    N, C, H, W, D = x_src.size()
    s_src = nets.style_encoder(x_src, y_src)
    return s_src


# @torch.no_grad()
# def generate_from_latent(nets, args, x_src, latent, outname):
#     N, C, H, W , D= x_src.size()
#     img_gen = nets.generator(x_src, latent)
#     save_image(img_gen, 1, outname)


@torch.no_grad()
def translate_and_reconstruct(
    generator, style_encoder, args, x_src, y_src, x_ref, y_ref, filename, logger
):
    N, C, H, W = x_src.size()
    s_ref = style_encoder(x_ref, y_ref)
    x_fake = generator(x_src, s_ref, masks=None)
    s_src = style_encoder(x_src, y_src)
    x_rec = generator(x_fake, s_src, masks=None)
    x_concat = [x_src, x_ref, x_fake, x_rec]
    x_concat = torch.cat(x_concat, dim=0)
    log_image(x_concat, N, logger, filename)
    del x_concat


@torch.no_grad()
def translate_using_latent(
    generator,
    mapping_network,
    args,
    x_src,
    y_trg_list,
    z_trg_list,
    psi,
    filename,
    logger,
):
    N, C, H, W = x_src.size()
    latent_dim = z_trg_list[0].size(1)
    x_concat = [x_src]

    for i, y_trg in enumerate(y_trg_list):
        z_many = torch.randn(10000, latent_dim).to(x_src.device)
        y_many = torch.LongTensor(10000).to(x_src.device).fill_(y_trg[0])
        s_many = mapping_network(z_many, y_many)
        s_avg = torch.mean(s_many, dim=0, keepdim=True)
        s_avg = s_avg.repeat(N, 1)

        for z_trg in z_trg_list:
            s_trg = mapping_network(z_trg, y_trg)
            s_trg = torch.lerp(s_avg, s_trg, psi)
            x_fake = generator(x_src, s_trg, masks=None)
            x_concat += [x_fake]

    x_concat = torch.cat(x_concat, dim=0)
    log_image(x_concat, N, logger, filename)


@torch.no_grad()
def translate_using_reference(
    generator, style_encoder, args, x_src, x_ref, y_ref, filename, logger
):
    N, C, H, W = x_src.size()
    wb = torch.ones(1, C, H, W).to(x_src.device)
    x_src_with_wb = torch.cat([wb, x_src], dim=0)

    s_ref = style_encoder(x_ref, y_ref)
    s_ref_list = s_ref.unsqueeze(1).repeat(1, N, 1)
    x_concat = [x_src_with_wb]
    for i, s_ref in enumerate(s_ref_list):
        x_fake = generator(x_src, s_ref, masks=None)
        x_fake_with_ref = torch.cat([x_ref[i : i + 1], x_fake], dim=0)
        x_concat += [x_fake_with_ref]

    x_concat = torch.cat(x_concat, dim=0)
    log_image(x_concat, N + 1, logger, filename)
    del x_concat


def log_image(x, ncol, logger, name):
    x = denormalize(x)
    grid = vutils.make_grid(x.cpu(), nrow=ncol, padding=0, normalize=True)
    logger.log({f"{name}": [wandb.Image(grid)]})


@torch.no_grad()
def debug_image(
    generator,
    style_encoder,
    mapping_network,
    args,
    x_src,
    y_src,
    x_ref,
    y_ref,
    step,
    logger,
):

    device = x_src.device
    N = x_src.size(0)

    # translate and reconstruct (reference-guided)
    filename = "cycle_consistency.jpg"
    translate_and_reconstruct(
        generator, style_encoder, args, x_src, y_src, x_ref, y_ref, filename, logger
    )

    # latent-guided image synthesis
    y_trg_list = [
        torch.tensor(y).repeat(N).to(device) for y in range(min(args.num_domains, 5))
    ]
    z_trg_list = (
        torch.randn(args.num_outs_per_domain, 1, args.latent_dim)
        .repeat(1, N, 1)
        .to(device)
    )
    for psi in [0.5, 0.7, 1.0]:
        filename = "latent_psi_%.1f.jpg" % (psi)
        translate_using_latent(
            generator,
            mapping_network,
            args,
            x_src,
            y_trg_list,
            z_trg_list,
            psi,
            filename,
            logger,
        )

    # reference-guided image synthesis
    filename = "reference.jpg"
    translate_using_reference(
        generator, style_encoder, args, x_src, x_ref, y_ref, filename, logger
    )
