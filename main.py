from data import Dataset_3d, get_train_loader, get_test_loader
from solver import Solver
from torch import nn
from munch import Munch
import pandas as pd
import numpy as np
import configparser
import argparse
import torch
import os

def main(args):
    # Setting Random Seeds
    np.random.seed(999)
    torch.manual_seed(999)
    torch.cuda.manual_seed(999)

    # Configure GPUs
    torch.backends.cudnn.benchmark = True
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("\nTorch GPUs: " + str(torch.cuda.device_count()))

    # Read config file
    config = configparser.ConfigParser()
    config.read('./config/config.ini')

    csv = pd.read_csv('/home/bashyamv/Research/3DStarGAN/3DStarGAN/train_df_modal.csv')
    # csv.columns = ['fname']
    # csv["Domain"] = 0

    solver = Solver(args)

    loaders = Munch(src=get_train_loader(csv, root='',
                                                    which='source',
                                                    img_size=182,
                                                    batch_size=1,
                                                    prob=0,
                                                    num_workers=20),
                            ref=get_train_loader(csv, root='',
                                                    which='reference',
                                                    img_size=182,
                                                    batch_size=1,
                                                    prob=0,
                                                    num_workers=20),
                            val=get_test_loader(csv, root='',
                                                    img_size=182,
                                                    batch_size=1,
                                                    shuffle=True,
                                                    num_workers=20))

    solver.train(loaders= loaders)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    # model arguments
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image resolution')
    parser.add_argument('--num_domains', type=int, default=4,
                        help='Number of domains')
    parser.add_argument('--latent_dim', type=int, default=512,
                        help='Latent vector dimension')
    parser.add_argument('--hidden_dim', type=int, default=50,
                        help='Hidden dimension of mapping network')
    parser.add_argument('--style_dim', type=int, default=16,
                        help='Style code dimension')


    # weight for objective functions
    parser.add_argument('--lambda_reg', type=float, default=1,
                        help='Weight for R1 regularization')
    parser.add_argument('--lambda_cyc', type=float, default=1,
                        help='Weight for cyclic consistency loss')
    parser.add_argument('--lambda_sty', type=float, default=1,
                        help='Weight for style reconstruction loss')
    parser.add_argument('--lambda_ds', type=float, default=1,
                        help='Weight for diversity sensitive loss')
    parser.add_argument('--ds_iter', type=int, default=50000,
                        help='Number of iterations to optimize diversity sensitive loss')
    parser.add_argument('--w_hpf', type=float, default=0,
                        help='weight for high-pass filtering')


    # training arguments
    parser.add_argument('--randcrop_prob', type=float, default=0.0,
                        help='Probabilty of using random-resized cropping')
    parser.add_argument('--total_iters', type=int, default=1000000,
                        help='Number of total iterations')
    parser.add_argument('--resume_iter', type=int, default=0,
                        help='Iterations to resume training/testing')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=1,
                        help='Batch size for validation')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for D, E and G')
    parser.add_argument('--f_lr', type=float, default=1e-6,
                        help='Learning rate for F')
    parser.add_argument('--beta1', type=float, default=0.0,
                        help='Decay rate for 1st moment of Adam')
    parser.add_argument('--beta2', type=float, default=0.99,
                        help='Decay rate for 2nd moment of Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--num_outs_per_domain', type=int, default=5,
                        help='Number of generated images per domain during sampling')


    # misc
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'train_with_covar','sample', 'eval', 'align', 'latent', 'generate_latent'],
                        help='This argument is used in solver')
    parser.add_argument('--num_workers', type=int, default=10,
                        help='Number of workers used in DataLoader')
    parser.add_argument('--seed', type=int, default=655,
                        help='Seed for random number generator')
    # directory for training
    parser.add_argument('--train_img_dir', type=str, default='data/celeba_hq/train',
                        help='Directory containing training images')
    parser.add_argument('--val_img_dir', type=str, default='data/celeba_hq/val',
                        help='Directory containing validation images')
    parser.add_argument('--sample_dir', type=str, default='expr/samples',
                        help='Directory for saving generated images')
    parser.add_argument('--checkpoint_dir', type=str, default='expr/checkpoints',
                        help='Directory for saving network checkpoints')


    parser.add_argument('--gen_latent_csv', type=str, required=False,
                        help='CSV to generate new images with specificed latent values')
    parser.add_argument('--train_covariates_csv', type=str, required=False,
                        help='CSV covariates information for training')
    # directory for calculating metrics
    parser.add_argument('--eval_dir', type=str, default='expr/eval',
                        help='Directory for saving metrics, i.e., FID and LPIPS')


    # directory for testing
    parser.add_argument('--result_dir', type=str, default='expr/results',
                        help='Directory for saving generated images and videos')
    parser.add_argument('--src_dir', type=str, default='/home/bashyamv/Research/Data/ADNI/Src_img',
                        help='Directory containing input source images')
    parser.add_argument('--ref_dir', type=str, default='/home/bashyamv/Research/Data/ADNI/Ref_img',
                        help='Directory containing input reference images')
    parser.add_argument('--inp_dir', type=str, default='assets/representative/custom/female',
                        help='input directory when aligning faces')
    parser.add_argument('--out_dir', type=str, default='assets/representative/celeba_hq/src/female',
                        help='output directory when aligning faces')


    # step size
    parser.add_argument('--print_every', type=int, default=1)
    parser.add_argument('--sample_every', type=int, default=1000)
    parser.add_argument('--save_every', type=int, default=10000)
    parser.add_argument('--eval_every', type=int, default=1000000)

    args = parser.parse_args()
    main(args)