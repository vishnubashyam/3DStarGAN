import argparse


def create_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment", type=str,
                        help="Name of the experiment", default='None')

    parser.add_argument(
        "--experiment_tag", type=str, default="None", help="Optional Tag for experiment"
    )
    # model arguments
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image resolution')
    parser.add_argument('--num_domains', type=int, default=1,
                        help='Number of domains')
    parser.add_argument('--latent_dim', type=int, default=16,
                        help='Latent vector dimension')
    parser.add_argument('--hidden_dim', type=int, default=1024,
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

    # training arguments
    parser.add_argument('--resume_iter', type=int, default=0,
                        help='Iterations to resume training/testing')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=32,
                        help='Batch size for Validation')
    parser.add_argument('--lr', type=float, default=2e-4,
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
                        choices=['train', 'train_with_covar', 'sample',
                                 'eval', 'align', 'latent', 'generate_latent'],
                        help='This argument is used in solver')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of workers used in DataLoader')
    parser.add_argument('--device', type=str, default='gpu',
                        help='Specify the device to use: gpu or cpu')

    # directory for testing
    parser.add_argument('--csv_path', type=str,
                        help='Path to train csv')
    parser.add_argument('--img_dir', type=str,
                        help='Directory containing input source images')
    parser.add_argument('--log_initial_debug_images ', action='store_true')

    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--sample_every', type=int, default=1000)
    parser.add_argument('--save_every', type=int, default=5000)
    parser.set_defaults(log_initial_debug_images=False)

    # parser.add_argument(
    #     "--data_augmentation",
    #     action="store_true",
    #     help="Use data augmentation during training and testing"
    # )
    # parser.add_argument(
    #     "--job_id",
    #     type=str,
    #     default='None',
    #     help="ID of the submission on the cluster"
    # )

    parser.add_argument("--model_name", type=str, default="StarGan")
    parser.add_argument(
        "--max_steps", type=int, help="Number of steps to train for", default=1000000
    )

    return parser
