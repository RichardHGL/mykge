import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--every', type=int, default=10)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dim', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--batch_size2', type=int, default=128)
    parser.add_argument('--p', type=int, default=1)
    parser.add_argument('--clipping_max_value', type=float, default=4.0)
    parser.add_argument('--optim', type=str, default='Adam')
    parser.add_argument('--n_sample', type=int, default=12)
    parser.add_argument('--mode', type=str, default='rand',
                        help='{rand, rela, popu, corr, adve, cach, 1all, kall')
    
    parser.add_argument('--rate', type=float, default=0.1)

# TransX
    parser.add_argument('--gamma', type=float, default=12.0)
    parser.add_argument('--bern', action='store_true')
    # unused
    parser.add_argument('--margin', type=float, default=4.0)
    parser.add_argument('--reg', type=float, default=1.0)
# ConvE
    parser.add_argument('--label_smooth_eps', type=float, default=0.1)
    parser.add_argument('--input_drop', type=float, default=0.2)
    parser.add_argument('--feat_drop', type=float, default=0.2)
    parser.add_argument('--hidd_drop', type=float, default=0.3)
# ConvTransE
    parser.add_argument('--channels', type=int, default=50)
    parser.add_argument('--kernel_size', type=int, default=3)
# Kbgan
    parser.add_argument('--n_pool', type=int, default=100)
    parser.add_argument('--model_D', type=str, default='TransH')
    parser.add_argument('--model_G', type=str, default='DistMult')
    parser.add_argument('--scratch', action='store_true')
    parser.add_argument('--D_file', type=str, default=None)
    parser.add_argument('--G_file', type=str, default=None)

# checkpoint
    parser.add_argument('--path', type=str, default='data')
    parser.add_argument('--data', type=str, default='FB15K237')
    parser.add_argument('--exp_path', type=str, default='experiment')
    parser.add_argument('--model', type=str, default='TransE')
    parser.add_argument('--v', type=float, default=0)
    parser.add_argument('--f', action='store_true')
    parser.add_argument('--adv', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--trained', action='store_true')
    parser.add_argument('--pretrained_name', type=str, default=None)

    return parser.parse_args()
