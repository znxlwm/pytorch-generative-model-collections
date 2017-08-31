import argparse, os
from GAN import GAN
from CGAN import CGAN
from LSGAN import LSGAN
from DRAGAN import DRAGAN
from ACGAN import ACGAN
from WGAN import WGAN
from infoGAN import infoGAN
from EBGAN import EBGAN
from BEGAN import BEGAN

"""parsing and configuration"""
def parse_args():
    desc = "Pytorch implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--gan_type', type=str, default='EBGAN',
                        choices=['GAN', 'CGAN', 'infoGAN', 'ACGAN', 'EBGAN', 'BEGAN', 'WGAN', 'DRAGAN', 'LSGAN'],
                        help='The type of GAN')#, required=True)
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion-mnist', 'celebA'],
                        help='The name of dataset')
    parser.add_argument('--epoch', type=int, default=25, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory name to save the model')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--lrG', type=float, default=0.0002)
    parser.add_argument('--lrD', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--gpu_mode', type=bool, default=True)

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --save_dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # --result_dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # --result_dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

        # declare instance for GAN
    if args.gan_type == 'GAN':
        gan = GAN(args)
    elif args.gan_type == 'CGAN':
        gan = CGAN(args)
    elif args.gan_type == 'ACGAN':
        gan = ACGAN(args)
    elif args.gan_type == 'infoGAN':
        gan = infoGAN(args, SUPERVISED = True)
    elif args.gan_type == 'EBGAN':
        gan = EBGAN(args)
    elif args.gan_type == 'WGAN':
        gan = WGAN(args)
    elif args.gan_type == 'DRAGAN':
        gan = DRAGAN(args)
    elif args.gan_type == 'LSGAN':
        gan = LSGAN(args)
    elif args.gan_type == 'BEGAN':
        gan = BEGAN(args)
    else:
        raise Exception("[!] There is no option for " + args.gan_type)

        # launch the graph in a session
    gan.train()
    print(" [*] Training finished!")

    # visualize learned generator
    gan.visualize_results(args.epoch)
    print(" [*] Testing finished!")

if __name__ == '__main__':
    main()
