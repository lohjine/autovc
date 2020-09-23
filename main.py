import os
import argparse
from solver_encoder import Solver
from data_loader import get_loader
from torch.backends import cudnn


def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training.
    cudnn.benchmark = True

    print('starting up')

    # Data loader.
    vcc_loader = get_loader(config.data_dir, config.batch_size, config.len_crop)

    print('data loaded')

    solver = Solver(vcc_loader, config)

    print('solver loaded, training...')

    solver.train()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--lambda_cd', type=float, default=1, help='weight for hidden code loss')
    parser.add_argument('--dim_neck', type=int, default=64) #16
    parser.add_argument('--dim_emb', type=int, default=19) # set to number of one-hot
    parser.add_argument('--dim_pre', type=int, default=512)
    parser.add_argument('--freq', type=int, default=16) #16

    # Training configuration.
    parser.add_argument('--data_dir', type=str, default='./autovc_train') #r'C:\Users\ACTUS\Desktop\pyscripts\waveglow\data\autovc_train') #
#    parser.add_argument('--data_dir', type=str, default=r'./spmel')
    parser.add_argument('--batch_size', type=int, default=2, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=1000000, help='number of total iterations')
    parser.add_argument('--len_crop', type=int, default=128, help='dataloader output sequence length')

    # Miscellaneous.
    parser.add_argument('--log_step', type=int, default=500)
    parser.add_argument('--checkpoint', type=int, default=20000)

    parser.add_argument('--resume', type=str, default='checkpoint/chkpt_140000')

    config = parser.parse_args()
    print(config)
    main(config)