import argparse


def add_args():
    parser = argparse.ArgumentParser(description="AT_PT")

    parser.add_argument('--batch-size', type=int, default=4096, metavar='N',
                        help='input batch size for training (default: %(default)s)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--lr', type=float, default=0.0007, metavar='LR',
                        help='learning rate (default: %(default)s)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: %(default)s)')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training (default: %(default)s)')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass (default: %(default)s)')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: %(default)s)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status (default: %(default)s)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model (default: %(default)s)')

    parser.add_argument('--model-path', type=str, default='my_model',
                        help='path to save model (default: %(default)s)')
    parser.add_argument('--dropout', type=float, default=0.27,
                        help='Dropout percentage (default: %(default)s)')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam'],
                        help='Optimizer name adam (default: %(default)s)')



# Run this many copies of the program on the given nodes.
    parser.add_argument('-n', type=int, default=0,
                        help='Run this many copies of the program on the given nodes. n > 0 for mpirun (default: %(default)s)')

    parser.add_argument('--hpu', action='store_true', default=False,
                        help='Use hpu device')
    parser.add_argument('--use_lazy_mode', action='store_true', default=False,
                        help='Enable lazy mode on hpu device, default eager mode')
    parser.add_argument('--hmp', dest='is_hmp', action='store_true', help='enable hmp mode')
    parser.add_argument('--hmp-bf16', default='ops_bf16_mnist.txt', help='path to bf16 ops list in hmp O1 mode')
    parser.add_argument('--hmp-fp32', default='ops_fp32_mnist.txt', help='path to fp32 ops list in hmp O1 mode')
    parser.add_argument('--hmp-opt-level', default='O1', help='choose optimization level for hmp')
    parser.add_argument('--hmp-verbose', action='store_true', help='enable verbose mode for hmp')

    
    
    
    
    
    parser.add_argument('--dl-worker-type', default='MP', type=lambda x: x.upper(),
                        choices = ["MT", "MP"], help='select multithreading or multiprocessing')
    parser.add_argument('--world_size', default=1, type=int, metavar='N',
                        help='number of total workers (default: 1)')
    parser.add_argument('--process-per-node', default=8, type=int, metavar='N',
                        help='Number of process per node')
    parser.add_argument('--start-epoch', type=int, default=0, metavar='N', help='starting epoch number, default 0')
    parser.add_argument('--checkpoint', default='', help='resume from checkpoint')
    parser.add_argument('--distributed', action='store_true', help='whether to enable distributed mode and run on multiple devices')






    args = parser.parse_args()

    return args
