import argparse


def add_args():
    parser = argparse.ArgumentParser(description="CT args generator")

    parser.add_argument('-n', type=int, default=0,
                        help='Run this many copies of the program on the given nodes. n > 0 for mpirun (default: %(default)s)')
    parser.add_argument('--minibatch-size', type=int, default=1,
                        help='Minibatch size (default: %(default)s)')
    parser.add_argument('--iterations', type=int, default=10,
                        help='Number of iterations to run (default: %(default)s)')

    precisionDict = {
        'float32':  0,
        'mixed':    1,
        'bfloat16': 2,
        'float16':  3,
    }

    precisionChoices = []
    for key, value in precisionDict.items():
        precisionChoices.append(key)

    parser.add_argument('--precision', type=str, default='float32', choices=precisionChoices,
                        help='Precision (default: %(default)s)')


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


    args.precision = precisionDict[args.precision]

    return args
