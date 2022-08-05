# The only purpose of this code is to generate arguments for CosmicTagger on Habana.
# An example is this code automatically generated a string for Habana based on
# "--precision 'mixed'".
from gen_args_args import add_args


def main():

    args = add_args()

    ctRunCommand = "python bin/exec.py<newline>" + \
    "mode=train<newline>" + \
    f"run.id='{args.precision}_{args.minibatch_size}x{args.iterations}'<newline>" + \
    "run.distributed=False<newline>" + \
    "data.data_directory=/lambda_stor/data/datascience/cosmic_tagging/<newline>" + \
    "framework=torch<newline>" + \
    "run.compute_mode=HPU<newline>" + \
    f"run.minibatch_size={args.minibatch_size}<newline>" + \
    f"run.iterations={args.iterations}<newline>" + \
    f"run.precision={args.precisionId}"


    # Deal with CT run.precision = mixed
    if args.precision == 'mixed':
        mixedPrecisionStr = f'<newline>--hmp --hmp-bf16={args.hmp_bf16} --hmp-fp32={args.hmp_fp32}'
        ctRunCommand += mixedPrecisionStr

    if args.n > 0:
        ctRunCommand = ctRunCommand.replace('run.distributed=False', 'run.distributed=True')
        mpirunCommand = f"mpirun -n {args.n} --bind-to core --map-by slot:PE=7 --rank-by core --report-bindings --allow-run-as-root<newline>"
        finalCommand = mpirunCommand + ctRunCommand
    else:
        finalCommand = ctRunCommand

    print(f'Command: \n{finalCommand}')

    print(f'\n\nCommand:')

    commandLine = finalCommand.split('<newline>')
    for line in commandLine:
        print(f'{line} \\')



    return 0


if __name__ == '__main__':
    import sys
    try:
        retVal = main()
    except KeyboardInterrupt:
        print('Received <Ctrl>+c')
        sys.exit(-1)

    sys.exit(retVal)
