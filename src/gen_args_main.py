# The only purpose of this code is to generate arguments for CosmicTagger on Habana.
# An example is this code automatically generated a string for Habana based on
# "--precision 'mixed'".
from gen_args_args import add_args


def main():

    args = add_args()

    if args.hpu:
        computeModeStr = "    run.compute_mode=HPU<newline>"
    else:
        computeModeStr = "    run.compute_mode=CPU<newline>"

    ctRunCommand = \
    f"    name={args.precision}_{args.minibatch_size}x{args.iterations}" + \
    "_${i}\n" + \
    "    python bin/exec.py<newline>" + \
    "    mode=train<newline>" + \
    "    run.id=${name}<newline>" + \
    "    run.distributed=False<newline>" + \
    "    data.data_directory=/lambda_stor/data/datascience/cosmic_tagging/<newline>" + \
    "    framework=torch<newline>" + \
    computeModeStr + \
    f"    run.minibatch_size={args.minibatch_size}<newline>" + \
    f"    run.iterations={args.iterations}<newline>" + \
    f"    run.precision={args.precisionId}<newline>"

    logStr = " > ${name}.log 2>&1 &"




    # Deal with CT run.precision = mixed
    if args.precision == 'mixed':
        mixedPrecisionStr = f'    --hmp --hmp-bf16={args.hmp_bf16} --hmp-fp32={args.hmp_fp32}'
        ctRunCommand += mixedPrecisionStr + logStr
    else:
        ctRunCommand += logStr

    if args.n > 0:
        ctRunCommand = ctRunCommand.replace('run.distributed=False', 'run.distributed=True')
        mpirunCommand = f"    mpirun -n {args.n} --bind-to core --map-by slot:PE=7 --rank-by core --report-bindings --allow-run-as-root<newline>"
        finalCommand = mpirunCommand + ctRunCommand
    else:
        finalCommand = ctRunCommand

    #print(f'Command: \n{finalCommand}')

    #print(f'\n\nCommand:')

    print('for i in {1..5}')
    print('do')

    commandLine = finalCommand.split('<newline>')
    for line in commandLine:
        if line.find('hmp') < 0:
            print(f'{line} \\')
        else:
            print(f'{line}')

    print('done')


    return 0


if __name__ == '__main__':
    import sys
    try:
        retVal = main()
    except KeyboardInterrupt:
        print('Received <Ctrl>+c')
        sys.exit(-1)

    sys.exit(retVal)
