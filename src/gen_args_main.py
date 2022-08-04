from gen_args_args import add_args
#from AT_PT_other import *


def main():

    args = add_args()

    ctRunCommand = """python bin/exec.py<newline>""" + \
    """mode=train<newline>""" + \
    """run.id='fp16_10x1'<newline>""" + \
    """run.distributed=True<newline>""" + \
    """data.data_directory=/lambda_stor/data/datascience/cosmic_tagging/<newline>""" + \
    """framework=torch<newline>""" + \
    """run.compute_mode=HPU<newline>""" + \
    """run.minibatch_size=1<newline>""" + \
    """run.iterations=10<newline>""" + \
    """run.precision=3"""

    if args.n > 0:
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
