# Process log.

# The logs have already been combined with:
# cat fp32_10x*.log | grep 'process last'

import csv

def writeCsvFileFromList(path, StocksData):
    """Writes a list to a CSV file."""

    #List = StocksData[lowerBound:upperBound]  # This does not work.
    # It probably should be upperBound + 1.
    List = StocksData

    outputFile = path

    with open( outputFile, 'w', newline='' ) as csvOut:
        cw = csv.writer( csvOut, delimiter=',', quotechar='"' )
        cw.writerows(List)
        print(f'Samples written to {outputFile}: {len(List)}')

import argparse


def add_args():
    parser = argparse.ArgumentParser(description="AT_PT")

    parser.add_argument('file', type=str, default='./logs/sweep1.log',
                        help='File to process (default: %(default)s)')

    args = parser.parse_args()

    return args


def main():

    args = add_args()

    with open(args.file, 'r') as log_file:
        lines = log_file.readlines()

    linesArray = [['iterations', 'throughput']]

    assert len(lines) == 15

    for line in lines:
        index = line.find('iterations')
        newLine = line[index:]
        print(f'newLine: {newLine}')
        splitLines = newLine.split(',')
        print(f'splitLines: {splitLines}')
        splitLines[0] = splitLines[0].replace('iterations: ', '')
        splitLines[1] = splitLines[1].replace(' throughput: ', '')
        splitLines[1] = splitLines[1].replace('"', '')
        splitLines[1] = splitLines[1].replace(' ', '')
        splitLines[1] = splitLines[1].replace('\n', '')
        linesArray.append([splitLines[0], splitLines[1]])

    writeCsvFileFromList('log.csv', linesArray)

    return 0


if __name__ == '__main__':
    import sys
    try:
        retVal = main()
    except KeyboardInterrupt:
        print('Received <Ctrl>+c')
        sys.exit(-1)

    sys.exit(retVal)
