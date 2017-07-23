"""
This files merges two data files, both in JTR format, into a single JTR data file.
It assumes that the structure of instances is identical for both input files
and only concatenates the two instances lists.
It also assumes that the global variables are identical in both input files.
"""

import json
import sys


def main():

    if len(sys.argv) != 4:
        print('Wrong arguments for merging two data files in Jack format into one. Usage:')
        print('\tpython3 merge_JTR_data_files.py input1.json input2.json output.json')
    else:
        # load input 1
        with open(sys.argv[1], 'r') as inputfile1:
            content1 = json.load(inputfile1)

        # load input 2
        with open(sys.argv[2], 'r') as inputfile2:
            content2 = json.load(inputfile2)

        # define new 'meta' field
        meta_ = "Merged Content of {} and {}".format(content1['meta'], content2['meta'])

        # define new 'globals' field. Note: so far assuming same globals in both input files.
        assert (content1['globals']) == content2['globals']
        globals_ = content1['globals']

        # concatenating instances of both input files
        instances_ = content1['instances'] + content2['instances']

        # defining the dictionary for dumping into json
        merged_content = {'meta': meta_, 'globals': globals_, 'instances': instances_}

        # sanity check: nothing unexpected got lost or added
        assert len(content1['instances']) + len(content2['instances']) == len(merged_content['instances'])

        # summary print
        print('Merged file {} with {} into {}'.format(sys.argv[1],sys.argv[2],sys.argv[3]))
        print('Number of instances: input1: {} input2: {} output: {}'\
            .format(len(content1['instances']), len(content2['instances']), len(merged_content['instances'])))

        # dump merged content into JTR output file.
        with open(sys.argv[3], 'w') as outputfile:
            json.dump(merged_content, outputfile)



if __name__ == "__main__":
    main()
