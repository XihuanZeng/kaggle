# this script add libffm score to the training and test set

import pandas as pd
import argparse
import os

def add_libffm_feature(model_input_dir, libffm_dir, train_or_test, cluster):
    output = []
    with open(os.path.join(libffm_dir, 'tmp_%s_output.txt' % train_or_test), 'rb') as f:
        while 1:
            line = f.readline()
            if line == '':
                break
            output.append(float(line.strip()))
    f.close()
    data = pd.read_csv(os.path.join(model_input_dir, '%s' % train_or_test, '%s%s' % (train_or_test, str(cluster))))
    data['libffm_score'] = output
    data.to_csv(os.path.join(model_input_dir, '%s' % train_or_test, '%s%s' % (train_or_test, str(cluster))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cluster',  type=int)
    args = parser.parse_args()
    k = args.cluster
    add_libffm_feature('train', k)
    add_libffm_feature('test', k)

if __name__ == '__main__':
    main()

