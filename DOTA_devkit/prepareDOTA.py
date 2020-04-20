import os
import argparse
import ImgSplit_multi_process
from DOTAdata import *

wordname_15 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']

def parse_args():
    parser = argparse.ArgumentParser(description='prepare dota1')
    parser.add_argument('--srcpath', default=r'../data')
    parser.add_argument('--dstpath', default=r'../datasets',
                        help='prepare data')
    args = parser.parse_args()

    return args



def prepare(srcpath, dstpath):
    if not os.path.exists(os.path.join(dstpath, 'test')):
        os.mkdir(os.path.join(dstpath, 'test'))
    if not os.path.exists(os.path.join(dstpath, 'val')):
        os.mkdir(os.path.join(dstpath, 'val'))
    if not os.path.exists(os.path.join(dstpath, 'train')):
        os.mkdir(os.path.join(dstpath, 'train'))
    split_train = ImgSplit_multi_process.splitbase(os.path.join(srcpath, 'train'),
                       os.path.join(dstpath, 'train'),
                      gap=200,
                      subsize=1024,
                      num_process=32
                      )
    split_train.splitdata(1)

    split_val = ImgSplit_multi_process.splitbase(os.path.join(srcpath, 'val'),
                       os.path.join(dstpath, 'val'),
                      gap=200,
                      subsize=1024,
                      num_process=32
                      )
    split_val.splitdata(1)

    split_test = SplitOnlyImage_multi_process.splitbase(os.path.join(srcpath,'test','images','images'),
                       os.path.join(dstpath, 'test', 'image'),
                      gap=200,
                      subsize=1024,
                      num_process=32
                      )
    split_test.splitdata(1)
    if not os.path.exists(os.path.join(dstpath, 'annotations')):
        os.mkdir(os.path.join(dstpath, 'annotations'))
    DOTA2COCOTrain(os.path.join(dstpath,'val'),os.path.join(dstpath,'annotations','instances_val.json'),wordname_15,difficult=-1)
    DOTA2COCOTrain(os.path.join(dstpath,'train'),os.path.join(dstpath,'annotations','instances_train.json'),wordname_15,difficult=-1)



if __name__ == '__main__':
    args = parse_args()
    srcpath = args.srcpath
    dstpath = args.dstpath
    prepare(srcpath,dstpath)
