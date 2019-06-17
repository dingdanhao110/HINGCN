import sys
import argparse
import ujson as json
import matplotlib.pyplot as plt

from time import time

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--log-path', type=str, required=True)
    parser.add_argument('--log-file', type=str, required=True)

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()

    epoches = []
    val_loss = []
    train_loss = []
    val_acc = []
    train_acc = []

    with open('{}{}.txt'.format(args.log_path,args.log_file), mode='r') as f:
        print(f)
        for line in f:
            if '{' not in line:
                continue

            line = json.loads(line)
            if 'train_loss' in line:
                train_loss.append(line['train_loss'])
            if 'val_loss' in line:
                epoches.append(line['epoch'])
                val_loss.append(line['val_loss'])

            if 'train_metric' in line:
                if line['epoch_progress']==0:
                    train_acc.append(line['train_metric']['accuracy'])
            if 'val_metric' in line:
                val_acc.append(line['val_metric']['accuracy'])

    l = min(len(train_loss),len(val_loss),len(val_acc),len(train_acc),len(epoches))
    while len(train_loss)>l:
        train_loss=train_loss[:-1]
    while len(val_loss)>l:
        val_loss=val_loss[:-1]
    while len(train_acc)>l:
        train_acc=train_acc[:-1]
    while len(val_acc)>l:
        val_acc=val_acc[:-1]
    while len(epoches) > l:
        epoches = epoches[:-1]

    plt.figure()
    plt.title(args.log_file)
    plt.subplot(211)
    plt.plot(epoches, train_loss, 'r', label='train')
    plt.plot(epoches, val_loss, 'g', label='val')
    plt.ylabel('loss')
    plt.xlabel('epoches')
    plt.legend(loc='upper right')

    plt.subplot(212)
    plt.plot(epoches, train_acc, 'r', label='train')
    plt.plot(epoches, val_acc, 'g', label='val')
    plt.ylabel('accuracy')
    plt.xlabel('epoches')
    plt.legend(loc='lower right')
    plt.suptitle(args.log_file)
    plt.show()
