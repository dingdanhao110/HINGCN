import subprocess
import ujson as json
import numpy as np
import sys
runs=10
#Top k HAN, variant2； adjust train_per in helper.py
args = [
    'python3',
    'train.py',
    '--problem-path',
    '../../../LineGraphGCN/data/dblp2/',
    '--problem',
    'dblp',
    '--lr-init',
    '0.001',
    '--weight-decay',
    '5e-4',
    '--dropout',
    '0.5',
    '--prep-class',
    'linear',
    '--n-train-samples',
    '10,10',
    '--prep-len',
    '128',
    '--in-edge-len',
    '128',
    '--n-head',
    '8',
    '--output-dims',
    '128,16,32,32',
    '--train-per',
    '0.2',
]
test_acc = []
test_macro = []
for seed in range(runs):
    process = subprocess.Popen(args+['--seed',str(seed)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    text = process.communicate()[1]

    lines = text.decode().split('\n')
    # print(lines)
    
    for line in lines:
        if '{' not in line:
            continue
        print(line)
        line = json.loads(line)
        if 'test_metric' in line:
            test_acc.append(line['test_metric']['accuracy'])
            test_macro.append(line['test_metric']['macro'])
    sys.stdout.flush()
test_acc = np.asarray(test_acc)
test_macro = np.asarray(test_macro)
print('average acc for {} runs is : {}'.format(len(test_acc), np.average(test_acc)))
print('average macro for {} runs is : {}'.format(len(test_macro), np.average(test_macro)))




