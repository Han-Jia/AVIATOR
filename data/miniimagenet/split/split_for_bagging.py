import numpy as np
from tqdm import tqdm
np.random.seed(10000)

data_dict = {}
with open('train.csv') as f:
    next(f)
    for line in f:
        path, class_name = line.strip().split(',')
        if class_name in data_dict:
            data_dict[class_name].append(path)
        else:
            data_dict[class_name] = [path]
                        
# sampling 10 bags
num_bag = 10
percent = 0.75
class_list = list(data_dict.keys())
num_class = len(class_list)
selected_class = int(percent * num_class)
perm = np.arange(num_class)

for index in tqdm(range(num_bag)):
    np.random.shuffle(perm)
    current_class_list = [class_list[e] for e in perm[:selected_class]]
    with open('train_' + str(index) + '.csv', 'w') as f:
        f.write('filename,label\n')
        for e in current_class_list:
            for term in data_dict[e]:
                f.write(','.join([term, e]) + '\n')