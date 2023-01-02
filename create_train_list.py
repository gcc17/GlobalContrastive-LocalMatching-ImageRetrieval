import pandas as pd
import os
from ipdb import set_trace

data_dir = os.path.join('datasets', 'data', 'landmark')
trainset_dir = os.path.join(data_dir, 'train')
testset_dir = os.path.join(data_dir, 'test')
indexset_dir = os.path.join(data_dir, 'index')
trainclean_fname = 'train_clean.csv'
train_fname = 'train.csv'
retrieval_fname = 'retrieval_solution_v2.1.csv'
recognition_fname = 'recognition_solution_v2.1.csv'
remapped_train_fname = 'train_list.txt'

train_df = pd.read_csv(os.path.join(trainset_dir, train_fname))
recognition_df = pd.read_csv(os.path.join(testset_dir, recognition_fname))
# image_id: new landmark id
new_train_id = {}
with open(os.path.join(data_dir, remapped_train_fname), "r") as fin:
    for line in fin:
        im_dir, new_id = line.strip().split(" ")
        image_id = im_dir.split("/")[-1].split(".")[0]
        new_train_id[image_id] = int(new_id)
# from train, old landmark id: image_id
old_train_id = {}
for _, row in train_df.iterrows():
    old_id = row['landmark_id']
    image_id = row['id']
    if old_id not in old_train_id.keys():
        old_train_id[old_id] = []
    old_train_id[old_id].append(image_id)

# for testset: map old landmark id to new landmark id
# image_id: new landmark id
new_test_id = {}
for _, row in recognition_df.iterrows():
    try:
        cur_label = int(row['landmarks'])
        test_image_id = row['id']
        train_image_list = old_train_id[cur_label]
        for train_image_id in train_image_list:
            if train_image_id in new_train_id.keys():
                if test_image_id not in new_test_id.keys():
                    new_test_id[test_image_id] = new_train_id[train_image_id]
                else:
                    if new_test_id[test_image_id] != new_train_id[train_image_id]:
                        print(f'[ERROR]: {test_image_id} different ids \
                            {new_test_id[test_image_id]} {new_train_id[train_image_id]}')
    except ValueError:
        continue

# store new_test_list
res_fname = 'new_test_list.txt'
with open(os.path.join(data_dir, res_fname), 'w') as f:
    for test_image_id, new_id in new_test_id.items():
        f.write(f'test/{test_image_id[0]}/{test_image_id[1]}/{test_image_id[2]}/{test_image_id}.jpg {new_id}\n')


