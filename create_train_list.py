import pandas as pd
import os
from ipdb import set_trace

trainset_dir = os.path.join('gldv2_dataset', 'train')
testset_dir = os.path.join('gldv2_dataset', 'test')
indexset_dir = os.path.join('gldv2_dataset', 'index')
trainclean_fname = 'train_clean.csv'
retrieval_fname = 'retrieval_solution_v2.1.csv'
recognition_fname = 'recognition_solution_v2.1.csv'

# The query images in test set only has 720, so do not use train set labels
"""
trainclean_df = pd.read_csv(os.path.join(trainset_dir, trainclean_fname))
all_labels = []
for _, row in trainclean_df.iterrows():
    all_labels.append(row['landmark_id'])
unique_labels = sorted(set(all_labels))
relabeling = {label: index for index, label in enumerate(unique_labels)}

recognition_df = pd.read_csv(os.path.join(testset_dir, recognition_fname))
usable_test_id_label = {}
for _, row in recognition_df.iterrows():
    try:
        cur_label = int(row['landmarks'])
        if cur_label not in relabeling.keys():
            continue
        cur_relabel = relabeling[cur_label]
        test_id = row['id']
        usable_test_id_label[test_id] = cur_relabel

    except ValueError:
        continue

retrieval_df = pd.read_csv(os.path.join(testset_dir, retrieval_fname))
retrieval_id_res = {}
for _, row in retrieval_df.iterrows():
    test_id = row['id']
    if row['images'] != 'None' and test_id in usable_test_id_label.keys():
        index_image_list = row["images"].split(' ')
        retrieval_id_res[test_id] = index_image_list
    if test_id in usable_test_id_label.keys() and test_id not in retrieval_id_res.keys():
        print(test_id)

print(len(retrieval_id_res))
print(len(usable_test_id_label))

"""

recognition_df = pd.read_csv(os.path.join(testset_dir, recognition_fname))
usable_test_id_label = {}
for _, row in recognition_df.iterrows():
    try:
        cur_label = int(row['landmarks'])
        test_id = row['id']
        usable_test_id_label[test_id] = cur_label

    except ValueError:
        continue

retrieval_df = pd.read_csv(os.path.join(testset_dir, retrieval_fname))
retrieval_id_res = {}
all_labels = []
for _, row in retrieval_df.iterrows():
    test_id = row['id']
    if test_id in usable_test_id_label.keys():
        if row['images'] != 'None':
            index_image_list = row['images'].split(' ')
            retrieval_id_res[test_id] = index_image_list
            all_labels.append(usable_test_id_label[test_id])
        else:
            del usable_test_id_label[test_id]

unique_labels = sorted(set(all_labels))
relabeling = {label: index for index, label in enumerate(unique_labels)}
for test_id in usable_test_id_label.keys():
    cur_label = usable_test_id_label[test_id]
    usable_test_id_label[test_id] = relabeling[cur_label]

# print(usable_test_id_label)
# print(retrieval_id_res)
# print(len(unique_labels))

# store retrieval_train_list
res_fname = 'retrieval_train_list.txt'
with open(os.path.join(testset_dir, res_fname), 'w') as f:
    for test_id, relabel in usable_test_id_label.items():
        f.write(f'test/{test_id[0]}/{test_id[1]}/{test_id[2]}/{test_id}.jpg {relabel}\n')
        for index_id in retrieval_id_res[test_id]:
            f.write(f'index/{index_id[0]}/{index_id[1]}/{index_id[2]}/{index_id}.jpg {relabel}\n')
