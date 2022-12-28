import os

import io
import numpy as np
from tools.util import ReadDatasetFile
import tools.delg_utils as delg_utils
from tools.revisitop.compute import compute_map
import torch
from torch.nn.functional import normalize
import cv2
import matplotlib.pyplot as plt
from skimage import feature
from skimage import io as skio
from scipy import spatial
from ipdb import set_trace
import pickle
import pydegensac
import copy
from concurrent import futures
from loftr_tools.match_single_pair import match_single_pair
import gc

from absl import app
from absl import flags


NUM_RERANK = 100
MAX_REPROJECTION_ERROR = 20.0
MAX_RANSAC_ITERATIONS = 1000
HOMOGRAPHY_CONFIDENCE = 1.0
MATCHING_THRESHOLD = 1.0
MAX_DISTANCE = 0.99
USE_RATIO_TEST = False
DRAW_MATCHES = False

_IMAGE_EXTENSION = '.jpg'

FLAGS = flags.FLAGS

dataset_name = 'oxford5k'
flags.DEFINE_string(
    'dataset_file_path', f'/home/mockbuild/Computer_Vision/retrieval_data/gnd_r{dataset_name}.mat',
    'Dataset file for Revisited Oxford or Paris dataset, in .mat format.')
flags.DEFINE_string(
    'images_dir', f'/home/mockbuild/Computer_Vision/{dataset_name}',
    'Directory where dataset images are located, all in .jpg format.')
flags.DEFINE_string(
    'query_global_path', f'output/features/{dataset_name}/query_global.npy',
    'Global features of query images')
flags.DEFINE_string(
    'index_global_path', f'output/features/{dataset_name}/index_global.npy',
    'Global features of index images')
flags.DEFINE_string(
    'query_local_path', f'output/features/{dataset_name}/query_local.pickle',
    'Local features of query images')
flags.DEFINE_string(
    'index_local_path', f'output/features/{dataset_name}/index_local.pickle',
    'Local features of index images')
flags.DEFINE_string(
    'match_vis_directory', f'output/matches/{dataset_name}',
    'Directory to visualize the local matching')
flags.DEFINE_boolean(
    'use_loftr', True,
    'whether use loftr model or not')

## The flags below are valid only when use_loftr is True.
flags.DEFINE_string(
    'loftr_weights_path', f'loftr_tools/weights/outdoor_ds.ckpt',
    'weights of the loftr model')
flags.DEFINE_enum(
    'loftr_rerank_method', 'NumofFeature', ['NumofFeature', 'RANSAC'], 
    'Use the number of features or RANSAC to rerank')
def global_search():
    """ rank by global descriptors """ 
    with open(FLAGS.query_global_path, 'rb') as f:
        query_global_features = np.load(f)
    with open(FLAGS.index_global_path, 'rb') as f:
        index_global_features = np.load(f)
    sim = np.dot(index_global_features, query_global_features.T)
    ranks = np.argsort(-sim, axis=0)
    #set_trace()
    #np.save("ranks_before_gv.npy", ranks)
    return ranks

def rerankGV_mulprocess_loftr(query_list, index_list, ground_truth, ranks_before_gv):
    print('>> mulprocess reranking using loftr...')
    ranks_after_gv = ranks_before_gv
    
    query_index_localfeatures = []
    for query_idx, query_image_name in enumerate(query_list):
        feat_dict = {}
        for k in range(NUM_RERANK):
            index_rank = ranks_before_gv[k, query_idx]
            index_image_name = index_list[index_rank]
        query_index_localfeatures.append((query_idx, feat_dict))
    gc.collect()
    
    ## mulprocess
    # with futures.ProcessPoolExecutor(max_workers=2) as executor:
    #     executor_dict = {executor.submit(localRank_loftr, \
    #         tuple_local_features, query_list, index_list, ground_truth, ranks_before_gv): \
    #         tuple_local_features for tuple_local_features in query_index_localfeatures}

    # for future in futures.as_completed(executor_dict):
    #     query_idx, inliers_numrerank = future.result()
    #     ranks_after_gv[:NUM_RERANK, query_idx] = ranks_before_gv[np.argsort(-1 * inliers_numrerank), query_idx] #-1
    ## single process
    for tuple_local_features in query_index_localfeatures:
        query_idx, inliers_numrerank = localRank_loftr(tuple_local_features, query_list, index_list, ground_truth, ranks_before_gv)
        ranks_after_gv[:NUM_RERANK, query_idx] = ranks_before_gv[np.argsort(-1 * inliers_numrerank), query_idx]
    #set_trace()
    return ranks_after_gv


def rerankGV_mulprocess(query_list, index_list, ground_truth, ranks_before_gv):
    print('>> mulprocess reranking ...')
    ranks_after_gv = ranks_before_gv
    with open(FLAGS.query_local_path, "rb") as f:
        query_local_features = pickle.load(f)
    with open(FLAGS.index_local_path, "rb") as f:
        index_local_features = pickle.load(f)
    
    query_index_localfeatures = []
    for query_idx, query_image_name in enumerate(query_list):
        feat_dict = {}
        feat_dict[query_image_name] = query_local_features[query_image_name]
        for k in range(NUM_RERANK):
            index_rank = ranks_before_gv[k, query_idx]
            index_image_name = index_list[index_rank]
            feat_dict[index_image_name] = index_local_features[index_image_name]
        query_index_localfeatures.append((query_idx, feat_dict))
    
    del query_local_features
    del index_local_features
    gc.collect()
    ## mulprocess
    # with futures.ProcessPoolExecutor(max_workers=24) as executor:
    #     executor_dict = {executor.submit(localRank, \
    #         tuple_local_features, query_list, index_list, ground_truth, ranks_before_gv): \
    #         tuple_local_features for tuple_local_features in query_index_localfeatures}

    # for future in futures.as_completed(executor_dict):
    #     query_idx, inliers_numrerank = future.result()
    #     ranks_after_gv[:NUM_RERANK, query_idx] = ranks_before_gv[np.argsort(-1 * inliers_numrerank), query_idx] #-1
    ## single process
    for tuple_local_features in query_index_localfeatures:
        query_idx, inliers_numrerank = localRank(tuple_local_features, query_list, index_list, ground_truth, ranks_before_gv)
        ranks_after_gv[:NUM_RERANK, query_idx] = ranks_before_gv[np.argsort(-1 * inliers_numrerank), query_idx]
    #set_trace()
    return ranks_after_gv

def localRank_loftr(tuple_local_features, query_list, index_list, ground_truth, ranks_before_gv):
    query_idx, part_local_features = tuple_local_features
    query_image_name = query_list[query_idx]
    print(">> Rerank {} {}".format(query_idx, query_image_name))
    query_im_array = cv2.imread(os.path.join(FLAGS.images_dir, query_image_name + _IMAGE_EXTENSION), cv2.IMREAD_GRAYSCALE)
    numrerank = np.zeros(NUM_RERANK)
    loftr_weight = torch.load(FLAGS.loftr_weights_path)['state_dict']
    for k in range(NUM_RERANK):
        if ranks_before_gv[k, query_idx] in ground_truth[query_idx]['junk']:
            continue
        index_image_name = index_list[ranks_before_gv[k, query_idx]]
        index_im_array = cv2.imread(os.path.join(FLAGS.images_dir, index_image_name + _IMAGE_EXTENSION), cv2.IMREAD_GRAYSCALE)
        
        
        query_im_feature_pos,index_im_feature_pos,_ = match_single_pair(query_im_array, index_im_array, loftr_weight)
        if FLAGS.loftr_rerank_method == 'NumofFeature':
            ## use the number of feature points to rerank
            numrerank[k] = index_im_feature_pos.shape[0]
        else:
            ## use RANSAC to rerank
            try:
                _, mask = pydegensac.findHomography(query_im_feature_pos, index_im_feature_pos,
                                            MAX_REPROJECTION_ERROR,
                                            HOMOGRAPHY_CONFIDENCE,
                                            MAX_RANSAC_ITERATIONS)
            except np.linalg.LinAlgError:
                numrerank[k] = 0
            numrerank[k] = int(copy.deepcopy(mask).astype(np.float32).sum())
        #set_trace()
    torch.cuda.empty_cache()
    return query_idx, numrerank

def localRank(tuple_local_features, query_list, index_list, ground_truth, ranks_before_gv):
    query_idx, part_local_features = tuple_local_features
    query_image_name = query_list[query_idx]
    print(">> Rerank {} {}".format(query_idx, query_image_name))

    query_im_array = skio.imread(os.path.join(FLAGS.images_dir, query_image_name + _IMAGE_EXTENSION))
    query_locations = part_local_features[query_image_name]["locations"]
    query_descriptors = part_local_features[query_image_name]["descriptors"]
    inliers_numrerank = np.zeros(NUM_RERANK)

    for k in range(NUM_RERANK):
        if ranks_before_gv[k, query_idx] in ground_truth[query_idx]['junk']:
            continue
        index_image_name = index_list[ranks_before_gv[k, query_idx]]
        index_im_array = skio.imread(os.path.join(FLAGS.images_dir, index_image_name + _IMAGE_EXTENSION))
        index_locations = part_local_features[index_image_name]["locations"]
        index_descriptors = part_local_features[index_image_name]["descriptors"]
        #set_trace()
        try:
            num_inliers, match_vis_bytes = compute_num_inliers(
                query_locations, query_descriptors, 
                index_locations, index_descriptors, 
                USE_RATIO_TEST, DRAW_MATCHES, 
                query_im_array, index_im_array)
            if DRAW_MATCHES:
                with open(os.path.join(FLAGS.match_vis_directory, \
                    f'{query_image_name}-{index_image_name}.{_IMAGE_EXTENSION}'), "wb") as f:
                    f.write(match_vis_bytes)
            inliers_numrerank[k] = num_inliers
            #set_trace()
        except:
            continue
    
    return query_idx, inliers_numrerank


def compute_putative_matching_keypoints(query_locations, query_descriptors,
        index_locations, index_descriptors,
        use_ratio_test=USE_RATIO_TEST, matching_threshold=float(MATCHING_THRESHOLD),
        max_distance=float(MAX_DISTANCE)):
    """Finds matches from `query_descriptors` to KD-tree of `index_descriptors`."""
    index_descriptor_tree = spatial.cKDTree(index_descriptors)

    if use_ratio_test:
        distances, matches = index_descriptor_tree.query(
            query_descriptors, k=2, n_jobs=-1)
        query_kp_cnt = query_locations.shape[0]
        index_kp_cnt = index_locations.shape[0]
        query_matching_locations = np.array([
            query_locations[i,]
            for i in range(query_kp_cnt)
            if distances[i][0] < matching_threshold*distances[i][1] ])
        index_matching_locations = np.array([
            index_locations[matches[i][0],]
            for i in range(index_kp_cnt)
            if distances[i][0] < matching_threshold*distances[i][1] ])

    else:
        _, matches = index_descriptor_tree.query(
              query_descriptors, distance_upper_bound=max_distance)

        query_kp_cnt = query_locations.shape[0]
        index_kp_cnt = index_locations.shape[0]

        query_matching_locations = np.array([
              query_locations[i,]
              for i in range(query_kp_cnt)
              if matches[i] != index_kp_cnt ])
        index_matching_locations = np.array([
              index_locations[matches[i],]
              for i in range(query_kp_cnt)
              if matches[i] != index_kp_cnt ])
    
    return query_matching_locations, index_matching_locations 


def compute_num_inliers(query_locations, query_descriptors, 
        index_locations, index_descriptors,
        use_ratio_test=False, draw_matches=True,
        query_im_array=None, index_im_array=None):
    """Returns the number of RANSAC inliers."""
    
    query_matching_locations, index_matching_locations = \
        compute_putative_matching_keypoints(query_locations, query_descriptors, 
            index_locations, index_descriptors, use_ratio_test=use_ratio_test)
    
    if query_matching_locations.shape[0] <= 4:  
        # Min keypoints supported by `pydegensac.findHomography()`
        return 0, b''

    try:
        _, mask = pydegensac.findHomography(query_matching_locations, index_matching_locations,
                                            MAX_REPROJECTION_ERROR,
                                            HOMOGRAPHY_CONFIDENCE,
                                            MAX_RANSAC_ITERATIONS)
    except np.linalg.LinAlgError:  # When det(H)=0, can't invert matrix.
        return 0, b''

    inliers = mask if mask is not None else []

    match_viz_bytes = b''
    if isinstance(query_im_array, np.ndarray) and isinstance(index_im_array, np.ndarray) and draw_matches:
        query_im_scale_factors = [1.0, 1.0]
        index_im_scale_factors = [1.0, 1.0]
        inlier_idxs = np.nonzero(inliers)[0]
        _, ax = plt.subplots()
        ax.axis('off')
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        feature.plot_matches(ax,
            query_im_array, index_im_array,
            query_matching_locations * query_im_scale_factors,
            index_matching_locations * index_im_scale_factors,
            np.column_stack((inlier_idxs, inlier_idxs)),
            only_matches=False)

        match_viz_io = io.BytesIO()
        plt.savefig(match_viz_io, format='jpeg', bbox_inches='tight', pad_inches=0)
        match_viz_bytes = match_viz_io.getvalue()
    
    return int(copy.deepcopy(mask).astype(np.float32).sum()), match_viz_bytes



def reportMAP(ground_truth, ranks):
    # evaluate ranks
    ks = [1, 5, 10]

    # search for easy
    gnd_t = []
    for i in range(len(ground_truth)):
        g = {}
        g['ok'] = np.concatenate([ground_truth[i]['easy']])
        g['junk'] = np.concatenate([ground_truth[i]['junk'], ground_truth[i]['hard']])
        gnd_t.append(g)
    mapE, apsE, mprE, prsE = compute_map(ranks, gnd_t, ks)

    # search for easy & hard
    gnd_t = []
    for i in range(len(ground_truth)):
        g = {}
        g['ok'] = np.concatenate([ground_truth[i]['easy'], ground_truth[i]['hard']])
        g['junk'] = np.concatenate([ground_truth[i]['junk']])
        gnd_t.append(g)
    mapM, apsM, mprM, prsM = compute_map(ranks, gnd_t, ks)

    # search for hard
    gnd_t = []
    for i in range(len(ground_truth)):
        g = {}
        g['ok'] = np.concatenate([ground_truth[i]['hard']])
        g['junk'] = np.concatenate([ground_truth[i]['junk'], ground_truth[i]['easy']])
        gnd_t.append(g)
    mapH, apsH, mprH, prsH = compute_map(ranks, gnd_t, ks)

    print('>> mAP E: {}, M: {}, H: {}'.format( 
          np.around(mapE*100, decimals=2), np.around(mapM*100, decimals=2), np.around(mapH*100, decimals=2)))
    print('>> mP@k{} E: {}, M: {}, H: {}'.format(np.array(ks), 
          np.around(mprE*100, decimals=2), np.around(mprM*100, decimals=2), np.around(mprH*100, decimals=2)))


def main(argv):
#   os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # torch.cuda.set_device(1)
    ranks_before_gv = global_search()
    query_list, index_list, ground_truth = ReadDatasetFile(
        FLAGS.dataset_file_path)
    #set_trace()
    print("Before local feature reranking")
    reportMAP(ground_truth, ranks_before_gv)
    print("After local feature reranking")
    if FLAGS.use_loftr == True:
        ranks_after_gv = rerankGV_mulprocess_loftr(query_list, index_list, ground_truth, ranks_before_gv)
    else:
        ranks_after_gv = rerankGV_mulprocess(query_list, index_list, ground_truth, ranks_before_gv)
    reportMAP(ground_truth, ranks_after_gv)


if __name__ == '__main__':
  app.run(main)
