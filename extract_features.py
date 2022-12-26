import os
import numpy as np
from tools.util import ReadDatasetFile
import tools.delg_utils as delg_utils
import torch
from torch.nn.functional import normalize
import cv2
import matplotlib.pyplot as plt
import pickle
from ipdb import set_trace

from absl import app
from absl import flags

from core.config import cfg
import core.config as config

""" common settings """
_MEAN = [0.406, 0.456, 0.485]
_SD = [0.225, 0.224, 0.229]
SCALE_LIST = [0.25, 0.3535, 0.5, 0.7071, 1.0, 1.4142, 2.0]
GLOBAL_SCALE_IND = [3, 4, 5]

IOU_THRES = 0.98
ATTN_THRES = 260.0
TOP_K = 1000

RF = 291.0
STRIDE = 16.0
PADDING = 145.0

_IMAGE_EXTENSION = '.jpg'

FLAGS = flags.FLAGS

dataset_name = 'paris6k'
flags.DEFINE_enum(
    'extract_feature_set', 'global', ['global', 'local'], 
    'Whether to extract global or local features')
flags.DEFINE_enum(
    'model_type', 'DELG', ['DELG', 'LoFTR'], 
    'Model to extract features')
flags.DEFINE_string(
    'weight_path', 'logs/r50_delg_s512.pyth', 
    'Path of the trained model weight.')
flags.DEFINE_string(
    'dataset_file_path', f'/home/chen/cv_proj/retrieval_data/gnd_r{dataset_name}.mat',
    'Dataset file for Revisited Oxford or Paris dataset, in .mat format.')
flags.DEFINE_string(
    'images_dir', f'/home/chen/cv_proj/retrieval_data/{dataset_name}_images',
    'Directory where dataset images are located, all in .jpg format.')
flags.DEFINE_enum(
    'image_set', 'query', ['query', 'index'],
    'Whether to extract features from query or index images.')
flags.DEFINE_string(
    'output_features_dir', f'output/features/{dataset_name}/',
    'Directory where features will be written to.')
flags.DEFINE_string(
    'config_fname', 'resnet_delg_8gpu.yaml',
    'Configuration file.')
flags.DEFINE_string(
    'config_directory', 'configs',
    'Directory containing configuration file.')


def setup_DELG_model():
    model = delg_utils.DelgExtraction()
    load_checkpoint(FLAGS.weight_path, model)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model


def load_checkpoint(checkpoint_file, model, optimizer=None):
    """Loads the checkpoint from the given file."""
    err_str = "Checkpoint '{}' not found"
    assert os.path.exists(checkpoint_file), err_str.format(checkpoint_file)
    # Load the checkpoint on CPU to avoid GPU mem spike
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    try:
        state_dict = checkpoint["model_state"]
    except KeyError:
        state_dict = checkpoint
    # Account for the DDP wrapper in the multi-gpu setting
    ms = model
    model_dict = ms.state_dict()

    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    if len(pretrained_dict) == len(state_dict):
        print('All params loaded')
    else:
        print('construct model total {} keys and pretrin model total {} keys.'.format(len(model_dict), len(state_dict)))
        print('{} pretrain keys load successfully.'.format(len(pretrained_dict)))
        not_loaded_keys = [k for k in state_dict.keys() if k not in pretrained_dict.keys()]
        print(('%s, ' * (len(not_loaded_keys) - 1) + '%s') % tuple(not_loaded_keys))
    model_dict.update(pretrained_dict)
    ms.load_state_dict(model_dict)
    #ms.load_state_dict(checkpoint["model_state"])
    # Load the optimizer state (commonly not done when fine-tuning)
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    #return checkpoint["epoch"]
    return checkpoint

def preprocess(im, scale_factor):
    im = im_scale(im, scale_factor) 
    im = im.transpose([2, 0, 1])
    im = im / 255.0
    im = color_norm(im, _MEAN, _SD)
    return im

def im_scale(im, scale_factor):
    h, w = im.shape[:2]
    h_new = int(round(h * scale_factor))
    w_new = int(round(w * scale_factor))
    im = cv2.resize(im, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
    return im.astype(np.float32)

def color_norm(im, mean, std):
    for i in range(im.shape[0]):
        im[i] = im[i] - mean[i]
        im[i] = im[i] / std[i]
    return im

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# According to models/research/delf/delf/python/delg/r50delg_gld_config.pbtxt
# 1. images scales 0.25, 0.3535, 0.5, 0.7071, 1.0, 1.414, 2.0
# and these scales are the same as extractor.py
# 2. global scale indices 3,4,5
# which means that global features should be scaled by 0.7071, 1.0, 1.414
# Reference: models/research/delf/delf/python/training/model/export_model_utils.py
def extract(im_array, model):
    input_data = torch.from_numpy(im_array)
    if torch.cuda.is_available(): 
        input_data = input_data.cuda() 
    
    global_feature, local_feature, att_score = model(input_data, targets=None)
    # set_trace()
    return global_feature, local_feature, att_score


def global_extract(img, model):
    """ multiscale process """
    # extract features for each scale, concat, and pool, normalization
    scale_cnt = len(GLOBAL_SCALE_IND)
    global_desc_list = []
    for scale_ind in GLOBAL_SCALE_IND:
        scale_factor = SCALE_LIST[scale_ind]
        im = preprocess(img.copy(), scale_factor)
        im_array = np.asarray([im], dtype=np.float32)
        global_feature, _, _ = extract(im_array, model)
        global_desc_list.append(global_feature)
    
    raw_global_descriptors = torch.zeros((scale_cnt, global_desc_list[0].shape[1]))
    for i in range(scale_cnt):
        raw_global_descriptors[i,:] = global_desc_list[i].detach().clone()
    unnormalized_global_descriptor = torch.sum(raw_global_descriptors, dim=0)
    global_descriptor = normalize(unnormalized_global_descriptor, p=2, dim=0)
    return global_descriptor


def local_extract(img, model):
    """ multiscale process """
    # extract features for each scale, and concat.
    output_boxes = []
    output_features = []
    output_scores = []
    output_scales = []
    output_original_scale_attn = None
    for scale_factor in SCALE_LIST:
        im = preprocess(img.copy(), scale_factor)
        im_array = np.asarray([im], dtype=np.float32)
        _, local_feature, att_score = extract(im_array, model)
       
        #tmp = delg_scores.squeeze().view(-1)
        #print(torch.median(tmp))

        selected_boxes, selected_features, \
        selected_scales, selected_scores, \
        selected_original_scale_attn = \
                    delg_utils.GetDelgFeature(local_feature, 
                                        att_score,
                                        scale_factor,
                                        RF,
                                        STRIDE,
                                        PADDING,
                                        ATTN_THRES)

        output_boxes.append(selected_boxes) if selected_boxes is not None else output_boxes
        output_features.append(selected_features) if selected_features is not None else output_features
        output_scales.append(selected_scales) if selected_scales is not None else output_scales
        output_scores.append(selected_scores) if selected_scores is not None else output_scores
        if selected_original_scale_attn is not None:
            output_original_scale_attn = selected_original_scale_attn
    if output_original_scale_attn is None:
        output_original_scale_attn = im.clone().uniform()
    # concat tensors precessed from different scales.
    output_boxes = delg_utils.concat_tensors_in_list(output_boxes, dim=0)
    output_features = delg_utils.concat_tensors_in_list(output_features, dim=0)
    output_scales = delg_utils.concat_tensors_in_list(output_scales, dim=0)
    output_scores = delg_utils.concat_tensors_in_list(output_scores, dim=0)
    # perform Non Max Suppression(NMS) to select top-k bboxes arrcoding to the attn_score.
    keep_indices, count = delg_utils.nms(boxes = output_boxes,
                              scores = output_scores,
                              overlap = IOU_THRES,
                              top_k = TOP_K)
    keep_indices = keep_indices[:TOP_K]

    output_boxes = torch.index_select(output_boxes, dim=0, index=keep_indices)
    output_features = torch.index_select(output_features, dim=0, index=keep_indices)
    output_scales = torch.index_select(output_scales, dim=0, index=keep_indices)
    output_scores = torch.index_select(output_scores, dim=0, index=keep_indices)
    output_locations = delg_utils.CalculateKeypointCenters(output_boxes)
    
    data = {
        'locations':to_numpy(output_locations),
        'descriptors':to_numpy(output_features),
        # 'scores':to_numpy(output_scores)
        # 'attention':to_numpy(output_original_scale_attn)
        }
    return data



def main(argv):
    config.load_cfg(FLAGS.config_directory, FLAGS.config_fname)
    cfg.freeze()

    if FLAGS.model_type == 'DELG': 
        model = setup_DELG_model()
    # Read list of images from dataset file.
    print('Reading list of images from dataset file...')
    # `ground_truth` is a list of {'easy/hard/junk': labels, 'bbx': bounding box coordinates}
    # len(query_list) = 70 for both oxford5k and paris6k
    # len(index_list) ~ 5k/7k

    query_list, index_list, ground_truth = ReadDatasetFile(
        FLAGS.dataset_file_path)
    if FLAGS.image_set == 'query':
        image_list = query_list
    else:
        image_list = index_list
    num_images = len(image_list)
    print("Done! Found %d images" % num_images)

    os.makedirs(FLAGS.output_features_dir, exist_ok=True)

    for i in range(num_images):
        image_name = image_list[i]
        input_image_fname = os.path.join(FLAGS.images_dir, \
            image_name + _IMAGE_EXTENSION)
        
        im = cv2.imread(input_image_fname)
        im = im.astype(np.float32, copy=False)
        if FLAGS.image_set == 'query':
            # Crop query image according to bounding box.
            # plt.imshow(im)
            # plt.show()
            # plt.close()
            bbox = [int(round(b)) for b in ground_truth[i]['bbx']]
            x0, y0, x1, y1 = bbox
            im = im[y0:y1, x0:x1]
            # plt.imshow(im)
            # plt.show()
            # plt.close()

        if FLAGS.extract_feature_set == 'global':
            global_descriptor = global_extract(im, model)
            if i == 0:
                global_features = torch.zeros(num_images, torch.numel(global_descriptor))
            global_features[i,:] = global_descriptor
        
        else:
            local_dict = local_extract(im, model)
            if i == 0:
                local_features = {}
            local_features[image_name] = local_dict

    
    if FLAGS.extract_feature_set == 'global':
        output_feature_fname = f'{FLAGS.image_set}_global.npy'
        with open(os.path.join(FLAGS.output_features_dir, output_feature_fname), 'wb') as f:
            np.save(f, to_numpy(global_features))
    else:
        output_feature_fname = f'{FLAGS.image_set}_local.pickle'
        with open(os.path.join(FLAGS.output_features_dir, output_feature_fname), 'wb') as f:
            pickle.dump(local_features, f, protocol=2)



if __name__ == '__main__':
  app.run(main)
