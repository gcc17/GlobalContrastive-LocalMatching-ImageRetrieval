# -*- coding: utf-8 -*-
"""utility functions"""
import sys, os
import struct
import numpy as np
import pickle
from scipy.io import matlab

_GROUND_TRUTH_KEYS = ['easy', 'hard', 'junk']
def ReadDatasetFile(dataset_file_path):
  """Reads dataset file in Revisited Oxford/Paris ".mat" format.

  Args:
    dataset_file_path: Path to dataset file, in .mat format.

  Returns:
    query_list: List of query image names.
    index_list: List of index image names.
    ground_truth: List containing ground-truth information for dataset. Each
      entry is a dict corresponding to the ground-truth information for a query.
      The dict may have keys 'easy', 'hard', or 'junk', mapping to a NumPy
      array of integers; additionally, it has a key 'bbx' mapping to a NumPy
      array of floats with bounding box coordinates.
  """
  cfg = matlab.loadmat(dataset_file_path)

  # Parse outputs according to the specificities of the dataset file.
  query_list = [str(im_array[0]) for im_array in np.squeeze(cfg['qimlist'])]
  index_list = [str(im_array[0]) for im_array in np.squeeze(cfg['imlist'])]
  ground_truth_raw = np.squeeze(cfg['gnd'])
  ground_truth = []
  for query_ground_truth_raw in ground_truth_raw:
    query_ground_truth = {}
    for ground_truth_key in _GROUND_TRUTH_KEYS:
      if ground_truth_key in query_ground_truth_raw.dtype.names:
        adjusted_labels = query_ground_truth_raw[ground_truth_key] - 1
        query_ground_truth[ground_truth_key] = adjusted_labels.flatten()

    query_ground_truth['bbx'] = np.squeeze(query_ground_truth_raw['bbx'])
    ground_truth.append(query_ground_truth)

  return query_list, index_list, ground_truth

def readkv(f):
    """read kv from file handle"""
    keylendata = f.read(4)
    if len(keylendata) != 4:
        print("wrongkey", file=sys.stderr)
        return None
    keylen = struct.unpack('I', keylendata)[0]
    if keylen > 5000:
        raise Exception('wrong key len %s' % (keylen))
    key = f.read(keylen)
    valuelen = struct.unpack('I', f.read(4))[0]
    if valuelen > 1000000000:
        raise Exception('wrong value len %s' % (valuelen))
    value = f.read(valuelen)
    return key, value


def writekv(f, k, v, flush=True):
    """writekv to file handle"""
    f.write(struct.pack('I', len(k)))
    f.write(k)
    f.write(struct.pack('I', len(v)))
    f.write(v)
    if flush:
        f.flush()


def loopkv(fin):
    """iter the kv file """
    while True:
        r = readkv(fin)
        if r is None:
            break
        key, value = r
        yield key, value


def getasciikey(key):
    #return key
    """convert 8bytes key to 'a,b' string contsign"""
    if len(key) == 8:
        key = struct.unpack('II', key)
        key = '%s,%s' % (key[0], key[1])
    return key


def l2_norm(fea):
    """l2 norm feature get cos dis"""
    if fea is None:
        return fea
    if len(fea.shape) > 1:
        fea = fea / np.linalg.norm(fea, axis=1)[:, None]
    else:
        fea = fea / np.linalg.norm(fea)
    return fea


def norm(x):
    n = np.sqrt(np.sum(x ** 2,1)).reshape(-1,1)
    return x / (n + 0.000001)


def walkfile(spath):
    """get files in input spath """
    files = os.listdir(spath)
    for file in files:
        tmppath = os.path.join(spath, file)
        if not os.path.isdir(tmppath):
            yield tmppath
        else:
            for lowfile in walkfile(tmppath):
                yield lowfile


def conmbineFeaPickle(COMBINE_DIR, OUTFILE):
    all_dict = {}
    files = os.listdir(COMBINE_DIR)
    for file in files:
        tmppath = os.path.join(COMBINE_DIR, file)
        print(tmppath)
        with open(tmppath,'rb') as fin:
            tmpres = pickle.load(fin)
            all_dict.update(tmpres)
    print(len(all_dict.keys()))
    with open(OUTFILE,'wb') as fout:
        pickle.dump(all_dict, fout, protocol=2)    
   

def visualize(path):
    searchfile = os.path.join(path, "searchresults.pickle")
    outhtml = os.path.join(path, "searchresults.html")
    with open(searchfile, "rb") as f:
        resdic = pickle.load(f)
    serverpath = 'http://127.0.0.0:8901/'
    print(serverpath)

    html_file = open(outhtml, "w")
    html_file.write('<html><meta charset=\"utf-8\"><body>\n')
    html_file.write('<p>\n')
    html_file.write('<table border="1">\n')

    for i, (k,v) in enumerate(sorted(resdic.items())):
        html_file.write('<td><img src="%s" width="150" height="150" /></td>' % \
                                (serverpath + "query/" + k))
        html_file.write("<td> %s %s</td>\n" % (i, k))

        recall = v[:10] # top10
        if len(recall) == 0:
            html_file.write('</tr>\n')
            continue
        for i in range(len(recall)):
            referhtml = serverpath + 'gallery/' + recall[i][0]
            html_file.write('<td><img src="%s" width="150" height="150" \
                /><br/><color="green">score: %s<br/> </td>' % (referhtml, str(recall[i][1])))
        html_file.write('</tr>\n')

    html_file.write("</table>\n</p>\n</body>\n</html>")
    print(path, "draw html finished ")



if __name__ == "__main__":
    if len(sys.argv)>1 :
        func = getattr(sys.modules[__name__], sys.argv[1])
        func(*sys.argv[2:])
    else:
        print('tools.py command', file=sys.stderr)
