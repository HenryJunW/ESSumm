"""
Utterance Community Detection (UCD)

input (automatic or manual meeting transcription):
data/meeting/ami/ES2004a.da-asr or ES2004a.da

output (utterance communities per meeting):
data/community/meeting/ami_[UCD parameter id]/ES2004a_comms.txt

output (POS tagged utterance communities per meeting):
data/community_tagged/meeting/ami_[UCD parameter id]/ES2004a_comms_tagged.txt

output (preprocessed meeting transcription):
data/utterance/meeting/ami_[UCD parameter id]/ES2004a_utterances.txt

output (grid search csv):
data/ami_params_create_community.csv'
"""
import os

import numpy as np

path_to_root = '/vulcanscratch/junwang/ESSumm/'
datapath_to_root = '/vulcanscratch/junwang/ESSumm/data/'
os.chdir(path_to_root)
import string
import core_rank
from data import utils
from data import clustering
from data.meeting import meeting_lists
from collections import Counter
from nltk import PerceptronTagger
from nltk import TweetTokenizer
from dictionary_tokenizer import DictionaryTokenizer
from sklearn.model_selection import ParameterGrid
import subprocess
import shlex
import torch, librosa
from fairseq.models.wav2vec import Wav2VecModel
import fairseq
import glob
from sklearn.cluster import MiniBatchKMeans
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import math
domain     = 'meeting' # meeting
dataset_id = 'ami'     # ami, icsi
# dataset_id = 'icsi'     # ami, icsi
language   = 'en'      # en
source     = 'asr'     # asr, manual
# source     = 'manual'     # asr, manual


# TF vector for each speech segments
def compute_tf(word_dict, l):
    tf = {}
    sum_nk = len(l)
    for word, count in word_dict.items():
        tf[word] = count / float(sum_nk)
    return tf

# define idf
def compute_idf(string_list):
    n = len(string_list)
    idf = dict.fromkeys(string_list[0].keys(), 0)
    for l in string_list:
        for word, count in l.items():
            if count > 0:
                idf[word] += 1

    for word, v in idf.items():
        idf[word] = math.log(n / (float(v) + 1))
    return idf

# define tf_idf 
def compute_tf_idf(tf, idf):
    tf_idf = dict.fromkeys(tf.keys(), 0)
    for word, v in tf.items():
        tf_idf[word] = v * idf[word]
    return tf_idf

# #########################
# ### RESOURCES LOADING ###
# #########################
if domain == 'meeting':
    path_to_stopwords    = path_to_root + 'resources/stopwords/meeting/stopwords.' + language + '.dat'
    path_to_filler_words = path_to_root + 'resources/stopwords/meeting/filler_words.' + language + '.txt'
    stopwords = utils.load_stopwords(path_to_stopwords)
    filler_words = utils.load_filler_words(path_to_filler_words)

    if dataset_id == 'ami':
        # ids = meeting_lists.ami_development_set + meeting_lists.ami_test_set
        ids = meeting_lists.ami_test_set
    elif dataset_id == 'icsi':
        # ids = meeting_lists.icsi_development_set + meeting_lists.icsi_test_set
        ids = meeting_lists.icsi_test_set

if language == 'en':
    path_to_word2vec_keys = path_to_root + 'resources/word2vec_keys.txt'
# tokenizer = DictionaryTokenizer(path_to_word2vec_keys) # highly time-consuming
# tokenizer = TweetTokenizer()
tagger = PerceptronTagger()

# ######################
# ### CORPUS LOADING ###
# ######################
corpus = {}
for id in ids:
    if domain == 'meeting':
        if dataset_id == 'ami' or dataset_id == 'icsi':
            if source == 'asr':
                path = path_to_root + 'data/meeting/' + dataset_id + '/' + id + '.da-asr'
            elif source == 'manual':
                path = path_to_root + 'data/meeting/' + dataset_id + '/' + id + '.da'
            # filler words will be removed during corpus loading
            corpus[id] = utils.read_ami_icsi(path, filler_words)

#################################################################################
# stage 1 : based on ASR time stamp, sox segments
for id in ids:
    print(id)

    if dataset_id == 'ami':
        path_wav = datapath_to_root + 'amicorpus/' + id + '/audio/' + id + '.Mix-Headset.wav'
    else:
        path_wav = datapath_to_root + 'icsi/' + id + '/' + id + '.interaction.wav'

    utterances_indexed = corpus[id]
    # start_time, end_time = [i[2] for i in utterances_indexed], [i[3] for i in utterances_indexed]
    # start_time, end_time = [i[2] for i in utterances_indexed], [i[3] for i in utterances_indexed]
    start_time, end_time = [], []
    for i in utterances_indexed:
        if i[2] != i[3]:
            start_time.append(i[2])
            end_time.append(i[3])

    # output community
    if domain == 'meeting':
        path_to_community = path_to_root + 'data/community/meeting/' + dataset_id + '/' + id + '_' + '/'
    if not os.path.exists(path_to_community):
        os.makedirs(path_to_community)
    # with open(path_to_community + id + '_comms.txt', 'w+') as txtfile:
    to_write = [(start_time[i], end_time[i]) for i in range(len(start_time))]
    time_stamp_path_to_write = path_to_community + id + '_comms.txt'
    np.savetxt(time_stamp_path_to_write, to_write)

    path_to_write = path_to_community + 'seg/'
    if not os.path.exists(path_to_write):
        os.makedirs(path_to_write)
    for j_wav in range(len(start_time)):
        path_to_write_file = path_to_write + 'seg_' + str(j_wav) + '.wav'
        duration_cur = end_time[j_wav] - start_time[j_wav]
        subprocess.call(f"/vulcanscratch/junwang/dependency/anaconda3/envs/fair/bin/sox {path_wav} {path_to_write_file} trim {start_time[j_wav]} {duration_cur}", shell=True)

################################################################################
  # stage 2 : Run wav2vec feature extraction
wav2vec2_checkpoint_path  = '/vulcanscratch/junwang/ESSumm/fairseq/xlsr_53_56k.pt'
checkpoint = torch.load(wav2vec2_checkpoint_path)
# https://github.com/pytorch/fairseq/issues/3181#issuecomment-771443998
model_base, cfg = fairseq.checkpoint_utils.load_model_ensemble([wav2vec2_checkpoint_path])
if type(model_base) is list:
    model_base = model_base[0]
model_base.eval()
# for xlsr_53_56k
wav2vec_encoder = model_base
for id in ids:
    print (id)
    txtfiles = []
    # output community
    if domain == 'meeting':
        path_to_community = path_to_root + 'data/community/meeting/' + dataset_id + '/' + id + '_' + '/'

    for file in glob.glob(path_to_community + 'seg/seg_*.wav'):
        txtfiles.append(file)
    textfiles_sorted = sorted(txtfiles, key=os.path.getmtime)
    features = []
    end_index = [0]

    # define wav2vect pretrained model
    wav2vec2_checkpoint_path = wav2vec2_checkpoint_path.split('/')[-1]
    # define number of clusters in the k-means, and number of segments to use in the final summary
    n_clusters = 32

    meeting_wav2vec_feature_folder = path_to_community + str(32) + '_' + wav2vec2_checkpoint_path
    if not os.path.exists(meeting_wav2vec_feature_folder):
        os.makedirs(meeting_wav2vec_feature_folder)
    meeting_wav2vec_feature_file = meeting_wav2vec_feature_folder + '/meeting_wav2vec_feature.npy'
    meeting_wav2vec_feature_index_file = meeting_wav2vec_feature_folder + '/meeting_wav2vec_feature_index.npy'
    # np.save(meeting_wav2vec_feature_file, features_aggregated_np)
    # np.save(meeting_wav2vec_feature_index_file, end_index)
# #

# ################################################################################
#  k-means:
    features_aggregated_np = np.load(meeting_wav2vec_feature_file)
    X = np.squeeze(features_aggregated_np)
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0).fit(X)

    y = kmeans.labels_
    feat_cols = ['pixel'+str(i) for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feat_cols)
    df['y'] = y
    df['label'] = df['y'].apply(lambda i: str(i))
    X, y = None, None
    print('Size of the dataframe: {}'.format(df.shape))

    # meeting_wav2vec_feature_index = end_index
    meeting_wav2vec_feature_index = np.load(meeting_wav2vec_feature_index_file)
    y = kmeans.labels_
    word_set = set(range(n_clusters))
    tf_meeting = []
    word_dict = []
    tf_output = []
    for i in range(len(meeting_wav2vec_feature_index[1:])):
        pre, cur = meeting_wav2vec_feature_index[i], meeting_wav2vec_feature_index[i + 1]
        cluster_IDs_speech_seg_cur = y[pre:cur]
        word_dict_cur = dict.fromkeys(word_set, 0)
        for word in cluster_IDs_speech_seg_cur:
            word_dict_cur[word] += 1
        tf_cur = compute_tf(word_dict_cur, cluster_IDs_speech_seg_cur)
        word_dict.append(word_dict_cur)
        tf_output.append(tf_cur)
        # if tf_cur is a dictionary, we only need the values
        tf_meeting.append(list(tf_cur.values()))

    idf = compute_idf(word_dict)

    if_idf_output = []
    if_idf_output_values = []
    for i in range(len(meeting_wav2vec_feature_index[1:])):
        if_idf_output_cur = compute_tf_idf(tf_output[i], idf)
        if_idf_output.append(if_idf_output_cur)
        if_idf_output_values.append(list(if_idf_output_cur.values()))

    # using tf idf
    tf_meeting_stack = np.stack(if_idf_output_values, 0)
    # using tf
    # tf_meeting_stack = np.stack(tf_meeting, 0)

    # for reproducability of the results
    np.random.seed(42)
    rndperm = np.random.permutation(df.shape[0])

    pca = PCA(n_components=4)
    pca_result = pca.fit_transform(tf_meeting_stack)
    #  the return parameters 'components_' is eigen vectors and 'explained_variance_' is eigen values
    print('Explained variation per principal componets: {}'.format(pca.explained_variance_ratio_))


    # calculate the eigenvector distances
    pca_principal_component = pca.components_[0]
    pca_principal_component_second = pca.components_[1]
    eculidean_distance_eigenvector = []
    for i in tf_meeting_stack:
        # calculate first two principal components, and weighted distance based on the eigenvectors ratio
        eculidean_distance_eigenvector_first = np.linalg.norm(pca_principal_component - i)
        eculidean_distance_eigenvector_second = np.linalg.norm(pca_principal_component_second - i)
        eculidean_distance_eigenvector_total = eculidean_distance_eigenvector_first * pca.explained_variance_ratio_[0] + \
                                                eculidean_distance_eigenvector_second * pca.explained_variance_ratio_[1]
        eculidean_distance_eigenvector.append(eculidean_distance_eigenvector_total)
    # sort the average distance and save the index, the distance smaller, the better
    index_speech_segment_sorted_index = np.argsort(eculidean_distance_eigenvector)
    # index is 0, so add 1 for each element, and then sort due to chrononical order, and only keep the first some elements
    number_segment_to_use = len(textfiles_sorted)
    index_speech_segment_sorted_index_transform = [i + 1 for i in index_speech_segment_sorted_index[:number_segment_to_use]]

    np.savetxt(os.path.join(meeting_wav2vec_feature_folder, 'tf_idf_pca4_index_speech_segment_segment_' + '.txt'), index_speech_segment_sorted_index_transform, fmt = '%i')
    np.savetxt(os.path.join(meeting_wav2vec_feature_folder, 'tf_idf_pca4_index_speech_segment_segment_' + '_bk.txt'), index_speech_segment_sorted_index_transform, fmt = '%i')