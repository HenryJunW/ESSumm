import numpy as np 
import os
from data import utils
import baseline.utils
import numpy as np
from data.meeting import meeting_lists
import sys
sys.setrecursionlimit(873 * 350 + 10)
import csv

def dict_mean(dict_list):
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict


rouge_1_list, rouge_2_list, rouge_l_list = [], [], []
path_to_root = '/vulcanscratch/junwang/ESSumm/'
os.chdir(path_to_root)
domain     = 'meeting' # meeting
# dataset_id = 'ami'     # ami, icsi
dataset_id = 'icsi'     # ami, icsi
language   = 'en'      # en
# source     = 'asr'     # asr, manual
source     = 'manual'     # asr, manual

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



for id in ids:
    print (id)
    if domain == 'meeting':
        path_to_community = path_to_root + 'data/community/meeting/' + dataset_id + '/' + id + '_' + '/'
    # define wav2vect pretrained model
    wav2vec2_checkpoint_path = 'xlsr_53_56k_updated.pt'
    # define number of clusters in the k-means, and number of segments to use in the final summary
    n_clusters = 32
    meeting_wav2vec_feature_folder = path_to_community + str(n_clusters) + '_' + wav2vec2_checkpoint_path

# read the wav2vect output seg ID
    with open(meeting_wav2vec_feature_folder + '/tf_idf_pca4_index_speech_segment_segment_.txt') as f:
        content = f.readlines()
    wav2vec_seg_id_index1 = np.asarray([x.strip() for x in content], dtype=int)
    wav2vec_seg_id = [x-1 for x in wav2vec_seg_id_index1]


    #  take the max 350 words after stripped stopwords
    max_words_budget = 350
    words_total, i, segment_list_output, summary_tmp = 0, 0, [], []
    while words_total <= max_words_budget and i < len(wav2vec_seg_id):
        words_wav_cur = corpus[id][wav2vec_seg_id[i]][4].split()
        words_wav_cur_strip = ' '.join(baseline.utils.strip_stopwords(words_wav_cur, stopwords))
        words_total += len(words_wav_cur_strip.split())
        summary_tmp.extend(words_wav_cur_strip.split()[:max_words_budget - len(summary_tmp)])
        segment_list_output.append(wav2vec_seg_id_index1[i])
        i = i + 1
    segment_list_output.sort()
    wav2vec_seg_id = [x - 1 for x in segment_list_output]
    summary_generated = []
    for i in wav2vec_seg_id:
        words_wav_cur_strip = ' '.join(baseline.utils.strip_stopwords(corpus[id][i][4].split(), stopwords))
        summary_generated.append(words_wav_cur_strip)
    # control the specific number of words instead of the seg number
    generated_summary_text_paragraph = ' '.join(summary_tmp)
    # write the generated summary
    path_to_to = path_to_root + 'ROUGE-2.0-1.2.1/distribute/test-summarization/system/'
    o = open(path_to_to + id + '_system.txt', 'w')
    o.write(generated_summary_text_paragraph)
    o.close()

print('Done!')