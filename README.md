# ESSumm: Extractive Speech Summarization from Untranscribed Meeting

The code base for [ESSumm: Extractive Speech Summarization from Untranscribed Meeting](https://www.isca-speech.org/archive/pdfs/interspeech_2022/wang22n_interspeech.pdf)
<br>Jun Wang

**NEWS**
- [22-06-15] 🔥 ESSumm is accepted at INTERSPEECH 2022.

## Abstract
<div style="text-align: justify">In this paper, we propose a novel architecture for direct extractive speech-to-speech summarization, ESSumm, which is an unsupervised model without dependence on intermediate transcribed text. Different from previous methods with text presentation, we are aimed at generating a summary directly from speech without transcription. First, a set of smaller speech segments are extracted based on speech signal's acoustic features. For each candidate speech segment, a distance-based summarization confidence score is designed for latent speech representation measure. Specifically, we leverage the off-the-shelf self-supervised convolutional neural network to extract the deep speech features from raw audio. Our approach automatically predicts the optimal sequence of speech segments that capture the key information with a target summary length. Extensive results on two well-known meeting datasets (AMI and ICSI corpora) show the effectiveness of our direct speech-based method to improve the summarization quality with untranscribed data. We also observe that our unsupervised speech-based method even performs on par with recent transcript-based summarization approaches, where extra speech recognition is required. </div>


<br>![network](https://github.com/HenryJunW/ESSumm/blob/main/figs/ESSumm_overview_figure.pdf)

### Features
* The first automatic speech summarization system with Wav2vec 2.0.
* Support major Meeting datasets: AMI and ICSI.

## Installation

See [installation instructions](INSTALL.md).

## Getting Started

See [Getting Started with TAG](GETTING_STARTED.md).



## Citation
Please cite our work if you found it useful,

```
@inproceedings{wang22n_interspeech,
  author={Jun Wang},
  title={{ESSumm: Extractive Speech Summarization from Untranscribed Meeting}},
  year=2022,
  booktitle={Proc. Interspeech 2022},
  pages={3243--3247},
  doi={10.21437/Interspeech.2022-945}
}

```

# License

This project is released under the [Apache 2.0 license](LICENSE).

# Acknowledgement

The source code of ESSumm is based on [CoreRank](https://github.com/bearblog/CoreRank) and [wav2vec 2.0](https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/README.md). 
