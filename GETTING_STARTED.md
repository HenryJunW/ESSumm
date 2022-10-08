# Getting Started

### Datasets for ESSumm

Two public corpora datasets: 

AMI Corpus: https://groups.inf.ed.ac.uk/ami/corpus/

ICSI Corpus: https://groups.inf.ed.ac.uk/ami/icsi/

Download the datasets following the instruction from the above websites to folder [data](https://github.com/HenryJunW/ESSumm/tree/main/data). The ground truth summary for both datasets are located within [ESSumm/data/meeting](https://github.com/HenryJunW/ESSumm/tree/main/data/meeting/).

### Segments Generation & Key-segments Extraction

The pipeline we are implementing is based on the [CoreRank](https://github.com/bearblog/CoreRank). Run it with

``` 
python ./data/ESSumm_utterance_community_detection.py
```

### Key-segments Concatenation

Based on the speech summary length constraint, concatenate the key segments together. Then evaluate the quality of the generated summary using ROUGE package.
``` 
python ./ESSumm_wav2vec.py
```