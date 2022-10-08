## Installation

### Requirements
- Linux with Python >= 3.8
- PyTorch >= 1.10.0
- fairseq: follow [fairseq installation instructions](https://github.com/facebookresearch/fairseq#requirements-and-installation). For the wav2vec 2.0 speech-feature extraction, download the pre-trained wav2vec 2.0 model located in the fairseq repo, https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/README.md, for example, 'xlsr_53_56k.pt'.

### Example conda environment setup
```bash
conda create -n essumm python=3.8 -y
conda activate essumm
git clone https://github.com/HenryJunW/ESSumm
cd ESSumm
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
```