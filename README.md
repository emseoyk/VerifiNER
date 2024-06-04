# VerifiNER
Official code for ACL 2024 paper: [VerifiNER: Verification-augmented NER via Knowledge-grounded Reasoning with Large Language Models](https://arxiv.org/abs/2402.18374).

## Overview
![](https://github.com/user-attachments/files/15546047/main_figure.pdf)

This is the overview of VerifiNER framework.
1. Using entity prediction by existing models, we (a) extract candidate spans to retrieve knowledge from KB and verify factuality of span accordingly. Then (b) using retrieved knowledge, we verify factuality of type by generating knowledge-grounded evidence.
2. Lastly, we take consistency voting to select a candidate that is the most contextually relevant, with the help of the reasoning ability of LLMs.

Please refer to the paper for details.

## Usage
### Factuality Verification

### Contextual Relevance Verification



## Citation
If you find our work useful, please cite our paper:
```
@misc{kim2024verifiner,
      title={VerifiNER: Verification-augmented NER via Knowledge-grounded Reasoning with Large Language Models}, 
      author={Seoyeon Kim and Kwangwook Seo and Hyungjoo Chae and Jinyoung Yeo and Dongha Lee},
      year={2024},
      eprint={2402.18374},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
