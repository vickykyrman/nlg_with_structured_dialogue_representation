# Exploring the Impact of Structured Dialogue Representation on Dialogue Response Generation

## Introduction

**Goal** <br>
Our primary goal is to explore the impact of structured dialogue representation on the dialogue response generation task. To achieve our main goal, we comparatively investigate the effect of different qualitative and quantitative scenarios for representing dialogue history on the predicted dialogue responses. 

**Methodology** <br>
- We generate dialogue responses by employing GODEL [(Peng et al., 2022)](#godel2022), a goal-directed transformer-based language model. (for more information on Godel, please visit the model's github page [here](https://github.com/microsoft/GODEL).) <br>
- We finetune Godel on the OpenDialKg dataset [(Moon et al., 2019)](#opendialkg), composed by 15K human crowd-sourced goal-driven dialogues on the topics of sports, music, books and music, where each turn is annotated with factual knowledge triples. <br>
- We produce 11 models corresponding to the 11 quantitative and qualitative scenarios for representing the dialogue history of turn. [image](./doc <br>
- The response quality of each model is evaluated on the basis of 4 automatic metrics, namely ROUGE, BLUE, METEOR and METRIC. A manual evaluation is conducted overhead inspired by the principles of the Gricean Maxims. Finally, given that the graphical representation of the dialogue history encapsulates solely factual information, additional perspectival information is extracted from the dataset and represented graphically. The enhanced graphs are then used to train a 12th model, in order to investigate the impact of a more holistic and informative graphical structural representation on model performance, and specifically on its ability to express perspectival information.



## References
@misc{peng2022godel,
author = {Peng, Baolin and Galley, Michel and He, Pengcheng and Brockett, Chris and Liden, Lars and Nouri, Elnaz and Yu, Zhou and Dolan, Bill and Gao, Jianfeng},
title = {GODEL: Large-Scale Pre-training for Goal-Directed Dialog},
howpublished = {arXiv},
year = {2022},
month = {June},
url = {https://www.microsoft.com/en-us/research/publication/godel-large-scale-pre-training-for-goal-directed-dialog/},
}
<a id="godel2022"></a>
[@peng2022godel]: Peng, B., Galley, M., He, P., Brockett, C., Liden, L., Nouri, E., Yu, Z., Dolan, B., Gao, J. (2022). *GODEL: Large-Scale Pre-training for Goal-Directed Dialog*. [arXiv](https://www.microsoft.com/en-us/research/publication/godel-large-scale-pre-training-for-goal-directed-dialog/).

@InProceedings{Moon2019opendialkg,
author = {Seungwhan Moon and Pararth Shah and Anuj Kumar and Rajen Subba},
title = {OpenDialKG: Explainable Conversational Reasoning with Attention-based Walks over Knowledge Graphs},
booktitle = {Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
month = {July},
year = {2019},
}
<a id="opendialkg"></a>
[@moon2019opendialkg]: Moon, S., Shah, P., Kumar, A., Subba, R. (2019). *OpenDialKG: Explainable Conversational Reasoning with Attention-based Walks over Knowledge Graphs*. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*. [Link](https://aclanthology.org/P19-1081.pdf)
