# fancy-nlp
[![Build Status](https://travis-ci.org/boat-group/fancy-nlp.svg?branch=master)](https://travis-ci.org/boat-group/fancy-nlp)
[![Coverage Status](https://coveralls.io/repos/github/boat-group/fancy-nlp/badge.svg?branch=master)](https://coveralls.io/github/boat-group/fancy-nlp?branch=master)
[![Commitizen friendly](https://img.shields.io/badge/commitizen-friendly-brightgreen.svg)](http://commitizen.github.io/cz-cli/)

`fancy-nlp` 是一套易用的自然语言处理工具，其直接面向应用场景，满足用户对自然语言处理任务的需求，使得用户无需处理复杂的预处理等中间过程，直接针对自然语言文本来完成多种NLP任务，实现所想即所得！

## fancy是什么寓意？

对于当前众多的NLP任务，例如序列标注、文本分类，大多数工具的设计都是偏向于模型的训练和评估。当普通用户希望将这些模型应用于实际业务场景中时，往往需要进行复杂的预处理和部署配置，这些过程往往和用户所期望的流程不符。因此`fancy`的寓意为**满足你的想象**，你可以在`fancy-nlp`中实现对NLP任务各个环节的一键式处理，高效将模型应用于实际的业务中。

## 安装

`fancy-nlp`当前支持在Python 3环境下使用：

```
pip install fancy-nlp
pip install git+https://www.github.com/keras-team/keras-contrib.git
```


## 使用指引

### 使用基础模型

当前fancy-nlp中默认加载了使用MSRA数据集训练的NER模型，其能够对中文文本中的组织机构（ORG）、地点（LOC）以及人物（PER）进行识别。当前的基础模型仅为便于用户直接体验，暂未进行深度的模型调用。当前你可以使用后续的**自定义模型**，来构建你的实体提取系统。

*注：我们将在随后不断优化多种场景（不同标注数据）的实体识别模型，以供用户直接使用*

```python
>>> from fancy_nlp.application import NER
# 获取NER实例
>>> ner_app = applications.NER()
# analyze: 输出文本中的实体信息
>>> ner_app.analyze('同济大学位于上海市杨浦区，校长为陈杰')
{'text': '同济大学位于上海市杨浦区，校长为陈杰',
 'entities': [{'name': '同济大学',
   'type': 'ORG',
   'score': 1.0,
   'beginOffset': 0,
   'endOffset': 4},
  {'name': '上海市',
   'type': 'LOC',
   'score': 1.0,
   'beginOffset': 6,
   'endOffset': 9},
  {'name': '杨浦区',
   'type': 'LOC',
   'score': 1.0,
   'beginOffset': 9,
   'endOffset': 12},
  {'name': '陈杰',
   'type': 'PER',
   'score': 1.0,
   'beginOffset': 16,
   'endOffset': 18}]}
# restrict_analyze: 限制输出结果中，每种实体只保留一个实体，取得分最高的实体
>>> ner_app.restrict_analyze('同济大学位于上海市杨浦区，校长为陈杰')
{'text': '同济大学位于上海市杨浦区，校长为陈杰',
 'entities': [{'name': '同济大学',
   'type': 'ORG',
   'score': 1.0,
   'beginOffset': 0,
   'endOffset': 4},
  {'name': '杨浦区',
   'type': 'LOC',
   'score': 1.0,
   'beginOffset': 9,
   'endOffset': 12},
  {'name': '陈杰',
   'type': 'PER',
   'score': 1.0,
   'beginOffset': 16,
   'endOffset': 18}]}
# predict: 查看具体的序列标注结果
>>> ner_app.predict('同济大学位于上海市杨浦区，校长为陈杰')
['B-ORG',
 'I-ORG',
 'I-ORG',
 'I-ORG',
 'O',
 'O',
 'B-LOC',
 'I-LOC',
 'I-LOC',
 'B-LOC',
 'I-LOC',
 'I-LOC',
 'O',
 'O',
 'O',
 'O',
 'B-PER',
 'I-PER']
``` 

### 使用自定义模型
在当前的商品画像构建业务中，我们为海量的商品建立了基础的商品画像信息，包括商品的品牌、类别、型号，已经品牌+类别、品牌+类别+型号所组成的商品SKU。使用`fancy-nlp`可以基于商品名的文本信息，分别使用一行代码，实现对商品品牌、型号等知识实体的提取，以及商品类别的分类

在当前的业务场景中，知识实体的提取准确率F1值可以达到**0.8692**，商品分类准确率可以达到**0.8428**。

```python
>>> from fancy_nlp.application import NER
# 获取NER实例
>>> ner_app = applications.NER()
# 加载你的训练集和验证集
>>> from fancy_nlp.utils import load_ner_data_and_labels
>>> train_data, train_labels = load_ner_data_and_labels('/your/path/to/train.txt')
>>> valid_data, valid_labels = load_ner_data_and_labels('/your/path/to/valid.txt')
# 开始训练模型
>>> ner_app.fit(train_data, train_labels, valid_data, valid_labels,
               ner_model_type='bilstm_cnn',
               char_embed_trainable=True,
               callback_list=['modelcheckpoint', 'earlystopping', 'swa'],
               checkpoint_dir='pretrained_models',
               model_name='dpa_ner_bilstm_cnn_crf',
               load_swa_model=True)
# 使用测试集评估模型效果
>>> test_data, test_labels = load_ner_data_and_labels('./your/path/to/test.txt')
>>> ner_app.score(test_data, test_labels)
Recall: 0.8922289546443909, Precision: 0.8474131187842217, F1: 0.8692437745364932
...
>>> ner_app.restrict_analyze('小米9SE骁龙712全息幻彩紫8GB+128GB游戏智能拍照手机')
{'text': '小米9SE骁龙712全息幻彩紫8GB+128GB游戏智能拍照手机',
 'entities': [{'name': '小米',
   'type': '品牌',
   'score': 0.9986118674278259,
   'beginOffset': 0,
   'endOffset': 2},
  {'name': '骁龙712',
   'type': '型号',
   'score': 0.9821863174438477,
   'beginOffset': 5,
   'endOffset': 10},
  {'name': '手机',
   'type': '类别',
   'score': 0.9981447458267212,
   'beginOffset': 30,
   'endOffset': 32}]}
>>> ner_app.analyze('小米9SE骁龙712全息幻彩紫8GB+128GB游戏智能拍照手机')
{'text': '小米9SE骁龙712全息幻彩紫8GB+128GB游戏智能拍照手机',
 'entities': [{'name': '小米',
   'type': '品牌',
   'score': 0.9986118674278259,
   'beginOffset': 0,
   'endOffset': 2},
  {'name': '9SE',
   'type': '型号',
   'score': 0.8843186497688293,
   'beginOffset': 2,
   'endOffset': 5},
  {'name': '骁龙712',
   'type': '型号',
   'score': 0.9821863174438477,
   'beginOffset': 5,
   'endOffset': 10},
  {'name': '手机',
   'type': '类别',
   'score': 0.9981447458267212,
   'beginOffset': 30,
   'endOffset': 32}]}

``` 


## 模型架构
### 知识实体识别
对于知识实体识别，我们使用字向量序列作为基础输入，并在此基础上：

- 加入分词特征，包括字所在词的词向量与位置向量；
- 加入邻接字特征，如bi-gram字向量，然后使用BiLSTM+CNN+CRF模型进行序列标注。

![知识实体识别模型架构](./img/entity_extract.png)

与基于词序列输入和基于字序列输入的模型相比，本实体识别方法可以显式利用句子中词的语义信息，同时还不会受分词错误的影响。

### 知识实体链接（待融合至代码库中）
对于实体链接模型，我们先使用基于注意力机制的Bi-LSTM模型抽取实体指称的语义特征，同时融合多种消歧特征：

- 文本相似度特征；
- 实体类型匹配特征；
- 基于文本中所有候选实体所构成的知识子图的多实体联合消歧特征，之后使用排序学习方法对候选实体列表进行排序，得到最佳匹配实体。

![实体链接模型架构](./img/entity_linking.png)

与传统方法相比，该链接模型融合多种消歧特征，能有效解决短文本上下文语境不丰富问题，提高泛化能力。

模型在多种特征组合场景的效果对比如下，数据集来自于[**CCKS 2019——中文短文本的实体链指**](https://biendata.com/competition/ccks_2019_el/)：

| batch | embed    | trainable | schema | encoder_type | crf |val_f1 |
|:-----:|:--------:|:---------:|:------:|:------------:|:---:|------:|
| 32    | c2v      | fix       | BIOES  | bilstm       |False|0.7399 |
| 32    | c2v      | fix       | BIOES  | bilstm       | True|0.7559 |
| 32    | c2v      | fix       | BIOES  | bigru        |False|0.7426 |
| 32    | c2v      | fix       | BIOES  | bigru        | True|0.7585 |
| 32    | c2v      | fix       | BIOES  | bilstm_cnn   |False|0.7593 |
| 32    | c2v      | fix       | BIOES  | bilstm_cnn   | True|0.7673 |
| 32    | c2v      | fix       | BIOES  | bigru_cnn    |False|0.7564 |
| 32    | c2v      | fix       | BIOES  | bigru_cnn    | True|0.7685 |


### 文本分类模型（待融合至代码库）
对于文本分类模型，我们集成了当前常用的文本分类模型，并进行了对比试验，效果如下：

| 序号 |    模型名   | Precision |  Recal | Macro-F1 | Accuracy | Time(s)/60015个样本 |
|:----:|:-----------:|:---------:|:------:|:--------:|:--------:|:-------------------:|
|   1  |     CNN     |   0.7532  | 0.7453 |  0.7462  |  0.8636  |        4.7972       |
|   2  |     LSTM    |   0.7446  | 0.7339 |  0.7389  |  0.8599  |        4.8093       |
|   3  |   Bi-LSTM   |   0.7436  | 0.7374 |  0.7384  |  0.8594  |        8.7405       |
|   4  |    DPCNN    |   0.7411  | 0.7326 |  0.7333  |  0.8462  |        5.6679       |
|   5  |     RCNN    |   0.7687  | 0.7633 |  0.7639  |  0.8744  |       11.1846       |
|   6  |     DCNN    |   0.6873  | 0.6743 |  0.6746  |  0.7955  |        5.2111       |
|   7  |    VDCNN    |   0.7119  | 0.6981 |  0.7049  |  0.8065  |       53.2978       |
|   8  | Att_Bi-LSTM |   0.7512  | 0.7448 |  0.7549  |  0.8644  |        9.0593       |
|   9  | CNN-Bi-LSTM |   0.7486  | 0.7411 |  0.7422  |  0.8563  |        3.4688       |
|  10  |   FastText  |   0.7313  | 0.7270 |  0.7274  |  0.8400  |        1.1811       |

效果最优的是RCNN模型，这也是我们当前实际采用的模型。若从模型速度和性能兼顾的角度来考虑，可以采用CNN或FastText模型。

## Acknowledgement

- 本项目在**2019腾讯广告犀牛鸟专项研究计划**的支持下，由AMS-PTP-创意优化组allene与同济大学大数据处理与智能分析实验室联合开发
- `fancy-nlp`在**CCKS 2019——中文短文本的实体链指**评测竞赛中取得了初赛第三名，复赛[第五名](https://biendata.com/competition/ccks_2019_el/final-leaderboard/)的成绩，且获得了该评测竞赛唯一的一项[技术创新奖](https://biendata.com/competition/ccks_2019_el/winners/)，原始可复现流程，请参考原始[repo](https://github.com/AlexYangLi/ccks2019_el)。

## Contribution

- 项目的代码规范符合PEP8标准，且每次提交会自动触发CI，并计算单测覆盖率
- 所有的代码提交请遵循[约定式提交规范](https://www.conventionalcommits.org/zh/v1.0.0-beta.4/)
- 所有的功能性代码请编写相应的单测模块
