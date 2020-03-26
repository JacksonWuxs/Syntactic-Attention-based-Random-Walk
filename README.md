# Syntactic-Attention-based-Random-Walk
 #### 摘要 Abstract

Arora 等人（2016）提出一个基于文本生成的随机游走模型，其以卓越的性能成为了句表征技术的新基线。本文指出通过在随机游走模型中添加一个简化的语法注意力机制可以进一步加强模型的语义表达能力。具体来说，本文提出了一个高性能、易于实现且高解释性的句子表征方法。同时，本文证明了句向量的极大似然估计等于动态修正词向量的加权平均。该改进模型在五个不同的语义文本相似度（Semantic Textual Similarity, STS）任务中比之前的模型平均提高了大约7.1% 的性能。同时，敏感性分析指出该模型在进行长句子表征时也相当
稳定。与Arora 等人的方法不同，本文的改进模型如人类一般同时基于通用语料（如，维基百科）和周围语境调整词语和语义和重要性。

Arora et al (2016) proposed a strong baseline of sentence embeddings using a random walk model of text generation. The current paper goes further, pointing out that adding a simplified syntactic attention mechanism can enhance semantic representations of sentences. Specifically, in this paper, we proposed a high-performance sentence representation method that is also simple to approach and high interpretable. We also proved that the maximum log likelihood of the sentence embedding equals to weighted average adjusted word vectors. This advanced method on average improves performance by 7.1% on five different semantic textual similarities (STS) tasks. The sensitivity analysis also finds that our advanced method has robust performance on long pieces of text. Unlike Arora et al.'s method, our approach adjusts the importance of words according to both general corpus (e.g. Wikipedia) and the surrounding context, just like a human. 

#### 下载 Download

* Step 1

  从pypi上下载最新版的SARW代码或者git clone本项目中的源代码。

  Download from Pypi or you can git clone the source code from this repository.

```bash
pip install SARW
```



* Step 2 / 1

  发送邮件wuxsmail@163.com获取最新的预训练模型

  Send a email to wuxsmail@163.com for the latest pre-trained model



* Step 2 / 2 

  或者给定一些语料、词向量和统计好的词频训练您自己的模型

  OR, train your model by given corpus, word embeddings and word frequency file

```python
>>> from SARW import trainSARW
>>> trainSARW(corpus, w2v_path, save_path, freq_path)
```



* Step 3

  加载模型并在您的工作中使用它

  load model and let it help you in your program

```python
>>> from SARW import loadSARW
>>> sarw_model = loadSARW(w2v_path, save_path, tokenizer=None)
>>> sarw_model.transform([
    	['This', 'is', 'the', 'first', 'sentence', '.'],
    	['This', 'is', 'another', 'once', '.']
    ])
```



#### 实验结果 Result

| 模型           | STS12 | STS13 | STS14 | STS15 | SICK14 |
| -------------- | ----- | ----- | ----- | ----- | ------ |
| avg-GloVe      | 52.1  | 49.4  | 54.1  | 56.1  | 65.9   |
| tfidf-GloVe    | 58.7  | 52.1  | 63.8  | 60.6  | 69.4   |
| GloVe+WR       | 56.2  | 56.6  | 68.6  | 71.7  | 72.2   |
| SCBOW-ATT-SUR  | 57.3  | 56.8  | 65.1  | 66.3  | -      |
| SCBOW-ATT-SPOS | 57.8  | 56.5  | 66.5  | 66.0  | -      |
| SCBOW-ATT-CCG  | 59.2  | 56.2  | 66.6  | 67.2  | -      |
| SARW           | 62.2  | 67.6  | 71.9  | 73.7  | 73.1   |

#### Contributors

* Xuansheng Wu

* Zhiyi Zhao

* Liangliang Liu

#### Reference

* [A Simple But Tough-to-Beat Baseline for Sentence Embeddings (Arora et al., 2016)](https://github.com/PrincetonML/SIF)



