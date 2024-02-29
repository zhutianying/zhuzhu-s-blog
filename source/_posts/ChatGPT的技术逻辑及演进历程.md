---
title: ChatGPT的技术逻辑及演进历程
date: 2024-02-29 21:16:00
tags:
    - AI
---

# 1.什么是**GPT**

ChatGPT里面有两个词，一个是Chat，指的是可以对话聊天。另外一个词，就是GPT。

GPT的全称，是Generative Pre-Trained Transformer（生成式预训练。模型）。

可以看到里面一共3个单词，Generative生成式、Pre-Trained预训练、和Transformer。

简单来说，GPT是一个自回归的**语言模型，**可以不断基于前文生成下一个词的**续写模型（**用前K个单词预测第K+1个单词**）。**

# 2.GPT之技术演进时间线

1. 2017年6月，Google发布论文《Attention is all you need》，首次提出Transformer模型，成为GPT发展的基础。论文地址：https://arxiv.org/abs/1706.03762
2. 2018年6月,OpenAI 发布论文《Improving Language Understanding by Generative Pre-Training》(通过生成式预训练提升语言理解能力)，首次提出GPT模型(Generative Pre-Training)。论文地址：https://paperswithcode.com/method/gpt 。
3. 2019年2月，OpenAI 发布论文《Language Models are Unsupervised Multitask Learners》（语言模型应该是一个无监督多任务学习者），提出GPT-2模型。论文地址: https://paperswithcode.com/method/gpt-2
4. 2020年5月，OpenAI 发布论文《Language Models are Few-Shot Learners》(语言模型应该是一个少量样本(few-shot)学习者，提出GPT-3模型。论文地址：https://paperswithcode.com/method/gpt-3
5. 2022年2月底，OpenAI 发布论文《Training language models to follow instructions with human feedback》（使用人类反馈指令流来训练语言模型），公布 Instruction GPT模型。论文地址：https://arxiv.org/abs/2203.02155

# 3.GPT之T-Transformer（2017）

## **3.1、上一代RNN模型的重大缺陷**

在Transformer模型出来前，RNN模型(循环神经网络)是典型的NLP模型架构。

**RNN**的基本原理是，从左到右浏览每个单词向量(比如说this is a dog)，保留每个单词的数据，后面的每个单词，都依赖于前面的单词。

**RNN**的关键问题**：前后需要顺序、依次计算。**可以想象一下，一本书、一篇文章，里面是有大量单词的**，而又因为顺序依赖性，不能并行，所以**效率很低。

## **3.2、Transformer之All in Attention**

简单理解，就是单词与单词之间的关联度，通过注意力(Attention) 这个向量来描述。

比如说 You are a good man(你是个好人)，AI在分析 You的注意力向量时，可能是这么分析的：

从Your are a good man这句话中，通过注意力机制进行测算，You和You（自身）的注意力关联概率最高(0.7,70%)，毕竟 你(you)首先是你(you) ；于是You,You的注意力向量是 0.7。

You和man（人）的注意力关联其次(0.5，50%)，你(you)是个人(man)，于是You,man的注意力向量是0.5。

You和good(好)的注意力关联度再次(0.4，40%)，你在人的基础上，还是一个好(good)人。于是You，good的注意力向量值是0.4。

You，are向量值是 0.3； You，a的向量值是0.2。

于是最终You的注意力向量列表是【0.7 、0.3、0.2、0.4、0.5】。

## **3.3、论文中对attention和Transfomer的价值描述**

在论文中，google对于attention和transfomer的描述，主要强调了**传统模型对顺序依赖存在**，Transformer模型可以替代当前的递归模型，**消减对输入输出的顺序依赖**。

## **3.4、Transformer机制的深远意义**

Transformer问世后，迅速取代循环神经网络RNN的系列变种，成为主流的模型架构基础。

如果说可以并行、速度更快都是技术特征，让行外人士、普罗大众还不够直观，那么从当前ChatGPT的震憾效果就可以窥知一二。

Transformer从根本上解决了**两个关键障碍**，其推出是**变革性的、革命性的、开创性的**。

### 3.4.1、摆脱了人工标注数据集（大幅降低人工数量 ）

**第一个关键障碍**就是：过往训练我们要训练一个深度学习模型，必须使用大规模的**标记好的数据集合(Data set)**来训练，这些数据集合**需要人工标注，成本极高**。

打个比方，就是机器学习需要大量教材，大量输入、输出的样本，让机器去学习、训练。这个教材需要量身制定，而且需求数量极大。好比 以前要10000、10万名老师编写教材，现在只需要10人，降低成千上万倍。

那么这块是怎么解决的呢？简单描述一下，就是通过**Mask机制**，遮挡已有文章中的句段，**让**AI去填空。

好比是一篇已有的文章、诗句，挡住其中一句，**让机器根据学习到的模型，依据上一句，去填补下一句**。

这样，**很多现成的文章、**wiki、论文等，就是天然的标注数据集了。

### 3.4.2、化顺序计算为并行计算，巨幅降低训练时间

除了人工标注之外，**顺序计算，单一流水线**的问题。这是**另一个关键障碍**。

Self-Attention机制，结合mask机制和算法优化，使得 一篇文章、一句话、一段话 能够并行计算。

（注意力向量）

# 4、GPT(Generative Pre-Training)（2018年6月）

## 4.1、GPT模型的核心主张1-预训练(pre-training)

GPT模型依托于**Transformer解除了顺序关联和依赖性的前提**，提出一个建设性的主张。

先通过**大量**的**无监督预训练(Unsupervised pre-training)**，

注：**无监督**是指不需要人介入，不需要标注数据集（**不需要教材和老师**）的预训练。

再通过**少量有监督微调（Supervised fine-tunning)**，来修正其理解能力。

打个比方，就好像我们培养一个小孩，分了两个阶段：

1)、**大规模自学阶段**（自学1000万本书，没有老师）：给AI提供充足的算力，让其基于Attention机制，**自学**。

2)、**小规模指导阶段**(教10本书)：依据10本书，举一反"三"

## 4.2、GPT模型的核心主张2-生成式(Generative)

在机器学习里，有判别式模式(discriminative model)和生成式模式(Generative model)两种区别。

GPT(Generative Pre-Training)顾名思义，采用了生成式模型。

简单来说，生成式模型相比判别式模型更适合**大数据**学习 ，后者更适合精确样本(人工标注的有效数据集）。要更好实现**预训练(Pre-Training)**，生成式模式会更合适。

生成式模式： https://en.wikipedia.org/wiki/Generative_model 

## 4.3、GPT vs Transfomer的模型改进

**GPT**训练了一个12层仅decoder的解码器**（decoder-only,没有encoder)，从而使得模型**更为简单**。

# 5、GPT-2（2019年2月）

## 5.1、GPT-2模型相比GPT-1的核心变化

前面提到，GPT的核心主张有Generative(生成式)、Pre-Training（预训练）同时，GPT训练有两步：

1)、**大规模自学阶段**（Pre-Training预训练,自学1000万本书，没有老师）：给AI提供充足的算力，让其基于Attention机制，自学。

2)、**小规模指导阶段**(fine-tuning微调，教10本书)：依据10本书，举一反"三"

GPT-2的时候，OpenAI将**有监督**fine-tuning微调阶段给直接去掉了，将其变成了一个**无监督的模型（不要人工、不要老师）**。

同时，增加了一个关键字**多任务(multitask)**。

## 5.2、为什么这么调整？试图解决zero-shot问题

GPT-2为什么这么调整？从论文描述来看，是为了尝试解决**zero-shot(零次学习问题)**。

**zero-shot(零次学习)** 是一个什么问题呢？简单可理解为推理能力。就是指面对未知事物时，**AI**也能自动认识它，即具备推理能力。

比如说，在去动物园前，我们告诉小朋友，**像熊猫一样，是黑白色，并且呈黑白条纹的类马动物就是斑马**，小朋友根据这个提示，**能够正确找到斑马**。小朋友没有见过斑马，但是却能一眼认出斑马（推理）。

## 5.3、multitask多任务如何理解？

传统ML中，如果要训练一个模型，就需要一个专门的标注数据集，训练一个专门的AI。

比如说，要训练一个能认出狗狗图像的机器人，就需要一个标注了狗狗的100万张图片，训练后，AI就能认出狗狗。这个AI，是专用AI，也叫single task。显然从宏观层面上，single task会更贵更浪费，因为能认识狗的专用AI，极大概率不认识什么是猫、什么是人。

而multitask多任务，就是主张不要训练专用AI，而是喂取了海量数据后，任意任务都可完成。显然，multitask更能**训练通用的AI**。

## **5.4、GPT-2的数据和训练规模**

数据集增加到800万网页，**40GB大小**。

而模型自身，**也达到最大15亿参数**、Transfomer堆叠至48层。**就像是模拟人类15亿神经元。**

# 6、GPT-3（2020年5月）

## **6.1、GPT-3的突破式效果进展**

1、GPT-3在翻译 、问题回答和完形填空中表现出强大的性能，同时能够解读单词、句子中使用新单词或执行3位数算术。

2、GPT-3可以生成新闻文章的样本，人类已然区分不出来。

## **6.2、GPT-3相比GPT-2的核心变化**

前面提到GPT-2在追求无监督、zero-shot（零次学习），但是其实在GPT-2论文中，OpenAI也提出结果不达预期。这显然是需要调整的，于是GPT-3就进行了相关调整。从标题《Language Models are Few-Shot Learners》(语言模型应该是一个少量样本(few-shot)学习者)也可看出。

并且，在训练过程中，OpenAI是同时开展了**4种模式**(Zero-shot、One-shot、Few-shot、Fine-tuning)， 会对比Zero-shot（零次学习） 、One-shot（单一样本学习）、Few-shot（少量样本学习），以及fine-tuning（人工微调）的方式。

最后在多数情况下，few-shot(少量样本）的综合表现，是在无监督模式下最优的，但稍弱于fine-tuning微调模式。

## **6.3、GPT-3的训练规模**

GPT-3采用了过滤前45TB的压缩文本，并且在**过滤后(after filtering)也仍有570GB的海量数据**。

在模型参数上，**从GPT-2的15亿，提升到1750亿，翻了110多倍。**

# **7、**Instruction **GPT**（2022年2月）

## 7.1、Instruction GPT相比GPT-3的核心变化

Instruction GPT是基于GPT-3的一轮增强优化，所以也被称为GPT-3.5。

前面提到，GPT-3主张few-shot少样本学习，同时坚持无监督学习。

但是事实上,few-shot的效果，显然是差于fine-tuning监督微调的方式的。

那么怎么办呢？走回fine-tuning监督微调？显然不是。

OpenAI给出新的答案：强化学习--在GPT-3的基础上，基于人工反馈(RHLF）训练一个reward model(奖励模型),再用reward model(奖励模型，RM)去训练学习模型。

简单来说就是**用机器(**AI)来训练机器(AI)

## **7.2、Insctruction** **GPT**的核心训练步骤

reinforcement learning from human feedback（RLHF）利用人类反馈的强化学习

- 第一步：**监督微调**-supervised fine-tuning (SFT) 

训练数据集：为人工撰写的提示和通过OpenAI API搜集的用户提示，人工标注出那些满意的回复。

采用监督学习的方法fintune GPT-3模型

- 第二步：**奖励模型-**reward model (RM) training 

训练数据集：人工对模型生成的多种回复进行排序（打分）。

利用排序信息训练一个reward model (RM)，这个RM训练好以后就可以为任意回复进行打分。

- 第三步：reinforcement learning via Proximal Policy Optimization (PPO) 

通过PPO算法对第一步得到的模型进一步进行强化学习。不断优化模型使得reward model (RM)的打分更高。

**第2步、第3步是完全可以迭代、循环多次进行的**。

## **7.3、**Instruction **GPT**的训练规模

**基础数据规模同GPT-3**，只是在其基础上增加了3个步骤（监督微调SFT、奖励模型训练Reward Model，增强学习优化PPO)。

多了OpenAI雇佣或有相关关系的标注人员(labler)和GPT-3 API的调用用户（customer）

# **8、**ChatGPT（2022年11月）

## **8.1、**ChatGPT和Instruction **GPT**

ChatGPT和InstructionGPT本质上是同一代际的，仅仅是在InstructionGPT的基础上，**增加了Chat功能（强大的上下文学习能力），以及将InstructionGPT发布后上述训练步骤的第二步、第三步循环迭代了多轮，同时开放到公众测试训练，以便产生更多有效标注数据。**

## 8.2、ChatGPT可以用来做什么



# 参考资料与拓展阅读：

ai.googleblog.com/2017/08/transformer-novel-neural-network.html

https://arxiv.org/abs/1706.03762

https://paperswithcode.com/method/gpt

https://paperswithcode.com/method/gpt-2

https://paperswithcode.com/method/gpt-3

https://arxiv.org/abs/2203.02155

https://zhuanlan.zhihu.com/p/464520503

https://zhuanlan.zhihu.com/p/82312421

https://cloud.tencent.com/developer/article/1656975

https://cloud.tencent.com/developer/article/1848106

https://zhuanlan.zhihu.com/p/353423931

https://zhuanlan.zhihu.com/p/353350370

https://juejin.cn/post/6969394206414471175

https://zhuanlan.zhihu.com/p/266202548

https://en.wikipedia.org/wiki/Generative_model

https://zhuanlan.zhihu.com/p/67119176

https://zhuanlan.zhihu.com/p/365554706

https://cloud.tencent.com/developer/article/1877406

https://zhuanlan.zhihu.com/p/34656727

https://zhuanlan.zhihu.com/p/590311003

https://zhuanlan.zhihu.com/p/590311003

https://zhuanlan.zhihu.com/p/446293526

https://zhuanlan.zhihu.com/p/264749298

https://zhuanlan.zhihu.com/p/112998607