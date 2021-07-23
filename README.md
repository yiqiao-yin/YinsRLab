# YinsRLab
[![YinsCapital](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://yinscapital.com/research/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

This is the library that I use to collect common machine learning algorithm I use for research project and industrial practice.

<p align="center">
  <img width="800" src="https://github.com/yiqiao-yin/YinsRLab/blob/master/figs/HELLO2.gif">
</p>
<p align="center">
	<img src="https://img.shields.io/badge/stars-30+-blue.svg"/>
	<img src="https://img.shields.io/badge/license-CC0-blue.svg"/>
</p>

# Installation

The installation of this package is simple. We recommend to use devtools to install from Github.

```
# Note: you must have these packages:
# quantmod, stats, xts, TTR, knitr, gtools, glmnet, fastAdaboost, e1071, keras, naivebayes, randomForest, iRF, BayesTree, gbm, ipred, pROC, Rtts, audio, beepr
# Please install them if you do not have them. 

# Install Package: Yin's RLab (i.e. YinsRLab)
devtools::install_github("yiqiao-yin/YinsRLab")
```

# Input

This package has all functions collected in *R* folder. Almost all functions following the following format:
- input $X$: this is the explanatory variables in the data set;
- input $y$: this is the response variable in the data set;
- input *cutoff*: this is a numerical value from 0 to 1 (default value is 0.9), implying that the algorithm will take the first 90% of the observation as training and the rest as testing;
- input parameters: this is dependent on each function (ex: for trees, there are number of trees, for SVM, there is gamma, etc..);
- input *cutoff.coefficient*: this is a value with default at 1, which means the algorithm is setting exactly mean of predicted scores as cutoff to convert scores into binary values.

# Author

Yiqiao Yin is a Data Scientist at Bayer. He has been a Research Assistant at Columbia University since 2017. Prior to his current position, he has been a Researcher at Simon Business School with professors from AQR Capital. 

He is admitted to Ph. D. program in Statistics at Columbia University and he will be working on machine learning and generalized big data problems.

# Acknowledge

This package is jointly designed with [Statistical Machine Learning](https://github.com/yiqiao-yin/Statistical-Machine-Learning) and one can refer to my [Advice](https://github.com/yiqiao-yin/YinsRLab/blob/master/Advice.md) to my audience and to those who are interested in developing their own packages.
