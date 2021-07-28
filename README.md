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

---

# Installation

The installation of this package is simple. We recommend to use devtools to install from Github.

```
# Note: you must have these packages:
# quantmod, stats, xts, TTR, knitr, gtools, glmnet, fastAdaboost, e1071, keras, naivebayes, randomForest, iRF, BayesTree, gbm, ipred, pROC, Rtts, audio, beepr
# Please install them if you do not have them. 

# Install Package: Yin's RLab (i.e. YinsRLab)
devtools::install_github("yiqiao-yin/YinsRLab")
```

## Input

This package has all functions collected in *R* folder. Almost all functions following the following format:
- input $X$: this is the explanatory variables in the data set;
- input $y$: this is the response variable in the data set;
- input *cutoff*: this is a numerical value from 0 to 1 (default value is 0.9), implying that the algorithm will take the first 90% of the observation as training and the rest as testing

## Production Workflow

How are these functions created in this package? Each function is designed to solve certain problems and they have been successful before. The entire production workflow consists of (i) Research and Development, and (ii) software production.

<p align="center">
  <img width="600" src="https://github.com/yiqiao-yin/YinsRLab/blob/master/figs/workflow.jpg">
</p>

---

# Author

I am currently a PhD student in Statistics at Columbia University. 

Prior to Columbia, I have held professional positions including an enterprise-level Data Scientist at Bayer and a quantitative researcher at AQR working on alternative quantitative approaches to portfolio management and factor-based trading, and a trader at T3 Trading on Wall Street. I supervise a small fund specializing in algorithmic trading (since 2011, performance is here). I also run my own YouTube Channel.

---

# Acknowledge

This package is jointly designed with [Statistical Machine Learning](https://github.com/yiqiao-yin/Statistical-Machine-Learning) and one can refer to my [Advice](https://github.com/yiqiao-yin/Statistical-Machine-Learning/blob/master/Story.md) to my audience and to those who are interested in developing their own packages.
