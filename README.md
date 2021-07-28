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


|    Content    |    Section    |
| ------------- | ------------- |
| Installation  | [here](https://github.com/yiqiao-yin/YinsRLab#installation) |
| - Input       | [here](https://github.com/yiqiao-yin/YinsRLab#input) |
| - Workflow    | [here](https://github.com/yiqiao-yin/YinsRLab#production-workflow) |
| Author        | [here](https://github.com/yiqiao-yin/YinsRLab#author) |
| Acknowledgement | [here](https://github.com/yiqiao-yin/YinsRLab#acknowledge) |
| Representation Learning | [ANN](https://github.com/yiqiao-yin/YinsRLab#artificial-neural-network), [CNN](https://github.com/yiqiao-yin/YinsRLab#convolutional-neural-network), [RNN](https://github.com/yiqiao-yin/YinsRLab#recurrent-neural-network) |


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

---

# Representation Learning

There are three major types of *Representation Learning*.

## Artificial Neural Network

The architecture of a basic Artificial Neural Network (ANN) is the following
$$
\text{input variables:} \rightarrow
[\vdots] \rightarrow [\vdots]
\rightarrow
\text{output: predictions}
$$

When the notion of neural network is mentioned, we assume the architecture above with any particular sets of parameters (number of layers, number of neurons per layer, ..., i.e. these are all tuning parameters). ANN uses two concepts: *forward propagation* and *backward propagation*. 

- *Forward Propagation*: This operation allows information to pass from the input layer all the way to the output layer. There are linear components as well as non-linear components. Recall the basic linear regression and logistic regression models. We have explanatory variables $X$ and we first construct a linear combination of $\sum_{j=1}^p w_j X_j$ while the running index $j$ indicates the $j$th variable. This is then fed into the activation function (the famous ones are ReLU and sigmoid) to create predictions $\hat{Y}$. 
- *Backward Propagation*: This operation allows us to compare the predictions made by the educated guesses generated from the model with the ground truth, i.e. we compare how far away predictions $\hat{Y}$ is from the truth $Y$. This allows us to figure out the exact loss of a model. This is considered as an optimization problem with the following objective function:
$$\min_w \mathcal{L}(\hat{Y}, Y)$$
and the loss function is a matter of choice by the scientist or user of the algorithm. A list of loss functions can be found [here](https://towardsdatascience.com/most-common-loss-functions-in-machine-learning-c7212a99dae0). This above objective function states the following. The goal is to minimize the loss generated by prediction $\hat{Y}$ and the ground truth $Y$ while subject to the constraint of parameters (or weights) of the model $w$. In other words, we are allowed to change our weights $w$ until we found ourselves satisfied with the error generated with $\hat{Y}$ and $Y$. The quantity of error margin is a matter of choice and it usually is carried out as a relative term from model to model (i.e. this means we need a benchmark to compare and we do not just rely on one model).
- *Regressor*: This type of ANN learns from $X$ and produces an estimated value of $Y$ while $Y$ is continuous. The common loss function is [MSE](https://towardsdatascience.com/https-medium-com-chayankathuria-regression-why-mean-square-error-a8cad2a1c96f). In a neural network architecture that is designed as a regressor (predict a continuous variable, i.e. like regression model), we output one neuron.
$$
\text{input variables:} \rightarrow
[\vdots] \rightarrow [\vdots]
\rightarrow
[.], 
\text{output: predictions}
$$

- *Classifier*: This type of ANN learns from $X$ and produes an estimated probability of $Y$ that is a particular discrete value (aka factor or class). The common loss function is [binary cross-entropy](https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a). For a neural network architecture that is designed as a classifier (predict a discrete variable, a class, or a label, i.e. like logistic model), we output a certain number of neurons (number should match the number of levels in the response variable). 
$$
\text{input variables:} \rightarrow
[\vdots] \rightarrow [\vdots]
\rightarrow
[:], 
\text{output: predictions}
$$
The above architecture assumes two-class classification (the output has two dots).
- *Optimizer*: The architecture of an ANN consists of input layer (which is the explanatory variables), hidden layer (if any), the output layer (tailored to the type of problems, regressor or classifier), and a loss function. Once the architecture is setup we can use an optimizer to find the weights that are used in the optimizer. A famous optimizer is called [gradient descent](https://towardsdatascience.com/gradient-descent-explained-9b953fc0d2c). The key of the gradient descent algorithm focuses on using iterated loops to update the parameters $w$ of the model a step at a time. At each step $s$, we compute the gradient of the loss function, i.e. $\nabla \mathcal{L}(\hat{Y}, Y)$. Next, we update the parameters: $w_s = w_{s+1} - \eta \cdot \nabla \mathcal{L}(\hat{Y}, Y)$. Here are some videos I posted: [Gradient Descent](https://www.youtube.com/watch?v=OtLSnzjT5ns), [Adam](https://www.youtube.com/watch?v=AqzK8LeRThM), [ANN Regressor Classifier Summary](https://www.youtube.com/watch?v=zhBLiMdqOdQ), related python scripts are posted [here](https://www.github.com/yiqiao-yin/YinsPy) 
- *Loss*: A loss function measures the distance between the predictions and the ground truth. The bigger the loss, the more mistakes there are in the model, and the worse the performance will be. The loss function is a choice given by the scientist or the user of the software package. A common loss function is *Mean Square Error*. Suppose we have ground truth $Y$ and the prediction $\hat{Y}$. The loss function *Mean Square Error* or *MSE* is defined as $\frac{1}{n} \sum_{i=1}^n (\hat{y_i} - y_i)^2$ while the running index $i$ indicates the observation. Another famous loss function is *Cross-entropy*. In binary classification problems, we recall the mathematical model of a Bernoulli random variable (i.e. recall the coin toss example). A random variable $X$ is said to have Bernoulli distribution or is a Bernoulli random variable if $\mathbb{P}(X=0) = 1-p$ and $\mathbb{P}(X=1) = p$ while $X = 0$ or $X = 1$. We can write $\mathbb{P}(X) = p^x (1-p)^{1-x}$ to be the mathematical model describing a coin-toss-like profile. Last, we take logarithm on the last step and obtain $x \log(p) + (1-p) \log(1-x)$. In reality, this $p$ is a predicted probability, so we replace it with $\hat{y}$. The $x$ is replaced with $y$ because it is the ground truth. Hence, we arrive with $y\log(\hat{y}) + (1 - y)\log(1 - \hat{y})$. A common list of lost function can be seen [here](https://towardsdatascience.com/most-common-loss-functions-in-machine-learning-c7212a99dae0).
- Sample code can be found in the following:
```
model <- keras_model_sequential() # sequential here means I am coding the structure layer by layer (it is not referring to sequential model)
# what is the model above?
# it is telling computer I want a keras sequential object
# this sequential object allows me to add dense layer or other NN layers
model %>%
  layer_dense(
    units = number.of.levels, # how many neurons
    input_shape = c(ncol(x_train)), # input shapes, this is the number of the columns
    activation = finalactivation, # "sigmoid", "tanh", "relu", .......
    use_bias = useBias # TRUE / FALSE (it is referring to beta_0 in the markdown file, i.e. recall beta_0 in linear model as well as logistic model)
  ) # adding one dense layer (or one layer)
summary(model)
```

## Convolutional Neural Network

Convolutional Neural Network (CNN) is built upon the understanding of a basic neural network. In addition, we make the assumption that we are accepting image data. This assumption implies two key information: (i) local information and its intrinsic value, (ii) geospatial architecture may have important relationship amonst each other. Based on this information and understanding of this type of data sets, we have a strong motivation of using a new tool to develop new features to feed in the neural network architecture.

The convolutional operation relies on combining matrices. Some basic matrix algebra can be found [here](https://towardsdatascience.com/basics-of-linear-algebra-for-data-science-9e93ada24e5c). The convolutional operation can be found [here](https://www.freecodecamp.org/news/an-intuitive-guide-to-convolutional-neural-networks-260c2de0a050/). The following digram is adopted from the sources above.

The convolutional operation can be illustrated in the following diagram. We have a filter (artificially designed) that intends to capture certain pattern in the picture. We apply this filter starting from the top left corner of the image. Then we roll this filter towards right first until we hit the last column. Once we are finished with one column we move the filter down one row and start from the left of the picture. We keep rolling this filter until we hit the bottom right corner of the image.
<p align="center"><img src="https://github.com/yiqiao-yin/Introduction-to-Machine-Learning-Big-Data-and-Application/blob/main/pics/basic-conv-op.png"></img></p>

The architecture below illustrates a simple Convolutional Neural Network (CNN) architecture and the basic forms of operation.
<p align="center"><img src="https://github.com/yiqiao-yin/Introduction-to-Machine-Learning-Big-Data-and-Application/blob/main/pics/basic-cnn.png"></img></p>

Sample code can be found in the following:
```
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = convl1, # number of filters
                kernel_size = convl1kernel, # the size of the filters
                activation = activation, # same with neural network => non-linear component => for general situation in case the assumptions of linearity break
                input_shape = input_shape # same with neural network => the only difference is instead of number of variables, we define number of rows and columns in the picture (matrix)
                ) %>%

  layer_conv_2d(filters = convl2, kernel_size = convl2kernel, activation = activation) %>%
  # layer_max_pooling_2d(pool_size = maxpooll1) %>% # maxpooling => given a 2x2 matrix, I 'm taking the maximum value => shrink dimension from 2x2=4 to 1 number 
  # layer_dropout(rate = 0.25) %>% # the percentage of data I drop every round of training
  layer_flatten() %>% # this function is a necessity and it reshapes the output matrices from the previous layer into a big long vector so that we can process neural network
  layer_dense(units = l1.units, activation = activation) %>% layer_dropout(rate = 0.5) %>%
  layer_dense(units = l2.units, activation = activation) %>% layer_dropout(rate = 0.5) %>%
  layer_dense(units = l3.units, activation = activation) %>% layer_dropout(rate = 0.5) %>%
  layer_dense(units = num_classes, activation = activationfinal)
```

Some additional sources:
- Computer Vision Feature Extraction: [post](https://towardsdatascience.com/computer-vision-feature-extraction-101-on-medical-images-part-1-edge-detection-sharpening-42ab8ef0a7cd)
- Building advanced model using Variational Auto-Encoder (VAE): [python](https://blog.keras.io/building-autoencoders-in-keras.html), [R](https://keras.rstudio.com/articles/examples/variational_autoencoder.html)
- Building advanced model using Generalized Adversarial Network (GAN): [python](https://keras.io/examples/generative/dcgan_overriding_train_step/), [R](https://blogs.rstudio.com/ai/posts/2018-08-26-eager-dcgan/)
- Avatarify: [github](https://github.com/alievk/avatarify-python). Please keep the application within a range of practice that does not harm the society of other people!
- Keras Examples: [python](https://keras.io/examples/), [R](https://tensorflow.rstudio.com/tutorials/)
- General AI Blog: [R AI Blog](https://blogs.rstudio.com/ai/)

## Recurrent Neural Network

LSTM is an tweak version of Recurrent Neural Network which forgets or remembers certain information over a long period of time.

Generalize intuition from above to the following:
- The previous cell state (i.e. the information that was present in the memory after the previous time step).
- The previous hidden state (i.e. this is the same as the output of the previous cell).
- The input at the current time step (i.e. the new information that is being fed in at that moment).

## Recurrent Neural Network (a sequential model)

Given data $X$ and $Y$, we want to feed information forward into a time stamp. Then we form some belief and we make some initial predictions. We investigate our beliefs by looking at the loss function of the initial guesses and the real value. We update our model according to error we observed. 

## Architecture: Feed-forward

Consider data with time stamp
$$X_{\langle 1 \rangle} \rightarrow X_{\langle 2 \rangle} \rightarrow \dots \rightarrow X_{\langle T \rangle}$$
and feed-forward architecture pass information through exactly as the following:
$$
\text{Information in:} \rightarrow
\begin{matrix}
Y_{\langle 1 \rangle}, \hat{Y}_{\langle 1 \rangle} & Y_{\langle 2 \rangle}, \hat{Y}_{\langle 2 \rangle} &       & Y_{\langle T \rangle}, \hat{Y}_{\langle T \rangle} \\
\uparrow               & \uparrow               &       & \uparrow \\
X_{\langle 1 \rangle} \rightarrow    & X_{\langle 2 \rangle} \rightarrow    & \dots \rightarrow & X_{\langle T \rangle} \\
\uparrow               & \uparrow               &       & \uparrow \\
w_{\langle 1 \rangle}, b_{0, \langle 1 \rangle}    & w_{\langle 2 \rangle}, b_{0, \langle 2 \rangle}    &       & w_{\langle T \rangle}, b_{0, \langle T \rangle} \\
\end{matrix}
\rightarrow
\text{Form beliefs about } Y_{\angle T \rangle}
$$

while the educated guesses $\hat{Y}_{\langle T \rangle}$ are our beliefs about real $Y$ at time stamp $T$. 

## Architecture: Feed-backward

Let us clearly define our loss function to make sure we have a proper grip of our mistakes. 
$$\mathcal{L} = \sum_t L(\hat{y}_{\langle t \rangle} - y_t)^2$$
and we can compute the gradient 
$$\triangledown = \frac{\partial \mathcal{L}}{\partial a}$$
and then with respect with parameters $w$ and $b$
$$\frac{\partial \triangledown}{\partial w}, \frac{\partial \triangledown}{\partial a}$$
and now with perspective of where we make our mistakes according to our parameters we can go backward

$$
\text{Information in:} \leftarrow
\begin{matrix}
Y_{\langle 1 \rangle}, \hat{Y}_{\langle 1 \rangle} & Y_{\langle 2 \rangle}, \hat{Y}_{\langle 2 \rangle} &       & Y_{\langle T \rangle}, \hat{Y}_{\langle T \rangle} \\
\downarrow               & \downarrow               &       & \downarrow \\
X_{\langle 1 \rangle} \leftarrow    & X_{\langle 2 \rangle} \leftarrow    & \dots \leftarrow & X_{\langle T \rangle} \\
\downarrow               & \downarrow               &       & \downarrow \\
\text{Update: } w_{\langle 1 \rangle}, b_{0, \langle 1 \rangle}    & w_{\langle 2 \rangle}, b_{0, \langle 2 \rangle}    &       & w_{\langle T \rangle}, b_{0, \langle T \rangle} \\
\end{matrix}
\leftarrow
\text{Form beliefs about } Y_{\angle T \rangle}
$$

and the *update* action in the above architecture is dependent on your optimizer specified in the algorithm.

