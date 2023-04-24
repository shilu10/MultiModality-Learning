
# Deep MultiModality Learning

## Project: MultiModality Entailment Classification

  

### Install

  

This project requires **Python 3.7** and the following Python libraries installed:

  

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [tensorflow](https://www.tensorflow.org/)



  

We recommend students install [Anaconda](https://www.continuum.io/downloads), a pre-packaged Python distribution that contains all of the necessary libraries and software for this project.

### MultiModality Learning
MultiModality is subfield of Machine learning, where we use multimodal data, instead of unimodal data. Because, sometimes it is better to learn from the multimodal data then unimodal data.
Video, Images, Text, and Audio are different modalities of data, Multimodality learning mimics the human behavior, where we used to  learn in multimodality fashion.

![Workflow of a typical multimodal. Three unimodal neural networks encode the different input modalities independently. After feature extraction, fusion modules combine the different modalities (optionally in pairs), and finally, the fused features are inserted into a classification network. ](https://assets-global.website-files.com/5d7b77b063a9066d83e1209c/639b05b4719574e870f698fe_multimodal%20wokflow.webp)

### Directory Structure
		train.py    		    : contains code for training the multimodality model.
		test.py     : contains code for testing the multimodality model.
		metrics.py              : contains code for metrics with custom implementation.
		model_builder.py        : contains code for building image encoder and text encoder and multimodality models.
		roberta_preprocessor.py : contains code for roberta preprocessor model.
		manual_train.py         : contains code for custom training with tensorflow.
		data_preprocessor.py    : contains code for data preprocessing.
		download_dataset.sh     : used to download the dataset from github.
		datasets/               : Downloaded dataset will be saved to this directory.
		notebooks/              : This directory contains the notebooks(kaggle notebooks).
		tests/                  : contains the test cases, which uses the pytest.
		

### Run

  

In a terminal or command window, navigate to the top-level project directory `MultiModality-Learning/` . And execute the below command for training the multimodality model.

  

```bash

python3 train.py 
Args:
--image-shape=(128, 128) 
--image-shape=(128, 128, 3) 
--batch-size=32
--save-path='models/'
--save-model=True
--image-dir="tweets_images"
--epoch=30
--lr=0.0001
--opt-type="adam"

```


  

### Data

  

The modified census dataset consists of approximately 32,000 data points, with each datapoint having 13 features. This dataset is a modified version of the dataset published in the paper *"Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid",* by Ron Kohavi. You may find this paper [online](https://www.aaai.org/Papers/KDD/1996/KDD96-033.pdf), with the original dataset hosted on [UCI](https://archive.ics.uci.edu/ml/datasets/Census+Income).

  

**Features**

-  `age`: Age

-  `workclass`: Working Class (Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked)

-  `education_level`: Level of Education (Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool)

-  `education-num`: Number of educational years completed

-  `marital-status`: Marital status (Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse)

-  `occupation`: Work Occupation (Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces)

-  `relationship`: Relationship Status (Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried)

-  `race`: Race (White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black)

-  `sex`: Sex (Female, Male)

-  `capital-gain`: Monetary Capital Gains

-  `capital-loss`: Monetary Capital Losses

-  `hours-per-week`: Average Hours Per Week Worked

-  `native-country`: Native Country (United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands)

  

**Target Variable**

-  `income`: Income Class (<=50K, >50K)
