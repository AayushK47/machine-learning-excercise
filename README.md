# Machine Learning Excersises

This repository contains all the the machine learning activities that I have 
performed on various datasets.

## List of all the Datasets

- Amazon Review Dataset
- Cancer Dataset
- Face Detection
- IMDB Review Dataset
- Kaggle Titanic Dataset
- Manual Neural Network
- MNIST Dataset
- Restaurant Review Dataset
- Yelp Review Datatset

### Amazon Review Dataset

This dataset consists of two features :
* Reviews : Review of a product left by user on amazon.com
* Label: Consists of 2 values, 0 or 1. 0 Means it is a negative review whereas 1 means it is a positive review

I used the bag of words model to convert the text into feature matrix and the performed supevised learning using GaussianNB classifier, Decision Tree Classifier, Random Forest Classifier and Gradient Boosting classifier.

### Cancer Dataset

This dataset consists of two features :

* **id** : Id of each patient
* **clump_thickness** : (1-10). Thickness of clump. Benign cells tend to be grouped in monolayers, while cancerous cells are often grouped in multilayers.
* **unif_cell_size** : (1-10). Cancer cells tend to vary in size and shape. That is why these parameters are valuable in determining whether the cells are cancerous or not.
* **unif_cell_shape** : (1-10). Uniformity of cell size/shape: Cancer cells tend to vary in size and shape. That is why these parameters are valuable in determining whether the cells are cancerous or not.
* **marg_adhesion** : (1-10). Normal cells tend to stick together. Cancer cells tends to loose this ability. So loss of adhesion is a sign of malignancy.
* **single_epith_cell_size** : (1-10). It is related to the uniformity mentioned above. Epithelial cells that are significantly enlarged may be a malignant cell.
* **bare_nuclei** : (1-10). This is a term used for nuclei that is not surrounded by cytoplasm (the rest of the cell). Those are typically seen in benign tumours.
* **bland_chrom** : (1-10). Describes a uniform "texture" of the nucleus seen in benign cells. In cancer cells the chromatin tend to be more coarse.
* **norm_nucleoli** : (1-10). Nucleoli are small structures seen in the nucleus. In normal cells the nucleolus is usually very small if visible at all. In cancer cells the nucleoli become more prominent, and sometimes there are more of them.
* **mitoses** : (1-10). Cancer is essentially a disease of uncontrolled mitosis
* **class** : (0-1) : 1 means cancer is malignant 0 means it is benign.

The dataset contains some missing values. After treating the missing values, I used Multinomial Naive Bayes Classifier to achive 95% accuracy.

### Face Detection

In this project, I used opencv in python to create a script to implement face detector. I used The very popular Viola Davis algorithm to detect face and various features of face like smile and eyes.
 
### IMDB Review Dataset

This dataset consists of two features :
* Reviews : Review of a movie left by user on imdb.com
* Label: Consists of 2 values, 0 or 1. 0 Means it is a negative review whereas 1 means it is a positive review

I used the bag of words model to convert the text into feature matrix and the performed supevised learning using GaussianNB classifier, Decision Tree Classifier, Random Forest Classifier and Gradient Boosting classifier.

### Kaggle Titanic Dataset

This is the dataset from a kaggle competition. Here's the link to the kaggle page that contains the dataset and the details of the various features :- https://www.kaggle.com/c/titanic/data

The task is to predict wheather a passenger survived or not.The dataset contains various missing values. After treating all the missing values and some feature engineering, I got 83% accuracy on Logistic Regressor.

### Manual Neural Network

Here, I created an artificial neural network model from scratch without using any external library.

### MNIST Dataset

The dataset consists of images of digits. Our task is to identify the digit in the image. I used 2 methods to complete the task. I used the softmax algorithm which gave the accuracy of 91.%. I also used convolutional neural network which gave accuracy of about 98%.

### Restaurant Review Dataset

This dataset consists of two features :
* Reviews : Review of a Restaurant left by user
* Label: Consists of 2 values, 0 or 1. 0 Means it is a negative review whereas 1 means it is a positive review

I used the bag of words model to convert the text into feature matrix and the performed supevised learning using GaussianNB classifier.

### Yelp Review Datatset

This dataset consists of two features :
* Reviews : Review of a place left by user on yelp.com
* Label: Consists of 2 values, 0 or 1. 0 Means it is a negative review whereas 1 means it is a positive review

I used the bag of words model to convert the text into feature matrix and the performed supevised learning using GaussianNB classifier, Decision Tree Classifier, Random Forest Classifier and Gradient Boosting classifier.
