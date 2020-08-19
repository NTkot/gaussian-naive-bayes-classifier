# Gaussian Naive Bayes Classifier

A simple C++ Gaussian Naive Bayes Classifier (GNBC) library using [Eigen 3.3.7](https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.zip). Make sure to read the short-readme for its usage.

#### Dependecies
- Eigen 3.3.7

#### Installation
Include header file and compile .cpp file along with your "main.cpp" file.  
**You may need to change the include path of your Eigen library**.  

#### Usage
You can test the library by compiling test.cpp (with bayes.cpp).  

###### Training
Declare a class object-classifier and train it using the following line:  

`GNBC classifierObject(trainDataX,trainDataY);`  

or  

`GNBC classifierObject();`  
`classifierObject.train(trainDataX,trainDataY);`

###### Classifying
- To classify a `Eigen::VectorXf observation` (vector that contains a single observation with length = numberOfFeatures), use:  
  `Eigen::VectorXf prediction = classifierObject.predict(observation);`  
  `int label = classifierObject.label(prediction);`  

  `prediction`: Vector that contains probabilities of each class (length = numberOfClasses)  
  `label`: Contains index of class predicted (index of prediction vector that corresponds to the biggest probability)


- To classify a `Eigen::MatrixXf observations` (matrix that contains n observations with dimension = n*numberOfFeatures), use:  
  `Eigen::MatrixXf predictions = classifierObject.predictMatrix(observations);`  
  `Eigen::VectorXi labels = classifierObject.labelMatrix(predictions);`  

  `predictions`: MatrixXf that contains probabilities for every observation for each class (with dimension = n*numberOfClasses)  
  `labels`: Contains indexes of classes predicted (indexes of predictions matrix that corresponds to the biggest probability row-wise)

You can view the *test.cpp* file for an example.

:warning: At training stage, **trainDataY** (integer vector containing outcomes of examples given) should follow **zero-based numbering**!  