## Linear Regression w/ Gradient Descent, Regularization
This program shows how step by step Linear Regression happens with a break down of each step in different functions, followed by an implementation of Regular/Batch/Stochastic Gradient Descent. Here we use more than 2 variables, i.e., example of Multivariate Linear Regression.

#### Linear Regression
In simple terms, it's (Supervised Learning) approach for predicting the next value given some data.

* In the case of the program, We first carry out Normalization. This process helps put the values to a certain scale, such that they revolve around a scalable and better understandable value. This is usually from [0,1].

* We then proceed with hypothesis function [theta] to minimize the loss.

      h[theta](x) = [theta]T x; where [theta]T = transpose of theta

* We try to choose this theta to minimize the square loss fn. 
      
      J(theta) = 1/m {(w) - yi}^2; where w = hypothesis fn
      
#### Gradient Descent
This algorithm is used to update the weights in the loss function to minimize the loss. Once we obtain the minimum loss, we can classify the relationship between the input variables and the output variable.

* After carrying the previous step out and obtaining the loss we have to update [theta] to minimize the loss. This is called the weight update step of the gradient descent algo.

      [theta]new = [theta] - d(J)/d([theta]) * a; where a = learning rate
      
* This is then inserted into the square loss function runs again to find the global minima of the square loss.

#### Regularization
Also included is regularized gradient descent. In brief terms, regularization is done to prevent overfitting, i.e, high variance of the predicted values. There are 2 types of regression, lasso(L1) and ridge(L2).

* Here we use Ridge Regression. We add an error term, making our new square loss eqn -
      
      J(theta) = 1/m {(w) - yi}^2 + [lambda] * {[theta]T * [theta]}; where [lambda] = regularization param
      
NOTE - The final value in the code have to self-printed, i.e., according to what the user wants to print. 
