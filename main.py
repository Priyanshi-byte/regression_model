import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def linear_regression_model(X:np.ndarray,y:np.ndarray, learning_rate:float=0.01,iteration:int=1000):
    """Generate the Linear regression model.

    Linear regression model is given by this equation: y=X*beta+beta0.
    To solve for the varibales beta and beta0, we use least square method 
    which minimizes error between actual and predicted values.
    We use Gradient Descent technique to find the derivate of the least square function
    with respect to beta and beta0 and update the values in each iteration using learning rate.

    Args:
        X (np.ndarray): Explanatory variable
        y (np.ndarray): Objecitve variable
        learning_rate (float, optional): Learning rate value. Defaults to 0.01.
        iteration (int, optional): Number of iterations. Defaults to 1000.

    Returns:
        tuple(np.ndarray,float): values of beta and beta0.
    """

    N=X.shape[0] #Sample_number
    m=X.shape[1] #Feature_number

    #Initialize beta and beta0
    beta=np.ones(m)
    beta0=1

    for _ in range(iteration):
        #Linear regression model: y=X*beta+beta0
        y_pred= X @ beta + beta0
        
        #Gradient calculation
        #Least sqaure function: E=1/N[y-(X*beta+beta0)]^2
        # We observe the following gradients: 
        # 1. dE/dbeta= -2/N*(Xt*(y-(X*beta+beta0)))
        # 2. dE/dbeta0= -2/N*(y-(X*beta+beta0)
        d_beta= -2/N * (X.T @ (y-y_pred))
        d_beta0= -2/N * np.sum(y-y_pred)

        #Parameters update
        #To update the parameters, we will use gradient descent:
        # 1. beta=beta-learning_rate*dE/dbeta
        # 2. beta=beta0-learning_rate*dE/dbeta0
        beta-=learning_rate*d_beta
        beta0-=learning_rate*d_beta0
    
    return beta,beta0

def linear_regression_prediction(X:np.ndarray,beta:np.ndarray,beta0:float)->np.ndarray:
    """Predicts y value using obtained coefficients.

    Args:
        X (np.ndarray): Explanatory variable
        beta (np.ndarray): beta coefficient
        beta0 (float): beta0 coefficient

    Returns:
        np.ndarray: Predicted objective variable
    """
    return X @ beta + beta0

def linear_regression_rmse(y:np.ndarray,y_pred:np.ndarray)->float:
    """Calcualtes root mean square value of model

    Args:
        y (np.ndarray): Objective variable
        y_pred (np.ndarray): Predicted objective variable

    Returns:
        float: Error value
    """
    return np.sqrt(np.mean((y-y_pred)**2))


if __name__ == "__main__":
    #Prepare data
    train_df=pd.read_csv("data/regression_train.csv")
    X_train=train_df.iloc[:,1:4].values
    y_train=train_df.iloc[:,4].values

    test_df=pd.read_csv("data/regression_test.csv")
    X_test=test_df.iloc[:,1:4].values
    y_test=test_df.iloc[:,4].values

    #Feature scaling - standardization
    #Since the features have quite varying values, we will standardize to mean 0 and sd 1
    X_train_mean = np.mean(X_train, axis=0)
    X_train_sd = np.std(X_train, axis=0)
    X_train= (X_train - X_train_mean) / X_train_sd

    X_test_mean = np.mean(X_test, axis=0)
    X_test_sd = np.std(X_test, axis=0)
    X_test= (X_test - X_test_mean) / X_test_sd

    #Generate model
    beta,beta0=linear_regression_model(X_train,y_train)

    #Perform plotting
    y_pred=linear_regression_prediction(X_test,beta,beta0)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test,y_pred)
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Ideal fit')
    plt.xlabel('Actual value')
    plt.ylabel('Predicted alue')
    plt.legend()
    plt.savefig("Actual vs predicted.png")

    #Evaluate
    mse_val=linear_regression_rmse(y_test,y_pred)
    print(f"Model performance: Root mean squared error value = {mse_val}")

        
