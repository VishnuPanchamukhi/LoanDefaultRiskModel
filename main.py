import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


dataset = pd.read_csv(r"C:\Vishnu\Coding\Python\Credit Risk Model\creditRiskDataset.csv")
inputFeatures = dataset.iloc[:, :-1].values
outputs = dataset.iloc[:, -1].values.reshape(-1, 1)


# feature scaling
# z score normalisation
means = np.mean(inputFeatures, axis=0)
stds = np.std(inputFeatures, axis=0)
# weighted scaling
sclaingWeights = np.array([2, 0.5, 2, 1.0, 1.2, 1.4, 1.7])
# apply sclaing
inputFeatures = (inputFeatures - means) / stds
inputFeatures *= sclaingWeights


# sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# logistic loss cost function
def computeCost(inputFeatures, outputs, weights, bias):
    m = len(outputs)
    
    z = np.dot(inputFeatures, weights) + bias
    f_wb = sigmoid(z)
    
    # add small value to avoid log0 issues
    smallValue = 0.000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001
    cost = -outputs * np.log(f_wb + smallValue) - (1 - outputs) * np.log(1 - f_wb + smallValue)
    totalCost = np.sum(cost) / m

    return totalCost


# calculate partial derivatives
def computeGradient(inputFeatures, outputs, weights, bias):
    m, n = inputFeatures.shape
    wPartialDerivatives = np.zeros(weights.shape)
    bPartialDerivative = 0.

    for i in range(m):
        prediction = sigmoid(np.dot(inputFeatures[i], weights) + bias)
        real = outputs[i]
        cost = prediction - real
        wPartialDerivatives += cost * inputFeatures[i].reshape(-1, 1)
        bPartialDerivative += cost
    wPartialDerivatives /= m
    bPartialDerivative /= m

    return wPartialDerivatives, bPartialDerivative


def gradientDescent(inputFeatures, outputs, weights, bias, learningRate, numIterations):
    m, n = outputs.shape
    
    # store cost at each iteration for visualisation
    jHistory = []
    
    for i in range(numIterations):
        # compute gradient
        wPartialDerivatives, bPartialDerivative = computeGradient(inputFeatures, outputs, weights, bias)    

        # update parameters simultaneously
        weights -= learningRate * wPartialDerivatives              
        bias -= learningRate * bPartialDerivative              
        
        # save cost each iteration
        cost = computeCost(inputFeatures, outputs, weights, bias)
        jHistory.append(cost)

        # Print cost
        if i % max(1, numIterations // 10) == 0 or i == numIterations - 1:
            print(f"Iteration {i:4}: Cost {cost:.4f}")

    return weights, bias, jHistory


# predict 1 if simgoid is greater than 0.5
def predict(inputFeatures, weights, bias):
    m,n = inputFeatures.shape
    p = np.zeros(m)

    for i in range(m):
        prediction = sigmoid(np.dot(inputFeatures[i], weights) + bias)
        if prediction > 0.5:
            p[i] = 1
        else:
            p[i] = 0
    
    return p


# inital weights and bias
weights = np.random.randn(7, 1)
bias = 0

learningRate = 0.01
numIterations = 5000

optimisedWeights, optimisedBias, costHistory = gradientDescent(inputFeatures, outputs, weights, bias, learningRate, numIterations)

# plot cost history
plt.plot(costHistory)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Reduction over Iterations")
plt.show()

predictions = predict(inputFeatures, optimisedWeights, optimisedBias)

accuracy = np.mean(predictions == outputs.flatten()) * 100
print(f"Training Accuracy: {accuracy:.2f}%")