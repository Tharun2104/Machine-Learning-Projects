import pickle
import time
import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt



def preprocess():
    """ 
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1 
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output: 
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args

    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_features + 1, 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    # Add bias term to train_data
    train_data = np.hstack((np.ones((n_data, 1)), train_data))  # Shape becomes N x (D+1)

    # Compute the sigmoid function
    theta = sigmoid(np.dot(train_data, initialWeights))  # theta = σ(w^T x)
    theta = theta.reshape(-1, 1) 

    # print("Shape of labeli:", labeli.shape)
    # print("Shape of theta:", theta.shape)

    # Compute the error function using Equation (2) (cross-entropy loss)
    error = -(1 / n_data) * np.sum(labeli * np.log(theta + 1e-10) + (1 - labeli) * np.log(1 - theta + 1e-10))

    # Compute the gradient of the error function using Equation (3)
    error_grad = (1 / n_data) * np.dot(train_data.T, (theta - labeli))

    return error, error_grad.flatten()

def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W 
     of Logistic Regression
     
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight 
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
         
     Output: 
         label: vector of size N x 1 representing the predicted label of 
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    data = np.insert(data, 0, 1, axis=1)
    posterior = np.dot(data,W)
    pred = sigmoid(posterior)
    prediction = np.argmax(pred, axis=1)
    label = np.reshape(prediction,(data.shape[0],1))

    return label

def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights_b: the weight vector of size (D + 1) x 10
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    train_data, labeli = args
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_feature + 1, n_class))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    # reshaping weights
    W = params.reshape(n_feature + 1,n_class)

    # Add bias term to train_data
    train_data = np.hstack((np.ones((n_data, 1)), train_data))

    # creating theta matrix
    theta = np.zeros((n_data, n_class))

    # Equation 5
    W_x_dot_product = np.dot(train_data, W)
    denominator = np.sum(np.exp(W_x_dot_product), axis=1).reshape(n_data, 1)
    theta_matrix = (np.exp(W_x_dot_product) / denominator)

    log_theta_matrix = np.log(theta_matrix)

    # The likelihood function with the negative logarithm Equation(7)
    error = (-1) * np.sum(np.sum(labeli * log_theta_matrix))

    # Equation(8)
    error_grad = (np.dot(train_data.T, (theta_matrix - labeli)))


    return error, error_grad.flatten()


def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    # Number of data points
    n_data = data.shape[0]

    # Add bias term to data
    data = np.hstack((np.ones((n_data, 1)), data))  # Add bias, shape becomes N x (D + 1)

    W_x_dot_product = np.dot(data, W)
    theta_matrix_sum = np.sum(np.exp(W_x_dot_product))
    posterior = np.exp(W_x_dot_product) / theta_matrix_sum

    for i in range(posterior.shape[0]):
        label[i] = np.argmax(posterior[i])
    label = label.reshape(label.shape[0], 1)


    return label

#########################################################################################################

"""
Record the total error with respect to each category in both training data and test data.
"""

def model_evaluation(W, train_data, train_label, test_data, test_label):
    """
    Evaluate the model on training and test data:
    - Print accuracy and total errors
    - Print category-wise errors
    - Return results for reporting
    """
    print("\n--- Model Evaluation ---")

    predicted_train = blrPredict(W, train_data)
    predicted_test = blrPredict(W, test_data)

    total_train_error = np.sum(predicted_train != train_label)
    total_test_error = np.sum(predicted_test != test_label)

    # accuracies
    train_accuracy = 100 * (1 - total_train_error / train_data.shape[0])
    test_accuracy = 100 * (1 - total_test_error / test_data.shape[0])

    print(f"\nOverall Results:")
    print(f"Training Error: {total_train_error}, Training Accuracy: {train_accuracy:.2f}%")
    print(f"Test Error: {total_test_error}, Test Accuracy: {test_accuracy:.2f}%")

    train_errors_by_each_category = np.zeros(n_class, dtype=int)
    test_errors_by_each_category = np.zeros(n_class, dtype=int)

    # Counting errors for each category
    for i in range(n_class):

        train_indices = (train_label == i).ravel()
        train_errors_by_each_category[i] = np.sum(predicted_train[train_indices] != train_label[train_indices])

        test_indices = (test_label == i).ravel()
        test_errors_by_each_category[i] = np.sum(predicted_test[test_indices] != test_label[test_indices])

    print("\nCategory-wise Training Errors:")
    for i in range(n_class):
        print(f"Category {i}: {train_errors_by_each_category[i]} errors")

    print("\nCategory-wise Test Errors:")
    for i in range(n_class):
        print(f"Category {i}: {test_errors_by_each_category[i]} errors")

    return {
        "train_error": total_train_error,
        "test_error": total_test_error,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "train_errors_by_each_category": train_errors_by_each_category,
        "test_errors_by_each_category": test_errors_by_each_category,
    }

#########################################################################################################


"""
Script for Logistic Regression
"""


train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()



# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros(n_feature + 1)  # initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 100}
for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))

# Find the accuracy on Training Dataset
print("\n\n--------------Logistic Regression-------------------")
predicted_label = blrPredict(W, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')


# Evaluate One-vs-All Logistic Regression Model
results_ova = model_evaluation(W, train_data, train_label, test_data, test_label)

print("\n--- Summary : One-vs-All Logistic Regression---")
print(f"Total Training Error: {results_ova['train_error']}")
print(f"Total Test Error: {results_ova['test_error']}")
print(f"Training Accuracy: {results_ova['train_accuracy']:.2f}%")
print(f"Testing Accuracy: {results_ova['test_accuracy']:.2f}%")



# Script for Extra Credit Part


# FOR EXTRA CREDIT ONLY
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class)).flatten() # flattened the initialweights array
opts_b = {'maxiter': 100}

args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

# Find the accuracy on Training Dataset

print("\n\n--------------Logistic Regression Multiclass Classification using SoftMax-------------------")

predicted_label_b = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')


# Evaluate Multi-Class Logistic Regression Model
results_mlr = model_evaluation(W_b, train_data, train_label, test_data, test_label)

print("\n--- Summary: Multi-Class Logistic Regression ---")
print(f"Total Training Error: {results_mlr['train_error']}")
print(f"Total Test Error: {results_mlr['test_error']}")
print(f"Training Accuracy: {results_mlr['train_accuracy']:.2f}%")
print(f"Testing Accuracy: {results_mlr['test_accuracy']:.2f}%")



"""
Script for Support Vector Machine
"""


print('\n\n--------------SVM-------------------\n\n')
##################
# YOUR CODE HERE #
##################


train_label = train_label.ravel()
validation_label = validation_label.ravel()
test_label = test_label.ravel()

# Function to train and evaluate SVM with different kernels and parameters
def train_and_evaluate_svm(kernel, C=1.0, gamma="scale", title="SVM"):

    start_time = time.time()
    model = SVC(kernel=kernel, C=C, gamma=gamma)
    model.fit(train_data, train_label)

    # Evaluate on training, validation, and testing datasets
    train_acc = accuracy_score(train_label, model.predict(train_data))
    validation_acc = accuracy_score(validation_label, model.predict(validation_data))
    test_acc = accuracy_score(test_label, model.predict(test_data))
    

    print(f"\n{title} Results:")
    print(f"Training Accuracy: {train_acc * 100:.2f}%")
    print(f"Validation Accuracy: {validation_acc * 100:.2f}%")
    print(f"Testing Accuracy: {test_acc * 100:.2f}%")
    print("--- %s seconds ---" % (time.time() - start_time))

    return train_acc, validation_acc, test_acc

# Experiment 1: SVM with linear kernel
print("\n\n------------- SVM with Linear Kernel -------------")
train_and_evaluate_svm(kernel="linear", title="SVM with Linear Kernel")

# Experiment 2: SVM with RBF kernel and gamma=1
print("\n\n------------- SVM with RBF Kernel (gamma=1) -------------")
train_and_evaluate_svm(kernel="rbf", gamma=1, title="SVM with RBF Kernel (gamma=1)")

# Experiment 3: SVM with RBF kernel and default gamma
print("\n\n------------- SVM with RBF Kernel (gamma=default) -------------")
train_and_evaluate_svm(kernel="rbf", gamma="scale", title="SVM with RBF Kernel (gamma=default)")

# Experiment 4: SVM with RBF kernel, default gamma, and varying C values
print("\n\n------------- SVM with RBF Kernel and Varying C Values -------------")

start_time = time.time()
C_values = [1] + list(range(10, 101, 10))
train_accuracies = []
validation_accuracies = []
test_accuracies = []

for C in C_values:
    print(f"\nEvaluating SVM with C={C}...")
    train_acc, validation_acc, test_acc = train_and_evaluate_svm(
        kernel="rbf", C=C, gamma="scale", title=f"SVM with RBF Kernel and C={C}"
    )
    train_accuracies.append(train_acc * 100)
    validation_accuracies.append(validation_acc * 100)
    test_accuracies.append(test_acc * 100)

print("--- %s seconds ---" % (time.time() - start_time))

# For verification accuracies are dumped into pickle File
# pickle.dump((train_accuracies, validation_accuracies, test_accuracies),open("SVM_params_different_cvalue.pickle","wb"))


# Plot accuracy vs. C values
plt.figure(figsize=(12, 7))
plt.plot(C_values, train_accuracies, marker="o", label="Training Accuracy", linestyle="--")
plt.plot(C_values, validation_accuracies, marker="o", label="Validation Accuracy", linestyle="-")
plt.plot(C_values, test_accuracies, marker="o", label="Testing Accuracy", linestyle=":")
plt.title("SVM Accuracy vs. C Values (RBF Kernel, gamma=default)", fontsize=14)
plt.xlabel("C Values", fontsize=12)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()





