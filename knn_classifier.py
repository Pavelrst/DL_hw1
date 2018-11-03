import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

import cs236605.dataloader_utils as dataloader_utils
from . import dataloaders


class KNNClassifier(object):
    def __init__(self, k):
        self.k = k
        self.x_train = None
        self.y_train = None
        self.n_classes = None

    def train(self, dl_train: DataLoader):
        """
        Trains the KNN model. KNN training is memorizing the training data.
        Or, equivalently, the model parameters are the training data itself.
        :param dl_train: A DataLoader with labeled training sample (should
            return tuples).
        :return: self
        """

        x_train, y_train = dataloader_utils.flatten(dl_train)
        self.x_train = x_train
        self.y_train = y_train
        self.n_classes = len(set(y_train.numpy()))
        return self

    def predict(self, x_test: Tensor):
        """
        Predict the most likely class for each sample in a given tensor.
        :param x_test: Tensor of shape (N,D) where N is the number of samples.
        :return: A tensor of shape (N,) containing the predicted classes.
        """

        # Calculate distances between training and test samples
        dist_matrix = self.calc_distances(x_test)
        #print("distances:", dist_matrix)
        # TODO: Implement k-NN class prediction based on distance matrix.
        # For each training sample we'll look for it's k-nearest neighbors.
        # Then we'll predict the label of that sample to be the majority
        # label of it's nearest neighbors.

        n_test = x_test.shape[0]
        y_pred = torch.zeros(n_test, dtype=torch.int64)
        #print("predict1")
        for i in range(n_test):
            # TODO:
            # - Find indices of k-nearest neighbors of test sample i
            # - Set y_pred[i] to the most common class among them

            # ====== YOUR CODE: ======
            tens = torch.ones(())
            indices = tens.new_empty((self.k),dtype=torch.int64)


            min = np.inf
            index_min = 0
            # run on a row and find k minimums.
            # iterate k times.
            for k in range(len(indices)):
                # run on a column and find one minimum
                for j in range(dist_matrix.size(0)):
                    # current value
                    mat_value = dist_matrix[j][i]
                    if mat_value<min:
                        #print("mat_value<min")
                        index_min=j
                        min=dist_matrix[j][i]
                        #print("temporal min:",min)

                #print(k,"'st min in col",i,"is:",min)
                # here we found a min. add it to list:
                indices[k] = index_min


                # set as infinity the last min value.
                dist_matrix[index_min][i] = np.inf
                #print("found min:", min)
            #print("indices:", indices)



            classes = {}
            for ind in range(len(indices)):
                if(self.y_train[indices[ind]] in classes):
                    classes[self.y_train[indices[ind]]] += 1
                else:
                    classes[self.y_train[indices[ind]]] = 1
            max=0
            for key in classes:
                if classes[key]>max:
                    max=classes[key]
                    y_pred[i]=key
            # inverse = [(value, key) for key, value in classes.items()]
            # y_pred[i] = max(inverse)[1]
            # ========================
        #print("predict end")
        return y_pred

    def calc_distances(self, x_test: Tensor):
        """
        Calculates the L2 distance between each point in the given test
        samples to each point in the training samples.
        :param x_test: Test samples. Should be a tensor of shape (Ntest,D).
        :return: A distance matrix of shape (Ntrain,Ntest) where Ntrain is the
            number of training samples. The entry i, j represents the distance
            between training sample i and test sample j.
        """

        # TODO: Implement L2-distance calculation as efficiently as possible.
        # Notes:
        # - Use only basic pytorch tensor operations, no external code.
        # - No credit will be given for an implementation with two explicit
        #   loops.
        # - Partial credit will be given for an implementation with only one
        #   explicit loop.
        # - Full credit will be given for a fully vectorized implementation
        #   (zero explicit loops). Hint: Open the expression (a-b)^2.

        dists = torch.tensor([])
        # ====== YOUR CODE: ======
        #print("calc_distances started")
        #print("Calculate -2ab")
        ab_mat = torch.mm(self.x_train,torch.transpose(x_test, 0, 1))
        ab_mat_t = torch.transpose(ab_mat, 0, 1)
        ab2_mat = torch.mul(ab_mat,2)
        #print("ab2_mat size:",ab2_mat.size()) # good size
        # No reshape

        # Calculate a^2 (test)
        a_mat = x_test
        a_mat_t = torch.transpose(x_test, 0, 1)
        a_mat_sqr = torch.mm(a_mat,a_mat_t)
        #print("a_mat_sqr:", a_mat_sqr.size())  # good size ([1000, 1000])
        a_mat_diag = torch.diag(a_mat_sqr, 0) #this is a row vector.
        #print("a_mat_diag:", a_mat_diag.size())
        #a_mat_diag = torch.transpose(a_mat_diag, 0, -1) #now it is a row vector.
        #DO WE NEED TRANSPOSE HERE?
        a_mat_expand = a_mat_diag.expand(ab_mat.size(0),a_mat_sqr.size(0))
        #print("a_mat_expand:", a_mat_expand.size())

        # Calculate b^2 (train)
        b_mat = self.x_train
        b_mat_t = torch.transpose(self.x_train, 0, 1)
        b_mat_sqr = torch.mm(b_mat, b_mat_t)
        #print("b_mat_sqr:", b_mat_sqr.size()) # good size
        b_mat_diag = torch.diag(b_mat_sqr, 0)  # this is a row vector.
        #print("b_mat_diag:", b_mat_diag.size())
        b_mat_expand = b_mat_diag.expand(ab_mat.size(1),b_mat_sqr.size(0))  # expand the diag to size of matrix.
        b_mat_expand = torch.transpose(b_mat_expand, 0, 1)
        #print("b_mat_expand:", b_mat_expand.size())

        # Calculate the dist matrix by: a^2-2ab+b^2
        dists = a_mat_expand + b_mat_expand - ab2_mat
        #print("dists size:", dists.size())
        #print("calc_distances stopped")
        # ========================

        return dists


def accuracy(y: Tensor, y_pred: Tensor):
    """
    Calculate prediction accuracy: the fraction of predictions in that are
    equal to the ground truth.
    :param y: Ground truth tensor of shape (N,)
    :param y_pred: Predictions vector of shape (N,)
    :return: The prediction accuracy as a fraction.
    """
    assert y.shape == y_pred.shape
    assert y.dim() == 1

    # TODO: Calculate prediction accuracy. Don't use an explicit loop.

    accuracy = None
    # ====== YOUR CODE: ======
    y_vector = y.numpy()
    y_pred_vector = y_pred.numpy()
    #print("y:",y)
    #print("y_pred:", y_pred)
    truth_vector= np.equal(y_pred_vector,y_vector)
    #print("truth_vector:", truth_vector)
    trues=np.sum(truth_vector)
    print(trues)
    print(y.size(0))
    # accuracy = Fraction(trues,y.shape)
    accuracy=trues/y.size(0)
    print(accuracy)
    # ========================

    return accuracy


def find_best_k(ds_train: Dataset, k_choices, num_folds):
    """
    Use cross validation to find the best K for the kNN model.

    :param ds_train: Training dataset.
    :param k_choices: A sequence of possible value of k for the kNN model.
    :param num_folds: Number of folds for cross-validation.
    :return: tuple (best_k, accuracies) where:
        best_k: the value of k with the highest mean accuracy across folds
        accuracies: The accuracies per fold for each k (list of lists).
    """

    accuracies = []

    for i, k in enumerate(k_choices):
        model = KNNClassifier(k)

        # TODO: Train model num_folds times with different train/val data.
        # Don't use any third-party libraries.
        # You can use your train/validation splitter from part 1 (even if
        # that means that it's not really k-fold CV since it will be a
        # different split each iteration), or implement something else.

        # ====== YOUR CODE: ======
        raise NotImplementedError()
        # ========================

    best_k_idx = np.argmax([np.mean(acc) for acc in accuracies])
    best_k = k_choices[best_k_idx]

    return best_k, accuracies
