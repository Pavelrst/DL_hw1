import numpy as np
import torch
from torch import Tensor
import matplotlib.pyplot as plt # need to delete
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

            # get indices of k smallest vals in matrix column
            curr_test_vec_dist = dist_matrix[:,i]
            k_sorted_idxs = np.argpartition(curr_test_vec_dist, self.k)
            indices = k_sorted_idxs[:self.k]

            classes = {}
            for ind in range(len(indices)):
                key = int(self.y_train[indices[ind]])
                #print("key:",key)
                if(key in classes):
                    #print("we are in if")
                    classes[key] += 1
                else:
                    #print("we are in else")
                    classes[key] = 1
                #print("temp classes:", classes)
            max=0
            #print("curr classes:",classes)
            for key in classes:
                if classes[key]>max:
                    max=classes[key]
                    y_pred[i]=key
            # inverse = [(value, key) for key, value in classes.items()]
            # y_pred[i] = max(inverse)[1]
            # ========================
        #print("predict end")
        #print("predict: ",y_pred)
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
    #print(trues)
    #print(y.size(0))
    # accuracy = Fraction(trues,y.shape)
    accuracy=trues/y.size(0)
    #print(accuracy)
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

        validation_ratio = 1/(num_folds-1)

        len1 = int(len(ds_train) * validation_ratio)
        len2 = int(len(ds_train) - len1)

        accuracy_of_all_folds = 0
        accuracy_fold_list = list()
        # we need to split and train num_folds times
        for fold in range(num_folds):


            temp_ds_train, temp_ds_valid = torch.utils.data.random_split(ds_train,[len1,len2])
            #print("ds_train len",len(temp_ds_train))
            #print("ds_valid len",len(temp_ds_valid))

            dl_train = torch.utils.data.DataLoader(temp_ds_train, shuffle=True)
            dl_valid = torch.utils.data.DataLoader(temp_ds_valid, shuffle=True)

            # now we need to train the model
            model.train(dl_train)
            # now validate:
            # predict validation data


            # get the truth labels, separate data form labels
            data_valid, labels_valid = dataloader_utils.flatten(dl_valid)

            pred = model.predict(data_valid)
            curr_accuracy = accuracy(labels_valid, pred)
            #print("current fold-",fold,"k-",k,"  accuracy", curr_accuracy)
            accuracy_fold_list.append(curr_accuracy)

            accuracy_of_all_folds += curr_accuracy / num_folds

        accuracies.append(accuracy_fold_list)
        # ========================

    best_k_idx = np.argmax([np.mean(acc) for acc in accuracies])
    best_k = k_choices[best_k_idx]

    return best_k, accuracies
