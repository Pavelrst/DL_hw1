import torch
from torch import Tensor
from torch.utils.data import DataLoader
from collections import namedtuple

from .losses import ClassifierLoss


class LinearClassifier(object):

    def __init__(self, n_features, n_classes, weight_std=0.001):
        """
        Initializes the linear classifier.
        :param n_features: Number or features in each sample.
        :param n_classes: Number of classes samples can belong to.
        :param weight_std: Standard deviation of initial weights.
        """
        self.n_features = n_features
        self.n_classes = n_classes

        # TODO: Create weights tensor of appropriate dimensions
        # Initialize it from a normal dist with zero mean and the given std.

        self.weights = None
        # ====== YOUR CODE: ======
        # n_features contains weights and bias
        # 𝑊 is of shape (n_features)×n_classes contains weights and bias
        matrix_size = [n_features,n_classes]
        self.weights = torch.empty(matrix_size, requires_grad=True)
        torch.nn.init.normal_(self.weights, mean=0, std=weight_std)
        # ========================

    def predict(self, x: Tensor):
        """
        Predict the class of a batch of samples based on the current weights.
        :param x: A tensor of shape (N,n_features) where N is the batch size.
        :return:
            y_pred: Tensor of shape (N,) where each entry is the predicted
                class of the corresponding sample. Predictions are integers in
                range [0, n_classes-1].
            class_scores: Tensor of shape (N,n_classes) with the class score
                per sample.
        """

        # TODO: Implement linear prediction.
        # Calculate the score for each class using the weights and
        # return the class y_pred with the highest score.

        y_pred, class_scores = None, None
        # ====== YOUR CODE: ======
        class_scores = torch.mm(x, self.weights)
        y_pred = torch.topk(class_scores, 1, dim=1)[1]
        y_pred = torch.transpose(y_pred, 0, 1)
        # ========================

        return y_pred, class_scores

    @staticmethod
    def evaluate_accuracy(y: Tensor, y_pred: Tensor):
        """
        Calculates the prediction accuracy based on predicted and ground-truth
        labels.
        :param y: A tensor of shape (N,) containing ground truth class labels.
        :param y_pred: A tensor of shape (N,) containing predicted labels.
        :return: The accuracy in percent.
        """

        # TODO: calculate accuracy of prediction.
        # Use the predict function above and compare the predicted class
        # labels to the ground truth labels to obtain the accuracy (in %).
        # Do not use an explicit loop.

        acc = None
        # ====== YOUR CODE: ======
        false_vector = torch.ne(y_pred, y)
        false = torch.sum(false_vector)
        false = int(false) / y.size(0)
        acc = 1 - false
        # ========================

        return acc * 100

    def train(self,
              dl_train: DataLoader,
              dl_valid: DataLoader,
              loss_fn: ClassifierLoss,
              learn_rate=0.1, weight_decay=0.001, max_epochs=100):

        Result = namedtuple('Result', 'accuracy loss')
        train_res = Result(accuracy=[], loss=[])
        valid_res = Result(accuracy=[], loss=[])

        print('Training', end='')
        for epoch_idx in range(max_epochs):
        #for epoch_idx in range(3):

            # TODO: Implement model training loop.
            # At each epoch, evaluate the model on the entire training set
            # (batch by batch) and update the weights.
            # Each epoch, also evaluate on the validation set.
            # Accumulate average loss and total accuracy for both sets.
            # The train/valid_res variables should hold the average loss and
            # accuracy per epoch.
            #
            # Don't forget to add a regularization term to the loss, using the
            # weight_decay parameter.

            total_correct = 0
            average_loss = 0

            # ====== YOUR CODE: ======
            print("epoch:",epoch_idx)
            # TODO:
            # evaluate the model on the entire training set
            # (batch by batch) and update the weights.

            # TODO: get train a batch - HOW get batches and iterate trough them?
            # x - samples.
            # y - truth labels.
            import cs236605.dataloader_utils as dataloader_utils
            x_train, y_train = dataloader_utils.flatten(dl_train)
            print("x_train size",x_train.size())

            # TODO: predict(batch) and get a:
            # x_scores - scores matrix.
            # y_predicted - predicted labels.
            y_predicted, x_scores = self.predict(x_train)
            accuracy_train = self.evaluate_accuracy(y_train, y_predicted)
            # TODO: initialize loss with params above.
            train_loss = loss_fn.loss(x_train, y_train, x_scores, y_predicted)

            # TODO: get validation set.
            x_valid, y_valid = dataloader_utils.flatten(dl_valid)

            # TODO: predict and get avg_loss + accuracy.
            y_predicted_valid, x_scores_valid = self.predict(x_valid)
            accuracy_valid = self.evaluate_accuracy(y_valid, y_predicted_valid)
            valid_loss = loss_fn.loss(x_valid, y_valid, x_scores_valid, y_predicted_valid)

            # TODO: calc grad of the loss.
            loss_grad = loss_fn.grad()
            # TODO: add a weight_decay to the loss grad
            loss_grad = loss_grad + torch.mul(loss_grad, weight_decay)
            # TODO: gradient descent step.
            grad_step = torch.mul(loss_grad, learn_rate)
            self.weights = self.weights - grad_step

            # TODO: Accumulate average loss and total accuracy for both sets.
            # Accumulate average loss and total accuracy for both sets.
            # The train/valid_res variables should hold the average loss and
            # accuracy per epoch.
            train_res.loss.append(train_loss)
            train_res.accuracy.append(accuracy_train)
            valid_res.loss.append(valid_loss)
            valid_res.accuracy.append(accuracy_valid)

            # ========================
            print('.', end='')

        print('')
        return train_res, valid_res

    def weights_as_images(self, img_shape, has_bias=True):
        """
        Create tensor images from the weights, for visualization.
        :param img_shape: Shape of each tensor image to create, i.e. (C,H,W).
        :param has_bias: Whether the weights include a bias component
            (assumed to be at the end).
        :return: Tensor of shape (n_classes, C, H, W).
        """

        # TODO: Convert the weights matrix into a tensor of images.
        # The output shape should be (n_classes, C, H, W).

        # ====== YOUR CODE: ======
        raise NotImplementedError()
        # ========================

        return w_images
