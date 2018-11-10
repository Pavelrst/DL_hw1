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
        y_pred=y_pred[0]
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
            #print("epoch:",epoch_idx)

            import cs236605.dataloader_utils as dataloader_utils

            # Iterate trough train batches and do GD step for each batch
            num_of_batches = 0
            for (x_train, y_train) in dl_train:
                num_of_batches += 1

                # Calc batch loss and accuracy and accumulate them.
                y_predicted, x_scores = self.predict(x_train)
                batch_accuracy = self.evaluate_accuracy(y_train, y_predicted)
                batch_loss = loss_fn.loss(x_train, y_train, x_scores, y_predicted)
                average_loss += batch_loss
                total_correct += batch_accuracy

                # Calc the grad of loos, add Regularization factor, GD step.
                loss_grad = loss_fn.grad()
                loss_grad += torch.mul(loss_grad, weight_decay)
                grad_step = torch.mul(loss_grad, learn_rate)
                self.weights = self.weights - grad_step

            # Calculate accuracy and loss on validation set
            x_valid, y_valid = dataloader_utils.flatten(dl_valid)
            y_predicted_valid, x_scores_valid = self.predict(x_valid)
            accuracy_valid = self.evaluate_accuracy(y_valid, y_predicted_valid)
            valid_loss = loss_fn.loss(x_valid, y_valid, x_scores_valid, y_predicted_valid)

            # Calc avg loss and acc across all train batches.
            # Append train/valid loss and acc to lists.
            average_loss = average_loss/num_of_batches
            total_correct = total_correct/num_of_batches
            train_res.loss.append(average_loss)
            train_res.accuracy.append(total_correct)
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

        # get only the data, not all the gradient features.
        # Otherwise I't wont be able to convert to numpy
        # Remove biases if needed
        if has_bias == False:
            weights = self.weights.data
        else:
            num_of_classes = self.weights.size(1)
            num_of_fetures = self.weights.size(0)
            num_of_fetures -= 1
            weights = self.weights[:num_of_fetures, :num_of_fetures].data

        # Set the size of output Tensor C x Ch x H x W
        size = (num_of_classes,)
        size+=(img_shape)
        w_images = torch.empty(size)

        for my_class in range(num_of_classes):
            # take column from weights, reshape and add to output Tensor
            image_tensor = weights[:,my_class]
            image_tensor = torch.reshape(image_tensor, img_shape)
            w_images[my_class] = image_tensor
        # ========================

        return w_images
