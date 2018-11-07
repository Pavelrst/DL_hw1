import abc
import torch


class ClassifierLoss(abc.ABC):
    """
    Represents a loss function of a classifier.
    """

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    @abc.abstractmethod
    def loss(self, *args, **kw):
        pass

    @abc.abstractmethod
    def grad(self):
        """
        :return: Gradient of the last calculated loss w.r.t. model
            parameters, as a Tensor of shape (D, C).
        """
        pass


class SVMHingeLoss(ClassifierLoss):
    def __init__(self, delta=1.0):
        self.delta = delta
        self.grad_ctx = {}

    def loss(self, x, y, x_scores, y_predicted):
        """
        Calculates the loss for a batch of samples.

        :param x: Batch of samples in a Tensor of shape (N, D).
        :param y: Ground-truth labels for these samples: (N,)
        :param x_scores: The predicted class score for each sample: (N, C).
        :param y_predicted: The predicted class label for each sample: (N,).
        :return: The classification loss as a Tensor of shape (1,).
        """

        assert x_scores.shape[0] == y.shape[0]
        assert y.dim() == 1

        # TODO: Implement SVM loss calculation based on the hinge-loss formula.
        #
        # Notes:
        # - Use only basic pytorch tensor operations, no external code.
        # - Partial credit will be given for an implementation with only one
        #   explicit loop.
        # - Full credit will be given for a fully vectorized implementation
        #   (zero explicit loops).
        #   Hint: Create a matrix M where M[i,j] is the margin-loss
        #   for sample i and class j (i.e. s_j - s_{y_i} + delta).

        loss = None
        # ====== YOUR CODE: ======
        # init margin_loss_matrix by delta and x_scores
        margin_loss_matrix = torch.add(x_scores, self.delta)

        # create tensor (1,2,3,4....999)
        inds = torch.tensor(range(x_scores.size(0)))

        true_scores = x_scores[inds,y]
        true_scores_mat = true_scores.expand(x_scores.size(1),x_scores.size(0))
        true_scores_mat = torch.transpose(true_scores_mat,0,1)

        margin_loss_matrix = margin_loss_matrix.sub(true_scores_mat)
        # set to zero negative values
        margin_loss_matrix[margin_loss_matrix<0]=0
        # finally we have the margin loss matrix
        # set to zero true scores
        margin_loss_matrix[inds,y]=0

        loss = torch.sum(margin_loss_matrix)/x_scores.size(0)
        # ========================

        # TODO: Save what you need for gradient calculation in self.grad_ctx
        # ====== YOUR CODE: ======
        self.grad_ctx['margin_loss_matrix'] = margin_loss_matrix
        self.grad_ctx['x_samples'] = x
        self.grad_ctx['truth_labels'] = y
        # ========================

        return loss

    def grad(self):

        # TODO: Implement SVM loss gradient calculation
        # Same notes as above. Hint: Use the matrix M from above, based on
        # it create a matrix G such that X^T * G is the gradient.

        grad = None
        # ====== YOUR CODE: ======
        # M - margin_loss_matrix. (saved in grad ctx)

        # lets create a [0,1] mask from M matrix.
        # this way we can calc the dLi/dwj which is xi or 0.
        G_matrix = self.grad_ctx['margin_loss_matrix']
        truth_labels = self.grad_ctx['truth_labels']
        #print("G_matrix", G_matrix)
        # create [0,1] mask from M_matrix
        G_matrix[G_matrix > 0] = 1
        #print("G_matrix", G_matrix)
        #print("G_matrix.size:", G_matrix.size())
        # Create tensor of rows sum
        G_matrix_rows_sum = torch.sum(G_matrix, 1)
        #print("G_matrix_rows_sum.size:", G_matrix_rows_sum.size())
        #print("G_matrix_rows_sum:", G_matrix_rows_sum)

        inds = torch.tensor(range(truth_labels.size(0)))
        G_matrix[inds, truth_labels] = -G_matrix_rows_sum[inds]
        #print("G_matrix.size:", G_matrix.size())
        #print("G_matrix:", G_matrix)

        # Now lets do XT*G
        x_samples = self.grad_ctx['x_samples']
        x_samples = torch.transpose(x_samples, 0, 1)
        grad = torch.mm(x_samples, G_matrix)

        # Now 1/N the grad
        N = x_samples.size(1)
        grad = torch.div(grad, N)
        #print("grad:",grad)
        # ========================

        return grad
