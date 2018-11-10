import numpy as np
import sklearn
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
from pandas import DataFrame
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, check_X_y


class LinearRegressor(BaseEstimator, RegressorMixin):
    """
    Implements Linear Regression prediction and closed-form parameter fitting.
    """

    def __init__(self, reg_lambda=0.1):
        self.reg_lambda = reg_lambda

    def predict(self, X):
        """
        Predict the class of a batch of samples based on the current weights.
        :param X: A tensor of shape (N,n_features_) where N is the batch size.
        :return:
            y_pred: np.ndarray of shape (N,) where each entry is the predicted
                value of the corresponding sample.
        """
        X = check_array(X)
        check_is_fitted(self, 'weights_')

        # TODO: Calculate the model prediction, y_pred

        y_pred = None
        # ====== YOUR CODE: ======
        y_pred = np.dot(X,self.weights_)
        # ========================

        return y_pred

    def fit(self, X, y):
        """
        Fit optimal weights to data using closed form solution.
        :param X: A tensor of shape (N,n_features_) where N is the batch size.
        :param y: A tensor of shape (N,) where N is the batch size.
        """
        X, y = check_X_y(X, y)

        # TODO: Calculate the optimal weights using the closed-form solution
        # Use only numpy functions.

        w_opt = None
        # ====== YOUR CODE: ======
        # w = (XTX+lI)^-1 * XTy = XTX_inv * XTy

        XTX = np.dot(np.transpose(X),X)

        I = np.eye(XTX.shape[0])
        XTX_reg = np.add(XTX,self.reg_lambda*I)

        XTX_inv = np.linalg.inv(XTX_reg)
        XTy = np.dot(np.transpose(X), y)
        w_opt = np.dot(XTX_inv, XTy)
        # ========================

        self.weights_ = w_opt
        return self

    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X)


class BiasTrickTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        :param X: A tensor of shape (N,D) where N is the batch size or of shape
            (D,) (which assumes N=1).
        :returns: A tensor xb of shape (N,D+1) where xb[:, 0] == 1
        """
        X = check_array(X)

        # TODO: Add bias term to X as the first feature.

        xb = None
        # ====== YOUR CODE: ======
        ones_size = X.shape[0]
        ones = np.ones((ones_size,1))
        xb = np.append(ones,X,1)
        # ========================

        return xb


class BostonFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Generates custom features for the Boston dataset.
    """
    def __init__(self, degree=2):
        self.degree = degree

        # TODO: Your custom initialization, if needed
        # Add any hyperparameters you need and save them as above
        # ====== YOUR CODE: ======
        #raise NotImplementedError()
        # ========================

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Transform features to new features matrix.
        :param X: Matrix of shape (n_samples, n_features_).
        :returns: Matrix of shape (n_samples, n_output_features_).
        """
        X = check_array(X)
        # check_is_fitted(self, ['n_features_', 'n_output_features_'])

        # TODO: Transform the features of X into new features in X_transformed
        # Note: You can count on the order of features in the Boston dataset
        # (this class is "Boston-specific"). For example X[:,1] is the second
        # feature ('ZN').

        X_transformed = None
        # ====== YOUR CODE: ======
        #print("X", X)
        poly = PolynomialFeatures(self.degree)
        X_transformed = poly.fit_transform(X[:,[1,2,3,5,6,7,8,9,10,11,12,13]])
        #print("X_transformed",X_transformed)
        # ========================

        return X_transformed


def top_correlated_features(df: DataFrame, target_feature, n=5):
    """
    Returns the names of features most strongly correlated (correlation is
    close to 1 or -1) with a target feature. Correlation is Pearson's-r sense.

    :param df: A pandas dataframe.
    :param target_feature: The name of the target feature.
    :param n: Number of top features to return.
    :return: A tuple of
        - top_n_features: Sequence of the top feature names
        - top_n_corr: Sequence of correlation coefficients of above features
        Both the returned sequences should be sorted so that the best (most
        correlated) feature is first.
    """

    # TODO: Calculate correlations with target and sort features by it

    # ====== YOUR CODE: ======
    feature_names = list(df.columns.values)
    target_vals_arr = df.loc[:, target_feature]
    tup_list = list()

    # iterate trough columns of dataframe (excluding the target feature)
    for name in feature_names:
        if(name != target_feature):
            feature_vals_arr = df.loc[:, name]
            corr_r = np.corrcoef(feature_vals_arr, target_vals_arr)
            corr_tup = (np.abs(corr_r[1][0]), name)
            tup_list.append(corr_tup)

    # Sort
    tup_list.sort(reverse=True)
    n_top = tup_list[:n]
    top_n_corr, top_n_features = zip(*n_top)
    top_n_corr = np.asarray(top_n_corr)
    top_n_features = np.asanyarray(top_n_features)

    # ========================

    return top_n_features, top_n_corr


def cv_best_hyperparams(model: BaseEstimator, X, y, k_folds,
                        degree_range, lambda_range):
    """
    Cross-validate to find best hyperparameters with k-fold CV.
    :param X: Training data.
    :param y: Training targets.
    :param model: sklearn model.
    :param lambda_range: Range of values for the regularization hyperparam.
    :param degree_range: Range of values for the degree hyperparam.
    :param k_folds: Number of folds for splitting the training data into.
    :return: A dict containing the best model parameters,
        with some of the keys as returned by model.get_params()
    """

    # TODO: Do K-fold cross validation to find the best hyperparameters
    #
    # Notes:
    # - You can implement it yourself or use the built in sklearn utilities
    #   (recommended). See the docs for the sklearn.model_selection package
    #   http://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
    # - If your model has more hyperparameters (not just lambda and degree)
    #   you should add them to the search.
    # - Use get_params() on your model to see what hyperparameters is has
    #   and their names. The parameters dict you return should use the same
    #   names as keys.
    # - You can use MSE or R^2 as a score.

    # ====== YOUR CODE: ======
    kf = sklearn.model_selection.KFold(n_splits=k_folds)
    best_params = 0
    min_mse = np.inf
    for curr_degree in degree_range:
        for curr_lambda in lambda_range:
            params = dict(linearregressor__reg_lambda=curr_lambda, bostonfeaturestransformer__degree=curr_degree)
            model.set_params(**params)
            mse = 0
            counter = 0
            for train_index, test_index in kf.split(X):
                counter = counter + 1
                model.fit(X[train_index],y[train_index])
                y_pred = model.predict(X[test_index])
                mse = mse + np.mean((y[test_index] - y_pred) ** 2)

            avg_mse = mse/counter
            print("avg_mse:", avg_mse, " labmda:", curr_lambda, " degree:", curr_degree)
            if  avg_mse < min_mse:
                best_params = params
                min_mse = avg_mse
    # ========================

    return best_params
