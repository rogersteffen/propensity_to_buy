import scipy
from sklearn.metrics import precision_recall_curve, f1_score, roc_auc_score, classification_report
import numpy as np
import scikitplot as skplt
from typing import Optional



class Evaluator(object):

    def __init__(self, ground_truth: np.ndarray, probabilities: np.ndarray, predictions: Optional[np.ndarray]=None,
                 threshold: Optional[float]=None):
        '''
        This will print out the common evaluation metrics for binary classification.  This is not written (or at least tested)
        to support multi-class classification (todo: future enhancement perhaps).
        :param ground_truth: array of ground truth labels
        :param probabilities: array of probabilities ... supports both just positive class and both positive and negative class
        :param predictions: if provided, the predicted class
        :param threshold: if provided, the cut off used to make the predictions
        '''
        self.ground_truth = ground_truth
        self.predictions = predictions

        if type(probabilities[0]) == np.ndarray:
            self.positive_class_probabilities = probabilities[:,1]
            self.negative_class_probabilities = probabilities[:,0]
            self.probabilities = probabilities
        else:
            # todo: revist this ... but first check the class of output
            self.positive_class_probabilities = probabilities
            self.negative_class_probabilities = 1 - probabilities
            self.probabilities = np.stack((self.negative_class_probabilities, self.positive_class_probabilities), axis=1)


        if threshold is not None:
            self.optimal_threshold = threshold
        else: # compute the optimal threshold if not given
            precision, recall, thresholds = precision_recall_curve(self.ground_truth, self.positive_class_probabilities)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            self.optimal_threshold = round(thresholds[np.argmax(f1_scores)],2)
            print(f"Computing optimal Threshold: {self.optimal_threshold}")

        if predictions is not None:
            self.predictions = predictions
        else:
            print("Computing predictions...")
            self.predictions = [1 if prob >= self.optimal_threshold else 0 for prob in self.positive_class_probabilities]

        self.auc = roc_auc_score(self.ground_truth, self.positive_class_probabilities)
        self.f1_score = round(f1_score(self.ground_truth, self.predictions,labels=None, pos_label=1,
                                 average='binary', sample_weight=None, zero_division='warn'),2)


    def get_confusion_matrix(self, title='Confusion matrix'):

        return skplt.metrics.plot_confusion_matrix(self.ground_truth, self.predictions, normalize=False,
                                                   title = title)

    def plot_confusion_matrix(self, title='Confusion matrix'):

        self.get_confusion_matrix(title)

    def get_classification_report(self):

        return classification_report(self.ground_truth, self.predictions)

    def plot_roc(self, title='ROC'):
        # print(type(self.probabilities))
        # print(self.probabilities)
        skplt.metrics.plot_roc(self.ground_truth, self.probabilities, title=title)

    def display_results(self, dataset_title='Test Dataset'):

        print(f"AUC: {self.auc:.3f}")
        print(f"Optimal Threshold: {self.optimal_threshold}")
        print(f"Maximum F1 Score: {self.f1_score}")

        print(self.get_classification_report())
        self.plot_confusion_matrix()
        self.plot_roc()

def main():
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 1])
    y_pred_proba = np.array([0.1, 0.8, 0.3, 0.7, 0.5, 0.2, 0.6, 0.4, 0.9, 0.05])
    e = Evaluator(y_true, y_pred_proba)
    e.display_results()


if __name__ == "__main__":
    main()
