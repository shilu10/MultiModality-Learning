from sklearn.metrics import classification_report, accuracy_score, f1_score


class Metrics: 
    def __init__(self, y_preds, y_trues, class_names, avg):
        """
            this class has a multiple methods that has multiple evulation metrics for the evaulating the prediction data.
            Attrs:
                y_preds(type: np.array): prediction from the model.
                y_trues(type; np.array): Actual value(true value) of the dataset.
                class_names(type: List): list of classes.
                avg(type: str): Type of method for averaging the multiclass values.
        """
        self.y_preds = y_preds 
        self.y_trues = y_trues 
        self.class_names = class_names
        self.avg = self.avg 

    @property
    def get_classification_report(self):
        """
            returns the classiction report.
        """
        report = classification_report(self.y_true, self.y_preds, target_names=self.class_names)
        return report

    @property
    def f1score(self): 
        """
            returns the f1 score.
        """
        score = f1_score(self.y_true. self.y_preds, average=self.avg)
        return score 
        
    @property
    def accuracy(self):
        """
            returns the accuracy
        """
        acc = accuracy_score(self.y_trues, y_preds)
        return acc 