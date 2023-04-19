from sklearn.metrics import classification_report, accuracy_score, f1_score


class Metrics: 
    def __init__(self, y_preds, y_trues, class_names, avg):
        self.y_preds = y_preds 
        self.y_trues = y_trues 
        self.class_names = class_names
        self.avg = self.avg 

    @property
    def get_classification_report(self):
        report = classification_report(self.y_true, self.y_preds, target_names=self.class_names)
        return report

    @property
    def f1score(self): 
        score = f1_score(self.y_true. self.y_preds, average=self.avg)
        return score 
        
    @property
    def accuracy(self):
        acc = accuracy_score(self.y_trues, y_preds)
        return acc 