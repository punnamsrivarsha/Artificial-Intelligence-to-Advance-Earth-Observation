class MetricsCalculator:
    @staticmethod
    def compute_iou(confusion_matrix):
        """
        Calculates IoU: Intersection / (Union)
        IoU = TP / (TP + FP + FN)
        """
        intersection = np.diag(confusion_matrix)
        ground_truth_set = confusion_matrix.sum(axis=1)
        predicted_set = confusion_matrix.sum(axis=0)
        union = ground_truth_set + predicted_set - intersection
        
        iou = intersection / (union + 1e-7)
        return iou, np.mean(iou)

    @staticmethod
    def compute_f1(confusion_matrix):
        # Implementation of F1 score from confusion matrix
        tp = np.diag(confusion_matrix)
        fp = confusion_matrix.sum(axis=0) - tp
        fn = confusion_matrix.sum(axis=1) - tp
        
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
        return f1