import torch
from sklearn.metrics import hamming_loss, f1_score, confusion_matrix


def compute_average_f1_score(predicted, truth, num_labels):
    assert isinstance(predicted, torch.Tensor)
    assert isinstance(truth, torch.Tensor)

    if num_labels > 1:
        weighted_avg_f1 = f1_score(truth, predicted, average='weighted')
        unweighted_avg_f1 = f1_score(truth, predicted, average='macro')
        all_f1 = f1_score(truth, predicted, average=None)
        return weighted_avg_f1, unweighted_avg_f1, all_f1
    else:
        avg_f1 = f1_score(truth, predicted, average='binary')
        all_f1 = f1_score(truth, predicted, average=None)
        return avg_f1, all_f1

def label_correctness(predictions, truths, num_labels=1):
    #counts up hamming distance and true accuracy
    # assert predictions.size(-1) == num_labels
    
    additional_scores = {}
    if len(predictions.size()) == 1:
        predictions = torch.sigmoid(predictions) > 0.5
    else:
        assert len(predictions.size()) == 2
        predictions = torch.max(predictions, dim=-1)[1]

    mcm = confusion_matrix(truths.squeeze().cpu(), predictions.squeeze().cpu())
    tn, fp, fn, tp = mcm.ravel()
    additional_scores['tn'] = tn
    additional_scores['tp'] = tp
    additional_scores['fn'] = fn
    additional_scores['fp'] = fp
    additional_scores['precision'] = tp / (tp + fp)	
    additional_scores['recall'] = tp / (tp + fn) 
    
    additional_scores['hamming_accuracy'] = 1 - hamming_loss(truths.squeeze().cpu(), predictions.squeeze().cpu())
    # print(additional_scores)
    if num_labels > 1:
        w_avg_f1, additional_scores['unweighted_f1'], additional_scores['all_f1s'] = compute_average_f1_score(truths.squeeze().cpu(), predictions.squeeze().cpu(), num_labels)
        return 1 - w_avg_f1, additional_scores
    else:
        w_avg_f1, additional_scores['all_f1s'] = compute_average_f1_score(truths.squeeze().cpu(), predictions.squeeze().cpu(), num_labels)
        return 1 - w_avg_f1, additional_scores
