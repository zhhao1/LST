import os, sys
import jiwer


def compute_single_metric(gt,pred,metric):
    if metric == 'cer':
        return jiwer.cer(gt,pred)
    elif metric == 'wer':
        return jiwer.wer(gt,pred)
    elif metric == 'mer':
        return jiwer.mer(gt,pred)
    elif metric == 'wil':
        return jiwer.wil(gt,pred)
    elif metric == 'wip':
        return jiwer.wip(gt,pred)
    else:
        raise KeyError("invalid metric: {} !".format(metric))

def compute_metrics(ground_truth:list,prediction:list,metrics:list)->dict:
    """compute the auto speech recognition (ASR) metrics, inlcuding:
    Character Error Rate (CER),
    Word Error Rate (WER), 
    Match Error Rate (MER), 
    Word Information Lost (WIL) and Word Information Preserved (WIP)

    Args:
        ground_truth (list): list of ground truth answer, e.g., ['apple','marry','mark twin']
        prediction (list): list of the prediction, e.g., ['appl','malli','mark twen']
        metrics (list): list of choices, i.e., ['cer','wer','mer','wil','wip']
    """   
    choices = ['cer','wer','mer','wil','wip']

    assert len(ground_truth) == len(prediction), 'length mis-match!'
    assert all([c in choices for c in metrics]), "metrics out of the pre-definition, i.e., ['cer','wer','mer','wil','wip']"

    results = dict([(c,0.0) for c in metrics])

    ## calculate the average value from all instances, traverse each metric
    for metric in metrics:
        score = compute_single_metric(ground_truth,prediction,metric)
        score = score * 100
        results[metric] = score
    
    return results


with open(os.path.join(sys.argv[1], 'hyp'), 'r') as f:
    hyp = f.readlines()
with open(os.path.join(sys.argv[1], 'ref'), 'r') as f:
    ref = f.readlines()

print(compute_metrics(ref,hyp,['wer']))
