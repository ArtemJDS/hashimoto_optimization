import numpy as np

def get_two_clusters_classification_performance(x):
    assert len(x) % 2 == 0, "Odd dimension of input"
    s1 = np.concatenate([np.zeros(len(x)//2), np.ones(len(x)//2)])
    perf = np.abs(x - s1).sum()/len(x)
    return min(perf, 1 - perf)


