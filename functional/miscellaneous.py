import numpy as np
from functional.pseudohashimoto import compute_M_weighted
from tqdm import tqdm

def get_two_clusters_classification_performance(x):
    assert len(x) % 2 == 0, "Odd dimension of input"
    s1 = np.concatenate([np.zeros(len(x)//2), np.ones(len(x)//2)])
    perf = np.abs(x - s1).sum()/len(x)
    return min(perf, 1 - perf)


def get_eigvals_path(C, 
                      range_start = 0.1, range_end = 1., 
                      n_steps = 1000,
                      ts_return = True, 
                      complex = False):
    l = []
    ts = np.linspace(range_start, range_end, n_steps)
    for t in (ts):
        try:
            if complex:
                e = np.linalg.eigvals(compute_M_weighted(1/t, C))
            else:
                e = np.linalg.eigvalsh(compute_M_weighted(1/t, C))
        except:
            e = np.zeros_like(e)
        l.append(np.sort(e))
    l = np.vstack(l)
    if ts_return:
        return l, ts
    else:
        return l
    

def get_eivals_field(C,
                     real_range_start = 0.1, real_range_end = 1.,
                     imag_range_start = 0.1, imag_range_end = 1.,
                     n_steps_real = 200,
                     n_steps_imag = 200,
                     ts_return = True,
                     eigval = True):
    
    l = []
    real_ts = np.linspace(real_range_start, real_range_end, n_steps_real)
    imag_ts = np.linspace(imag_range_start, imag_range_end, n_steps_imag)

    ts = np.meshgrid(real_ts, imag_ts)

    ts = np.array([ts[0].flatten(),ts[1].flatten()]).T 
    ts_iter = ts[:,0] + ts[:, 1] * 1j
    for t in tqdm(ts_iter):
        if eigval:
            e = np.linalg.eigvals(compute_M_weighted(1/t, C))
            e = np.abs(e).min()
        else:
            e = np.linalg.det(compute_M_weighted(1/t, C))
            

        l.append(e)
    l = np.vstack(l)
    
    if ts_return:
        return l, ts
    else:
        return l
    

