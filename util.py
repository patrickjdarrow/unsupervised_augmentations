'''
TODO:
    - np arr < - > Torch tensor helpers
'''


def quick_log(fn, arr):
    '''
    Quick and dirty logger by date (learningnets\logs)
    '''
    
    import numpy as np
    import os
    from datetime import date
    
    date = date.today().strftime("%m-%d-%Y")
    
    dp = os.path.join('logs', date)
    os.makedirs(dp, exist_ok=True)
    
    if not fn.endswith('.npy'):
        fn += '.npy'
    
    fp = os.path.join(dp, fn)
    np.save(fp, arr)
    
    
def get_w2v(dataset, dim):
    '''
    W2V retrieval convenience fn
    '''
    import numpy as np
    
    return np.load(f'glove\\{dataset}_{dim}d.npy')