import random
import numpy as np
import torch


def seed_all(seed,deterministic=False,threads=4):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark=not deterministic
    torch.use_deterministic_algorithms(deterministic)


def rng_state():
    state=np.random.get_state()
    return {'python':random.getstate(),'numpy':[state[0],state[1].tolist(),state[2],state[3],state[4]],
            'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    n=state['numpy']; np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] and torch.cuda.is_available(): torch.cuda.set_rng_state_all(state['cuda'])
