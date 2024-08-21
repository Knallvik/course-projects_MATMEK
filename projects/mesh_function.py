import numpy as np


def mesh_function(f, t):
    y = np.empty_like(t, dtype = float)
    for i, t_ in enumerate(t):
        y[i] = f(t_)
    return y

def func(t):
    if t>=0 and t<=3:
        return np.exp(-t)
    elif t>3 and t<=4:
        return np.exp(-3*t)
    else: print('outside of range')

def test_mesh_function():
    t = np.array([1, 2, 3, 4])
    f = np.array([np.exp(-1), np.exp(-2), np.exp(-3), np.exp(-12)])
    print(f)
    fun = mesh_function(func, t)
    print(fun)
    assert np.allclose(fun, f)
