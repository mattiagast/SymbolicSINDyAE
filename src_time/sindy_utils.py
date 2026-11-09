import numpy as np
from scipy.special import binom
from scipy.integrate import odeint


def library_size(n, poly_order, use_sine=False, include_constant=True):
    l = 0
    for k in range(poly_order+1):
        l += int(binom(n+k-1,k))
    if use_sine:
        l += 1
    if not include_constant:
        l -= 1
    return l


def sindy_library(X, poly_order, include_sine=False, t=None):
    m,n = X.shape
    l = library_size(n, poly_order, include_sine, True)
    library = np.ones((m,l))
    index = 1

    for i in range(n):
        library[:,index] = X[:,i]
        index += 1

    if poly_order > 1:
        for i in range(n):
            for j in range(i,n):
                library[:,index] = X[:,i]*X[:,j]
                index += 1

    if poly_order > 2:
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n):
                    library[:,index] = X[:,i]*X[:,j]*X[:,k]
                    index += 1

    if poly_order > 3:
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n):
                    for q in range(k,n):
                        library[:,index] = X[:,i]*X[:,j]*X[:,k]*X[:,q]
                        index += 1
                    
    if poly_order > 4:
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n):
                    for q in range(k,n):
                        for r in range(q,n):
                            library[:,index] = X[:,i]*X[:,j]*X[:,k]*X[:,q]*X[:,r]
                            index += 1

    if include_sine:
        if t is None:
            # Fallback per sicurezza o debug
            library[:, index] = 0 
        else:
            # Assicuriamoci che t sia un array della stessa lunghezza di X
            t = np.array(t).reshape((m,))
            # library[:, index] = np.cos(5.0 * t)
            library[:, index] = np.cos(4.998442071146355 * t)
        index += 1

    return library


def sindy_fit(RHS, LHS, coefficient_threshold):
    m,n = LHS.shape
    Xi = np.linalg.lstsq(RHS,LHS, rcond=None)[0]
    
    for k in range(10):
        small_inds = (np.abs(Xi) < coefficient_threshold)
        Xi[small_inds] = 0
        for i in range(n):
            big_inds = ~small_inds[:,i]
            if np.where(big_inds)[0].size == 0:
                continue
            Xi[big_inds,i] = np.linalg.lstsq(RHS[:,big_inds], LHS[:,i], rcond=None)[0]
    return Xi


def sindy_simulate(x0, t, Xi, poly_order, include_sine):
    m = t.size
    n = x0.size
    
    # La funzione f ora deve accettare t e passarlo a sindy_library
    # Nota: odeint passa (x, t), solve_ivp passa (t, x)
    def f(x, t_val):
        # x.reshape((1,n)) per avere la riga singola per sindy_library
        # t_val è lo scalare del tempo corrente fornito dall'integratore
        theta = sindy_library(np.array(x).reshape((1, n)), poly_order, include_sine, t=t_val)
        return np.dot(theta, Xi).reshape((n,))

    x = odeint(f, x0, t)
    return x

