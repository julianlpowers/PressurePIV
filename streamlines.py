import scipy as sp
import numpy as np

def fit_circle(x, y, threshold=0.99):
    def calc_R(xc, yc):
        return np.sqrt((x - xc)**2 + (y - yc)**2)

    def f(c):
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    def calculate_r2(Ri, R):
        ss_res = np.sum((Ri - R)**2)
        ss_tot = np.sum((Ri - np.mean(Ri))**2)
        return 1 - (ss_res / ss_tot)

    center_estimate = np.mean(x), np.mean(y)
    center = sp.optimize.least_squares(f, center_estimate).x
    Ri = calc_R(*center)
    R = Ri.mean()
    r2 = calculate_r2(Ri, R)
    #print(r2)

    x_new,y_new = x.copy(), y.copy()
    while r2 < threshold and len(x_new) > 5:
        x_new = x_new[1:-1]
        y_new = y_new[1:-1]
        center_estimate = np.mean(x_new), np.mean(y_new)
        center = sp.optimize.least_squares(f, center_estimate).x
        Ri = calc_R(*center)
        R = Ri.mean()
        r2 = calculate_r2(Ri, R)

    return center, R



def f(t,y):
    return vel_interp(y.T) * (np.linalg.norm(y.T,axis=-1) > 25.4e-3)[:,...]

def central_ivp(f,span=0.2,y0=[0,0],num_eval_points=100):
    sol_forward = sp.integrate.solve_ivp(f,
                                         t_span=[0,span/2],
                                         y0=y0,
                                         t_eval=np.linspace(0,span/2,num_eval_points//2),
                                         vectorized=True
    )
    sol_backward = sp.integrate.solve_ivp(lambda t,y: -f(t,y),
                                          t_span=[0, span/2], 
                                          y0=y0, 
                                          t_eval=np.linspace(0,span/2,num_eval_points//2),
                                          vectorized=True
    )
    
    sol = {}
    sol['t'] = np.concatenate((-sol_backward.t[::-1], sol_forward.t[1:]))
    sol['y'] = np.concatenate((sol_backward.y[:,::-1], sol_forward.y[:,1:]), axis=1)
    sol['success'] = sol_forward.success and sol_backward.success
    return sol

