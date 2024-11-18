import numpy as np
import matplotlib.pyplot as plt
from extrapolate import NearestNDInterpolator, LinearNDInterpolator, Poly2DExtrapolator
from geometry_tools import compute_offset, compute_curvature, compute_tangents


def build_grid(ncv,strech_factor,h_wm):
    nfa = ncv + 1
    y_cv = np.zeros((ncv))
    y_fa = np.zeros((nfa))

    y_fa[0] = 0
    tmp_inc = 1

    for ifa in range(1,nfa):
        y_fa[ifa] = y_fa[ifa-1] + tmp_inc
        y_cv[ifa-1] = 0.5* (y_fa[ifa] + y_fa[ifa-1])
        tmp_inc = tmp_inc*strech_factor

        y_max = np.max(y_cv)
        
    for ifa in range(nfa):
        y_fa[ifa] = y_fa[ifa]/y_max*h_wm
        
    for icv in range(ncv):
        y_cv[icv] = y_cv[icv]/y_max*h_wm
    return y_cv, y_fa



def solve_u_tw(ut, h_wm, mu_lam=1.78e-5, rho=1.225, ncv=30, stretch_factor=2):
    kappa = 0.41
    Aplus = 17
    tol = 1e-7
    max_iter = 100

    tau_wall_lam = mu_lam*ut/h_wm

    y_cv, y_fa = build_grid(ncv, stretch_factor, h_wm)
    nfa = len(y_fa)

    Iter = 0
    done = False
    tau_wall = 0.7
    
    while not done:
        tau_wall_prev = tau_wall
        mu_fa = mu_lam * np.ones((nfa))
        for ifa in range(1,nfa):
            nu = mu_lam#/rho
            utau = np.sqrt(tau_wall/rho)
            D = 1 - np.exp(-y_fa[ifa]*utau/nu/Aplus)
            D = D**2
            mut = rho*kappa*y_fa[ifa]*utau*D
            mu_fa[ifa] = mu_fa[ifa] + mut

        A = np.zeros((ncv,ncv))
        b = np.zeros((ncv))
        A[ncv-1,ncv-1] = 1
        b[ncv-1] = ut
        for icv in range(1,ncv-1):
            superdiag = mu_fa[icv+1]/(y_cv[icv+1]-y_cv[icv])
            subdiag = mu_fa[icv]/(y_cv[icv]-y_cv[icv-1])
            diag = -(superdiag+subdiag)
            A[icv,icv] = diag
            A[icv,icv+1] = superdiag
            A[icv,icv-1] = subdiag
            b[icv] = 0

        b[0]=0
        superdiag = mu_fa[1]/(y_cv[1]-y_cv[0])
        diag = -(superdiag+mu_fa[0]/(y_cv[0]-y_fa[0]))
        A[0,0] = diag
        A[0,1] = superdiag

        u_wm = np.linalg.solve(A,b)
        tau_wall = mu_fa[0]*( u_wm[0] - 0.0)/(y_cv[0]-y_fa[0])

        if abs((tau_wall - tau_wall_prev)/tau_wall_lam) < tol:
            y1_plus = y_cv[0]/nu*np.sqrt(tau_wall/rho)
            assert y1_plus < 0.1
            done = True

        if not done:
            Iter+=1
            
        if Iter>max_iter:
            break
        
    return u_wm,y_cv,tau_wall,Iter

#def solve_u_tw_V2():
    

def compute_Pw(u,n,kappa,rho,Pt):
    g = 1+n*kappa
    # if u[0]>1.9:
    #     print(g)
    return Pt - rho*np.trapz(u**2*kappa/g,x=n)

def compute_Pw_V2(xt,yt,ut,vt,Pt,profile,rho,mu_lam,kappa_t):
    Xt = np.array([xt,yt]).T
    Ut = np.array([ut,vt]).T
    distances = np.linalg.norm(profile[None,:,:]-Xt[:,None,:],axis=-1)
    h_wm = np.min(distances,axis=1)
    Xw   = profile[np.argmin(distances,axis=1)]
    Pw = np.zeros_like(Pt)
    tangents = NearestNDInterpolator(profile,compute_tangents(profile))(Xt)
    u_tang = np.einsum('Nd, Nd -> N',tangents,Ut)
    kappa_w = NearestNDInterpolator(profile,compute_curvature(profile))(Xt)
    # plt.plot(Xt[:,0],kappa_t)
    # plt.show()
    for i in range(len(Pw)):
        u,n = solve_u_tw(abs(u_tang[i]), h_wm[i], mu_lam, rho)[0:2]
        #u,n = np.array(20*[abs(u_tang[i])]), np.linspace(0,h_wm[i],20)
        kappa = 1 / (1/kappa_w[i]*(1-n/h_wm[i]) + 1/kappa_t[i]*(n/h_wm[i]))
        #kappa = kappa_w[i]
        Pw[i] = compute_Pw(u,n,kappa,rho,Pt[i])
    return Xw,Pw

def compute_Pw_V3(xt,yt,vel_fun,Pt,profile,rho,kappa_t):
    Xt = np.array([xt,yt]).T
    Ut = np.array([ut,vt]).T
    distances = np.linalg.norm(profile[None,:,:]-Xt[:,None,:],axis=-1)
    h_wm = np.min(distances,axis=1)
    Xw   = profile[np.argmin(distances,axis=1)]
    Pw = np.zeros_like(Pt)
    tangents = NearestNDInterpolator(profile,compute_tangents(profile))(Xt)
    u_tang = np.einsum('Nd, Nd -> N',tangents,Ut)
    kappa_w = NearestNDInterpolator(profile,compute_curvature(profile))(Xt)
    # plt.plot(Xt[:,0],kappa_t)
    # plt.show()
    for i in range(len(Pw)):
        u,n = solve_u_tw(abs(u_tang[i]), h_wm[i], mu_lam, rho)[0:2]
        #u,n = np.array(20*[abs(u_tang[i])]), np.linspace(0,h_wm[i],20)
        kappa = 1 / (1/kappa_w[i]*(1-n/h_wm[i]) + 1/kappa_t[i]*(n/h_wm[i]))
        #kappa = kappa_w[i]
        Pw[i] = compute_Pw(u,n,kappa,rho,Pt[i])
    return Xw,Pw


    




if __name__ == '__main__':

    x_c,C_p,C_f,H,Re_theta,Re_delta_star = tuple(np.loadtxt(r'data\Afoil_surface_data.dat',skiprows=1).T)
    plt.plot(x_c,Re_theta + Re_delta_star)
    plt.show()

    ut = 0.3
    h_wm = 0.1*0.085
    mu_lam = 1e-8
    rho = 2
    Pt = 1
    kappa = 0.0
    
    u,n = solve_u_tw(ut, h_wm, mu_lam, rho)[0:2]

    Pw = compute_Pw(u,n,kappa,rho,Pt)

    plt.plot(u,n)
    plt.show()
    