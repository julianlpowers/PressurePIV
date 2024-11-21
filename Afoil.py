import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from mesh import Mesh2D
from GFI import GFI
from streamlines import fit_circle, central_ivp
from tqdm import tqdm
from geometry_tools import compute_offset,contains_points
from eqwm import solve_u_tw
from extrapolate import Poly2DExtrapolator

######################################################################
#                             PARAMETERS                             #
######################################################################
if True:
    NOISE = 0.00
    RESOLUTION = '400x400'

    METHOD = 'raw'

    OFFSET_CALC_STREAMLINES = 3e-2
    OFFSET_AVOID_BAD_DATA = 1e-2 
    CIRCLE_FIT_POINTS = 20
    STREAMLINE_ODE_TIME_SPAN = 0.05
    VELOCITY_NORMALIZATION_EXPONENT = 1
    OMEGA_THRESHOLD = 100

    OFFSET_EXTRAPOLATION = 1e-2

    SAVE_TO_FILE = False
    SAVE_FILENAME = fr'figs\AFOIL_{METHOD}.npz'
    #SAVE_FILENAME = fr'figs\AFOIL_raw.npz'
    #SAVE_FILNAME = fr'figs\AFOIL_extrapolation_2mm.npz'


######################################################################
#                             READ DATA                              #
######################################################################
if print('Reading data...') or True:
    data = np.genfromtxt(fr'data\Afoil-{RESOLUTION}.csv', delimiter=',',skip_header=1)
    profile = np.genfromtxt(r'data\Afoil.dat', skip_header=1)[::-1]
    profile = profile[1:-1]
    x_taps, Cp_taps = tuple(np.loadtxt(r'data\Afoil_surface_data.dat',skiprows=1).T)[0:2]

    #x,y,rho,u,v,p,uu,vv,ww,_uv = tuple(data.T[0:10])
    rho,u,v,p,uu,vv,ww,_uv = tuple(data.T[0:8])
    x,y = tuple(data.T[-3:-1])

    uv = -1*_uv
    M2g = 0.15**2 * 1.4#0.0159138441085815*2 # = 2*(P/Pinf)_max = Minf^2 * gamma
    RT  = 1/M2g
    Cp = (p-1)/0.5/M2g

    if 'extrap' in METHOD:
        mask = ~contains_points(compute_offset(profile,OFFSET_EXTRAPOLATION),np.array([x,y]).T)
        mask *= contains_points(compute_offset(profile,10),np.array([x,y]).T)
    if 'raw' in METHOD:
        mask = ~contains_points(compute_offset(profile,0.002),np.array([x,y]).T)
        mask *= contains_points(compute_offset(profile,10),np.array([x,y]).T)
    if 'streamlines' in METHOD:
        mask = ~contains_points(profile,np.array([x,y]).T)

    mask *= (np.linalg.norm(np.array([x,y]).T - np.array([1,-0.01184]).T, axis=-1) > 0.01)
    mask *= (np.linalg.norm(np.array([x,y]).T - np.array([1.01188,-0.01251]).T, axis=-1) > 0.01)

    x,y,rho,u,v,p,uu,vv,ww,uv,Cp = [a[mask] for a in [x,y,rho,u,v,p,uu,vv,ww,uv,Cp]]

    #add noise
    np.random.seed(0)
    u *= 1+NOISE*np.random.randn(u.shape[0])
    v *= 1+NOISE*np.random.randn(v.shape[0])

    mesh = Mesh2D(np.array([x,y]).T,
                  alpha = None,
                  scalars = {'x':x, 'y':y, 'rho':rho, 'u':u, 'v':v, 'p':p, 'uu':uu, 'vv':vv, 'ww':ww, 'uv':uv, 'Cp':Cp},
                 )

   

    plt.figure(1) 
    Vmag = np.sqrt(u**2+v**2) 
    plt.plot(*tuple(profile.T),'k')
    mesh.color(Vmag)
    plt.colorbar()
    plt.title(r'$|V|/V_\infty$')
    skip=1
    plt.quiver(x[::skip],y[::skip],(u/Vmag)[::skip],(v/Vmag)[::skip])
    plt.show() 



######################################################################
#                             COMPUTE GRADP                          #
######################################################################
if print('Computing pressure gradient...') or True:
    derivs = mesh.derivatives(max_neighbors = 4,
                            max_radius=np.inf,
                            derivatives_order=2)

    u,v = mesh.scalars['u'], mesh.scalars['v']
    ux,uy,vx,vy = derivs['u_x'],derivs['u_y'],derivs['v_x'],derivs['v_y']
    uux,uvx,uvy,vvy = derivs['uu_x'],derivs['uv_x'],derivs['uv_y'],derivs['vv_y']

    om = abs(vx-uy)
    #divR  = np.array([uux+uvy, uvx+vvy]).T
    divR  = np.array([uvy, uvx]).T
    con = np.array([ux*u+uy*v,vx*u+vy*v]).T
    gradP = -con-divR
    gradP *= rho[:,None]

    # plt.figure(2)
    # mesh.color(om)
    # plt.colorbar()
    # plt.show()



######################################################################
#            STREAMLINE AND WALL CORRECTION FOR  GRADP               #
######################################################################
if 'streamlines' in METHOD and (print('Computing streamlines...') or True):
    U_normalized = u/Vmag**VELOCITY_NORMALIZATION_EXPONENT
    V_normalized = v/Vmag**VELOCITY_NORMALIZATION_EXPONENT
    vel_interp = sp.interpolate.LinearNDInterpolator(mesh.nodes, np.array([U_normalized,V_normalized]).T,fill_value=0)

    def f(t,y):
        return vel_interp(y.T)

    center_list = []
    radius_list = []
    center_perp_list = []
    radius_perp_list = []
    initial_point_list = []

    #compute streamlines for data semi-close to wall
    plot_tic = 0
    mask = contains_points(compute_offset(profile,OFFSET_CALC_STREAMLINES), mesh.nodes) & ~contains_points(compute_offset(profile,OFFSET_AVOID_BAD_DATA), mesh.nodes)  
    for i in tqdm(np.arange(len(gradP))[mask]):
        initial_point = mesh.nodes[i]
        initial_point_list.append(initial_point)
        Vmag_initial_point = Vmag[i]

        sol = central_ivp(f, 
                        span=STREAMLINE_ODE_TIME_SPAN, 
                        y0=initial_point, 
                        num_eval_points=CIRCLE_FIT_POINTS
        )

        x_sol, y_sol = (sol['y'])[:,~contains_points(compute_offset(profile,OFFSET_AVOID_BAD_DATA), sol['y'].T)]
        center, radius = fit_circle(x_sol, y_sol)
        center_list.append(center)
        radius_list.append(radius)
        nhat = (initial_point - center)/np.linalg.norm(initial_point - center)
        dP_dn = 1 / radius * Vmag_initial_point**2    

        sol_perp = central_ivp(lambda t,y: np.array([-f(t,y)[...,1],f(t,y)[...,0]]).T, 
                                span=STREAMLINE_ODE_TIME_SPAN, 
                                y0=initial_point, 
                                num_eval_points=CIRCLE_FIT_POINTS
        )
        x_sol_perp, y_sol_perp = (sol_perp['y'])[:,~contains_points(compute_offset(profile,OFFSET_AVOID_BAD_DATA), sol_perp['y'].T)]
        center_perp, radius_perp = fit_circle(x_sol_perp, y_sol_perp)
        center_perp_list.append(center_perp)
        radius_perp_list.append(radius_perp)
        shat = (initial_point - center_perp)/np.linalg.norm(initial_point - center_perp)
        dP_ds = 1 / radius_perp * Vmag_initial_point**2

        gradP[i] =  dP_dn * nhat + dP_ds * shat - divR[i] 

        if plot_tic % 10 == 0:
            plt.plot(*tuple(initial_point), 'bo')

            plt.plot(x_sol, y_sol, 'r')
            circle = plt.Circle(center, radius, color='b', fill=False, linestyle='--')
            plt.gca().add_artist(circle)
            

            plt.plot(x_sol_perp,y_sol_perp,'r')
            circle = plt.Circle(center_perp, radius_perp, color='b', fill=False, linestyle='--')
            plt.gca().add_artist(circle)

        plot_tic += 1

    #compute wall curvature
    for i in range(CIRCLE_FIT_POINTS//2 + 1 , len(profile)-CIRCLE_FIT_POINTS//2 - 1 , 10):
        center,radius = fit_circle(*tuple(profile[i-CIRCLE_FIT_POINTS//2:i+CIRCLE_FIT_POINTS].T))
        center_list.append(center)
        radius_list.append(radius)
        initial_point_list.append(profile[i])

    #use wall model and streamline interpolation for points very close to wall
    mask = contains_points(compute_offset(profile,OFFSET_AVOID_BAD_DATA),mesh.nodes) 
    for i in tqdm(np.arange(len(gradP))[mask]):
        initial_point = mesh.nodes[i]

        ut = sp.interpolate.NearestNDInterpolator(mesh.nodes[~mask],Vmag[~mask])(initial_point)
        if sp.interpolate.NearestNDInterpolator(mesh.nodes[~mask],om[~mask])(initial_point)   < OMEGA_THRESHOLD:
            Vmag_initial_point = ut
        else:
            #EQWM
            pt = sp.interpolate.NearestNDInterpolator(mesh.nodes[~mask],mesh.nodes[~mask])(initial_point)
            ht = np.min(np.linalg.norm(pt[None,:] - profile))
            u_wm,y_cv,tau_wall,Iter = solve_u_tw(ut,ht, rho=2, mu_lam=1/2.1e6)
            h = np.min(np.linalg.norm(initial_point[None,:]-profile))
            Vmag_initial_point = sp.interpolate.interp1d(y_cv,u_wm)(h)

        center = sp.interpolate.LinearNDInterpolator(initial_point_list,center_list)(initial_point)
        radius = sp.interpolate.LinearNDInterpolator(initial_point_list,radius_list)(initial_point) 
        nhat = (initial_point - center)/np.linalg.norm(initial_point - center)
        gradP[i] =  1 / radius * Vmag_initial_point**2 * nhat



######################################################################
#                       INTEGRATE TO COMPUTE Cp                      #
# ####################################################################
if print('Integrating pressure gradient...') or True:
    Pb,P = GFI(mesh, gradP, compute_interior = 'extrap' in METHOD, LOS_block=[np.array([[0,0],[1.5,0]])])[0:2]

    xb = mesh.boundary_face_centers()[0]
    profile_offset = compute_offset(profile, 0.03)
    br = np.argmin( np.linalg.norm(xb-profile_offset[0] ,axis=-1) )
    bl = np.argmin( np.linalg.norm(xb-profile_offset[-1],axis=-1) )
    N  = xb.shape[0] 
    inds = list(range(bl,N)) + list(range(0,br)) if bl>br else range(bl,br)

    rho_inf = 2
    q_inf = 1

    Cp_PIV = Pb[inds] * rho_inf / q_inf
    Cp_PIV = Cp_PIV - Cp_PIV[-1] + Cp_taps[-1]
    x_PIV = xb[inds,0]
    

    Xw = profile[np.argmin(np.linalg.norm(profile[None,:,:]-xb[:,None,:],axis=-1),axis=1)[inds]]
    if 'extrap' in METHOD:
        Cp_PIV = Poly2DExtrapolator(mesh.cell_centers(),
                                  P,
                                  max_neighbors=6,
                                  poly_order=1)(Xw)
        Cp_PIV *= rho_inf / q_inf
        Cp_PIV = Cp_PIV - Cp_PIV[-1] + Cp_taps[-1]
        x_PIV = Xw[:,0]  
    
    

######################################################################
#                             PLOT RESULTS                           #
######################################################################
if True:
    if SAVE_TO_FILE:
        np.savez(SAVE_FILENAME, x = x_PIV, Cp=Cp_PIV)

    plt.figure(2)

    x_taps, Cp_taps = x_taps[x_taps<0.95], Cp_taps[x_taps<0.95]
    x_PIV, Cp_PIV = x_PIV[x_PIV<0.95], Cp_PIV[x_PIV<0.95]
    Cp_PIV = Cp_PIV - Cp_PIV[0] + Cp_taps[0]

    plt.plot(x_taps,Cp_taps,label='exact')
    plt.plot(x_PIV,Cp_PIV,label='PIV')
    plt.xlabel('x/c'); plt.ylabel('Cp'); plt.title('A-Airfoil'); plt.legend() 
    
    plt.gca().invert_yaxis()
    plt.show()
