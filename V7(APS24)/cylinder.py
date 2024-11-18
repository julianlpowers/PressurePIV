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
    METHOD = 'streamlines_2mm'

    OFFSET_CALC_STREAMLINES = 2e-3
    OFFSET_AVOID_BAD_DATA = 1e-3 
    CIRCLE_FIT_POINTS = 20
    STREAMLINE_ODE_TIME_SPAN = 0.005
    VELOCITY_NORMALIZATION_EXPONENT = 1

    OFFSET_EXTRAPOLATION = 2e-3

    SAVE_TO_FILE = False
    SAVE_FILENAME = fr'figs\CYLINDER_{METHOD}.npz'
    #SAVE_FILENAME = fr'figs\CYLINDER_raw.npz'
    #SAVE_FILNAME = fr'figs\CYLINDER_extrapolation_2mm.npz'



######################################################################
#                             READ DATA                              #
######################################################################
if True:
    data = pd.read_excel(r'data\tunnel\10_8_24.xlsx')
    theta_taps = data['theta.1'].to_numpy()
    Cp_taps = data['Cp.1'].to_numpy()

    th = np.linspace(0,2*np.pi,1000)
    profile = 25.4e-3 * np.array([np.cos(th), np.sin(th)]).T

    data = pd.read_csv(r'data\tunnel\PIV\cylinder0001_2.csv', delimiter=';')
    #data = pd.read_csv(r'data\tunnel\PIV\cylinder_frontview_test0001.csv', delimiter=';')
    x = data['x [mm]'].to_numpy() * 1e-3
    y = (data['y [mm]'].to_numpy()+60) * 1e-3
    u = data['Velocity u [m/s]'].to_numpy()
    v = data['Velocity v [m/s]'].to_numpy()
    uu = data['Reynolds stress Rxx [(m/s)^2]'].to_numpy()
    uv = data['Reynolds stress Rxy [(m/s)^2]'].to_numpy()
    vv = data['Reynolds stress Ryy [(m/s)^2]'].to_numpy()
    DV = data['Uncertainty V [m/s]'].to_numpy()

    if 'extrap' in METHOD:
        mask = ~contains_points(compute_offset(profile,OFFSET_EXTRAPOLATION),np.array([x,y]).T)
        mask *= contains_points(compute_offset(profile,20e-3),np.array([x,y]).T)
    else:
        mask = ~contains_points(profile,np.array([x,y]).T)

    x,y,u,v,uu,uv,vv,DV = [a[mask] for a in [x,y,u,v,uu,uv,vv,DV]]
    mesh = Mesh2D(np.array([x,y]).T,
                  alpha = None,
                  scalars = {'x':x, 'y':y, 'u':u, 'v':v, 'uu':uu, 'uv':uv, 'vv':vv, 'DV':DV},
                 )


    plt.figure(1)
    Vmag = (u**2 + v**2)**0.5
    mesh.color(Vmag)
    plt.plot(*tuple(profile.T),'k',linewidth=1)
    plt.colorbar();plt.ylim([0,None]); plt.xlabel('x [m]'); plt.ylabel('y [m]'); plt.title('|V| [m/s]')

    plt.show()

    



######################################################################
#                             COMPUTE GRADP                          #
######################################################################
if True:
    derivs = mesh.derivatives(max_neighbors = 4,
                            max_radius=np.inf,
                            derivatives_order=2)

    u,v = mesh.scalars['u'], mesh.scalars['v']
    ux,uy,vx,vy = derivs['u_x'],derivs['u_y'],derivs['v_x'],derivs['v_y']
    uux,uvx,uvy,vvy = derivs['uu_x'],derivs['uv_x'],derivs['uv_y'],derivs['vv_y']

    om = abs(vx-uy)

    #divR = np.array([uux+uvy, uvx+vvy]).T
    divR = np.array([uvy, uvx]).T
    con = np.array([ux*u+uy*v,vx*u+vy*v]).T
    gradP = -con-divR

    # plt.figure(2)
    # mesh.color(om / (3.9/50.8e-3))
    # plt.colorbar()
    # plt.show()



######################################################################
#            STREAMLINE AND WALL CORRECTION FOR  GRADP               #
######################################################################
if 'streamlines' in METHOD:
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

        if plot_tic % 50 == 0:
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
        if sp.interpolate.NearestNDInterpolator(mesh.nodes[~mask],om[~mask])(initial_point)/ (3.9/50.8e-3) < 15:
            Vmag_initial_point = ut
        else:
            #EQWM
            pt = sp.interpolate.NearestNDInterpolator(mesh.nodes[~mask],mesh.nodes[~mask])(initial_point)
            ht = np.min(np.linalg.norm(pt[None,:] - profile))
            u_wm,y_cv,tau_wall,Iter = solve_u_tw(ut,ht)
            h = np.min(np.linalg.norm(initial_point[None,:]-profile))
            Vmag_initial_point = sp.interpolate.interp1d(y_cv,u_wm)(h)

        center = sp.interpolate.LinearNDInterpolator(initial_point_list,center_list)(initial_point)
        radius = sp.interpolate.LinearNDInterpolator(initial_point_list,radius_list)(initial_point) 
        nhat = (initial_point - center)/np.linalg.norm(initial_point - center)
        gradP[i] =  1 / radius * Vmag_initial_point**2 * nhat


    
######################################################################
#                       INTEGRATE TO COMPUTE Cp                      #
# ####################################################################
if True: 
    Pb,P = GFI(mesh,gradP,compute_interior='extrap' in METHOD)

    xb = mesh.boundary_face_centers()[0]
    bl = np.argmin( np.linalg.norm(xb-np.array([-0.025026,0.005197]) ,axis=-1) )
    br = np.argmin( np.linalg.norm(xb-np.array([ 0.024961,0.005579]) ,axis=-1) )
    N = xb.shape[0]
    inds = list(range(bl,N)) + list(range(0,br)) if bl>br else range(bl,br)



    q_inf = (1.53-0.65)*10
    rho_inf = 1.225

    Cp_PIV = Pb[inds] * rho_inf / q_inf
    Cp_PIV = Cp_PIV - Cp_PIV[-1] + Cp_taps[-1]
    theta_PIV = -180/np.pi * np.arctan2(xb[inds,1],xb[inds,0]) + 180


    Xw = profile[np.argmin(np.linalg.norm(profile[None,:,:]-xb[:,None,:],axis=-1),axis=1)[inds]]
    if 'extrap' in METHOD:
        Cp_PIV = Poly2DExtrapolator(mesh.cell_centers(),
                                    P,
                                    max_neighbors=6,
                                    poly_order=1)(Xw)
        Cp_PIV *= rho_inf / q_inf
        Cp_PIV = Cp_PIV - Cp_PIV[-1] + Cp_taps[-1]
        theta_PIV = -180/np.pi * np.arctan2(Xw[:,1],Xw[:,0]) + 180



######################################################################
#                             PLOT RESULTS                           #
######################################################################
if True:
    if SAVE_TO_FILE:
        np.savez(SAVE_FILENAME, theta = theta_PIV, Cp=Cp_PIV)

    plt.figure(2)
    plt.plot(theta_PIV, Cp_PIV, label='PIV + streamline correction')
    plt.errorbar(theta_taps, Cp_taps, yerr=0.03, fmt='o', mfc='none', barsabove=True, label='taps')

    plt.xlabel('theta [deg]'); plt.ylabel('Cp'); #plt.legend(['PIV + 0','PIV + wm','taps'])
    plt.title(fr'Cylinder, $Re_D=16500$, offset={OFFSET_CALC_STREAMLINES*1e3}mm')
    plt.show()