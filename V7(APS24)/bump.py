import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from mesh import Mesh2D, PIVmesh
from GFI import GFI
from extrapolate import NearestNDInterpolator, LinearNDInterpolator, Poly2DExtrapolator
from geometry_tools import compute_offset, compute_curvature, compute_tangents
import dill
from eqwm import solve_u_tw, compute_Pw
from matplotlib.tri import Triangulation, LinearTriInterpolator, CubicTriInterpolator, UniformTriRefiner

OFFSET = 4 #mm
CIRCLE_FIT_POINTS = 20
STREAMLINE_ODE_TIME_SPAN = 0.005
VELOCITY_NORMALIZATION_EXPONENT = 0.25


with open(r'data\bump_PIV.pkl', 'rb') as file:
    mesh = dill.load(file)


    
x_bump = np.linspace(-0.8,1,5000)
y_bump = 0.085*np.exp(-(x_bump/0.195)**2)

offset = 0.20*0.085
scalars = mesh.scalars.copy()
mesh_offset = Mesh2D(mesh.nodes,
                     alpha=None,
                     avoid_regions=[compute_offset(np.array([list(x_bump)+[1,-0.8],
                                                             list(y_bump)+[-5,-5]]).T,
                                                   -offset)],
                     scalars=scalars)



scalars['U'] *= 1 + np.random.randn(*scalars['U'].shape)*0.00
scalars['V'] *= 1 + np.random.randn(*scalars['V'].shape)*0.00

    
derivs = mesh_offset.derivatives(max_neighbors = 10,
                                 max_radius=np.inf,
                                 derivatives_order=2)
    
gradV = np.array([[ derivs['U_x'], derivs['U_y'] ],
                  [ derivs['V_x'], derivs['V_y'] ]])

V = np.array([ derivs['U_'], derivs['V_'] ]).T

divR = np.array([derivs['UU_x']*1 + derivs['UV_y'],
                 derivs['UV_x'] + derivs['VV_y']*1]).T 

#Niave Formulation
gradP = -2*np.einsum('NX, UXN -> NU', V, gradV) - 2*divR

u,v = mesh_offset.scalars['U'],mesh_offset.scalars['V']
ux,uy,vx,vy = derivs['U_x'],derivs['U_y'],derivs['V_x'],derivs['V_y']

kappa = (u**2*vx - v**2*uy + u*v*(vy-ux)) / (u**2+v**2)**1.5
mesh_offset.scalars['kappa'] = kappa
mesh_offset.scalars['Vmag'] = (u**2 + v**2)**0.5
con = 1
kappa_norm = np.sign(kappa)*(1-np.exp(-abs(kappa)*con))

# #Improved Formulation
# divR = np.array([derivs['UU_x']*0 + derivs['UV_y'],
#                  derivs['UV_x'] + derivs['VV_y']*0]).T 
# om = vx-uy
# #gradP = 2*np.array([om*v,-om*u]).T - 2*divR

# plt.figure('no_data')
# Vmag = (mesh_offset.scalars['U']**2 + mesh_offset.scalars['V']**2)**0.5
# var = om
# im = mesh_offset.color(var)

# plt.plot(x_bump,y_bump)
# plt.xlabel('x/L')
# plt.ylabel('y/L')
# plt.title('Streamline Curvature')
# plt.xlim(-0.8, 1)
# plt.ylim(0, 0.45)
# plt.gca().set_aspect('equal', adjustable='box')
# plt.colorbar(im,fraction=0.02, pad=0.04)
# #plt.savefig('no_data.png',dpi=500)

# plt.show()

##########################################################################################

xb = mesh_offset.boundary_face_centers()[0]
bl = np.argmin( xb[:,0] + 2*xb[:,1])
br = np.argmin(-xb[:,0] + 2*xb[:,1])
N  = xb.shape[0] 
inds = list(range(bl,N)) + list(range(0,br)) if bl>br else range(bl,br)

bump = np.array([x_bump,y_bump]).T



kappa = -NearestNDInterpolator(bump,compute_curvature(bump))(xb[inds])
# plt.figure()
# plt.plot(xb[inds,0],kappa)
# plt.plot(xb[inds,0],mesh_offset.boundary_scalars()[0]['kappa'][inds])
##plt.show()
###########################################################################################
plt.figure()

Pbt,P = GFI(mesh_offset,gradP,compute_interior=False)[0:2]
Pb = Pbt #- mesh_offset.boundary_scalars()[0]['U']**2 - mesh_offset.boundary_scalars()[0]['V']**2
##mesh_offset.color(P,on_volumes=True)
Xw = bump[np.argmin(np.linalg.norm(bump[None,:,:]-xb[:,None,:],axis=-1),axis=1)[inds]]

np.savez(r'figs\BUMP_naiveGradP_nn_0noise_20offset.npz',x=xb[:,0][inds],cp=Pb[inds]-Pb[inds][0])


# plt.figure()
# xv = mesh_offset.cell_centers()
# Pw_extrap = Poly2DExtrapolator(xv,P,
#                               max_neighbors=6,
#                               poly_order=1)(Xw)
# plt.plot(Xw[:,0],Pw_extrap-Pw_extrap[0])

# #np.savez('polyextrap_offset_20h_order_1_neighbors_6.npz',x=bump[::10],Cp=Pw_extrap-Pw_extrap[0])

# Pb = Pb[inds]
# Pb -= Pb[0]
# plt.plot(xb[inds][:,0],Pb)

# np.savez(r'figs\BUMP_naiveGradP_linear_0noise_20offset.npz',x=Xw[:,0],cp=Pw_extrap-Pw_extrap[0])

# x_act, Cp_act = tuple(np.loadtxt(r'data\SpeedBump-ReL-2M-Cp.dat',skiprows=2).T)
# plt.plot(x_act, Cp_act)

# plt.show()




# h_wm = np.min(np.linalg.norm(bump[None,:,:]-xb[:,None,:],axis=-1),axis=1)[inds]

# Pw = np.zeros_like(Pb)
# #ut = mesh_offset.boundary_scalars()[0]['Vmag'][inds]
# tangents = NearestNDInterpolator(bump,compute_tangents(bump))(xb)
# ut = np.einsum('Nd, dN -> N',
#                tangents,
#                [mesh_offset.boundary_scalars()[0]['U'],
#                 mesh_offset.boundary_scalars()[0]['V']])[inds]
# kappa_t = mesh_offset.boundary_scalars()[0]['kappa'][inds]
# kappa_w = -NearestNDInterpolator(bump,compute_curvature(bump))(xb[inds])
# kappa_w *= np.sign(kappa_t*kappa_w)


# for i in range(len(Pw)):
#     rho = 2
#     mu_lam = 1e-6
#     u,n = solve_u_tw(abs(ut[i]), h_wm[i], mu_lam, rho)[0:2]
#     kappa = kappa_w[i]#1 / (1/kappa_w[i]*(1-n/h_wm[i]) + 1/kappa_t[i]*(n/h_wm[i]))
#     Pw[i] = compute_Pw(u,n,-kappa,rho,Pb[i])
# plt.plot(Xw[:,0],Pw)

# #np.savez(r'figs\BUMP_improvedGradP_WM_0noise_20offset.npz',x=Xw[:,0],cp=Pw)
    
# #############################################################################################

# plt.show()
# plt.gcf().show()


