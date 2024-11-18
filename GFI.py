import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import time
from mesh import Mesh2D
from symbolic import diff
from geometry_tools import line_of_sight
from tqdm import tqdm


def GFI(mesh,  gradP_in, D=2, Niter=15, compute_interior=False, eps=1e-10, gradP_on_cells=False, LOS_block=None):
    '''
    INPUTS:
    -----------------
    mesh                    : a 2D Mesh object
    gradP (Nv x D)          : list of pressure gradients
    D                       : dimension of problem (2 or 3)
    Niter                   : number of Jacobi iterations
    compute_interior        : whether to compute the pressure in the interior of the mesh as well
    eps                     : small number to bypass divide by zero errors
    gradP_on_cells          : whether gradP is defined for mesh nodes (False) or for cell centers (True)
    LOS_block [(N,2,D),...] : list of linestrings defining 'line-of-sight' influence. points on either side of linestrings do not influence one another

    OUTPUTS:
    -----------------
    Pb (Nb)          : list of boundary pressures
    P  (Nv)          : list of interior pressures
    K  (D x Nv x Nv) : effective processing matrix, K @ gradP -> P
    '''

    xb = np.concatenate(mesh.boundary_face_centers()) # (Nb x D) : list of boundary points
    dS = np.concatenate(mesh.boundary_face_normals()) # (Nb x D) : list of out-facing boundary surface normal vectors 
    xv = mesh.cell_centers()                          # (Nv x D) : list of volume center points
    dV = mesh.cell_volumes()                          # (Nv)     : list of element volumes

    print('Interpolating...')
    if gradP_on_cells:
        gradP = gradP_in
    else:
        gradP = sp.interpolate.LinearNDInterpolator(mesh.nodes, gradP_in)(xv)
    print('Done Interpolating')
    
    Nb = xb.shape[0]
    Nv = xv.shape[0]  
    
    
    Ab = np.zeros((D,Nb,Nv))            
    r = xv - xb[:,None]
    r_norm = r / (np.linalg.norm(r,axis=-1)**D)[:,:,None]
    Ab = -1/2/(D-1)/np.pi * np.einsum('kjd, j -> dkj ',r_norm,dV)
    if not LOS_block == None: 
        Ab*= line_of_sight(xb,xv,LOS_block)


    Bb = np.zeros((Nb,Nb))    
    dL = np.concatenate(mesh.boundary_face_parallels())
    r = xb - xb[:,None]

    r_hat = r / np.linalg.norm(r,axis=-1)[:,:,None]
    r_hat[~np.isfinite(r_hat)] = 0
    
    r_m = np.linalg.norm(r - dL/2,axis=-1)
    r_p = np.linalg.norm(r + dL/2,axis=-1)
    
    sin_m = 0.5 * np.einsum('Kkd, kd -> Kk', r_hat,dS) / r_m
    sin_p = 0.5 * np.einsum('Kkd, kd -> Kk', r_hat,dS) / r_p

    theta = np.arcsin(sin_p) + np.arcsin(sin_m)
    Bb = 1/2/(D-1)/np.pi * theta
    if not LOS_block == None: 
        Bb*= line_of_sight(xb,xb,LOS_block)
    
    

    J = np.eye(Nb)- 2*Bb
    G = np.einsum('dki, id -> k', Ab, gradP)

   
    Pb = 2*G
    invJ = np.eye(Nb)
    L = np.tril(J,k=-1)
    U = np.triu(J,k=1)
    
    for _ in range(Niter):
        Pb = 2*G - (L+U) @ Pb #J.diagonal() = ones
        invJ = np.eye(Nb) - (L+U) @ invJ
        Pb = Pb - np.mean(Pb)   #nudge Pb towards 0 average pressure
        invJ = (np.eye(Nb) - np.ones((Nb,Nb)) / Nb) @ invJ
        
    #Kb = np.einsum('kK, dKi -> dki', 2*invJ, Ab)

    
    
    ### ####################################################################################
            
    P = None
    K = None  
    
    if compute_interior:
        
        # A  = np.zeros((D,Nv,Nv))
        # r = xv-xv[:,None]
        # r_norm = r / (np.linalg.norm(r,axis=-1)**D + 1e-10)[:,:,None]
        # A = -1/2/(D-1)/np.pi * np.einsum('ijd, j -> dij ',r_norm,dV)
        # if not LOS_block == None: 
        #     A*= line_of_sight(xv,xv,LOS_block)

        A = np.zeros((D,Nv,Nv))
        for d in range(D):
            for i in tqdm(range(Nv)):
                ri = xv - xv[i] #rij = ri[j]
                ri_norm = ri / (np.linalg.norm(ri,axis=-1)**D + 1e-10)[:,None] #ri_norm[j,:] = rij/|rij|^D
                A[d,i,:] = -1/2/(D-1)/np.pi * ri_norm[:,d] * dV #A[d,i,j] = "__" * r_norm[i,j,d] * dV[j]


##        B  = np.zeros((Nv,Nb))
##        r = xb - xv[:,None]        
##        r_m = r - dL/2
##        r_p = r + dL/2
##        theta = np.arctan2(r_p[...,1],r_p[...,0]) - np.arctan2(r_m[...,1],r_m[...,0])
##        theta = abs(2*np.arcsin(np.sin(theta/2)))
##        B = 1/2/(D-1)/np.pi * theta

        B = np.zeros((Nv,Nb))    
        dL = np.concatenate(mesh.boundary_face_parallels())
        r = xb - xv[:,None]

        r_hat = r / np.linalg.norm(r,axis=-1)[:,:,None]
        r_hat[~np.isfinite(r_hat)] = 0
        
        r_m = np.linalg.norm(r - dL/2,axis=-1)
        r_p = np.linalg.norm(r + dL/2,axis=-1)
        
        sin_m = 0.5 * np.einsum('Kkd, kd -> Kk', r_hat,dS) / r_m
        sin_p = 0.5 * np.einsum('Kkd, kd -> Kk', r_hat,dS) / r_p

        theta = np.arcsin(sin_p) + np.arcsin(sin_m)
        B = 1/2/(D-1)/np.pi * theta

        #K = A + 2*np.einsum('ik, kK, dKj -> dij',B, invJ, Ab)

        P = np.einsum('dij, jd -> i',A,gradP) + np.einsum('ik, k -> i',B,Pb)
        
    return Pb,P
 

if __name__ == '__main__':
    
    def p(x,y):
        th = np.arctan2(y,x)
        r  = (x**2+y**2)**0.5
        p = 2/r**2*np.cos(2*th) - 1/r**4
        return p

    p_x = diff(p,'x')
    p_y = diff(p,'y')


    N=200

    x,y = 2*[np.linspace(-3,3,N)]
    x,y = np.meshgrid(x,y)
    x,y = x.flatten(),y.flatten()
    mask = x**2 + y**2 > 1
    x,y = x[mask], y[mask]
    nodes = np.array([x,y]).T

    
    mesh = Mesh2D(nodes,alpha=None)
    mesh.plot()
    
    gradP = np.array([p_x(x,y), p_y(x,y)]).T
    shat = np.array([-y,x]).T / np.linalg.norm(np.array([-y,x]).T,axis=-1)[:,None]

    gradP -= np.einsum('ni,ni -> n',gradP, shat)[:,None] * shat
    

    Pb,P = GFI(mesh, gradP,compute_interior=False)
    
    Pb_list = mesh.disconnect_boundary_scalar(Pb)
    is_hole = mesh.boundary_is_hole()
    circ_ind = [ind for ind in range(len(is_hole)) if  is_hole[ind]][0]
    
    Pb_circ = Pb_list[circ_ind]
    x_circ,y_circ = tuple(mesh.boundary_face_centers()[circ_ind].T)
    th_circ = np.arctan2(y_circ,x_circ)

    
    plt.figure()  
    plt.plot(Pb_circ - np.mean(Pb_circ))
    plt.plot(p(x_circ,y_circ)-np.mean(p(x_circ,y_circ)))


    # plt.figure()
    # mesh.color(P,on_volumes=True)
    # plt.colorbar()
    
    # plt.figure()
    # mesh.color(p)
    
    plt.show()

    

    



    
    

   

    
    





