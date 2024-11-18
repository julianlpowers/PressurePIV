from mesh import Mesh2D
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from geometry_tools import compute_curvature, compute_normals, compute_tangents
from scipy.spatial import cKDTree
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.special import factorial as fact


def Poly2DExtrapolator(nodes,p,max_neighbors=10, max_radius=np.inf, poly_order=2, deriv_query='_'):
    mesh = Mesh2D(nodes, scalars={'p':p})
    derivs = mesh.derivatives(max_neighbors=max_neighbors,
                              max_radius=np.inf,
                              derivatives_order=poly_order)
    x_query = deriv_query.count('x')
    y_query =  deriv_query.count('y')

    assert poly_order >= max(x_query,y_query)

    #TODO: make this faster by removing the for loop
    def interp(points):            
        P=[]
        for point in points:
            nearest_ind = np.argmin(np.linalg.norm(point-nodes,axis=-1))
            nearest_node = nodes[nearest_ind]
            dx,dy = tuple((point-nearest_node).T)

            value = 0
            for name, coeff in derivs.items():
                nx = name.split('_')[-1].count('x') 
                ny = name.split('_')[-1].count('y') 
                summand = coeff[nearest_ind]
                summand*= 0 if nx<x_query else fact(nx)/fact(nx-x_query) * dx**(nx-x_query)
                summand*= 0 if ny<y_query else fact(ny)/fact(ny-y_query) * dy**(ny-y_query)
                value += summand
            P.append(value)
                
        return np.array(P)
    
    return interp


NearestNDInterpolator = sp.interpolate.NearestNDInterpolator
LinearNDInterpolator = sp.interpolate.LinearNDInterpolator



class NearestTwoAverageInterpolator:
    def __init__(self, points, values):
        self.points = np.asarray(points)
        self.values = np.asarray(values)
        
        self.spline_x = InterpolatedUnivariateSpline(np.arange(len(points)), self.points[:, 0], k=1)
        self.spline_y = InterpolatedUnivariateSpline(np.arange(len(points)), self.points[:, 1], k=1)
        self.spline_val = InterpolatedUnivariateSpline(np.arange(len(points)), self.values, k=1)
        
        # Creating a kd-tree for efficient nearest-neighbor search
        spline_points = np.column_stack((self.spline_x(np.linspace(0, len(points)-1, 1000)),
                                         self.spline_y(np.linspace(0, len(points)-1, 1000))))
        self.tree = cKDTree(spline_points)
    
    def __call__(self, target_points):
        target_points = np.asarray(target_points)
        
        # Find the nearest points on the spline using kd-tree
        dist, idx = self.tree.query(target_points)
        t = np.linspace(0, len(self.points)-1, 1000)[idx]
        
        # Interpolate the value at the nearest point on the spline
        interpolated_values = self.spline_val(t)
        
        return interpolated_values


def TSLGradPExtrapolator(gradP_points,    gradP,
                         velocity_points=[0,0], velocity=0,
                         wall_shape=np.zeros((100,2)),
                         rho=2):
    
    velocity_func  = NearestNDInterpolator(velocity_points,velocity)
    curvature_func = NearestNDInterpolator(wall_shape,compute_curvature(wall_shape))
    normals_func   = NearestNDInterpolator(wall_shape,compute_normals(wall_shape))
    tangents_func  = NearestNDInterpolator(wall_shape,compute_tangents(wall_shape))
    gradP_func     = NearestNDInterpolator(gradP_points,gradP)
    

    def interp(points):
        velocities = velocity_func(points)
        curvatures = curvature_func(points)
        normals    = normals_func(points)
        tangents   = tangents_func(points)
        gradP      = gradP_func(points)

        tangent_velocities=np.einsum('Nd, Nd -> N', velocities, tangents)
    
        dp_ds = np.einsum('Nd, Nd -> N', gradP, tangents)
        dp_dn = -rho * tangent_velocities**2 * curvatures

        dp_dx = dp_dn * normals[:, 0] + dp_ds * tangents[:, 0]
        dp_dy = dp_dn * normals[:, 1] + dp_ds * tangents[:, 1]

        return np.array([dp_dx,dp_dy]).T
    
    return interp


def MixedGradPExtrapolator(gradP_points,    gradP,
                           velocity_points=[0,0], velocity=0,
                           wall_shape=np.zeros((100,2)),
                           rho=2,
                           wall_model_frac_func=lambda d: np.where(d>0.1,0,1) ):
    
    def interp(points):
        wall_distances = np.min(np.linalg.norm(points[:,None,:]-wall_shape[None,:,:],axis=-1),axis=-1)
        wall_model_frac = wall_model_frac_func(wall_distances)

        wall_interp = TSLGradPExtrapolator(gradP_points,gradP, velocity_points, velocity, wall_shape=wall_shape,rho=rho)
        outer_interp = NearestNDInterpolator(gradP_points,gradP)
        
        return wall_model_frac[:,None] * wall_interp(points) + (1-wall_model_frac)[:,None] * outer_interp(points)

    return interp
        
        
    
    



if __name__ == '__main__':

    N=100

    x,y = [a.flatten() for a in np.meshgrid(*[np.linspace(-3,3,N)]*2)]
    x,y = [a[x**2+y**2 > 1] for a in (x,y)]
    nodes = np.array([x,y]).T

    p=np.sin(x)*np.cos(y)

    interp = Poly2DExtrapolator(nodes,p,
                                max_neighbors=15,
                                max_radius=np.inf,
                                poly_order=4,
                                deriv_query='_x')
    

    x,y = [a.flatten() for a in np.meshgrid(*[np.linspace(-3,3,N)]*2)]
    nodes = np.array([x,y]).T

    p_est = interp(nodes)
    p_act = np.cos(x)*np.cos(y)




    mesh = Mesh2D(nodes)
    plt.figure()
    mesh.color(p_act)
    plt.colorbar()

    plt.figure()
    mesh.color(p_est-p_act)
    plt.colorbar()


    
    
    plt.show()       
    
