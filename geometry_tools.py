import numpy as np
import matplotlib.pyplot as plt
from shapely import intersects,LinearRing,LineString
from matplotlib import path
from scipy.interpolate import CubicSpline, interp1d


def calculate_arc_lengths(points):
    """Calculate the cumulative arc lengths of the line defined by `points`."""
    arc_lengths = [0]
    for i in range(1, len(points)):
        segment_length = np.linalg.norm(points[i] - points[i - 1])
        arc_lengths.append(arc_lengths[-1] + segment_length)
    return np.array(arc_lengths)

def resample_line(points, num_new_points):
    """Resample `points` to `num_new_points` with approximately the same arc length distribution."""
    # Calculate the cumulative arc lengths of the original points
    arc_lengths = calculate_arc_lengths(points)
    total_length = arc_lengths[-1]

    # Create the target arc length distribution for the new points
    target_arc_lengths = np.linspace(0, total_length, num_new_points)

    # Interpolate new points based on the target arc lengths
    new_points = []
    for dim in range(points.shape[1]):
        interp_func = interp1d(arc_lengths, points[:, dim], kind='linear')
        new_points.append(interp_func(target_arc_lengths))
    
    return np.column_stack(new_points)



def compute_curvature(points):
    # Fit a cubic spline to interpolate points
    t = np.arange(len(points))
    spl = CubicSpline(t, points, bc_type='not-a-knot')
    
    # Compute the first and second derivatives of the spline
    dx = spl.derivative(nu=1)
    ddx = spl.derivative(nu=2)
    
    # Calculate curvature
    numerator = dx(t)[:, 0] * ddx(t)[:, 1] - dx(t)[:, 1] * ddx(t)[:, 0]
    denominator = (dx(t)[:, 0]**2 + dx(t)[:, 1]**2)**(3/2)
    curvature = -numerator / denominator

    curvature[-1] = curvature[-2]
    curvature[0]  = curvature[1]
    return curvature

def compute_curvature_V2(points):
    # Calculate differences between consecutive points
    dx = np.diff(points[:, 0])
    dy = np.diff(points[:, 1])
    
    # Calculate first derivatives
    dx_dt = np.gradient(points[:, 0], edge_order=2)
    dy_dt = np.gradient(points[:, 1], edge_order=2)
    
    # Calculate second derivatives
    d2x_dt2 = np.gradient(dx_dt, edge_order=2)
    d2y_dt2 = np.gradient(dy_dt, edge_order=2)
    
    # Calculate curvature
    numerator = dx_dt * d2y_dt2 - dy_dt * d2x_dt2
    denominator = (dx_dt**2 + dy_dt**2)**(3/2)
    curvature = numerator / denominator

    return curvature
   

def compute_tangents(points):
    t = np.arange(len(points))
    spl = CubicSpline(t, points, bc_type='not-a-knot')
    
    
    # Compute the first derivative of the spline to get tangent vectors
    dx = spl.derivative(nu=1)
    
    # Calculate normal vectors by rotating tangent vectors by 90 degrees
    tangents = dx(t)/np.linalg.norm(dx(t),axis=-1)[:,None]
    return tangents

def compute_normals(points):
    tangents = compute_tangents(points)
    normals = np.zeros_like(tangents)
    normals[:, 0] = -tangents[:, 1]
    normals[:, 1] = tangents[:, 0]
    return normals
    

def compute_offset(points,distance,side='right',join_style='mitre'):
    offset_points = LineString(points).parallel_offset(distance,side,join_style=join_style,mitre_limit=5).coords
    return np.array(offset_points)

def contains_points(loop, points):
    return path.Path(loop).contains_points(points)

def reorganize_points(points, d=1):
    """
    Reorganizes the list of points such that no two consecutive points are more than d apart.
    
    Parameters:
    - points: List of points defining the curve (N, 2).
    - d: Maximum allowable distance between consecutive points.
    
    Returns:
    - Reorganized list of points.
    """
    
    n = points.shape[0]
    
    # Calculate distances between consecutive points
    distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
    
    # Find the index where distance exceeds d
    cut_index = np.where(distances > d)[0]
    
    if len(cut_index) == 0:
        # All distances are within the limit
        return points
    
    cut_index = cut_index[0]  # Only one cut is needed

    # Split and rearrange points
    new_order = np.concatenate([points[cut_index + 1:], points[:cut_index + 1]])
    
    return new_order

def intersects(seg1,seg2,tol=1e-6):
    def ccw(A,B,C):
        return (C[...,1]-A[...,1]) * (B[...,0]-A[...,0]) > (B[...,1]-A[...,1]) * (C[...,0]-A[...,0]) - tol
    a1 = seg1[...,0,:]
    b1 = seg1[...,1,:]
    a2 = seg2[...,0,:]
    b2 = seg2[...,1,:]
    return (ccw(a1,a2,b2) != ccw(b1,a2,b2)) * (ccw(a1,b1,a2) != ccw(a1,b1,b2))

def line_of_sight(a,b,boundary_faces):
    LOS = np.ones((a.shape[0],b.shape[0]))
    A = np.zeros((a.shape[0],b.shape[0],2,2))
    A[:,:,0,:] = a[:,None]
    A[:,:,1,:] = b[None,:]
    
    for B in boundary_faces:
        LOS *= ~np.sum(intersects(A.reshape(-1,2,2)[:,None],B[None,:]),
                       axis=-1,
                       dtype=np.bool_).reshape(A.shape[:-2])

    return LOS
        
    


    

    

if __name__ == '__main__':
    # Load airfoil profile from Afoil.dat
    points = np.genfromtxt(r'data\Afoil.dat', skip_header=1)[::-1]

    points = points[1:-1]

    # Compute offset points
    offset_distance = 0.01
    offset_points = compute_offset(points, offset_distance)

    # Plot original and offset profiles
    plt.figure()
    plt.plot(points[:, 0], points[:, 1], label='Original Profile')
    plt.plot(offset_points[:, 0], offset_points[:, 1], label='Offset Profile')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Airfoil Profile and Offset')
    plt.axis('equal')
    plt.show()





    
