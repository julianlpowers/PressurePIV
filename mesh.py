import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib
from geometry_tools import contains_points, resample_line, calculate_arc_lengths, compute_offset,intersects



class Mesh2D:
    def __init__(self, nodes, scalars={}, alpha=np.inf, max_cell_aspect_ratio=np.inf, avoid_regions=[]):
        '''
        INPUTS:
        -------------------------------------------------
        self
        nodes                  (N x 2)                   : nodal points
        scalars                {p0:(N), p1:(N), ...}     : dictionary of scalar values at nodal points
        alpha                  float                     : alpha shape parameter, reject cells with diameter > alpha
        max_cell_aspect_ratio  float                     : reject cells with high aspect ratios
        avoid_regions          [(n0 x 2), (n1 x 2), ...] : reject cells that fall within specified closed curves
        '''
        
        if alpha==None:
            alpha = 2*np.linalg.norm(nodes[0]-nodes[1])

        self.nodes = nodes
        self.scalars = scalars
        node_mask = np.ones_like(nodes[:,0], dtype=np.bool_)
        for loop in avoid_regions:
            node_mask *= ~contains_points(loop,nodes)
        self.nodes = self.nodes[node_mask] # remove bad nodes
        for name,val in self.scalars.items():
            self.scalars[name] = val[node_mask]
        

    
        self.cells = sp.spatial.Delaunay(self.nodes).simplices # triangulations
        cell_nodes = self.nodes[self.cells]
        
        diameters = np.max(np.linalg.norm(cell_nodes[:,:,None] - cell_nodes[:,None, :], axis=-1),axis=(1,2))
        aspect_ratios = diameters**2 / self.cell_volumes()
        self.cells = self.cells[ (diameters < alpha) * (aspect_ratios < max_cell_aspect_ratio) ] # remove large or thin (bad) cells

        centers = self.cell_centers()
        cell_mask = np.ones_like(centers[:,0], dtype=np.bool_)
        node_mask = np.ones_like(nodes[:,0], dtype=np.bool_)
        for loop in avoid_regions:
            cell_mask *= ~contains_points(loop,centers)
        self.cells = self.cells[cell_mask] # remove cells outside of "bad" regions (not our domain)
            

                
        # find boundary faces (edges)
        mesh_edges = set()
        for cell in self.cells:
            edges = [ (cell[i],cell[i+1]) for i in range(len(cell)-1)] + [(cell[-1],cell[0]) ]
            for edge in edges:
                assert edge not in mesh_edges, 'bad mesh, duplicate directed edge'
                if edge[::-1] in mesh_edges:
                    mesh_edges.remove(edge[::-1])
                else:
                    mesh_edges.add(edge)

        # remove edge conflicts
        edge_dict = {}
        for i,j in mesh_edges:
            if i in edge_dict:
                edge_dict[j] = i
            else:
                edge_dict[i] = j
                
        # find boundary cycles
        visited = set()
        cycles = []
        for i in edge_dict:
            if i not in visited:
                cycle = []
                j=i
                while j not in visited:
                    visited.add(j)
                    cycle.append((j,edge_dict[j]))
                    j = edge_dict[j]
                cycles.append(cycle)
                 
 
        self.boundary_faces = cycles
        
        #format the cycles
        self.boundaries = [[node[0] for node in cycle] for cycle in cycles]


        
    def cell_nodes(self):
        return self.nodes[self.cells]
    
    def boundary_face_nodes(self):
        return [self.nodes[boundary] for boundary in self.boundary_faces]

    def boundary_nodes(self):
        return [self.nodes[boundary] for boundary in self.boundaries]

    def exterior(self): 
        return np.concatenate(self.boundaries)
        
    def interior(self):
        all_inds = np.arange(self.nodes.shape[0])
        mask = np.ones_like(all_inds,dtype=np.bool_)
        mask[self.exterior()] = 0
        return all_inds[mask]
    
    def exterior_nodes(self):
        return self.nodes[self.exterior()]
        
    def interior_nodes(self):
        return self.nodes[self.interior()]

    def cell_centers(self):
        return np.mean(self.cell_nodes(),axis=1)

    def boundary_face_centers(self):
        return [np.mean(boundary,axis=1) for boundary in self.boundary_face_nodes()]

    def boundary_face_parallels(self):
        return [ boundary[:,1,:]-boundary[:,0,:] for boundary in self.boundary_face_nodes()]
    
    def boundary_face_normals(self):
        return [ boundary @ np.array([[0,-1],[1,0]]) for boundary in self.boundary_face_parallels()]

    def boundary_sizes(self):
        return [len(bound) for bound in self.boundaries]

    def cell_volumes(self):
        triangles = self.cell_nodes()
        return 0.5 * np.abs(np.cross(triangles[:,1] - triangles[:,0],
                                     triangles[:,2] - triangles[:,0], axis=1))

    def volume_integral(self,f):
        if callable(f):
            x,y = tuple(self.cell_centers().T)
            return np.sum(self.cell_volumes()*f(x,y))     
        else:
            assert len(list(f)) == self.nodes.shape[0]
            ff = np.mean(f[self.cells],axis=1)
            return np.sum(self.cell_volumes()*ff) 

    def surface_flux(self,v):
        pass


    def derivatives(self, max_neighbors=10, max_radius=np.inf, derivatives_order=2, needs_extrap=[]):        
        kdtree = sp.spatial.KDTree(self.nodes)
        d,i = kdtree.query(self.nodes,k=max_neighbors+1)
        d=d[:,0:]
        i=i[:,0:]
        far = d>max_radius
        extrap_ind = np.isin(i,needs_extrap)

        # create derivative mask matrix
        x,y = tuple(self.nodes.T)
        dx = x[i]-x[:,None]
        dy = y[i]-y[:,None]
        dx[far] = 0
        dy[far] = 0
        dx[extrap_ind] = 0
        dy[extrap_ind] = 0
        

        d_list = []
        d_names = [] 
        for m in range(0, derivatives_order + 1):
            for n in range(m+1):
                d_list += [ dx**(m-n) * dy**n ]
                d_names += [ 'x'*(m-n) + 'y'*n ]         
        A = np.stack(d_list,axis=2)
    
        # setup local differences matrix    
        p_list = []
        for p_name in self.scalars:
            p = self.scalars[p_name].copy()
            p = p[i]
            p[far] = 0
            p[extrap_ind]=0
            p_list.append(p)
        p_list =np.stack(p_list)

        # get output
        out = np.einsum('Ndk, sNk -> sNd',np.linalg.pinv(A), p_list)
        
        derivs_dict = {}
        for p_ind,p_name in enumerate(self.scalars):
            for d_ind,d_name in enumerate(d_names):
                name = p_name + '_' + d_name
                deriv = out[p_ind,:,d_ind]
                derivs_dict[name] = deriv
                
        return derivs_dict

    def boundary_scalars(self):
        b_scalars = []
        for bound in self.boundary_faces:
            p_dict = {}
            for p_name in self.scalars:
                p_dict[p_name] = np.mean(self.scalars[p_name][bound],axis=-1)
            b_scalars.append(p_dict)
        return b_scalars
                

    def evaluate_at_nodes(self, f):
        x,y = tuple(self.nodes.T)
        return f(x,y)

    def inherit_scalars_from(self, mesh, interp=sp.interpolate.LinearNDInterpolator):
        # interpolate scalars onto new mesh
        for p_name in mesh.scalars:
            p = mesh.scalars[p_name]
            P = interp(mesh.nodes,p)(self.nodes)
            self.scalars[p_name] = P
    

    def boundary_is_hole(self):
        is_hole = []
        #check to see if loop goes clockwise
        for points in self.boundary_nodes():
            x = points[:,0]
            y = points[:,1]
            area = np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]))
            area += (x[0] - x[-1]) * (y[0] + y[-1])  # Closing the loop
            is_hole.append(area > 0)
        return is_hole
            
        
    def flip_boundary_directions(self, flip):
        for k in range(len(self.boundaries)):
            if flip[k]:
                new_faces = []
                for node in self.boundary_faces[k]:
                    new_faces.append((node[1],node[0]))
                self.boundary_faces[k] = new_faces
                self.boundaries[k].reverse()
                
    def disconnect_boundary_scalar(self,p):
        inds = np.cumsum([0] + self.boundary_sizes())
        return [p[inds[i] : inds[i+1]] for i in range(len(inds)-1)]

        
                
        

    def plot(self):
        plt.triplot(self.nodes[:,0],self.nodes[:,1],self.cells,color='black')
        plt.plot(self.cell_centers()[:,0],self.cell_centers()[:,1],'kx')
        plt.plot(self.interior_nodes()[:,0],self.interior_nodes()[:,1],'ro')

        for xb in self.boundary_face_centers():
            plt.plot(xb[:,0],xb[:,1],'bo')

        for k in range(len(self.boundary_face_centers())):
            xb = self.boundary_face_centers()[k]
            dS = self.boundary_face_normals()[k]
            plt.quiver(xb[:,0],xb[:,1],dS[:,0],dS[:,1])
        
        plt.axis('equal')
            
            
    def color(self,f,on_volumes=False,scale_axes=1.0,**kwargs):
        triangulation = matplotlib.tri.Triangulation(self.nodes[:,0]*scale_axes, self.nodes[:,1]*scale_axes, self.cells) 
        if callable(f):
            x,y = tuple(self.nodes.T)
            z = f(x,y)
        else:
            if on_volumes:
                z = f.copy()
            else:
                assert len(list(f)) == self.nodes.shape[0]
                z = np.mean(f[self.cells],axis=1)
            
        im = plt.tripcolor(triangulation,z,**kwargs)
        plt.axis('equal')
        return im
            
    

    def info(self):
        num_interior_points = self.interior().shape[0]
        num_exterior_points = self.exterior().shape[0]
        return num_interior_points, num_exterior_points

    def in_mesh(self,points):
        boundary_is_hole = self.boundary_is_hole()
        mask = np.ones_like(points[:,0],dtype=np.bool_)
        for boundary_ind, boundary in enumerate(self.boundary_nodes()):
            is_hole = boundary_is_hole[boundary_ind]
            loop = matplotlib.path.Path(boundary)
            if is_hole:
                mask = mask * ~loop.contains_points(points)
            else:
                mask = mask * loop.contains_points(points)      
        return mask
        


def PIVmesh(mesh,Nx=100,Ny=100,pad=1e-6,alpha=np.inf):

    # create a rough rectangular grid around given mesh nodes
    x,y = tuple(mesh.nodes.T)
    minx,maxx = np.min(x)+pad, np.max(x)-pad
    miny,maxy = np.min(y)+pad, np.max(y)-pad

    X,Y = np.meshgrid(np.linspace(minx,maxx,Nx),np.linspace(miny,maxy,Ny))
    X=X.flatten()
    Y=Y.flatten()
    Nodes = np.array([X,Y]).T
    
    # remove nodes outside of original mesh      
    Nodes = Nodes[mesh.in_mesh(Nodes)]

    # interpolate scalars onto new mesh
    new_mesh = Mesh2D(Nodes,alpha=alpha)
    new_mesh.inherit_scalars_from(mesh)
    
        
    return new_mesh

def shrink(mesh, alpha=None):
    Nodes = mesh.interior_nodes()
    
    # interpolate scalars onto new mesh
    new_mesh = Mesh2D(Nodes,alpha=alpha)
    new_mesh.inherit_scalars_from(mesh)
    
    return new_mesh

def expand(mesh, boundaries):
    dL = np.min(np.linalg.norm(mesh.nodes[0][None,:]-mesh.nodes[1:],axis=-1),axis=-1)

    resampled_boundaries = [resample_line(bound, int(calculate_arc_lengths(bound)[-1]/dL)) for bound in boundaries]
    new_boundary_nodes = np.concatenate(resampled_boundaries)

    min_distance = np.min(np.linalg.norm(mesh.nodes[None,:,:]-new_boundary_nodes[:,None,:],axis=-1))
    print(min_distance,dL)

    new_nodes = mesh.nodes
    for d in np.linspace(0,min_distance,int(min_distance/dL)+1,endpoint=False):
        new_nodes = np.concatenate([new_nodes]+[compute_offset(bound,d) for bound in resampled_boundaries])

    return Mesh2D(new_nodes,avoid_regions=boundaries)
        
    

    
    
    
if __name__ == '__main__':
    np.random.seed(0)
    x,y = tuple(np.random.uniform(-3,3,[2,200**2]))
    mask = (x**2+y**2>1) * ((abs(x)-3)**2+(abs(y)-3)**2>0.3)
    x,y = x[mask],y[mask]
    nodes = np.array([x,y]).T

    def franke(x,y):
        f = (
            + 0.75 * np.exp( -(9*x-2)**2/4  -(9*y-2)**2/4  )
            + 0.75 * np.exp( -(9*x+1)**2/49 -(9*y+1)**2/10 )
            + 0.50 * np.exp( -(9*x-7)**2/4  -(9*y-3)**2/4  )
            - 0.20 * np.exp( -(9*x-4)**2    -(9*y-7)**2    )
            )
        return f
    
    def cylinder_pressure(x,y):
        th = np.arctan2(y,x)
        r  = (x**2+y**2)**0.5
        p = 2/r**2*np.cos(2*th) - 1/r**4
        return p
        

    p = cylinder_pressure(x,y)
    
    mesh = Mesh2D(nodes,scalars={'p': p},alpha=0.2, max_cell_aspect_ratio=np.inf)

    PIV = PIVmesh(mesh, Nx=20,Ny=20)
    derivs = PIV.derivatives(max_neighbors=20,
                              max_radius=np.inf,
                              derivatives_order=3)
    
    
    PIV.plot()

    PIV_small = shrink(PIV)
    plt.figure()
    PIV_small.plot()

##    plt.figure()
##    PIV.color(diff(cylinder_pressure,'x')(*tuple(PIV.nodes.T)))
##    plt.colorbar()
##    
##    plt.figure()
##    PIV.color(derivs['p_x'])
##    plt.colorbar()
##
##    plt.figure()
##    PIV.color(derivs['p_x']-diff(cylinder_pressure,'x')(*tuple(PIV.nodes.T)))
##    plt.colorbar()

    plt.figure()
    th = np.linspace(0,2*np.pi,500,endpoint=False)
    bound = np.array([np.cos(th),np.sin(th)]).T
    PIV_big = expand(PIV_small,[bound])
    PIV_big.plot()
    
    plt.show()
    
    

    
        
        
    
