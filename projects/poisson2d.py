import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from sympy.utilities.lambdify import implemented_function
from poisson import Poisson
import itertools

x, y = sp.symbols('x,y')

class Poisson2D:
    r"""Solve Poisson's equation in 2D::

        \nabla^2 u(x, y) = f(x, y), x, y in [0, Lx] x [0, Ly]

    with homogeneous Dirichlet boundary conditions.
    """

    def __init__(self, Lx, Ly, Nx, Ny, bcx=(0,0), bcy=(0,0)):
        self.px = Poisson(Lx, Nx) # we can reuse some of the code from the 1D case
        self.py = Poisson(Ly, Ny)
        self.Nx = Nx
        self.Ny = Ny
        self.bcx = bcx
        self.bcy = bcy

    def create_mesh(self):
        """Return a 2D Cartesian mesh
        """
        xi, yj = np.meshgrid(self.px, self.py, indexing = 'ij', sparse = True)
        return xi, yj

    def laplace(self):
        """Return a vectorized Laplace operator"""
        dx = self.Lx/self.Nx
        dy = self.Ly/self.Ny
        
        Dx = 1/dx**2 * sparse.diags([-1,2-1], [-1,0,1], (self.Nx+1,self.Nx+1),'lil') # crate second derivate matricx for x 
                                                                       #(number of columns must be Nx to match that x is along the rows of U)
        Dx[0, :4] = 2, -5, 4, -1
        Dx[-1, -4:] = -1, 4, -5, 2
        
        Dy = 1/dx**2 * sparse.diags([-1,2-1], [-1,0,1], (self.Ny+1,self.Ny+1),'lil')
        Dy[0, :4] = 2, -5, 4, -1
        Dy[-1, -4:] = -1, 4, -5, 2


        A = sparse.kron(Dx,sparse.eye(self.Ny+1)) + sparse.kron(Dy,sparse.eye(self.Nx+1)) 
        
        for i in range(self.Nx+1):
            for j in range(self.Ny+1):
                if i%self.Nx==0 or j%self.Ny==0:
                    ij = (self.Ny+1)*i+j
                    A[ij,:] = 0
                    A[ij,ij] = 1
        """
        B = np.ones((self.Nx+1,self.Ny+1), dtype=bool)
        B[1:-1,1:-1] = 0
        bnds = np.where(B.ravel()==1)[0]
        for i in bnds:
            A[i,i] = 0
        """
            
        return A #vectorized laplace operator for dirichlet boundary conditions

    def assemble(self, f=None):
        """Return assemble coefficient matrix A and right hand side vector b"""
        A = self.laplace()
        
        xi, yi = self.create_mesh()
        f_num = sp.lambdify((x,y), f)(xi,yi)
        f_num[0,:] = self.bcx[0]
        f_num[-1,:] = self.bcx[1]
        f_num[:,0] = self.bcy[0]
        f_num[:,-1] = self.bcy[1]
        
        return A, b


    def l2_error(self, u, ue):
        """Return l2-error

        Parameters
        ----------
        u : array
            The numerical solution (mesh function)
        ue : Sympy function
            The analytical solution
        """
        xi, yi = self.create_mesh()
        ue_vec = sp.lambdify((x,y), ue)(xi,yi)
        
        return np.sqrt(self.dx*np.sum(ue_vec.t,u))

    def __call__(self, f=implemented_function('f', lambda x, y: 2)(x, y)):
        """Solve Poisson's equation with a given righ hand side function

        Parameters
        ----------
        N : int
            The number of uniform intervals
        f : Sympy function
            The right hand side function f(x, y)

        Returns
        -------
        The solution as a Numpy array

        """
        A, b = self.assemble(f=f)
        return sparse.linalg.spsolve(A, b.ravel()).reshape((self.px.N+1, self.py.N+1))

def test_poisson2d():
    solver = Poisson2D(Lx=1, Ly=1, Nx=100, Ny=100, bcx=())
    u = solver()
    ue = 1/2*(x**2+y**2)
    assert solver.l2_error(u,ue)<1e-12

