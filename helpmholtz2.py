import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type
from petsc4py import PETSc
from dolfinx.fem.petsc import LinearProblem
import ufl

k=20
nx,ny=80,80
#maillage du carré oméga
domain=mesh.create_unit_square(MPI.COMM_WORLD,nx,ny)

#espace d'eléments finis P1
V=fem.functionspace(domain,("Lagrange",1))

#définition des bords
fdim=domain.topology.dim-1
def left(x): return np.isclose(x[0],0)
def right(x): return np.isclose(x[0],1)
def bottom(x): return np.isclose(x[1],0)
def top(x): return np.isclose(x[1],1)   
left_facets=mesh.locate_entities_boundary(domain,fdim,left)
right_facets=mesh.locate_entities_boundary(domain,fdim,right)
bottom_facets=mesh.locate_entities_boundary(domain,fdim,bottom)
top_facets=mesh.locate_entities_boundary(domain,fdim,top)

#marquage des facettes(on n' besoin que du bord droit pour le terme de robin)
right_marker=1
marked_facets=np.hstack([right_facets])
marked_values=np.hstack([np.full(len(right_facets),right_marker,dtype=np.int32)])
sorted_facets=np.argsort(marked_facets)
facet_tag=mesh.meshtags(domain,fdim,marked_facets[sorted_facets],marked_values[sorted_facets])
ds=ufl.Measure("ds",domain=domain,subdomain_data=facet_tag)

#condition de dirichlet u=1 sur le bord gauche
uD=fem.Function(V)
uD.interpolate(lambda x: np.full(x.shape[1],PETSc.ScalarType(1)))
left_dofs=fem.locate_dofs_topological(V,fdim,left_facets)
bc=fem.dirichletbc(uD,left_dofs)

#formulation variationnelle en ufl
u=ufl.TrialFunction(V)
v=ufl.TestFunction(V)
a=(ufl.inner(ufl.grad(u),ufl.grad(v))*ufl.dx-(k**2)*ufl.inner(u,v)*ufl.dx+1j*k*ufl.inner(u,v)*ds(right_marker))
L=0
#résolution du problème linéaire
problem = LinearProblem(
    a, L,
    bcs=[bc],
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu"
    },
    petsc_options_prefix="helmholtz"
)
uh=problem.solve()

