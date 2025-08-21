from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
from fenics import *

tol = 1E-14
l = 1   # unit cell 2D dimensions
w = 1   # unit cell length
r= 0.4  # inclusion radius
# set_log_level(1) # mark function

# # pick one dimension
d=2

#Domain definition and mesh
mesh = Mesh('geometry' + str(d)+'.xml')
subdomains = MeshFunction("size_t", mesh, 'geometry'+str(d)+'_physical_region.xml')
facets = MeshFunction("size_t", mesh, 'geometry'+str(d)+'_facet_region.xml')


xdmf_file = XDMFFile("mesh"+str(d)+".xdmf")
xdmf_file.write(mesh)
xdmf_file.close()

xdmf_file = XDMFFile("subdomains"+str(d)+".xdmf")
xdmf_file.write(subdomains)
xdmf_file.close()

xdmf_file = XDMFFile("facets"+str(d)+".xdmf")
xdmf_file.write(facets)
xdmf_file.close()

# class used to define the domain for Dirichlet BC
class Box(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

# class used to define the periodic boundary map
class PeriodicBoundary(SubDomain):
    def inside(self, x, on_boundary):
        # the boundary minus the top and right sides
        return on_boundary and not (near(x[0], 1.) or near(x[1], 1.) or near(x[d-1], 1.))

    def map(self, x, y):
        for j in range(d):
            if near(x[j], l ):
                y[j] = x[j] - 1
            else:
                y[j] = x[j]

u_fe = dolfin.VectorElement(
   family="CG",
   cell=mesh.ufl_cell(),
   degree=1)
V    = FunctionSpace(mesh,u_fe,constrained_domain=PeriodicBoundary())
V_   = FunctionSpace(mesh,u_fe)
W    = TensorFunctionSpace(mesh, 'CG', 2)

# Material constants # Stress sigma and strain epsilon
Eps_ini = 0
Eps = Constant(np.zeros((d,d)))


Ef   = 10e3    # volume 0 in GMSH, fiber
nuf  = 0.2
Em   = 250e3  # volume 1 in GMSH, matrix <-- this one is followed even if it's called volume 0
num  = 0.3

material_parameters = [(Ef, nuf), (Em, num)]
nphases = len(material_parameters)

def epsilon(u):
    return sym(grad(u))
def sigma(u, i, Eps):
    E, nu = material_parameters[i]
    lmbda = E*nu/(1+nu)/(1-2*nu)
    mu = E/2/(1+nu)
    return lmbda*tr(epsilon(u) + Eps)*Identity(d) + 2*mu*(epsilon(u)+Eps)

#Define variational problem

# Variational problems are defined in terms of trial and test functions
v    = TestFunction(V)
u   = TrialFunction(V)
dx = Measure('dx', domain=mesh, subdomain_data=subdomains)
d    = u.geometric_dimension()
F    = sum([inner(sigma(u, i, Eps), epsilon(v))*dx(i) for i in range(nphases)])
a, L = lhs(F), rhs(F)



## Problem Solving
u    = Function(V)  # Note the re-use of the variable u as both a TrialFunction in the variational problem and a function to store the solution
u.rename('U', 'U')
u_   = Function(V_)
u_.rename('U_', 'U_')
u_tot= Function(V_)
u_tot.rename('U_tot', 'U_tot')
stress = Function(W)
stress.rename('stress','stress')

# define vector lengths with problem dimension
if d==2:
    d_array = 3
    bc   = DirichletBC(V, [0,0], Box())
if d==3:
    d_array = 6
    bc   = DirichletBC(V, [0,0,0], Box())

# Compute macro strain
def get_macro_strain(i):
    """returns the macroscopic strain for the 3/6 elementary load cases"""
    Eps_Voigt = np.zeros(d_array)
    Eps_Voigt[i] = 1
    a = np.zeros((d,d))
    for i in range(d):
        for j in range(d):
            if i == j:
                a[i,j] = Eps_Voigt[i]
            else:
                a[i,j] = Eps_Voigt[d_array-i-j]/2
    return a

def stress2Voigt(s):
    if d == 2:
        return as_vector([s[0,0], s[1,1], s[0,1]])
    elif d ==3:
        return as_vector([s[0,0], s[1,1], s[2,2], s[2,1], s[2,0], s[1,0]])

xdmf_file = XDMFFile("displacement.xdmf")
xdmf_file_s = XDMFFile("stress.xdmf")
xdmf_file.write(u_tot, 0.)
xdmf_file_s.write(stress, 0.)

Chom = np.zeros((d_array, d_array))
# for (j, case) in enumerate(["Exx", "Eyy", "Ezz", "Eyz", "Exz", "Exy"]):
for j in range(d_array):
    # print("Solving {} case...".format(case))
    macro_strain = get_macro_strain(j)
    Eps.assign(Constant(macro_strain))
    solve(lhs(F) == rhs(F), u, [])

    Sigma = np.zeros(d_array)
    for k in range(d_array):
         Sigma[k] = assemble(sum([stress2Voigt(sigma(u, i, Eps))[k]*dx(i) for i in range(nphases)]))

         # print (Sigma)
    Chom[j, :] = Sigma
    stress = project(sigma(u,i,Eps), V=W)

# project macroscopic deformation on Vector space
    if d == 2:
        expression=Expression(
            ('Eps00*x[0]+Eps01*x[1]',
             'Eps01*x[0]+Eps11*x[1]'),
            Eps00=macro_strain[0,0],
            Eps11=macro_strain[1,1],
            Eps01=macro_strain[0,1],
            element=u_fe)


    elif d == 3:
        expression=Expression(
        ('Eps00*x[0]+Eps01*x[1]+Eps02*x[2]',
         'Eps01*x[0]+Eps11*x[1]+Eps12*x[2]',
         'Eps02*x[0]+Eps12*x[1]+Eps22*x[2]'),
        Eps00=macro_strain[0,0],
        Eps11=macro_strain[1,1],
        Eps22=macro_strain[2,2],
        Eps01=macro_strain[0,1],
        Eps02=macro_strain[0,2],
        Eps12=macro_strain[1,2],
        element=u_fe)

    dolfin.project(
             expression,
             V=V_,
             function=u_tot)

# project fluctuation on Vector space

    dolfin.project(
        u,
        V=V_,
        function=u_)
    u_tot.vector()[:] += u_.vector()[:]
    xdmf_file.write(u_tot, float(j+1))
    xdmf_file_s.write(stress, float(j+1))

xdmf_file.close()
xdmf_file_s.close()

#print results
print(np.array_str(Chom, precision=0))

lmbda_hom = Chom[0, 1]
mu_hom = Chom[d_array-1, d_array-1]
print(lmbda_hom, mu_hom)
E_hom = mu_hom*(3*lmbda_hom + 2*mu_hom)/(lmbda_hom + mu_hom)
nu_hom = lmbda_hom/(lmbda_hom + mu_hom)/2
print("Apparent Young modulus:", E_hom)
print("Apparent Poisson ratio:", nu_hom)


Shom = np.linalg.inv(Chom)
print (np.array_str(Shom, precision=0))

if d == 2:
    E1hom_PS = 1./Shom[0,0]
    E2hom_PS = 1./Shom[1,1]
    nu12hom_PS = -E1hom_PS*Shom[0,1]

    nu12hom = nu12hom_PS/(1+nu12hom_PS)
    E1hom = E1hom_PS*(1-nu_hom**2)
    E2hom = E2hom_PS*(1-nu_hom**2)

    print ("E1hom_PS = "+str(E1hom_PS))
    print ("E2hom_PS = "+str(E2hom_PS))
    print ("nu12hom_PS = "+str(nu12hom_PS))

    print ("E1hom= "+str(E1hom))
    print ("E2hom = "+str(E2hom))
    print ("nu12hom = "+str(nu12hom))

if d == 3:
    E1hom = 1./Shom[0,0]
    E2hom = 1./Shom[1,1]
    E3hom = 1./Shom[2,2]
    G23hom = 1./2/Shom[3,3]
    G13hom = 1./2/Shom[4,4]
    G12hom = 1./2/Shom[5,5]

    print ("E1hom = "+str(E1hom))
    print ("E2hom = "+str(E2hom))
    print ("E3hom = "+str(E3hom))
    print ("G23hom = "+str(G23hom))
    print ("G13hom = "+str(G13hom))
    print ("G12hom = "+str(G12hom))
