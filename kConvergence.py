
from netgen.geom2d import unit_square
from ngsolve import *
from math import pi
ngsglobals.msg_level = 2

'''
This code corresponds to Example 1 (Figure 1) of the [arXiv preprint](https://arxiv.org/abs/2404.13578)

Tested with NGSolve version 6.2.2404

k-Convergence test for fully transient poroelasticity problem
Manufactured solutions on a squared domain
Pure Dirichlet BCs
HDG approximation in terms of stress, pressure & solid/fluid velocities
Crank-Nicolson scheme for time discretization
'''

def weaksym(k, h, dt, tend):
    
    # k: polynomial degree
    # h: mesh size
    # dt: time step
    # tend: final time

   # ********* Model coefficients and parameters ********* #
    
    t = Parameter(0.0)
    
    #porosity
    phi = 0.5
    
    #Biot-Willis coefficient
    alpha = 1

    #Constrained specific storage
    s = 1

    #saturating fluid density
    rhoF = 10
    
    #solid matrix density
    rhoS = 10
    
    #Density interaction matrix
    rho0  = phi*rhoF + (1 - phi)*rhoS
    rho01 = rhoF
    rho1  = rhoF/phi
    R = CoefficientFunction(( rho0,   rho01,   rho01,   rho1),   dims = (2,2) )
    
    # Permeability*(dynamic fluid viscosity)**{-1}
    beta = 1
    
    #Lamé coef. corresponding to C
    mu  = 50 
    lam = 100
    
    #needed for A = C**{-1}
    a1 = 0.5 / mu
    a2 = lam / (4.0 * mu * (lam + mu))
    

    # ******* Exact solutions for error analysis ****** #
    
   # ******* Exact solutions for error analysis ****** #
    
    #Exact displacement
    exactu_0 = x*cos(pi*y)*cos(t)
    exactu_1 = y*sin(pi*x)*sin(t)
    exactu = CoefficientFunction((exactu_0, exactu_1)) 
    
    #Exact solid velocity
    exactSv_0 = -x*cos(pi*y)*sin(t)
    exactSv_1 = y*sin(pi*x)*cos(t)
    exactSv = CoefficientFunction((exactSv_0, exactSv_1)) 
    
    #Exact solid acceleration
    exacta_0 = -x*cos(pi*y)*cos(t)
    exacta_1 = -y*sin(pi*x)*sin(t)
    exacta = CoefficientFunction((exacta_0, exacta_1)) 
    
    #Strain tensor
    epsu_00 = cos(pi*y)*cos(t)
    epsu_01 = (y*pi*cos(pi*x)*sin(t))/2 - (x*pi*sin(pi*y)*cos(t))/2
    epsu_11 = sin(pi*x)*sin(t)
    epsu = CoefficientFunction((epsu_00, epsu_01, epsu_01, epsu_11), dims = (2,2)) 
    
    #Stress tensor
    exactSigma = 2*mu*epsu + lam*Trace(epsu)*Id(2)
    
    #Fluid pressure
    exactp = sin(pi*x*y)*cos(t)
    
    # grad p
    exactgradp_0 = pi*y*cos(pi*x*y)*cos(t)
    exactgradp_1 = pi*x*cos(pi*x*y)*cos(t)
    exactgradp = CoefficientFunction((exactgradp_0, exactgradp_1))
    
    #Fluid velocity
    exactFv_0 = - (exp(-(beta*t)/rho1)*(beta*rho01*x*cos(pi*y) - beta*y*pi*cos(pi*x*y)))/(rho1**2 + beta**2) - ((y*pi*cos(pi*x*y) - rho01*x*cos(pi*y))*(sin(t) + (beta*cos(t))/rho1))/(rho1*(beta**2/rho1**2 + 1))
    
    exactFv_1 = (exp(-(beta*t)/rho1)*(rho01*rho1*y*sin(pi*x) + beta*x*pi*cos(pi*x*y)))/(rho1**2 + beta**2) - exp(-(beta*t)/rho1)*((exp((beta*t)/rho1)*cos(t)*(rho01*rho1*y*sin(pi*x) + beta*x*pi*cos(pi*x*y)))/(rho1**2 + beta**2) - (exp((beta*t)/rho1)*sin(t)*(beta*rho01*y*sin(pi*x) - rho1*x*pi*cos(pi*x*y)))/(rho1**2 + beta**2))
    
    exactFv = CoefficientFunction((exactFv_0, exactFv_1))
    
    #Source terms
    F0 = alpha*y*pi*cos(pi*x*y)*cos(t) - lam*pi*cos(pi*x)*sin(t) - rho0*x*cos(pi*y)*cos(t) - mu*pi*(cos(pi*x)*sin(t) - x*pi*cos(pi*y)*cos(t)) - (rho01*exp(-(beta*t)/rho1)*(y*pi*cos(pi*x*y) - rho01*x*cos(pi*y))*(beta**2 + rho1**2*exp((beta*t)/rho1)*cos(t) - beta*rho1*exp((beta*t)/rho1)*sin(t)))/(rho1*(rho1**2 + beta**2))
    
    F1 = (exp(-(beta*t)/rho1)*(rho01**2*rho1**2*y*exp((beta*t)/rho1)*sin(pi*x)*sin(t) - beta**2*rho01*x*pi*cos(pi*x*y) - rho0*rho1**3*y*exp((beta*t)/rho1)*sin(pi*x)*sin(t) - beta*rho01**2*rho1*y*sin(pi*x) + lam*rho1**3*pi*exp((beta*t)/rho1)*sin(pi*y)*cos(t) + mu*rho1**3*pi*exp((beta*t)/rho1)*sin(pi*y)*cos(t) + beta**2*lam*rho1*pi*exp((beta*t)/rho1)*sin(pi*y)*cos(t) + beta**2*mu*rho1*pi*exp((beta*t)/rho1)*sin(pi*y)*cos(t) + beta*rho01**2*rho1*y*exp((beta*t)/rho1)*sin(pi*x)*cos(t) - beta**2*rho0*rho1*y*exp((beta*t)/rho1)*sin(pi*x)*sin(t) + alpha*rho1**3*x*pi*cos(pi*x*y)*exp((beta*t)/rho1)*cos(t) - rho01*rho1**2*x*pi*cos(pi*x*y)*exp((beta*t)/rho1)*cos(t) + mu*rho1**3*y*pi**2*exp((beta*t)/rho1)*sin(pi*x)*sin(t) + alpha*beta**2*rho1*x*pi*cos(pi*x*y)*exp((beta*t)/rho1)*cos(t) + beta**2*mu*rho1*y*pi**2*exp((beta*t)/rho1)*sin(pi*x)*sin(t) + beta*rho01*rho1*x*pi*cos(pi*x*y)*exp((beta*t)/rho1)*sin(t)))/(rho1*(rho1**2 + beta**2))
    
    F = CoefficientFunction( (F0, F1) )
    
    g = (beta*rho01*sin(pi*x)*sin(t) - rho01*rho1*sin(pi*x)*cos(t) + beta*x**2*pi**2*sin(pi*x*y)*cos(t) + rho1*x**2*pi**2*sin(pi*x*y)*sin(t))/(rho1**2 + beta**2) + ((beta*cos(t) + rho1*sin(t))*(rho01*cos(pi*y) + y**2*pi**2*sin(pi*x*y)))/(rho1**2 + beta**2) + alpha*sin(pi*x)*cos(t) - alpha*cos(pi*y)*sin(t) + (exp(-(beta*t)/rho1)*(rho01*rho1*sin(pi*x) - beta*x**2*pi**2*sin(pi*x*y)))/(rho1**2 + beta**2) - s*sin(pi*x*y)*sin(t) - (beta*exp(-(beta*t)/rho1)*(rho01*cos(pi*y) + y**2*pi**2*sin(pi*x*y)))/(rho1**2 + beta**2)
        
    # ******* Mesh of the unit square ****** #

    mesh = Mesh(unit_square.GenerateMesh(maxh=h))

    # ********* Finite dimensional spaces ********* #

    S = L2(mesh, order =k)
    V = VectorL2(mesh, order =k+1)
    hatS = VectorFacetFESpace(mesh, order=k+1, dirichlet="bottom|left|right|top")
    hatF = VectorFacetFESpace(mesh, order=k+1)
    fes = FESpace([S, S, S, S, V, V, hatS, hatF])
    
    # ********* test and trial functions for product space ****** #
    
    sigma1, sigma12, sigma2, p, uS, uF, uShat, uFhat = fes.TrialFunction()
    tau1,   tau12,   tau2,   q, vS, vF, vShat, vFhat = fes.TestFunction()
    
    sigma = CoefficientFunction(( sigma1, sigma12, sigma12, sigma2), dims = (2,2) )
    
    tau   = CoefficientFunction(( tau1,   tau12,   tau12,   tau2),   dims = (2,2) )
    
    Asigma = a1 * sigma - a2 * Trace(sigma) *  Id(mesh.dim)
    
    sigmap = sigma - alpha*p*Id(mesh.dim)
    tauq   = tau   - alpha*q*Id(mesh.dim)
    
    U = CoefficientFunction(( uS[0],   uF[0],   uS[1],   uF[1]),   dims = (2,2) )
    V = CoefficientFunction(( vS[0],   vF[0],   vS[1],   vF[1]),   dims = (2,2) )
    
    n = specialcf.normal(mesh.dim)
    h = specialcf.mesh_size 
    
    dS = dx(element_boundary=True)
    
    jump_uS = uS - uShat
    jump_uF = uF - uFhat

    jump_vS = vS - vShat
    jump_vF = vF - vFhat
    
    
    # ********* Bilinear forms ****** #

    a = BilinearForm(fes, condense=True)
    a += (1/dt)*InnerProduct(U*R, V)*dx 
    a += (1/dt)*InnerProduct(Asigma, tau)*dx + (1/dt)*InnerProduct(s*p, q)*dx
    
    a +=   0.5*InnerProduct(beta*uF, vF)*dx
    
    a +=   0.5*InnerProduct(sigmap, grad(vS))*dx    - 0.5*InnerProduct(p, div(vF))*dx 
    a += - 0.5*InnerProduct( sigmap*n, jump_vS)*dS  + 0.5*InnerProduct(jump_vF, p*n)*dS
    
    a += - 0.5*InnerProduct(tauq, grad(uS))*dx      + 0.5*InnerProduct(q, div(uF))*dx 
    a +=   0.5*InnerProduct(tauq*n, jump_uS)*dS     - 0.5*InnerProduct(jump_uF, q*n)*dS 
    
    a +=   0.5*((k+1)**2/h)*jump_uS*jump_vS*dS + 0.5*((k+1)**2/h)*jump_uF*jump_vF*dS 
    
    a.Assemble()

    inv_A = a.mat.Inverse(freedofs=fes.FreeDofs(coupling=True))
    
    
    M = BilinearForm(fes)
    M += (1/dt)*InnerProduct(U*R, V)*dx 
    M += (1/dt)*InnerProduct(Asigma, tau)*dx + (1/dt)*InnerProduct(s*p, q)*dx
    
    M += - 0.5*InnerProduct(beta*uF, vF)*dx
    
    M += - 0.5*InnerProduct(sigmap, grad(vS))*dx    + 0.5*InnerProduct(p, div(vF))*dx
    M +=   0.5*InnerProduct( sigmap*n, jump_vS)*dS  - 0.5*InnerProduct(jump_vF*n, p)*dS 
    
    M +=   0.5*InnerProduct(tauq, grad(uS))*dx      - 0.5*InnerProduct(q, div(uF))*dx  
    M += - 0.5*InnerProduct(tauq*n, jump_uS)*dS     + 0.5*InnerProduct(jump_uF*n, q)*dS
    
    M += - 0.5*((k+1)**2/h)*jump_uS*jump_vS*dS - 0.5*((k+1)**2/h)*jump_uF*jump_vF*dS
    
    M.Assemble()
    
    # Right-hand side

    ft = LinearForm(fes)
    ft += F * vS * dx 
    ft += - exactp*(vFhat.Trace()*n) *ds(definedon=mesh.Boundaries("bottom|left|right|top"))
    ft += g*q*dx

    # ********* instantiation of initial conditions ****** #
    
    u0 = GridFunction(fes) 
    u0.components[0].Set(exactSigma[0,0])
    u0.components[1].Set(exactSigma[0,1])
    u0.components[2].Set(exactSigma[1,1])
    u0.components[3].Set(exactp)
    u0.components[4].Set(exactSv)
    u0.components[5].Set(exactFv)
    u0.components[6].Set(exactSv, dual=True)
    u0.components[7].Set(exactFv, dual=True)

    
    ft.Assemble()
    
    res = u0.vec.CreateVector()
    b0  = u0.vec.CreateVector()
    b1  = u0.vec.CreateVector()
    
    b0.data = ft.vec

    t_intermediate = dt # time counter within one block-run
    
    # ********* Time loop ************* # 

    while t_intermediate < tend:

        t.Set(t_intermediate)
        ft.Assemble()
        b1.data = ft.vec
     
        res.data = M.mat*u0.vec + 0.5*(b0.data + b1.data)

        u0.vec[:] = 0.0 
        u0.components[6].Set(exactSv, BND)#, dual=True)#BDN --->  Dirichlet Boundary

        res.data = res - a.mat * u0.vec

        res.data += a.harmonic_extension_trans * res

        u0.vec.data += inv_A * res
        
        u0.vec.data += a.inner_solve * res
        u0.vec.data += a.harmonic_extension * u0.vec
        
        b0.data = b1.data
        t_intermediate += dt
        
        print('t=%g' % t_intermediate)

    
    # ********* L2-errors at time tend ****** #
    
    gfsigma1, gfsigma12, gfsigma2, gfp, gfSv, gfFv = u0.components[0:6]

    gfsigma = CoefficientFunction(( gfsigma1, gfsigma12, gfsigma12, gfsigma2), dims = (2,2) )
    velo = CoefficientFunction((gfSv[0] - exactSv[0],   gfFv[0] - exactFv[0],   gfSv[1] - exactSv[1],   gfFv[1] - exactFv[1]), dims = (2,2))
    
    # Solid/fluid velocities error
    norm_V= InnerProduct(velo*R, velo)
    norm_V = Integrate(norm_V, mesh)
    norm_V = sqrt(norm_V)
    
    # Stess/pressure error
    norm_s  = InnerProduct(a1*(exactSigma - gfsigma) - a2*Trace(exactSigma - gfsigma)*  Id(mesh.dim), exactSigma - gfsigma) \
            + InnerProduct(s*(exactp - gfp), exactp - gfp)
    norm_s = Integrate(norm_s, mesh)
    norm_s = sqrt(norm_s)
    
    return norm_s, norm_V


# ********* Error collector ************* # 

def collecterrors(maxk, h, dt, tend):
    l2e_s = []
    l2e_r = []
    for k in range(0, maxk):
        er_1, er_2 = weaksym(k, h, dt, tend)
        l2e_s.append(er_1)
        l2e_r.append(er_2)
    return l2e_s, l2e_r


# ********* Convergence table ************* # 

def hconvergenctauEble(e_1, e_2, maxk):
    print("============================================================")
    print(" k   Errors_s   Error_u   ")
    print("------------------------------------------------------------")
    
    for i in range(maxk):
        print(" %-4d %8.2e    %8.2e   " % (i, e_1[i], 
               e_2[i]))

    print("============================================================")


# ============= MAIN DRIVER ==============================

maxk = 8 #number of k refinements
dt = 10e-6 #time step
tend = 0.3 #final time
h = 1/4 #mesh size

er_s, er_u = collecterrors(maxk, h, dt, tend)
hconvergenctauEble(er_s, er_u, maxk)