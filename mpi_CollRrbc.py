import numpy as np
import pylab as plt
from matplotlib import cm
import h5py as hp
from mpi4py import MPI
import time
from datetime import datetime
import random 
import scipy.integrate as integrate

############### Grid Parameters ###############

Lx, Ly, Lz = 1.0, 1.0, 1.0

Nx, Ny, Nz = 33, 33, 33

hx, hy, hz = Lx/(Nx-1), Ly/(Ny-1), Lz/(Nz-1)

x = np.linspace(0, Lx, Nx, endpoint=True)        
y = np.linspace(0, Ly, Ny, endpoint=True)
z = np.linspace(0, Lz, Nz, endpoint=True)    

hx2, hy2, hz2 = hx*hx, hy*hy, hz*hz

idx2, idy2, idz2 = 1.0/hx2, 1.0/hy2, 1.0/hz2

###############################################

############# MPI Parallelization #############

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

locNx = int(Nx/nprocs) + 1
xSize = locNx + 2

bn = 1 + locNx*rank
en = bn + locNx 

rootRank = rank == 0
frstRank = rank == 0
lastRank = rank == nprocs - 1

lftRank = rank - 1
rgtRank = rank + 1

xSt, ySt, zSt = 1, 1, 1
xEn, yEn, zEn = locNx+1, Ny-1, Nz-1

kbn = bn
if frstRank:
    lftRank = MPI.PROC_NULL
    xSize = locNx + 1
    xEn = locNx

    kbn = bn-1

ken = en
if lastRank:
    rgtRank = MPI.PROC_NULL
    xSize = locNx + 1
    xEn = locNx

    ken = Nx
    en = Nx-1

if rootRank:
    print('# Grid', Nx, Ny, Nz)
    print('#No. of Processors =',nprocs)

print('#', rank, bn, en)

###############################################

############# Fields Initialization ###########

# Field variables
U = np.zeros([xSize, Ny, Nz])
V = np.zeros([xSize, Ny, Nz])
W = np.zeros([xSize, Ny, Nz])
T = np.zeros([xSize, Ny, Nz])
P = np.zeros([xSize, Ny, Nz])

# Auxilliary variables
Pp = np.zeros([xSize, Ny, Nz])

# RHS Terms
Hx = np.zeros_like(U)
Hy = np.zeros_like(V)
Hz = np.zeros_like(W)
Ht = np.zeros_like(T)   
Pp = np.zeros_like(P)

# Initialize values
P.fill(1.0)
T[:, :, 0:Nz] = 1 - z[0:Nz]

###############################################

############### Flow Parameters ###############

Ra = 1.0e4
Pr = 1.0
Ta = 0.0

nu, kappa = np.sqrt(Pr/Ra), 1.0/np.sqrt(Ra*Pr)

if rootRank:
    print('#', 'Ra=', Ra, 'Pr=', Pr)

###############################################

########### Simulation Parameters #############

# Time step
dt = 0.01

# Final time
tMax = 0.1

# Number of iterations at which output is sent to standard I/O
opInt = 1

# File writing interval
fwInt = 2

# Tolerance value in Jacobi iterations
VpTolerance = 1.0e-5

# Tolerance value in Poisson iterations
PoissonTolerance = 1.0e-3

# Omega for SOR
gssor = 1.6

# Maximum iterations for iterative solvers
maxCount = 1e4

if rootRank:
    print('# Tolerance', VpTolerance, PoissonTolerance)

###############################################


def writeSoln(U, V, W, P, T, time):

    fName = "Soln_{0:09.5f}.h5".format(time)
    print("#Writing solution file: ", fName)        
    f = hp.File(fName, "w")

    dset = f.create_dataset("U", data = U)
    dset = f.create_dataset("V", data = V)
    dset = f.create_dataset("W", data = W)
    dset = f.create_dataset("T", data = T)
    dset = f.create_dataset("P", data = P)

    f.close()


def getDiv(U, V, W):

    divMat = ((U[xSt+1:xEn+1, ySt:yEn, zSt:zEn] - U[xSt-1:xEn-1, ySt:yEn, zSt:zEn])*0.5/hx +
              (V[xSt:xEn, ySt+1:yEn+1, zSt:zEn] - V[xSt:xEn, ySt-1:yEn-1, zSt:zEn])*0.5/hy +
              (W[xSt:xEn, ySt:yEn, zSt+1:zEn+1] - W[xSt:xEn, ySt:yEn, zSt-1:zEn-1])*0.5/hz)
    
    locdivMax = np.max(abs(divMat))

    globdivMax = comm.reduce(locdivMax, op=MPI.MAX, root=0)

    return globdivMax


def data_transfer(F):
    if nprocs > 1:

        comm.Send(F[en-1, :, :], dest = rgtRank)
        comm.Recv(F[en, :, :], source = rgtRank)

        comm.Send(F[bn, :, :], dest = lftRank)
        comm.Recv(F[bn-1, :, :], source = lftRank)  


def computeNLinDiff_X(U, V, W):

    Hx[xSt:xEn, ySt:yEn, zSt:zEn] = (((U[xSt+1:xEn+1, ySt:yEn, zSt:zEn] - 2.0*U[xSt:xEn, ySt:yEn, zSt:zEn] + U[xSt-1:xEn-1, ySt:yEn, zSt:zEn])/hx2 + 
                                      (U[xSt:xEn, ySt+1:yEn+1, zSt:zEn] - 2.0*U[xSt:xEn, ySt:yEn, zSt:zEn] + U[xSt:xEn, ySt-1:yEn-1, zSt:zEn])/hy2 + 
                                      (U[xSt:xEn, ySt:yEn, zSt+1:zEn+1] - 2.0*U[xSt:xEn, ySt:yEn, zSt:zEn] + U[xSt:xEn, ySt:yEn, zSt-1:zEn-1])/hz2)*0.5*nu -
                                       U[xSt:xEn, ySt:yEn, zSt:zEn]*(U[xSt+1:xEn+1, ySt:yEn, zSt:zEn] - U[xSt-1:xEn-1, ySt:yEn, zSt:zEn])/(2.0*hx) -
                                       V[xSt:xEn, ySt:yEn, zSt:zEn]*(U[xSt:xEn, ySt+1:yEn+1, zSt:zEn] - U[xSt:xEn, ySt-1:yEn-1, zSt:zEn])/(2.0*hy) - 
                                       W[xSt:xEn, ySt:yEn, zSt:zEn]*(U[xSt:xEn, ySt:yEn, zSt+1:zEn+1] - U[xSt:xEn, ySt:yEn, zSt-1:zEn-1])/(2.0*hz))

    return Hx[xSt:xEn, ySt:yEn, zSt:zEn]


def computeNLinDiff_Y(U, V, W):

    Hy[xSt:xEn, ySt:yEn, zSt:zEn] = (((V[xSt+1:xEn+1, ySt:yEn, zSt:zEn] - 2.0*V[xSt:xEn, ySt:yEn, zSt:zEn] + V[xSt-1:xEn-1, ySt:yEn, zSt:zEn])/hx2 + 
                                      (V[xSt:xEn, ySt+1:yEn+1, zSt:zEn] - 2.0*V[xSt:xEn, ySt:yEn, zSt:zEn] + V[xSt:xEn, ySt-1:yEn-1, zSt:zEn])/hy2 + 
                                      (V[xSt:xEn, ySt:yEn, zSt+1:zEn+1] - 2.0*V[xSt:xEn, ySt:yEn, zSt:zEn] + V[xSt:xEn, ySt:yEn, zSt-1:zEn-1])/hz2)*0.5*nu -
                                       U[xSt:xEn, ySt:yEn, zSt:zEn]*(V[xSt+1:xEn+1, ySt:yEn, zSt:zEn] - V[xSt-1:xEn-1, ySt:yEn, zSt:zEn])/(2.0*hx) -
                                       V[xSt:xEn, ySt:yEn, zSt:zEn]*(V[xSt:xEn, ySt+1:yEn+1, zSt:zEn] - V[xSt:xEn, ySt-1:yEn-1, zSt:zEn])/(2.0*hy) - 
                                       W[xSt:xEn, ySt:yEn, zSt:zEn]*(V[xSt:xEn, ySt:yEn, zSt+1:zEn+1] - V[xSt:xEn, ySt:yEn, zSt-1:zEn-1])/(2.0*hz))

    return Hy[xSt:xEn, ySt:yEn, zSt:zEn]


def computeNLinDiff_Z(U, V, W):
    global Hz
    global Nz, Ny, Nx, Nx, Ny, Nz

    Hz[xSt:xEn, ySt:yEn, zSt:zEn] = (((W[xSt+1:xEn+1, ySt:yEn, zSt:zEn] - 2.0*W[xSt:xEn, ySt:yEn, zSt:zEn] + W[xSt-1:xEn-1, ySt:yEn, zSt:zEn])/hx2 + 
                                      (W[xSt:xEn, ySt+1:yEn+1, zSt:zEn] - 2.0*W[xSt:xEn, ySt:yEn, zSt:zEn] + W[xSt:xEn, ySt-1:yEn-1, zSt:zEn])/hy2 + 
                                      (W[xSt:xEn, ySt:yEn, zSt+1:zEn+1] - 2.0*W[xSt:xEn, ySt:yEn, zSt:zEn] + W[xSt:xEn, ySt:yEn, zSt-1:zEn-1])/hz2)*0.5*nu -
                                       U[xSt:xEn, ySt:yEn, zSt:zEn]*(W[xSt+1:xEn+1, ySt:yEn, zSt:zEn] - W[xSt-1:xEn-1, ySt:yEn, zSt:zEn])/(2.0*hx) -
                                       V[xSt:xEn, ySt:yEn, zSt:zEn]*(W[xSt:xEn, ySt+1:yEn+1, zSt:zEn] - W[xSt:xEn, ySt-1:yEn-1, zSt:zEn])/(2.0*hy) - 
                                       W[xSt:xEn, ySt:yEn, zSt:zEn]*(W[xSt:xEn, ySt:yEn, zSt+1:zEn+1] - W[xSt:xEn, ySt:yEn, zSt-1:zEn-1])/(2.0*hz))


    return Hz[xSt:xEn, ySt:yEn, zSt:zEn]


def computeNLinDiff_T(U, V, W, T):
    global Ht
    global Nz, Ny, Nx

    Ht[xSt:xEn, ySt:yEn, zSt:zEn] = (((T[xSt+1:xEn+1, ySt:yEn, zSt:zEn] - 2.0*T[xSt:xEn, ySt:yEn, zSt:zEn] + T[xSt-1:xEn-1, ySt:yEn, zSt:zEn])/hx2 + 
                                      (T[xSt:xEn, ySt+1:yEn+1, zSt:zEn] - 2.0*T[xSt:xEn, ySt:yEn, zSt:zEn] + T[xSt:xEn, ySt-1:yEn-1, zSt:zEn])/hy2 + 
                                      (T[xSt:xEn, ySt:yEn, zSt+1:zEn+1] - 2.0*T[xSt:xEn, ySt:yEn, zSt:zEn] + T[xSt:xEn, ySt:yEn, zSt-1:zEn-1])/hz2)*0.5*kappa -
                                       U[xSt:xEn, ySt:yEn, zSt:zEn]*(T[xSt+1:xEn+1, ySt:yEn, zSt:zEn] - T[xSt-1:xEn-1, ySt:yEn, zSt:zEn])/(2.0*hx)-
                                       V[xSt:xEn, ySt:yEn, zSt:zEn]*(T[xSt:xEn, ySt+1:yEn+1, zSt:zEn] - T[xSt:xEn, ySt-1:yEn-1, zSt:zEn])/(2.0*hy) - 
                                       W[xSt:xEn, ySt:yEn, zSt:zEn]*(T[xSt:xEn, ySt:yEn, zSt+1:zEn+1] - T[xSt:xEn, ySt:yEn, zSt-1:zEn-1])/(2.0*hz))

    return Ht[xSt:xEn, ySt:yEn, zSt:zEn]


def uJacobi(rho):

    jCnt = 0
    while True:

        U[xSt:xEn, ySt:yEn, zSt:zEn] =(1.0/(1+nu*dt*(idx2 + idy2 + idz2))) * (rho[xSt:xEn, ySt:yEn, zSt:zEn] + 
                                       0.5*nu*dt*idx2*(U[xSt-1:xEn-1, ySt:yEn, zSt:zEn] + U[xSt+1:xEn+1, ySt:yEn, zSt:zEn]) +
                                       0.5*nu*dt*idy2*(U[xSt:xEn, ySt-1:yEn-1, zSt:zEn] + U[xSt:xEn, ySt+1:yEn+1, zSt:zEn]) +
                                       0.5*nu*dt*idz2*(U[xSt:xEn, ySt:yEn, zSt-1:zEn-1] + U[xSt:xEn, ySt:yEn, zSt+1:zEn+1]))          

        data_transfer(U)

        imposeUBCs(U)
        
        locmaxErr = np.amax(np.fabs(rho[xSt:xEn, ySt:yEn, zSt:zEn] - (U[xSt:xEn, ySt:yEn, zSt:zEn] - 0.5*nu*dt*(
                            (U[xSt-1:xEn-1, ySt:yEn, zSt:zEn] - 2.0*U[xSt:xEn, ySt:yEn, zSt:zEn] + U[xSt+1:xEn+1, ySt:yEn, zSt:zEn])/hx2 +
                            (U[xSt:xEn, ySt-1:yEn-1, zSt:zEn] - 2.0*U[xSt:xEn, ySt:yEn, zSt:zEn] + U[xSt:xEn, ySt+1:yEn+1, zSt:zEn])/hy2 +
                            (U[xSt:xEn, ySt:yEn, zSt-1:zEn-1] - 2.0*U[xSt:xEn, ySt:yEn, zSt:zEn] + U[xSt:xEn, ySt:yEn, zSt+1:zEn+1])/hz2))))
        

        totmaxErr = comm.allreduce(locmaxErr, op=MPI.MAX)

        if totmaxErr < VpTolerance:
            break
        
        jCnt += 1
        if jCnt > maxCount:
            print("ERROR: Jacobi not converging in U. Aborting")
            quit()

    return U[xSt:xEn, ySt:yEn, zSt:zEn]        


def vJacobi(rho):
        
    jCnt = 0
    while True:

        V[xSt:xEn, ySt:yEn, zSt:zEn] =(1.0/(1+nu*dt*(idx2 + idy2 + idz2))) * (rho[xSt:xEn, ySt:yEn, zSt:zEn] + 
                                       0.5*nu*dt*idx2*(V[xSt-1:xEn-1, ySt:yEn, zSt:zEn] + V[xSt+1:xEn+1, ySt:yEn, zSt:zEn]) +
                                       0.5*nu*dt*idy2*(V[xSt:xEn, ySt-1:yEn-1, zSt:zEn] + V[xSt:xEn, ySt+1:yEn+1, zSt:zEn]) +
                                       0.5*nu*dt*idz2*(V[xSt:xEn, ySt:yEn, zSt-1:zEn-1] + V[xSt:xEn, ySt:yEn, zSt+1:zEn+1]))  


        data_transfer(V)

        imposeVBCs(V)


        locmaxErr = np.amax(np.fabs(rho[xSt:xEn, ySt:yEn, zSt:zEn] - (V[xSt:xEn, ySt:yEn, zSt:zEn] - 0.5*nu*dt*(
                        (V[xSt-1:xEn-1, ySt:yEn, zSt:zEn] - 2.0*V[xSt:xEn, ySt:yEn, zSt:zEn] + V[xSt+1:xEn+1, ySt:yEn, zSt:zEn])/hx2 +
                        (V[xSt:xEn, ySt-1:yEn-1, zSt:zEn] - 2.0*V[xSt:xEn, ySt:yEn, zSt:zEn] + V[xSt:xEn, ySt+1:yEn+1, zSt:zEn])/hy2 +
                        (V[xSt:xEn, ySt:yEn, zSt-1:zEn-1] - 2.0*V[xSt:xEn, ySt:yEn, zSt:zEn] + V[xSt:xEn, ySt:yEn, zSt+1:zEn+1])/hz2))))
    
        totmaxErr = comm.allreduce(locmaxErr, op=MPI.MAX)

        if totmaxErr < VpTolerance:
            break
    
        jCnt += 1
        if jCnt > maxCount:
            print("ERROR: Jacobi not converging in V. Aborting")
            quit()
    
    return V[xSt:xEn, ySt:yEn, zSt:zEn]


def wJacobi(rho):
        
    jCnt = 0
    while True:

        W[xSt:xEn, ySt:yEn, zSt:zEn] =(1.0/(1+nu*dt*(idx2 + idy2 + idz2))) * (rho[xSt:xEn, ySt:yEn, zSt:zEn] + 
                                       0.5*nu*dt*idx2*(W[xSt-1:xEn-1, ySt:yEn, zSt:zEn] + W[xSt+1:xEn+1, ySt:yEn, zSt:zEn]) +
                                       0.5*nu*dt*idy2*(W[xSt:xEn, ySt-1:yEn-1, zSt:zEn] + W[xSt:xEn, ySt+1:yEn+1, zSt:zEn]) +
                                       0.5*nu*dt*idz2*(W[xSt:xEn, ySt:yEn, zSt-1:zEn-1] + W[xSt:xEn, ySt:yEn, zSt+1:zEn+1]))         

        data_transfer(W)
    
        imposeWBCs(W)


        locmaxErr = np.amax(np.fabs(rho[xSt:xEn, ySt:yEn, zSt:zEn] - (W[xSt:xEn, ySt:yEn, zSt:zEn] - 0.5*nu*dt*(
                        (W[xSt-1:xEn-1, ySt:yEn, zSt:zEn] - 2.0*W[xSt:xEn, ySt:yEn, zSt:zEn] + W[xSt+1:xEn+1, ySt:yEn, zSt:zEn])/hx2 +
                        (W[xSt:xEn, ySt-1:yEn-1, zSt:zEn] - 2.0*W[xSt:xEn, ySt:yEn, zSt:zEn] + W[xSt:xEn, ySt+1:yEn+1, zSt:zEn])/hy2 +
                        (W[xSt:xEn, ySt:yEn, zSt-1:zEn-1] - 2.0*W[xSt:xEn, ySt:yEn, zSt:zEn] + W[xSt:xEn, ySt:yEn, zSt+1:zEn+1])/hz2))))
    
        totmaxErr = comm.allreduce(locmaxErr, op=MPI.MAX)

        if totmaxErr < VpTolerance:
            break

        jCnt += 1
        if jCnt > maxCount:
            print("ERROR: Jacobi not converging in W. Aborting")
            quit()
    
    return W[xSt:xEn, ySt:yEn, zSt:zEn]       


def TJacobi(rho):
        
    jCnt = 0
    while True:

        T[xSt:xEn, ySt:yEn, zSt:zEn] =(1.0/(1+kappa*dt*(idx2 + idy2 + idz2))) * (rho[xSt:xEn, ySt:yEn, zSt:zEn] + 
                                       0.5*kappa*dt*idx2*(T[xSt-1:xEn-1, ySt:yEn, zSt:zEn] + T[xSt+1:xEn+1, ySt:yEn, zSt:zEn]) +
                                       0.5*kappa*dt*idy2*(T[xSt:xEn, ySt-1:yEn-1, zSt:zEn] + T[xSt:xEn, ySt+1:yEn+1, zSt:zEn]) +
                                       0.5*kappa*dt*idz2*(T[xSt:xEn, ySt:yEn, zSt-1:zEn-1] + T[xSt:xEn, ySt:yEn, zSt+1:zEn+1])) 

        data_transfer(T)

        imposeTBCs(T)

        locmaxErr = np.amax(np.fabs(rho[xSt:xEn, ySt:yEn, zSt:zEn] - (T[xSt:xEn, ySt:yEn, zSt:zEn] - 0.5*kappa*dt*(
                        (T[xSt-1:xEn-1, ySt:yEn, zSt:zEn] - 2.0*T[xSt:xEn, ySt:yEn, zSt:zEn] + T[xSt+1:xEn+1, ySt:yEn, zSt:zEn])/hx2 +
                        (T[xSt:xEn, ySt-1:yEn-1, zSt:zEn] - 2.0*T[xSt:xEn, ySt:yEn, zSt:zEn] + T[xSt:xEn, ySt+1:yEn+1, zSt:zEn])/hy2 +
                        (T[xSt:xEn, ySt:yEn, zSt-1:zEn-1] - 2.0*T[xSt:xEn, ySt:yEn, zSt:zEn] + T[xSt:xEn, ySt:yEn, zSt+1:zEn+1])/hz2))))
    
        totmaxErr = comm.allreduce(locmaxErr, op=MPI.MAX)

        if totmaxErr < VpTolerance:
            break
    
        jCnt += 1
        if jCnt > maxCount:
            print("ERROR: Jacobi not converging in T. Aborting")
            quit()
    
    return T[xSt:xEn, ySt:yEn, zSt:zEn]       


def PoissonSolver(rho):
            
    jCnt = 0
    
    while True:
    
        Pp[xSt:xEn, ySt:yEn, zSt:zEn] = (1.0/(-2.0*(idx2 + idy2 + idz2))) * (rho[xSt:xEn, ySt:yEn, zSt:zEn] - 
                                       idx2*(Pp[xSt-1:xEn-1, ySt:yEn, zSt:zEn] + Pp[xSt+1:xEn+1, ySt:yEn, zSt:zEn]) -
                                       idy2*(Pp[xSt:xEn, ySt-1:yEn-1, zSt:zEn] + Pp[xSt:xEn, ySt+1:yEn+1, zSt:zEn]) -
                                       idz2*(Pp[xSt:xEn, ySt:yEn, zSt-1:zEn-1] + Pp[xSt:xEn, ySt:yEn, zSt+1:zEn+1]))   

        data_transfer(Pp)

        imposePpBCs(Pp)
    
        locmaxErr = np.amax(np.fabs(rho[xSt:xEn, ySt:yEn, zSt:zEn] -((
                        (Pp[xSt-1:xEn-1, ySt:yEn, zSt:zEn] - 2.0*Pp[xSt:xEn, ySt:yEn, zSt:zEn] + Pp[xSt+1:xEn+1, ySt:yEn, zSt:zEn])/hx2 +
                        (Pp[xSt:xEn, ySt-1:yEn-1, zSt:zEn] - 2.0*Pp[xSt:xEn, ySt:yEn, zSt:zEn] + Pp[xSt:xEn, ySt+1:yEn+1, zSt:zEn])/hy2 +
                        (Pp[xSt:xEn, ySt:yEn, zSt-1:zEn-1] - 2.0*Pp[xSt:xEn, ySt:yEn, zSt:zEn] + Pp[xSt:xEn, ySt:yEn, zSt+1:zEn+1])/hz2))))
    
        totmaxErr = comm.allreduce(locmaxErr, op=MPI.MAX)

        if totmaxErr < PoissonTolerance:
            break
    
        jCnt += 1
        if jCnt > maxCount:
            print("ERROR: Poisson solver not converging. Aborting")
            quit()
    
    return Pp[xSt:xEn, ySt:yEn, zSt:zEn]     



def imposeUBCs(U):
    U[0, :, :], U[-1, :, :] = 0.0, 0.0
    U[:, 0, :], U[:, -1, :] = 0.0, 0.0
    U[:, :, 0], U[:, :, -1] = 0.0, 0.0

def imposeVBCs(V):
    V[0, :, :], V[-1, :, :] = 0.0, 0.0  
    V[:, 0, :], V[:, -1, :] = 0.0, 0.0  
    V[:, :, 0], V[:, :, -1] = 0.0, 0.0

def imposeWBCs(W):
    W[0, :, :], W[-1, :, :] = 0.0, 0.0 
    W[:, 0, :], W[:, -1, :] = 0.0, 0.0
    W[:, :, 0], W[:, :, -1] = 0.0, 0.0  

def imposeTBCs(T):
    T[0, :, :], T[-1, :, :] = T[1, :, :], T[-2, :, :]
    T[:, 0, :], T[:, -1, :] = T[:, 1, :], T[:, -2, :]
    T[:, :, 0], T[:, :, -1] = 1.0, 0.0

def imposePBCs(P):
    P[0, :, :], P[-1, :, :] = P[1, :, :], P[-2, :, :]
    P[:, 0, :], P[:, -1, :] = P[:, 1, :], P[:, -2, :]
    P[:, :, 0], P[:, :, -1] = P[:, :, 1], P[:, :, -2]

def imposePpBCs(Pp):
    Pp[0, :, :], Pp[-1, :, :] = 0.0, 0.0 #Pp[1, :, :], Pp[-2, :, :]
    Pp[:, 0, :], Pp[:, -1, :] = 0.0, 0.0 #Pp[:, 1, :], Pp[:, -2, :]
    Pp[:, :, 0], Pp[:, :, -1] = 0.0, 0.0 #Pp[:, :, 1], P[:, :, -2]


def main():
    iCnt = 1
    time = 0

    while True:

        t1 = datetime.now()

        if iCnt % opInt == 0:
            locU = np.sum(np.sqrt(U[kbn:ken, ySt:yEn, zSt:zEn]**2.0 + V[kbn:ken, ySt:yEn, zSt:zEn]**2.0 + W[kbn:ken, ySt:yEn, zSt:zEn]**2.0))
            globU = comm.reduce(locU, op=MPI.SUM, root=0)
        
            locWT = np.sum(W[kbn:ken, ySt:yEn, zSt:zEn]*T[kbn:ken, ySt:yEn, zSt:zEn])
            totalWT = comm.reduce(locWT, op=MPI.SUM, root=0)

            maxDiv = getDiv(U, V, W)

            if rootRank:
                Re = globU/(nu*Nx*Ny*Nz)
                Nu = 1.0 + totalWT/(kappa*Nx*Ny*Nz)
                print("%f    %f    %f    %f" %(time, Re, Nu, maxDiv))           


        Hx[xSt:xEn, ySt:yEn, zSt:zEn] = computeNLinDiff_X(U, V, W)
        Hy[xSt:xEn, ySt:yEn, zSt:zEn] = computeNLinDiff_Y(U, V, W)
        Hz[xSt:xEn, ySt:yEn, zSt:zEn] = computeNLinDiff_Z(U, V, W)
        Ht[xSt:xEn, ySt:yEn, zSt:zEn] = computeNLinDiff_T(U, V, W, T)  

        Hx[xSt:xEn, ySt:yEn, zSt:zEn] = U[xSt:xEn, ySt:yEn, zSt:zEn] + dt*(Hx[xSt:xEn, ySt:yEn, zSt:zEn] - np.sqrt((Ta*Pr)/Ra)*(-V[xSt:xEn, ySt:yEn, zSt:zEn]) - (P[xSt+1:xEn+1, ySt:yEn, zSt:zEn] - P[xSt-1:xEn-1, ySt:yEn, zSt:zEn])/(2.0*hx))
        uJacobi(Hx)

        Hy[xSt:xEn, ySt:yEn, zSt:zEn] = V[xSt:xEn, ySt:yEn, zSt:zEn] + dt*(Hy[xSt:xEn, ySt:yEn, zSt:zEn] - np.sqrt((Ta*Pr)/Ra)*(U[xSt:xEn, ySt:yEn, zSt:zEn]) - (P[xSt:xEn, ySt+1:yEn+1, zSt:zEn] - P[xSt:xEn, ySt-1:yEn-1, zSt:zEn])/(2.0*hy))
        vJacobi(Hy)

        Hz[xSt:xEn, ySt:yEn, zSt:zEn] = W[xSt:xEn, ySt:yEn, zSt:zEn] + dt*(Hz[xSt:xEn, ySt:yEn, zSt:zEn] - ((P[xSt:xEn, ySt:yEn, zSt+1:zEn+1] - P[xSt:xEn, ySt:yEn, zSt-1:zEn-1])/(2.0*hz)) + T[xSt:xEn, ySt:yEn, zSt:zEn])
        wJacobi(Hz)

        Ht[xSt:xEn, ySt:yEn, zSt:zEn] = T[xSt:xEn, ySt:yEn, zSt:zEn] + dt*Ht[xSt:xEn, ySt:yEn, zSt:zEn]
        TJacobi(Ht)   

        rhs = np.zeros([Nx, Ny, Nz])

        rhs[xSt:xEn, ySt:yEn, zSt:zEn] = ((U[xSt+1:xEn+1, ySt:yEn, zSt:zEn] - U[xSt-1:xEn-1, ySt:yEn, zSt:zEn])/(2.0*hx) +
                                          (V[xSt:xEn, ySt+1:yEn+1, zSt:zEn] - V[xSt:xEn, ySt-1:yEn-1, zSt:zEn])/(2.0*hy) +
                                          (W[xSt:xEn, ySt:yEn, zSt+1:zEn+1] - W[xSt:xEn, ySt:yEn, zSt-1:zEn-1])/(2.0*hz))/dt

        tp1 = datetime.now()
        Pp[xSt:xEn, ySt:yEn, zSt:zEn] = PoissonSolver(rhs)
        tp2 = datetime.now()
        #print(tp2-tp1)

        P[xSt:xEn, ySt:yEn, zSt:zEn] = P[xSt:xEn, ySt:yEn, zSt:zEn] + Pp[xSt:xEn, ySt:yEn, zSt:zEn]

        U[xSt:xEn, ySt:yEn, zSt:zEn] = U[xSt:xEn, ySt:yEn, zSt:zEn] - dt*(Pp[xSt+1:xEn+1, ySt:yEn, zSt:zEn] - Pp[xSt-1:xEn-1, ySt:yEn, zSt:zEn])/(2.0*hx)
        V[xSt:xEn, ySt:yEn, zSt:zEn] = V[xSt:xEn, ySt:yEn, zSt:zEn] - dt*(Pp[xSt:xEn, ySt+1:yEn+1, zSt:zEn] - Pp[xSt:xEn, ySt-1:yEn-1, zSt:zEn])/(2.0*hy)
        W[xSt:xEn, ySt:yEn, zSt:zEn] = W[xSt:xEn, ySt:yEn, zSt:zEn] - dt*(Pp[xSt:xEn, ySt:yEn, zSt+1:zEn+1] - Pp[xSt:xEn, ySt:yEn, zSt-1:zEn-1])/(2.0*hz)

        data_transfer(U)
        data_transfer(V)
        data_transfer(W)
        data_transfer(T)
        data_transfer(P)

        imposeUBCs(U)                               
        imposeVBCs(V)                               
        imposeWBCs(W)                               
        imposePBCs(P)                               
        imposeTBCs(T)           

        if time > tMax:
            break   

        time = time + dt

        iCnt = iCnt + 1

        t2 = datetime.now()

        #print("Time taken in one time step marching=",t2-t1)

main()
