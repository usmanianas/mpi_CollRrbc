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

Nx, Ny, Nz = 16, 16, 16

hx, hy, hz = Lx/(Nx-1), Ly/(Ny-1), Lz/(Nz-1)

x = np.linspace(0, Lx, Nx, endpoint=True)        
y = np.linspace(0, Ly, Ny, endpoint=True)
z = np.linspace(0, Lz, Nz, endpoint=True)    

hx2, hy2, hz2 = hx*hx, hy*hy, hz*hz

idx2, idy2, idz2 = 1.0/hx2, 1.0/hy2, 1.0/hz2

###############################################


############# Fields Initialization ###########

# Field variables
U = np.zeros([Nx, Ny, Nz])
V = np.zeros([Nx, Ny, Nz])
W = np.zeros([Nx, Ny, Nz])
T = np.zeros([Nx, Ny, Nz])
P = np.zeros([Nx, Ny, Nz])

# Auxilliary variables
Pp = np.zeros([Nx, Ny, Nz])

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


############# MPI Parallelization #############

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

locn = int(Nz/nprocs)
bn = 1 + locn*rank
en = bn + locn 

rootRank = rank == 0

lftRank = rank - 1
rgtRank = rank + 1

kbn = bn
if rank == 0:
    kbn = bn-1
    lftRank = MPI.PROC_NULL

if rank == nprocs-1:
    en = Nx-1
    rgtRank = MPI.PROC_NULL

ken = en
if rank == nprocs-1:
    ken = Nx

if rank == 0:
    print('# Grid', Nx, Ny, Nz)
    print('#No. of Processors =',nprocs)

print('#', rank, bn, en)

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

# Number of iterations after which output must be printed to standard I/O
opInt = 1

# File writing interval
fwInt = 2

# Tolerance value in Jacobi iterations
VpTolerance = 1.0e-5

# Tolerance value in Poisson iterations
PoissonTolerance = 1.0e-3

gssor = 1.6  # omega for SOR

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

    divMat = ((U[bn+1:en+1, 1:Ny-1, 1:Nz-1] - U[bn-1:en-1, 1:Ny-1, 1:Nz-1])*0.5/hx +
              (V[bn:en, 2:Ny, 1:Nz-1] - V[bn:en, 0:Ny-2, 1:Nz-1])*0.5/hy +
              (W[bn:en, 1:Ny-1, 2:Nz] - W[bn:en, 1:Ny-1, 0:Nz-2])*0.5/hz)
    
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

    Hx[bn:en, 1:Ny-1, 1:Nz-1] = (((U[bn+1:en+1, 1:Ny-1, 1:Nz-1] - 2.0*U[bn:en, 1:Ny-1, 1:Nz-1] + U[bn-1:en-1, 1:Ny-1, 1:Nz-1])/hx2 + 
                                (U[bn:en, 2:Ny, 1:Nz-1] - 2.0*U[bn:en, 1:Ny-1, 1:Nz-1] + U[bn:en, 0:Ny-2, 1:Nz-1])/hy2 + 
                                (U[bn:en, 1:Ny-1, 2:Nz] - 2.0*U[bn:en, 1:Ny-1, 1:Nz-1] + U[bn:en, 1:Ny-1, 0:Nz-2])/hz2)*0.5*nu -
                              U[bn:en, 1:Ny-1, 1:Nz-1]*(U[bn+1:en+1, 1:Ny-1, 1:Nz-1] - U[bn-1:en-1, 1:Ny-1, 1:Nz-1])/(2.0*hx) -
                              V[bn:en, 1:Ny-1, 1:Nz-1]*(U[bn:en, 2:Ny, 1:Nz-1] - U[bn:en, 0:Ny-2, 1:Nz-1])/(2.0*hy) - 
                              W[bn:en, 1:Ny-1, 1:Nz-1]*(U[bn:en, 1:Ny-1, 2:Nz] - U[bn:en, 1:Ny-1, 0:Nz-2])/(2.0*hz))

    return Hx[bn:en, 1:Ny-1, 1:Nz-1]


def computeNLinDiff_Y(U, V, W):

    Hy[bn:en, 1:Ny-1, 1:Nz-1] = (((V[bn+1:en+1, 1:Ny-1, 1:Nz-1] - 2.0*V[bn:en, 1:Ny-1, 1:Nz-1] + V[bn-1:en-1, 1:Ny-1, 1:Nz-1])/hx2 + 
                                (V[bn:en, 2:Ny, 1:Nz-1] - 2.0*V[bn:en, 1:Ny-1, 1:Nz-1] + V[bn:en, 0:Ny-2, 1:Nz-1])/hy2 + 
                                (V[bn:en, 1:Ny-1, 2:Nz] - 2.0*V[bn:en, 1:Ny-1, 1:Nz-1] + V[bn:en, 1:Ny-1, 0:Nz-2])/hz2)*0.5*nu -
                              U[bn:en, 1:Ny-1, 1:Nz-1]*(V[bn+1:en+1, 1:Ny-1, 1:Nz-1] - V[bn-1:en-1, 1:Ny-1, 1:Nz-1])/(2.0*hx) -
                              V[bn:en, 1:Ny-1, 1:Nz-1]*(V[bn:en, 2:Ny, 1:Nz-1] - V[bn:en, 0:Ny-2, 1:Nz-1])/(2.0*hy) - 
                              W[bn:en, 1:Ny-1, 1:Nz-1]*(V[bn:en, 1:Ny-1, 2:Nz] - V[bn:en, 1:Ny-1, 0:Nz-2])/(2.0*hz))

    return Hy[bn:en, 1:Ny-1, 1:Nz-1]


def computeNLinDiff_Z(U, V, W):
    global Hz
    global Nz, Ny, Nx, Nx, Ny, Nz

    Hz[bn:en, 1:Ny-1, 1:Nz-1] = (((W[bn+1:en+1, 1:Ny-1, 1:Nz-1] - 2.0*W[bn:en, 1:Ny-1, 1:Nz-1] + W[bn-1:en-1, 1:Ny-1, 1:Nz-1])/hx2 + 
                                (W[bn:en, 2:Ny, 1:Nz-1] - 2.0*W[bn:en, 1:Ny-1, 1:Nz-1] + W[bn:en, 0:Ny-2, 1:Nz-1])/hy2 + 
                                (W[bn:en, 1:Ny-1, 2:Nz] - 2.0*W[bn:en, 1:Ny-1, 1:Nz-1] + W[bn:en, 1:Ny-1, 0:Nz-2])/hz2)*0.5*nu -
                              U[bn:en, 1:Ny-1, 1:Nz-1]*(W[bn+1:en+1, 1:Ny-1, 1:Nz-1] - W[bn-1:en-1, 1:Ny-1, 1:Nz-1])/(2.0*hx) -
                              V[bn:en, 1:Ny-1, 1:Nz-1]*(W[bn:en, 2:Ny, 1:Nz-1] - W[bn:en, 0:Ny-2, 1:Nz-1])/(2.0*hy) - 
                              W[bn:en, 1:Ny-1, 1:Nz-1]*(W[bn:en, 1:Ny-1, 2:Nz] - W[bn:en, 1:Ny-1, 0:Nz-2])/(2.0*hz))


    return Hz[bn:en, 1:Ny-1, 1:Nz-1]


def computeNLinDiff_T(U, V, W, T):
    global Ht
    global Nz, Ny, Nx

    Ht[bn:en, 1:Ny-1, 1:Nz-1] = (((T[bn+1:en+1, 1:Ny-1, 1:Nz-1] - 2.0*T[bn:en, 1:Ny-1, 1:Nz-1] + T[bn-1:en-1, 1:Ny-1, 1:Nz-1])/hx2 + 
                                (T[bn:en, 2:Ny, 1:Nz-1] - 2.0*T[bn:en, 1:Ny-1, 1:Nz-1] + T[bn:en, 0:Ny-2, 1:Nz-1])/hy2 + 
                                (T[bn:en, 1:Ny-1, 2:Nz] - 2.0*T[bn:en, 1:Ny-1, 1:Nz-1] + T[bn:en, 1:Ny-1, 0:Nz-2])/hz2)*0.5*kappa -
                              U[bn:en, 1:Ny-1, 1:Nz-1]*(T[bn+1:en+1, 1:Ny-1, 1:Nz-1] - T[bn-1:en-1, 1:Ny-1, 1:Nz-1])/(2.0*hx)-
                              V[bn:en, 1:Ny-1, 1:Nz-1]*(T[bn:en, 2:Ny, 1:Nz-1] - T[bn:en, 0:Ny-2, 1:Nz-1])/(2.0*hy) - 
                              W[bn:en, 1:Ny-1, 1:Nz-1]*(T[bn:en, 1:Ny-1, 2:Nz] - T[bn:en, 1:Ny-1, 0:Nz-2])/(2.0*hz))

    return Ht[bn:en, 1:Ny-1, 1:Nz-1]


def uJacobi(rho):

    jCnt = 0
    while True:

        U[bn:en, 1:Ny-1, 1:Nz-1] =(1.0/(1+nu*dt*(idx2 + idy2 + idz2))) * (rho[bn:en, 1:Ny-1, 1:Nz-1] + 
                                       0.5*nu*dt*idx2*(U[bn-1:en-1, 1:Ny-1, 1:Nz-1] + U[bn+1:en+1, 1:Ny-1, 1:Nz-1]) +
                                       0.5*nu*dt*idy2*(U[bn:en, 0:Ny-2, 1:Nz-1] + U[bn:en, 2:Ny, 1:Nz-1]) +
                                       0.5*nu*dt*idz2*(U[bn:en, 1:Ny-1, 0:Nz-2] + U[bn:en, 1:Ny-1, 2:Nz]))          

        data_transfer(U)

        imposeUBCs(U)
        
        locmaxErr = np.amax(np.fabs(rho[bn:en, 1:Ny-1, 1:Nz-1] - (U[bn:en, 1:Ny-1, 1:Nz-1] - 0.5*nu*dt*(
                            (U[bn-1:en-1, 1:Ny-1, 1:Nz-1] - 2.0*U[bn:en, 1:Ny-1, 1:Nz-1] + U[bn+1:en+1, 1:Ny-1, 1:Nz-1])/hx2 +
                            (U[bn:en, 0:Ny-2, 1:Nz-1] - 2.0*U[bn:en, 1:Ny-1, 1:Nz-1] + U[bn:en, 2:Ny, 1:Nz-1])/hy2 +
                            (U[bn:en, 1:Ny-1, 0:Nz-2] - 2.0*U[bn:en, 1:Ny-1, 1:Nz-1] + U[bn:en, 1:Ny-1, 2:Nz])/hz2))))
        

        totmaxErr = comm.allreduce(locmaxErr, op=MPI.MAX)

        if totmaxErr < VpTolerance:
            #print(jCnt)
            break
        
        jCnt += 1
        if jCnt > maxCount:
                print("ERROR: Jacobi not converging in U. Aborting")
                quit()

    return U[bn:en, 1:Ny-1, 1:Nz-1]        


def vJacobi(rho):
        
    jCnt = 0
    while True:

        V[bn:en, 1:Ny-1, 1:Nz-1] =(1.0/(1+nu*dt*(idx2 + idy2 + idz2))) * (rho[bn:en, 1:Ny-1, 1:Nz-1] + 
                                       0.5*nu*dt*idx2*(V[bn-1:en-1, 1:Ny-1, 1:Nz-1] + V[bn+1:en+1, 1:Ny-1, 1:Nz-1]) +
                                       0.5*nu*dt*idy2*(V[bn:en, 0:Ny-2, 1:Nz-1] + V[bn:en, 2:Ny, 1:Nz-1]) +
                                       0.5*nu*dt*idz2*(V[bn:en, 1:Ny-1, 0:Nz-2] + V[bn:en, 1:Ny-1, 2:Nz]))  


        data_transfer(V)

        imposeVBCs(V)


        locmaxErr = np.amax(np.fabs(rho[bn:en, 1:Ny-1, 1:Nz-1] - (V[bn:en, 1:Ny-1, 1:Nz-1] - 0.5*nu*dt*(
                        (V[bn-1:en-1, 1:Ny-1, 1:Nz-1] - 2.0*V[bn:en, 1:Ny-1, 1:Nz-1] + V[bn+1:en+1, 1:Ny-1, 1:Nz-1])/hx2 +
                        (V[bn:en, 0:Ny-2, 1:Nz-1] - 2.0*V[bn:en, 1:Ny-1, 1:Nz-1] + V[bn:en, 2:Ny, 1:Nz-1])/hy2 +
                        (V[bn:en, 1:Ny-1, 0:Nz-2] - 2.0*V[bn:en, 1:Ny-1, 1:Nz-1] + V[bn:en, 1:Ny-1, 2:Nz])/hz2))))
    
        totmaxErr = comm.allreduce(locmaxErr, op=MPI.MAX)

        if totmaxErr < VpTolerance:
            #print(jCnt)
            break
    
        jCnt += 1
        if jCnt > maxCount:
            print("ERROR: Jacobi not converging in V. Aborting")
            quit()
    
    return V[bn:en, 1:Ny-1, 1:Nz-1]


def wJacobi(rho):
        
    jCnt = 0
    while True:

        W[bn:en, 1:Ny-1, 1:Nz-1] =(1.0/(1+nu*dt*(idx2 + idy2 + idz2))) * (rho[bn:en, 1:Ny-1, 1:Nz-1] + 
                                       0.5*nu*dt*idx2*(W[bn-1:en-1, 1:Ny-1, 1:Nz-1] + W[bn+1:en+1, 1:Ny-1, 1:Nz-1]) +
                                       0.5*nu*dt*idy2*(W[bn:en, 0:Ny-2, 1:Nz-1] + W[bn:en, 2:Ny, 1:Nz-1]) +
                                       0.5*nu*dt*idz2*(W[bn:en, 1:Ny-1, 0:Nz-2] + W[bn:en, 1:Ny-1, 2:Nz]))         

        data_transfer(W)
    
        imposeWBCs(W)


        locmaxErr = np.amax(np.fabs(rho[bn:en, 1:Ny-1, 1:Nz-1] - (W[bn:en, 1:Ny-1, 1:Nz-1] - 0.5*nu*dt*(
                        (W[bn-1:en-1, 1:Ny-1, 1:Nz-1] - 2.0*W[bn:en, 1:Ny-1, 1:Nz-1] + W[bn+1:en+1, 1:Ny-1, 1:Nz-1])/hx2 +
                        (W[bn:en, 0:Ny-2, 1:Nz-1] - 2.0*W[bn:en, 1:Ny-1, 1:Nz-1] + W[bn:en, 2:Ny, 1:Nz-1])/hy2 +
                        (W[bn:en, 1:Ny-1, 0:Nz-2] - 2.0*W[bn:en, 1:Ny-1, 1:Nz-1] + W[bn:en, 1:Ny-1, 2:Nz])/hz2))))
    
        totmaxErr = comm.allreduce(locmaxErr, op=MPI.MAX)

        if totmaxErr < VpTolerance:
            #print(jCnt)
            break


        jCnt += 1
        if jCnt > maxCount:
            print("ERROR: Jacobi not converging in W. Aborting")
            quit()
    
    return W[bn:en, 1:Ny-1, 1:Nz-1]       


def TJacobi(rho):
        
    jCnt = 0
    while True:

        T[bn:en, 1:Ny-1, 1:Nz-1] =(1.0/(1+kappa*dt*(idx2 + idy2 + idz2))) * (rho[bn:en, 1:Ny-1, 1:Nz-1] + 
                                       0.5*kappa*dt*idx2*(T[bn-1:en-1, 1:Ny-1, 1:Nz-1] + T[bn+1:en+1, 1:Ny-1, 1:Nz-1]) +
                                       0.5*kappa*dt*idy2*(T[bn:en, 0:Ny-2, 1:Nz-1] + T[bn:en, 2:Ny, 1:Nz-1]) +
                                       0.5*kappa*dt*idz2*(T[bn:en, 1:Ny-1, 0:Nz-2] + T[bn:en, 1:Ny-1, 2:Nz])) 

        data_transfer(T)

        imposeTBCs(T)

        locmaxErr = np.amax(np.fabs(rho[bn:en, 1:Ny-1, 1:Nz-1] - (T[bn:en, 1:Ny-1, 1:Nz-1] - 0.5*kappa*dt*(
                        (T[bn-1:en-1, 1:Ny-1, 1:Nz-1] - 2.0*T[bn:en, 1:Ny-1, 1:Nz-1] + T[bn+1:en+1, 1:Ny-1, 1:Nz-1])/hx2 +
                        (T[bn:en, 0:Ny-2, 1:Nz-1] - 2.0*T[bn:en, 1:Ny-1, 1:Nz-1] + T[bn:en, 2:Ny, 1:Nz-1])/hy2 +
                        (T[bn:en, 1:Ny-1, 0:Nz-2] - 2.0*T[bn:en, 1:Ny-1, 1:Nz-1] + T[bn:en, 1:Ny-1, 2:Nz])/hz2))))
    
        totmaxErr = comm.allreduce(locmaxErr, op=MPI.MAX)

        if totmaxErr < VpTolerance:
            #print(jCnt)
            break
    
        jCnt += 1
        if jCnt > maxCount:
            print("ERROR: Jacobi not converging in T. Aborting")
            quit()
    
    return T[bn:en, 1:Ny-1, 1:Nz-1]       



def PoissonSolver(rho):
            
    jCnt = 0
    
    while True:
    
        '''
        Ppp = Pp.copy()
        for i in range(1,Nx-1):
            for j in range(1,Ny-1):
                for k in range(1,Nz-1):
                    Pp[i,j,k] = (1.0-gssor)*Ppp[i,j,k] + (gssor/(-2.0*(idx2 + idy2 + idz2))) * (rho[i, j, k] - 
                                       idx2*(Pp[i+1, j, k] + Pp[i-1, j, k]) -
                                       idy2*(Pp[i, j+1, k] + Pp[i, j-1, k]) -
                                       idz2*(Pp[i, j, k+1] + Pp[i, j, k-1]))
        '''
            
        Pp[bn:en, 1:Ny-1, 1:Nz-1] = (1.0/(-2.0*(idx2 + idy2 + idz2))) * (rho[bn:en, 1:Ny-1, 1:Nz-1] - 
                                       idx2*(Pp[bn-1:en-1, 1:Ny-1, 1:Nz-1] + Pp[bn+1:en+1, 1:Ny-1, 1:Nz-1]) -
                                       idy2*(Pp[bn:en, 0:Ny-2, 1:Nz-1] + Pp[bn:en, 2:Ny, 1:Nz-1]) -
                                       idz2*(Pp[bn:en, 1:Ny-1, 0:Nz-2] + Pp[bn:en, 1:Ny-1, 2:Nz]))   

        data_transfer(Pp)

        imposePpBCs(Pp)
    
        locmaxErr = np.amax(np.fabs(rho[bn:en, 1:Ny-1, 1:Nz-1] -((
                        (Pp[bn-1:en-1, 1:Ny-1, 1:Nz-1] - 2.0*Pp[bn:en, 1:Ny-1, 1:Nz-1] + Pp[bn+1:en+1, 1:Ny-1, 1:Nz-1])/hx2 +
                        (Pp[bn:en, 0:Ny-2, 1:Nz-1] - 2.0*Pp[bn:en, 1:Ny-1, 1:Nz-1] + Pp[bn:en, 2:Ny, 1:Nz-1])/hy2 +
                        (Pp[bn:en, 1:Ny-1, 0:Nz-2] - 2.0*Pp[bn:en, 1:Ny-1, 1:Nz-1] + Pp[bn:en, 1:Ny-1, 2:Nz])/hz2))))
    
        totmaxErr = comm.allreduce(locmaxErr, op=MPI.MAX)

        if totmaxErr < PoissonTolerance:
            #print(jCnt)
            break
    
        jCnt += 1
        if jCnt > maxCount:
            print("ERROR: Poisson solver not converging. Aborting")
            quit()
    
    return Pp[bn:en, 1:Ny-1, 1:Nz-1]     



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


iCnt = 1
time = 0

while True:

    t1 = datetime.now()

    if iCnt % opInt == 0:

        locU = np.sum(np.sqrt(U[kbn:ken, 1:Ny-1, 1:Nz-1]**2.0 + V[kbn:ken, 1:Ny-1, 1:Nz-1]**2.0 + W[kbn:ken, 1:Ny-1, 1:Nz-1]**2.0))
        globU = comm.reduce(locU, op=MPI.SUM, root=0)
    
        #locWT = np.sum(W[kbn:ken, :, :]*T[kbn:ken, :, :])
        locWT = np.sum(W[kbn:ken, 1:Ny-1, 1:Nz-1]*T[kbn:ken, 1:Ny-1, 1:Nz-1])
        totalWT = comm.reduce(locWT, op=MPI.SUM, root=0)

        maxDiv = getDiv(U, V, W)

        if rootRank:
            Re = globU/(nu*Nx*Ny*Nz)
            Nu = 1.0 + totalWT/(kappa*Nx*Ny*Nz)
            print("%f    %f    %f    %f" %(time, Re, Nu, maxDiv))           


    Hx[bn:en, 1:Ny-1, 1:Nz-1] = computeNLinDiff_X(U, V, W)
    Hy[bn:en, 1:Ny-1, 1:Nz-1] = computeNLinDiff_Y(U, V, W)
    Hz[bn:en, 1:Ny-1, 1:Nz-1] = computeNLinDiff_Z(U, V, W)
    Ht[bn:en, 1:Ny-1, 1:Nz-1] = computeNLinDiff_T(U, V, W, T)  

    Hx[bn:en, 1:Ny-1, 1:Nz-1] = U[bn:en, 1:Ny-1, 1:Nz-1] + dt*(Hx[bn:en, 1:Ny-1, 1:Nz-1] - np.sqrt((Ta*Pr)/Ra)*(-V[bn:en, 1:Ny-1, 1:Nz-1]) - (P[bn+1:en+1, 1:Ny-1, 1:Nz-1] - P[bn-1:en-1, 1:Ny-1, 1:Nz-1])/(2.0*hx))
    uJacobi(Hx)

    Hy[bn:en, 1:Ny-1, 1:Nz-1] = V[bn:en, 1:Ny-1, 1:Nz-1] + dt*(Hy[bn:en, 1:Ny-1, 1:Nz-1] - np.sqrt((Ta*Pr)/Ra)*(U[bn:en, 1:Ny-1, 1:Nz-1]) - (P[bn:en, 2:Ny, 1:Nz-1] - P[bn:en, 0:Ny-2, 1:Nz-1])/(2.0*hy))
    vJacobi(Hy)

    Hz[bn:en, 1:Ny-1, 1:Nz-1] = W[bn:en, 1:Ny-1, 1:Nz-1] + dt*(Hz[bn:en, 1:Ny-1, 1:Nz-1] - ((P[bn:en, 1:Ny-1, 2:Nz] - P[bn:en, 1:Ny-1, 0:Nz-2])/(2.0*hz)) + T[bn:en, 1:Ny-1, 1:Nz-1])
    wJacobi(Hz)

    Ht[bn:en, 1:Ny-1, 1:Nz-1] = T[bn:en, 1:Ny-1, 1:Nz-1] + dt*Ht[bn:en, 1:Ny-1, 1:Nz-1]
    TJacobi(Ht)   

    rhs = np.zeros([Nx, Ny, Nz])

    rhs[bn:en, 1:Ny-1, 1:Nz-1] = ((U[bn+1:en+1, 1:Ny-1, 1:Nz-1] - U[bn-1:en-1, 1:Ny-1, 1:Nz-1])/(2.0*hx) +
                                (V[bn:en, 2:Ny, 1:Nz-1] - V[bn:en, 0:Ny-2, 1:Nz-1])/(2.0*hy) +
                                (W[bn:en, 1:Ny-1, 2:Nz] - W[bn:en, 1:Ny-1, 0:Nz-2])/(2.0*hz))/dt

    tp1 = datetime.now()
    Pp[bn:en, 1:Ny-1, 1:Nz-1] = PoissonSolver(rhs)
    tp2 = datetime.now()
    #print(tp2-tp1)

    P[bn:en, 1:Ny-1, 1:Nz-1] = P[bn:en, 1:Ny-1, 1:Nz-1] + Pp[bn:en, 1:Ny-1, 1:Nz-1]

    U[bn:en, 1:Ny-1, 1:Nz-1] = U[bn:en, 1:Ny-1, 1:Nz-1] - dt*(Pp[bn+1:en+1, 1:Ny-1, 1:Nz-1] - Pp[bn-1:en-1, 1:Ny-1, 1:Nz-1])/(2.0*hx)
    V[bn:en, 1:Ny-1, 1:Nz-1] = V[bn:en, 1:Ny-1, 1:Nz-1] - dt*(Pp[bn:en, 2:Ny, 1:Nz-1] - Pp[bn:en, 0:Ny-2, 1:Nz-1])/(2.0*hy)
    W[bn:en, 1:Ny-1, 1:Nz-1] = W[bn:en, 1:Ny-1, 1:Nz-1] - dt*(Pp[bn:en, 1:Ny-1, 2:Nz] - Pp[bn:en, 1:Ny-1, 0:Nz-2])/(2.0*hz)

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

#main()
