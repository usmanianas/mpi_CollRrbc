from datetime import datetime
from mpi4py import MPI
import numpy as np
import h5py as hp

############### Grid Parameters ###############

Lx, Ly, Lz = 1.0, 1.0, 1.0

Nx, Ny, Nz = 32, 32, 32

hx, hy, hz = Lx/(Nx), Ly/(Ny), Lz/(Nz)

x = np.linspace(0, Lx + hx, Nx + 2, endpoint=True) - hx/2
y = np.linspace(0, Ly + hx, Ny + 2, endpoint=True) - hy/2
z = np.linspace(0, Lz + hx, Nz + 2, endpoint=True) - hz/2

hx2, hy2, hz2 = hx*hx, hy*hy, hz*hz

idx2, idy2, idz2 = 1.0/hx2, 1.0/hy2, 1.0/hz2

###############################################

############# MPI Parallelization #############

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

locNx = int(Nx/nprocs)
xSize = locNx + 2

rootRank = rank == 0
frstRank = rank == 0
lastRank = rank == nprocs - 1

lftRank = rank - 1
rgtRank = rank + 1

xSt, ySt, zSt = 1, 1, 1
xEn, yEn, zEn = locNx+1, Ny-1, Nz-1

if frstRank:
    lftRank = MPI.PROC_NULL

if lastRank:
    rgtRank = MPI.PROC_NULL

x0 = slice(xSt, xEn)
xm1 = slice(xSt-1, xEn-1)
xp1 = slice(xSt+1, xEn+1)

y0 = slice(ySt, yEn)
ym1 = slice(ySt-1, yEn-1)
yp1 = slice(ySt+1, yEn+1)

z0 = slice(zSt, zEn)
zm1 = slice(zSt-1, zEn-1)
zp1 = slice(zSt+1, zEn+1)

if rootRank:
    print('# Grid', Nx, Ny, Nz)
    print('#No. of Processors =',nprocs)

print('#', rank, xSt, xEn, xSize)

###############################################

############# Fields Initialization ###########

# Field variables
U = np.zeros([xSize, Ny+2, Nz+2])
V = np.zeros([xSize, Ny+2, Nz+2])
W = np.zeros([xSize, Ny+2, Nz+2])
T = np.zeros([xSize, Ny+2, Nz+2])
P = np.zeros([xSize, Ny+2, Nz+2])

# Auxilliary variables
Pp = np.zeros([xSize, Ny+2, Nz+2])

# RHS Terms
Hx = np.zeros_like(U)
Hy = np.zeros_like(V)
Hz = np.zeros_like(W)
Ht = np.zeros_like(T)   
Pp = np.zeros_like(P)

# Initialize values
P.fill(1.0)
T[:, :, :] = 1 - z[:]

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
tMax = 1.0

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
maxCount = 1000

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

    divMat = ((U[xp1, y0, z0] - U[xm1, y0, z0])*0.5/hx +
              (V[x0, yp1, z0] - V[x0, ym1, z0])*0.5/hy +
              (W[x0, y0, zp1] - W[x0, y0, zm1])*0.5/hz)
    
    locdivMax = np.max(abs(divMat))

    globdivMax = comm.reduce(locdivMax, op=MPI.MAX, root=0)

    return globdivMax


def data_transfer(F):
    if nprocs > 1:
        comm.Irecv(F[-1, :, :], source = rgtRank)
        comm.Irecv(F[0, :, :], source = lftRank)  

        comm.Send(F[-2, :, :], dest = rgtRank)
        comm.Send(F[1, :, :], dest = lftRank)


def computeNLinDiff_X(U, V, W):
    global Hx
    global hx2, hy2, hz2
    global nu, hx, hy, hz
    global xSt, xEn, ySt, yEn, zSt, zEn

    Hx[x0, y0, z0] = (((U[xp1, y0, z0] - 2.0*U[x0, y0, z0] + U[xm1, y0, z0])/hx2 + 
                       (U[x0, yp1, z0] - 2.0*U[x0, y0, z0] + U[x0, ym1, z0])/hy2 + 
                       (U[x0, y0, zp1] - 2.0*U[x0, y0, z0] + U[x0, y0, zm1])/hz2)*0.5*nu -
                        U[x0, y0, z0]*(U[xp1, y0, z0] - U[xm1, y0, z0])/(2.0*hx) -
                        V[x0, y0, z0]*(U[x0, yp1, z0] - U[x0, ym1, z0])/(2.0*hy) - 
                        W[x0, y0, z0]*(U[x0, y0, zp1] - U[x0, y0, zm1])/(2.0*hz))

    return Hx[x0, y0, z0]


def computeNLinDiff_Y(U, V, W):
    global Hy
    global hx2, hy2, hz2
    global nu, hx, hy, hz
    global xSt, xEn, ySt, yEn, zSt, zEn

    Hy[x0, y0, z0] = (((V[xp1, y0, z0] - 2.0*V[x0, y0, z0] + V[xm1, y0, z0])/hx2 + 
                       (V[x0, yp1, z0] - 2.0*V[x0, y0, z0] + V[x0, ym1, z0])/hy2 + 
                       (V[x0, y0, zp1] - 2.0*V[x0, y0, z0] + V[x0, y0, zm1])/hz2)*0.5*nu -
                        U[x0, y0, z0]*(V[xp1, y0, z0] - V[xm1, y0, z0])/(2.0*hx) -
                        V[x0, y0, z0]*(V[x0, yp1, z0] - V[x0, ym1, z0])/(2.0*hy) - 
                        W[x0, y0, z0]*(V[x0, y0, zp1] - V[x0, y0, zm1])/(2.0*hz))

    return Hy[x0, y0, z0]


def computeNLinDiff_Z(U, V, W):
    global Hz
    global hx2, hy2, hz2
    global nu, hx, hy, hz
    global xSt, xEn, ySt, yEn, zSt, zEn

    Hz[x0, y0, z0] = (((W[xp1, y0, z0] - 2.0*W[x0, y0, z0] + W[xm1, y0, z0])/hx2 + 
                       (W[x0, yp1, z0] - 2.0*W[x0, y0, z0] + W[x0, ym1, z0])/hy2 + 
                       (W[x0, y0, zp1] - 2.0*W[x0, y0, z0] + W[x0, y0, zm1])/hz2)*0.5*nu -
                        U[x0, y0, z0]*(W[xp1, y0, z0] - W[xm1, y0, z0])/(2.0*hx) -
                        V[x0, y0, z0]*(W[x0, yp1, z0] - W[x0, ym1, z0])/(2.0*hy) - 
                        W[x0, y0, z0]*(W[x0, y0, zp1] - W[x0, y0, zm1])/(2.0*hz))


    return Hz[x0, y0, z0]


def computeNLinDiff_T(U, V, W, T):
    global Ht
    global hx2, hy2, hz2
    global kappa, hx, hy, hz
    global xSt, xEn, ySt, yEn, zSt, zEn

    Ht[x0, y0, z0] = (((T[xp1, y0, z0] - 2.0*T[x0, y0, z0] + T[xm1, y0, z0])/hx2 + 
                       (T[x0, yp1, z0] - 2.0*T[x0, y0, z0] + T[x0, ym1, z0])/hy2 + 
                       (T[x0, y0, zp1] - 2.0*T[x0, y0, z0] + T[x0, y0, zm1])/hz2)*0.5*kappa -
                        U[x0, y0, z0]*(T[xp1, y0, z0] - T[xm1, y0, z0])/(2.0*hx)-
                        V[x0, y0, z0]*(T[x0, yp1, z0] - T[x0, ym1, z0])/(2.0*hy) - 
                        W[x0, y0, z0]*(T[x0, y0, zp1] - T[x0, y0, zm1])/(2.0*hz))

    return Ht[x0, y0, z0]


def uJacobi(rho):

    jCnt = 0
    while True:

        U[x0, y0, z0] = (1.0/(1 + nu*dt*(idx2 + idy2 + idz2))) * (rho[x0, y0, z0] + 
                                       0.5*nu*dt*idx2*(U[xm1, y0, z0] + U[xp1, y0, z0]) +
                                       0.5*nu*dt*idy2*(U[x0, ym1, z0] + U[x0, yp1, z0]) +
                                       0.5*nu*dt*idz2*(U[x0, y0, zm1] + U[x0, y0, zp1]))          

        imposeUBCs(U)
        
        locmaxErr = np.amax(np.fabs(rho[x0, y0, z0] - (U[x0, y0, z0] - 0.5*nu*dt*(
                            (U[xm1, y0, z0] - 2.0*U[x0, y0, z0] + U[xp1, y0, z0])/hx2 +
                            (U[x0, ym1, z0] - 2.0*U[x0, y0, z0] + U[x0, yp1, z0])/hy2 +
                            (U[x0, y0, zm1] - 2.0*U[x0, y0, z0] + U[x0, y0, zp1])/hz2))))
        

        totmaxErr = comm.allreduce(locmaxErr, op=MPI.MAX)

        if totmaxErr < VpTolerance:
            break
        
        jCnt += 1
        if jCnt > maxCount:
            print("ERROR: Jacobi not converging in U. Aborting")
            quit()

    return U[x0, y0, z0]        


def vJacobi(rho):
        
    jCnt = 0
    while True:

        V[x0, y0, z0] = (1.0/(1 + nu*dt*(idx2 + idy2 + idz2))) * (rho[x0, y0, z0] + 
                                       0.5*nu*dt*idx2*(V[xm1, y0, z0] + V[xp1, y0, z0]) +
                                       0.5*nu*dt*idy2*(V[x0, ym1, z0] + V[x0, yp1, z0]) +
                                       0.5*nu*dt*idz2*(V[x0, y0, zm1] + V[x0, y0, zp1]))  

        imposeVBCs(V)

        locmaxErr = np.amax(np.fabs(rho[x0, y0, z0] - (V[x0, y0, z0] - 0.5*nu*dt*(
                        (V[xm1, y0, z0] - 2.0*V[x0, y0, z0] + V[xp1, y0, z0])/hx2 +
                        (V[x0, ym1, z0] - 2.0*V[x0, y0, z0] + V[x0, yp1, z0])/hy2 +
                        (V[x0, y0, zm1] - 2.0*V[x0, y0, z0] + V[x0, y0, zp1])/hz2))))
    
        totmaxErr = comm.allreduce(locmaxErr, op=MPI.MAX)

        if totmaxErr < VpTolerance:
            break
    
        jCnt += 1
        if jCnt > maxCount:
            print("ERROR: Jacobi not converging in V. Aborting")
            quit()
    
    return V[x0, y0, z0]


def wJacobi(rho):
        
    jCnt = 0
    while True:

        W[x0, y0, z0] = (1.0/(1 + nu*dt*(idx2 + idy2 + idz2))) * (rho[x0, y0, z0] + 
                                       0.5*nu*dt*idx2*(W[xm1, y0, z0] + W[xp1, y0, z0]) +
                                       0.5*nu*dt*idy2*(W[x0, ym1, z0] + W[x0, yp1, z0]) +
                                       0.5*nu*dt*idz2*(W[x0, y0, zm1] + W[x0, y0, zp1]))         

        imposeWBCs(W)

        locmaxErr = np.amax(np.fabs(rho[x0, y0, z0] - (W[x0, y0, z0] - 0.5*nu*dt*(
                        (W[xm1, y0, z0] - 2.0*W[x0, y0, z0] + W[xp1, y0, z0])/hx2 +
                        (W[x0, ym1, z0] - 2.0*W[x0, y0, z0] + W[x0, yp1, z0])/hy2 +
                        (W[x0, y0, zm1] - 2.0*W[x0, y0, z0] + W[x0, y0, zp1])/hz2))))
    
        totmaxErr = comm.allreduce(locmaxErr, op=MPI.MAX)

        if totmaxErr < VpTolerance:
            break

        jCnt += 1
        if jCnt > maxCount:
            print("ERROR: Jacobi not converging in W. Aborting")
            quit()
    
    return W[x0, y0, z0]       


def TJacobi(rho):
        
    jCnt = 0
    while True:
        T[x0, y0, z0] = (1.0/(1 + kappa*dt*(idx2 + idy2 + idz2))) * (rho[x0, y0, z0] + 
                                       0.5*kappa*dt*idx2*(T[xm1, y0, z0] + T[xp1, y0, z0]) +
                                       0.5*kappa*dt*idy2*(T[x0, ym1, z0] + T[x0, yp1, z0]) +
                                       0.5*kappa*dt*idz2*(T[x0, y0, zm1] + T[x0, y0, zp1])) 

        imposeTBCs(T)

        locmaxErr = np.amax(np.fabs(rho[x0, y0, z0] - (T[x0, y0, z0] - 0.5*kappa*dt*(
                        (T[xm1, y0, z0] - 2.0*T[x0, y0, z0] + T[xp1, y0, z0])/hx2 +
                        (T[x0, ym1, z0] - 2.0*T[x0, y0, z0] + T[x0, yp1, z0])/hy2 +
                        (T[x0, y0, zm1] - 2.0*T[x0, y0, z0] + T[x0, y0, zp1])/hz2))))
    
        totmaxErr = comm.allreduce(locmaxErr, op=MPI.MAX)

        if totmaxErr < VpTolerance:
            break
    
        jCnt += 1
        if jCnt > maxCount:
            print("ERROR: Jacobi not converging in T. Aborting")
            quit()
    
    return T[x0, y0, z0]       


def PoissonSolver(rho):
    jCnt = 0
    
    while True:
        Pp[x0, y0, z0] = (1.0/(-2.0*(idx2 + idy2 + idz2))) * (rho[x0, y0, z0] - 
                                       idx2*(Pp[xm1, y0, z0] + Pp[xp1, y0, z0]) -
                                       idy2*(Pp[x0, ym1, z0] + Pp[x0, yp1, z0]) -
                                       idz2*(Pp[x0, y0, zm1] + Pp[x0, y0, zp1]))   

        imposePpBCs(Pp)
    
        locmaxErr = np.amax(np.fabs(rho[x0, y0, z0] -((
                        (Pp[xm1, y0, z0] - 2.0*Pp[x0, y0, z0] + Pp[xp1, y0, z0])/hx2 +
                        (Pp[x0, ym1, z0] - 2.0*Pp[x0, y0, z0] + Pp[x0, yp1, z0])/hy2 +
                        (Pp[x0, y0, zm1] - 2.0*Pp[x0, y0, z0] + Pp[x0, y0, zp1])/hz2))))
    
        totmaxErr = comm.allreduce(locmaxErr, op=MPI.MAX)

        if totmaxErr < PoissonTolerance:
            break
    
        jCnt += 1
        if jCnt > 50*maxCount:
            print("ERROR: Poisson solver not converging. Aborting")
            quit()
    
    return Pp[x0, y0, z0]     


def imposeUBCs(U):
    data_transfer(U)

    if frstRank:
        U[0, :, :] = -U[1, :, :]
    if lastRank:
        U[-1, :, :] = -U[-2, :, :]

    U[:, 0, :], U[:, -1, :] = -U[:, 1, :], -U[:, -2, :]
    U[:, :, 0], U[:, :, -1] = -U[:, :, 1], -U[:, :, -2]


def imposeVBCs(V):
    data_transfer(V)

    if frstRank:
        V[0, :, :] = -V[1, :, :]
    if lastRank:
        V[-1, :, :] = -V[-2, :, :]

    V[:, 0, :], V[:, -1, :] = -V[:, 1, :], -V[:, -2, :]  
    V[:, :, 0], V[:, :, -1] = -V[:, :, 1], -V[:, :, -2]


def imposeWBCs(W):
    data_transfer(W)
    
    if frstRank:
        W[0, :, :] = -W[1, :, :]
    if lastRank:
        W[-1, :, :] = -W[-2, :, :]

    W[:, 0, :], W[:, -1, :] = -W[:, 1, :], -W[:, -2, :]
    W[:, :, 0], W[:, :, -1] = -W[:, :, 1], -W[:, :, -2]  


def imposeTBCs(T):
    data_transfer(T)

    if frstRank:
        T[0, :, :] = T[1, :, :]
    if lastRank:
        T[-1, :, :] = T[-2, :, :]

    T[:, 0, :], T[:, -1, :] = T[:, 1, :], T[:, -2, :]
    T[:, :, 0], T[:, :, -1] = 2.0 - T[:, :, 1], -T[:, :, -2]


def imposePBCs(P):
    data_transfer(P)

    if frstRank:
        P[0, :, :] = P[1, :, :]
    if lastRank:
        P[-1, :, :] = P[-2, :, :]

    P[:, 0, :], P[:, -1, :] = P[:, 1, :], P[:, -2, :]
    P[:, :, 0], P[:, :, -1] = P[:, :, 1], P[:, :, -2]


def imposePpBCs(Pp):
    data_transfer(Pp)

    if frstRank:
        Pp[0, :, :] = Pp[1, :, :]
    if lastRank:
        Pp[-1, :, :] = Pp[-2, :, :]

    Pp[:, 0, :], Pp[:, -1, :] = Pp[:, 1, :], Pp[:, -2, :]
    Pp[:, :, 0], Pp[:, :, -1] = Pp[:, :, 1], Pp[:, :, -2]


def main():
    iCnt = 1
    time = 0

    rhs = np.zeros([xSize, Ny+2, Nz+2])

    t1 = datetime.now()

    while True:
        if iCnt % opInt == 0:
            locU = np.sum(np.sqrt(U[x0, y0, z0]**2.0 + V[x0, y0, z0]**2.0 + W[x0, y0, z0]**2.0))
            locWT = np.sum(W[x0, y0, z0]*T[x0, y0, z0])

            globU = comm.reduce(locU, op=MPI.SUM, root=0)
            totalWT = comm.reduce(locWT, op=MPI.SUM, root=0)

            maxDiv = getDiv(U, V, W)

            if rootRank:
                Re = globU/(nu*Nx*Ny*Nz)
                Nu = 1.0 + totalWT/(kappa*Nx*Ny*Nz)
                print("%f    %f    %f    %f" %(time, Re, Nu, maxDiv))           


        Hx[x0, y0, z0] = computeNLinDiff_X(U, V, W)
        Hy[x0, y0, z0] = computeNLinDiff_Y(U, V, W)
        Hz[x0, y0, z0] = computeNLinDiff_Z(U, V, W)
        Ht[x0, y0, z0] = computeNLinDiff_T(U, V, W, T)  

        Hx[x0, y0, z0] = U[x0, y0, z0] + dt*(Hx[x0, y0, z0] - np.sqrt((Ta*Pr)/Ra)*(-V[x0, y0, z0]) - (P[xp1, y0, z0] - P[xm1, y0, z0])/(2.0*hx))
        uJacobi(Hx)

        Hy[x0, y0, z0] = V[x0, y0, z0] + dt*(Hy[x0, y0, z0] - np.sqrt((Ta*Pr)/Ra)*( U[x0, y0, z0]) - (P[x0, yp1, z0] - P[x0, ym1, z0])/(2.0*hy))
        vJacobi(Hy)

        Hz[x0, y0, z0] = W[x0, y0, z0] + dt*(Hz[x0, y0, z0] + T[x0, y0, z0] - (P[x0, y0, zp1] - P[x0, y0, zm1])/(2.0*hz))
        wJacobi(Hz)

        Ht[x0, y0, z0] = T[x0, y0, z0] + dt*Ht[x0, y0, z0]
        TJacobi(Ht)   

        rhs.fill(0.0)
        rhs[x0, y0, z0] = ((U[xp1, y0, z0] - U[xm1, y0, z0])/(2.0*hx) +
                           (V[x0, yp1, z0] - V[x0, ym1, z0])/(2.0*hy) +
                           (W[x0, y0, zp1] - W[x0, y0, zm1])/(2.0*hz))/dt

        #if rank == 0:
            #print(rhs[:, 1, 1])

        #if rank == 1:
        #    print(rhs[:, 1, 1])
        #exit(0)

        Pp[x0, y0, z0] = PoissonSolver(rhs)

        P[x0, y0, z0] = P[x0, y0, z0] + Pp[x0, y0, z0]
        U[x0, y0, z0] = U[x0, y0, z0] - dt*(Pp[xp1, y0, z0] - Pp[xm1, y0, z0])/(2.0*hx)
        V[x0, y0, z0] = V[x0, y0, z0] - dt*(Pp[x0, yp1, z0] - Pp[x0, ym1, z0])/(2.0*hy)
        W[x0, y0, z0] = W[x0, y0, z0] - dt*(Pp[x0, y0, zp1] - Pp[x0, y0, zm1])/(2.0*hz)

        imposeUBCs(U)
        imposeVBCs(V)
        imposeWBCs(W)
        imposePBCs(P)
        imposeTBCs(T)

        if time > tMax:
            break   

        iCnt = iCnt + 1
        time = time + dt

    t2 = datetime.now()

    if rootRank:
        print("Time taken for simulation =", t2-t1)

main()
