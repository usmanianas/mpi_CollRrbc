import scipy.integrate as integrate
from datetime import datetime
from mpi4py import MPI
import numpy as np
import h5py as hp

############### Simulation Parameters ###############

# Rayleigh Number
Ra = 5.0e4

# Prandtl Number
Pr = 7

# Taylor Number
Ta = 1e5

# Choose the grid sizes as indices from below list so that there are 2^n + 2 grid points
# Size index: 0 1 2 3  4  5  6  7   8   9   10   11   12   13    14
# Grid sizes: 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384
sInd = np.array([6, 6, 6])

# N should be of the form 2^n
# Then there will be 2^n + 2 points in total, including 2 ghost points
sLst = [2**x for x in range(12)]

# Dimensions of computational domain
Lx, Ly, Lz = 1.0, 1.0, 1.0

# Depth of each V-cycle in multigrid
VDepth = 3

# Number of V-cycles to be computed
vcCnt = 10

# Number of iterations during pre-smoothing
preSm = 5

# Number of iterations during post-smoothing
pstSm = 5

# Tolerance value for iterative solver
tolerance = 1.0e-6

# Time step
dt = 0.01

# Final time
tMax = 0.1

# Number of iterations at which output is sent to standard I/O
opInt = 1

# File writing interval
fwInt = 10

# Enable/Disable Parallel I/O
# WARNING: Parallel h5py truncates floats to a lower precision
mpiH5Py = False

# Tolerance value in Jacobi iterations
VpTolerance = 1.0e-5

# Omega for SOR
gssor = 1.6

# Maximum iterations for iterative solvers
maxCount = 1000

# Maximum number of iterations at coarsest level of multigrid solver
maxCountPp = 1000

###############################################

Nx, Ny, Nz = sLst[sInd[0]], sLst[sInd[1]], sLst[sInd[2]]

hx, hy, hz = Lx/(Nx), Ly/(Ny), Lz/(Nz)

xCord = np.linspace(0, Lx + hx, Nx + 2, endpoint=True) - hx/2
yCord = np.linspace(0, Ly + hx, Ny + 2, endpoint=True) - hy/2
zCord = np.linspace(0, Lz + hx, Nz + 2, endpoint=True) - hz/2

hx2, hy2, hz2 = hx*hx, hy*hy, hz*hz

idx2, idy2, idz2 = 1.0/hx2, 1.0/hy2, 1.0/hz2

# Get array of grid sizes are tuples corresponding to each level of V-Cycle
N = [(sLst[x[0]], sLst[x[1]], sLst[x[2]]) for x in [sInd - y for y in range(VDepth + 1)]]

# Define array of grid spacings along X
mghx = [hx*(2**x) for x in range(VDepth+1)]

# Define array of grid spacings along Y
mghy = [hy*(2**x) for x in range(VDepth+1)]

# Define array of grid spacings along Z
mghz = [hz*(2**x) for x in range(VDepth+1)]

# Square of mghx, used in finite difference formulae
mghx2 = [x*x for x in mghx]

# Square of mghy, used in finite difference formulae
mghy2 = [x*x for x in mghy]

# Square of mghz, used in finite difference formulae
mghz2 = [x*x for x in mghz]

# Cross product of mghy and mghz, used in finite difference formulae
hyhz = [mghy2[i]*mghz2[i] for i in range(VDepth + 1)]

# Cross product of mghx and mghz, used in finite difference formulae
hzhx = [mghx2[i]*mghz2[i] for i in range(VDepth + 1)]

# Cross product of mghx and mghy, used in finite difference formulae
hxhy = [mghx2[i]*mghy2[i] for i in range(VDepth + 1)]

# Cross product of mghx, mghy and mghz used in finite difference formulae
hxhyhz = [mghx2[i]*mghy2[i]*mghz2[i] for i in range(VDepth + 1)]

# Factor in denominator of Gauss-Seidel iterations
gsFactor = [1.0/(2.0*(hyhz[i] + hzhx[i] + hxhy[i])) for i in range(VDepth + 1)]

# Integer specifying the level of V-cycle at any point while solving
vLev = 0

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
xEn, yEn, zEn = locNx+1, Ny+1, Nz+1

gSt = rank*locNx
gEn = (rank + 1)*locNx

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
T[:, :, :] = 1 - zCord[:]

###############################################

############### Flow Parameters ###############

nu, kappa = np.sqrt(Pr/Ra), 1.0/np.sqrt(Ra*Pr)

if rootRank:
    print('#', 'Ra=', Ra, 'Pr=', Pr)

###############################################


def writeSoln(U, V, W, P, T, time):
    global mpiH5Py
    global gSt, gEn
    global rootRank
    global Nx, Ny, Nz

    fName = "Soln_{0:09.5f}.h5".format(time)
    if rootRank:
        print("#Writing solution file: ", fName)        

    f = None
    dShape = Nx, Ny, Nz

    if mpiH5Py:
        f = hp.File(fName, "w", driver='mpio', comm=comm)

        uDset = f.create_dataset("U", dShape, dtype = 'f')
        vDset = f.create_dataset("V", dShape, dtype = 'f')
        wDset = f.create_dataset("W", dShape, dtype = 'f')
        tDset = f.create_dataset("T", dShape, dtype = 'f')
        pDset = f.create_dataset("P", dShape, dtype = 'f')

        for rCnt in range(nprocs):
            if rCnt == rank:
                uDset[gSt:gEn,:,:] = U[1:-1, 1:-1, 1:-1]
                vDset[gSt:gEn,:,:] = V[1:-1, 1:-1, 1:-1]
                wDset[gSt:gEn,:,:] = W[1:-1, 1:-1, 1:-1]
                tDset[gSt:gEn,:,:] = T[1:-1, 1:-1, 1:-1]
                pDset[gSt:gEn,:,:] = P[1:-1, 1:-1, 1:-1]

        f.close()

    else:
        if rootRank:
            f = hp.File(fName, "w")

        dFull = comm.gather(U[1:-1, 1:-1, 1:-1], root=0)
        if rootRank:
            dFull = np.concatenate(dFull)
            dDset = f.create_dataset("U", data = dFull)

        dFull = comm.gather(V[1:-1, 1:-1, 1:-1], root=0)
        if rootRank:
            dFull = np.concatenate(dFull)
            dDset = f.create_dataset("V", data = dFull)

        dFull = comm.gather(W[1:-1, 1:-1, 1:-1], root=0)
        if rootRank:
            dFull = np.concatenate(dFull)
            dDset = f.create_dataset("W", data = dFull)

        dFull = comm.gather(T[1:-1, 1:-1, 1:-1], root=0)
        if rootRank:
            dFull = np.concatenate(dFull)
            dDset = f.create_dataset("T", data = dFull)

        dFull = comm.gather(P[1:-1, 1:-1, 1:-1], root=0)
        if rootRank:
            dFull = np.concatenate(dFull)
            dDset = f.create_dataset("P", data = dFull)

        if rootRank:
            f.close()


def getDiv(U, V, W):
    divMat = ((U[xp1, y0, z0] - U[xm1, y0, z0])*0.5/hx +
              (V[x0, yp1, z0] - V[x0, ym1, z0])*0.5/hy +
              (W[x0, y0, zp1] - W[x0, y0, zm1])*0.5/hz)
    
    locdivMax = np.max(abs(divMat[x0, y0, z0]))

    globdivMax = comm.reduce(locdivMax, op=MPI.MAX, root=0)

    return globdivMax


def data_transfer(F):
    reqRgt = comm.Irecv(F[-1, :, :], source = rgtRank)
    reqLft = comm.Irecv(F[0, :, :], source = lftRank)  

    comm.Send(F[-2, :, :], dest = rgtRank)
    comm.Send(F[1, :, :], dest = lftRank)

    reqRgt.wait()
    reqLft.wait()


def computeNLinDiff_X(U, V, W):
    global Hx
    global hx2, hy2, hz2
    global nu, hx, hy, hz

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


############### Multigrid Solver ###############

# The root function of MG-solver. And H is the RHS
def multigrid(H):
    global N
    global vcCnt
    global tolerance
    global pData, rData

    rData[0] = H
    chMat = np.zeros(N[0])

    for i in range(vcCnt):
        v_cycle()

        chMat = laplace(pData[0])
        locmaxRes = np.amax(np.abs(H[1:-1, 1:-1, 1:-1] - chMat[1:-1, 1:-1, 1:-1]))
        totmaxRes = comm.allreduce(locmaxRes, op=MPI.MAX)
        if totmaxRes < tolerance:
            break

    return pData[0]


# Multigrid V-cycle without the use of recursion
def v_cycle():
    global vLev
    global VDepth
    global pstSm, preSm

    vLev = 0

    # Pre-smoothing
    smooth(preSm)

    for vCnt in range(VDepth):
        # Compute residual
        calcResidual()

        # Copy smoothed pressure for later use
        sData[vLev] = np.copy(pData[vLev])

        # Restrict to coarser level
        restrict()

        # Reinitialize pressure at coarser level to 0 - this is critical!
        pData[vLev].fill(0.0)

        # If the coarsest level is reached, solve. Otherwise, keep smoothing
        if vLev == VDepth:
            #solve()
            smooth(preSm)
        else:
            smooth(preSm)

    # Prolongation operations
    for vCnt in range(VDepth):
        # Prolong pressure to next finer level
        prolong()

        # Add previously stored smoothed data
        pData[vLev] += sData[vLev]

        # Post-smoothing
        smooth(pstSm)


# Smoothens the solution sCount times using Gauss-Seidel smoother
def smooth(sCount):
    global N
    global vLev
    global gsFactor
    global rData, pData
    global hyhz, hzhx, hxhy, hxhyhz

    n = N[vLev]
    for iCnt in range(sCount):
        imposePpBCs(pData[vLev])

        # Vectorized Red-Black Gauss-Seidel
        # Update red cells
        # 0, 0, 0 configuration
        pData[vLev][1:-1:2, 1:-1:2, 1:-1:2] = (hyhz[vLev]*(pData[vLev][2::2, 1:-1:2, 1:-1:2] + pData[vLev][:-2:2, 1:-1:2, 1:-1:2]) +
                                               hzhx[vLev]*(pData[vLev][1:-1:2, 2::2, 1:-1:2] + pData[vLev][1:-1:2, :-2:2, 1:-1:2]) +
                                               hxhy[vLev]*(pData[vLev][1:-1:2, 1:-1:2, 2::2] + pData[vLev][1:-1:2, 1:-1:2, :-2:2]) -
                                              hxhyhz[vLev]*rData[vLev][1:-1:2, 1:-1:2, 1:-1:2]) * gsFactor[vLev]

        # 1, 1, 0 configuration
        pData[vLev][2::2, 2::2, 1:-1:2] = (hyhz[vLev]*(pData[vLev][3::2, 2::2, 1:-1:2] + pData[vLev][1:-1:2, 2::2, 1:-1:2]) +
                                           hzhx[vLev]*(pData[vLev][2::2, 3::2, 1:-1:2] + pData[vLev][2::2, 1:-1:2, 1:-1:2]) +
                                           hxhy[vLev]*(pData[vLev][2::2, 2::2, 2::2] + pData[vLev][2::2, 2::2, :-2:2]) -
                                          hxhyhz[vLev]*rData[vLev][2::2, 2::2, 1:-1:2]) * gsFactor[vLev]

        # 1, 0, 1 configuration
        pData[vLev][2::2, 1:-1:2, 2::2] = (hyhz[vLev]*(pData[vLev][3::2, 1:-1:2, 2::2] + pData[vLev][1:-1:2, 1:-1:2, 2::2]) +
                                           hzhx[vLev]*(pData[vLev][2::2, 2::2, 2::2] + pData[vLev][2::2, :-2:2, 2::2]) +
                                           hxhy[vLev]*(pData[vLev][2::2, 1:-1:2, 3::2] + pData[vLev][2::2, 1:-1:2, 1:-1:2]) -
                                          hxhyhz[vLev]*rData[vLev][2::2, 1:-1:2, 2::2]) * gsFactor[vLev]

        # 0, 1, 1 configuration
        pData[vLev][1:-1:2, 2::2, 2::2] = (hyhz[vLev]*(pData[vLev][2::2, 2::2, 2::2] + pData[vLev][:-2:2, 2::2, 2::2]) +
                                           hzhx[vLev]*(pData[vLev][1:-1:2, 3::2, 2::2] + pData[vLev][1:-1:2, 1:-1:2, 2::2]) +
                                           hxhy[vLev]*(pData[vLev][1:-1:2, 2::2, 3::2] + pData[vLev][1:-1:2, 2::2, 1:-1:2]) -
                                          hxhyhz[vLev]*rData[vLev][1:-1:2, 2::2, 2::2]) * gsFactor[vLev]

        data_transfer(pData[vLev])

        # Update black cells
        # 1, 0, 0 configuration
        pData[vLev][2::2, 1:-1:2, 1:-1:2] = (hyhz[vLev]*(pData[vLev][3::2, 1:-1:2, 1:-1:2] + pData[vLev][1:-1:2, 1:-1:2, 1:-1:2]) +
                                             hzhx[vLev]*(pData[vLev][2::2, 2::2, 1:-1:2] + pData[vLev][2::2, :-2:2, 1:-1:2]) +
                                             hxhy[vLev]*(pData[vLev][2::2, 1:-1:2, 2::2] + pData[vLev][2::2, 1:-1:2, :-2:2]) -
                                            hxhyhz[vLev]*rData[vLev][2::2, 1:-1:2, 1:-1:2]) * gsFactor[vLev]

        # 0, 1, 0 configuration
        pData[vLev][1:-1:2, 2::2, 1:-1:2] = (hyhz[vLev]*(pData[vLev][2::2, 2::2, 1:-1:2] + pData[vLev][:-2:2, 2::2, 1:-1:2]) +
                                             hzhx[vLev]*(pData[vLev][1:-1:2, 3::2, 1:-1:2] + pData[vLev][1:-1:2, 1:-1:2, 1:-1:2]) +
                                             hxhy[vLev]*(pData[vLev][1:-1:2, 2::2, 2::2] + pData[vLev][1:-1:2, 2::2, :-2:2]) -
                                            hxhyhz[vLev]*rData[vLev][1:-1:2, 2::2, 1:-1:2]) * gsFactor[vLev]

        # 0, 0, 1 configuration
        pData[vLev][1:-1:2, 1:-1:2, 2::2] = (hyhz[vLev]*(pData[vLev][2::2, 1:-1:2, 2::2] + pData[vLev][:-2:2, 1:-1:2, 2::2]) +
                                             hzhx[vLev]*(pData[vLev][1:-1:2, 2::2, 2::2] + pData[vLev][1:-1:2, :-2:2, 2::2]) +
                                             hxhy[vLev]*(pData[vLev][1:-1:2, 1:-1:2, 3::2] + pData[vLev][1:-1:2, 1:-1:2, 1:-1:2]) -
                                            hxhyhz[vLev]*rData[vLev][1:-1:2, 1:-1:2, 2::2]) * gsFactor[vLev]

        # 1, 1, 1 configuration
        pData[vLev][2::2, 2::2, 2::2] = (hyhz[vLev]*(pData[vLev][3::2, 2::2, 2::2] + pData[vLev][1:-1:2, 2::2, 2::2]) +
                                         hzhx[vLev]*(pData[vLev][2::2, 3::2, 2::2] + pData[vLev][2::2, 1:-1:2, 2::2]) +
                                         hxhy[vLev]*(pData[vLev][2::2, 2::2, 3::2] + pData[vLev][2::2, 2::2, 1:-1:2]) -
                                        hxhyhz[vLev]*rData[vLev][2::2, 2::2, 2::2]) * gsFactor[vLev]

    imposePpBCs(pData[vLev])


# Compute the residual and store it into iTemp array
def calcResidual():
    global vLev
    global iTemp, rData, pData

    iTemp[vLev].fill(0.0)
    iTemp[vLev] = rData[vLev] - laplace(pData[vLev])


# Restricts the data from an array of size 2^n to a smaller array of size 2^(n - 1)
def restrict():
    global N
    global vLev
    global iTemp, rData

    pLev = vLev
    vLev += 1

    n = N[vLev]
    rData[vLev][1:-1, 1:-1, 1:-1] = (iTemp[pLev][1:-1:2, 1:-1:2, 1:-1:2] + iTemp[pLev][2::2, 2::2, 2::2] +
                                     iTemp[pLev][1:-1:2, 1:-1:2, 2::2] + iTemp[pLev][2::2, 2::2, 1:-1:2] +
                                     iTemp[pLev][1:-1:2, 2::2, 1:-1:2] + iTemp[pLev][2::2, 1:-1:2, 2::2] +
                                     iTemp[pLev][2::2, 1:-1:2, 1:-1:2] + iTemp[pLev][1:-1:2, 2::2, 2::2])/8


# Solves at coarsest level using the Gauss-Seidel iterative solver
def solve():
    global N, vLev
    global gsFactor
    global tolerance
    global maxCountPp
    global pData, rData
    global hyhz, hzhx, hxhy, hxhyhz

    n = N[vLev]

    jCnt = 0
    while True:
        imposePpBCs(pData[vLev])

        # Vectorized Red-Black Gauss-Seidel
        # Update red cells
        # 0, 0, 0 configuration
        pData[vLev][1:-1:2, 1:-1:2, 1:-1:2] = (hyhz[vLev]*(pData[vLev][2::2, 1:-1:2, 1:-1:2] + pData[vLev][:-2:2, 1:-1:2, 1:-1:2]) +
                                               hzhx[vLev]*(pData[vLev][1:-1:2, 2::2, 1:-1:2] + pData[vLev][1:-1:2, :-2:2, 1:-1:2]) +
                                               hxhy[vLev]*(pData[vLev][1:-1:2, 1:-1:2, 2::2] + pData[vLev][1:-1:2, 1:-1:2, :-2:2]) -
                                              hxhyhz[vLev]*rData[vLev][1:-1:2, 1:-1:2, 1:-1:2]) * gsFactor[vLev]

        # 1, 1, 0 configuration
        pData[vLev][2::2, 2::2, 1:-1:2] = (hyhz[vLev]*(pData[vLev][3::2, 2::2, 1:-1:2] + pData[vLev][1:-1:2, 2::2, 1:-1:2]) +
                                           hzhx[vLev]*(pData[vLev][2::2, 3::2, 1:-1:2] + pData[vLev][2::2, 1:-1:2, 1:-1:2]) +
                                           hxhy[vLev]*(pData[vLev][2::2, 2::2, 2::2] + pData[vLev][2::2, 2::2, :-2:2]) -
                                          hxhyhz[vLev]*rData[vLev][2::2, 2::2, 1:-1:2]) * gsFactor[vLev]

        # 1, 0, 1 configuration
        pData[vLev][2::2, 1:-1:2, 2::2] = (hyhz[vLev]*(pData[vLev][3::2, 1:-1:2, 2::2] + pData[vLev][1:-1:2, 1:-1:2, 2::2]) +
                                           hzhx[vLev]*(pData[vLev][2::2, 2::2, 2::2] + pData[vLev][2::2, :-2:2, 2::2]) +
                                           hxhy[vLev]*(pData[vLev][2::2, 1:-1:2, 3::2] + pData[vLev][2::2, 1:-1:2, 1:-1:2]) -
                                          hxhyhz[vLev]*rData[vLev][2::2, 1:-1:2, 2::2]) * gsFactor[vLev]

        # 0, 1, 1 configuration
        pData[vLev][1:-1:2, 2::2, 2::2] = (hyhz[vLev]*(pData[vLev][2::2, 2::2, 2::2] + pData[vLev][:-2:2, 2::2, 2::2]) +
                                           hzhx[vLev]*(pData[vLev][1:-1:2, 3::2, 2::2] + pData[vLev][1:-1:2, 1:-1:2, 2::2]) +
                                           hxhy[vLev]*(pData[vLev][1:-1:2, 2::2, 3::2] + pData[vLev][1:-1:2, 2::2, 1:-1:2]) -
                                          hxhyhz[vLev]*rData[vLev][1:-1:2, 2::2, 2::2]) * gsFactor[vLev]

        data_transfer(pData[vLev])

        # Update black cells
        # 1, 0, 0 configuration
        pData[vLev][2::2, 1:-1:2, 1:-1:2] = (hyhz[vLev]*(pData[vLev][3::2, 1:-1:2, 1:-1:2] + pData[vLev][1:-1:2, 1:-1:2, 1:-1:2]) +
                                             hzhx[vLev]*(pData[vLev][2::2, 2::2, 1:-1:2] + pData[vLev][2::2, :-2:2, 1:-1:2]) +
                                             hxhy[vLev]*(pData[vLev][2::2, 1:-1:2, 2::2] + pData[vLev][2::2, 1:-1:2, :-2:2]) -
                                            hxhyhz[vLev]*rData[vLev][2::2, 1:-1:2, 1:-1:2]) * gsFactor[vLev]

        # 0, 1, 0 configuration
        pData[vLev][1:-1:2, 2::2, 1:-1:2] = (hyhz[vLev]*(pData[vLev][2::2, 2::2, 1:-1:2] + pData[vLev][:-2:2, 2::2, 1:-1:2]) +
                                             hzhx[vLev]*(pData[vLev][1:-1:2, 3::2, 1:-1:2] + pData[vLev][1:-1:2, 1:-1:2, 1:-1:2]) +
                                             hxhy[vLev]*(pData[vLev][1:-1:2, 2::2, 2::2] + pData[vLev][1:-1:2, 2::2, :-2:2]) -
                                            hxhyhz[vLev]*rData[vLev][1:-1:2, 2::2, 1:-1:2]) * gsFactor[vLev]

        # 0, 0, 1 configuration
        pData[vLev][1:-1:2, 1:-1:2, 2::2] = (hyhz[vLev]*(pData[vLev][2::2, 1:-1:2, 2::2] + pData[vLev][:-2:2, 1:-1:2, 2::2]) +
                                             hzhx[vLev]*(pData[vLev][1:-1:2, 2::2, 2::2] + pData[vLev][1:-1:2, :-2:2, 2::2]) +
                                             hxhy[vLev]*(pData[vLev][1:-1:2, 1:-1:2, 3::2] + pData[vLev][1:-1:2, 1:-1:2, 1:-1:2]) -
                                            hxhyhz[vLev]*rData[vLev][1:-1:2, 1:-1:2, 2::2]) * gsFactor[vLev]

        # 1, 1, 1 configuration
        pData[vLev][2::2, 2::2, 2::2] = (hyhz[vLev]*(pData[vLev][3::2, 2::2, 2::2] + pData[vLev][1:-1:2, 2::2, 2::2]) +
                                         hzhx[vLev]*(pData[vLev][2::2, 3::2, 2::2] + pData[vLev][2::2, 1:-1:2, 2::2]) +
                                         hxhy[vLev]*(pData[vLev][2::2, 2::2, 3::2] + pData[vLev][2::2, 2::2, 1:-1:2]) -
                                        hxhyhz[vLev]*rData[vLev][2::2, 2::2, 2::2]) * gsFactor[vLev]

        locmaxErr = np.amax(np.abs(rData[vLev] - laplace(pData[vLev]))[1:-1, 1:-1, 1:-1])
        totmaxErr = comm.allreduce(locmaxErr, op=MPI.MAX)

        if totmaxErr < tolerance:
            break

        jCnt += 1
        if jCnt > maxCountPp:
            print("ERROR: Gauss-Seidel solver not converging. Aborting")
            quit()

    imposePpBCs(pData[vLev])


# Interpolates the data from an array of size 2^n to a larger array of size 2^(n + 1)
def prolong():
    global vLev
    global pData

    pLev = vLev
    vLev -= 1

    pData[vLev][1:-1:2, 1:-1:2, 1:-1:2] = pData[vLev][2::2, 1:-1:2, 1:-1:2] = pData[vLev][1:-1:2, 2::2, 1:-1:2] = pData[vLev][1:-1:2, 1:-1:2, 2::2] = \
    pData[vLev][2::2, 2::2, 1:-1:2] = pData[vLev][1:-1:2, 2::2, 2::2] = pData[vLev][2::2, 1:-1:2, 2::2] = pData[vLev][2::2, 2::2, 2::2] = pData[pLev][1:-1, 1:-1, 1:-1]


# Computes the 3D laplacian of function
def laplace(function):
    global vLev
    global mghx2, mghy2, mghz2

    laplacian = np.zeros_like(function)
    laplacian[1:-1, 1:-1, 1:-1] = ((function[:-2, 1:-1, 1:-1] - 2.0*function[1:-1, 1:-1, 1:-1] + function[2:, 1:-1, 1:-1])/mghx2[vLev] + 
                                   (function[1:-1, :-2, 1:-1] - 2.0*function[1:-1, 1:-1, 1:-1] + function[1:-1, 2:, 1:-1])/mghy2[vLev] +
                                   (function[1:-1, 1:-1, :-2] - 2.0*function[1:-1, 1:-1, 1:-1] + function[1:-1, 1:-1, 2:])/mghz2[vLev])

    return laplacian


def initMGArrays():
    global N
    global pData, rData, sData, iTemp
    global sInd, sLst, VDepth, maxCountPp

    # Update VDepth according to number of procs
    maxDepth = min(min(sInd), sLst.index(int(Nx/nprocs))) - 1

    if VDepth > maxDepth:
        if rootRank:
            print("\nWARNING: Specified V-Cycle depth exceed maximum attainable value for grid and processor count.")
            print("Using new VDepth value = " + str(maxDepth))

        VDepth = maxDepth

    # Update List of grid sizes accordingly
    N = [(sLst[x[0]], sLst[x[1]], sLst[x[2]]) for x in [sInd - y for y in range(VDepth + 1)]]
    N = [(int(x[0]/nprocs), x[1], x[2]) for x in N]

    # Finally update max iterations variable
    maxSolCount = 100*N[-1][0]*N[-1][1]*N[-1][2]

    if maxSolCount > maxCountPp:
        if rootRank:
            print("\nWARNING: Maximum iteration count for Poisson solver is too low.")
            print("Using new maximum iteration count = " + str(maxSolCount))

        maxCountPp = maxSolCount

    nList = np.array(N)

    pData = [np.zeros(tuple(x)) for x in nList + 2]

    rData = [np.zeros_like(x) for x in pData]
    sData = [np.zeros_like(x) for x in pData]
    iTemp = [np.zeros_like(x) for x in pData]


############### Boundary Conditions ###############


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
    iCnt = 0
    time = 0

    initMGArrays()

    rhs = np.zeros([xSize, Ny+2, Nz+2])

    t1 = datetime.now()

    # Write output at t = 0
    locU = np.sum(np.sqrt(U[x0, y0, z0]**2.0 + V[x0, y0, z0]**2.0 + W[x0, y0, z0]**2.0))
    locWT = np.sum(W[x0, y0, z0]*T[x0, y0, z0])

    globU = comm.reduce(locU, op=MPI.SUM, root=0)
    totalWT = comm.reduce(locWT, op=MPI.SUM, root=0)

    maxDiv = getDiv(U, V, W)

    writeSoln(U, V, W, P, T, 0.0)

    if rootRank:
        Re = globU/(nu*Nx*Ny*Nz)
        Nu = 1.0 + totalWT/(kappa*Nx*Ny*Nz)
        print("%f    %f    %f    %f" %(time, Re, Nu, maxDiv))           

    while True:
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

        Pp = multigrid(rhs)

        P[x0, y0, z0] = P[x0, y0, z0] + Pp[x0, y0, z0]
        U[x0, y0, z0] = U[x0, y0, z0] - dt*(Pp[xp1, y0, z0] - Pp[xm1, y0, z0])/(2.0*hx)
        V[x0, y0, z0] = V[x0, y0, z0] - dt*(Pp[x0, yp1, z0] - Pp[x0, ym1, z0])/(2.0*hy)
        W[x0, y0, z0] = W[x0, y0, z0] - dt*(Pp[x0, y0, zp1] - Pp[x0, y0, zm1])/(2.0*hz)

        imposeUBCs(U)
        imposeVBCs(V)
        imposeWBCs(W)
        imposePBCs(P)
        imposeTBCs(T)

        iCnt = iCnt + 1
        time = time + dt

        if iCnt % opInt == 0:
            uSqr = U[x0, y0, z0]**2.0 + V[x0, y0, z0]**2.0 + W[x0, y0, z0]**2.0
            locU = integrate.simps(integrate.simps(integrate.simps(uSqr, zCord[z0]), yCord[y0]), xCord[x0])
            globU = comm.reduce(locU, op=MPI.SUM, root=0)

            wT = W[x0, y0, z0]*T[x0, y0, z0]
            locWT = integrate.simps(integrate.simps(integrate.simps(wT, zCord[z0]), yCord[y0]), xCord[x0])
            totalWT = comm.reduce(locWT, op=MPI.SUM, root=0)

            maxDiv = getDiv(U, V, W)

            if rootRank:
                Re = np.sqrt(globU)/nu
                Nu = 1.0 + totalWT/kappa
                print("%f    %f    %f    %f" %(time, Re, Nu, maxDiv))           

        if iCnt % fwInt == 0:
            writeSoln(U, V, W, P, T, time)

        if time + dt/2.0 > tMax:
            break   

    t2 = datetime.now()

    if rootRank:
        print("Time taken for simulation =", t2-t1)

main()
