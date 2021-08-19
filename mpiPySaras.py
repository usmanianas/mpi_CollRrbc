import scipy.integrate as integrate
from datetime import datetime
from mpi4py import MPI
import numpy as np
import h5py as hp

############### Simulation Parameters ###############

# Rayleigh Number
Ra = 1.0e4

# Prandtl Number
Pr = 0.71

# Taylor Number
Ta = 0.0

# Choose the grid sizes as indices from below list so that there are 2^n + 2 grid points
# Size index: 0 1 2 3  4  5  6  7   8   9   10   11   12   13    14
# Grid sizes: 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384
sInd = np.array([5, 5, 5])

# Flag to switch between uniform and non-uniform grid with tan-hyp stretching
nuFlag = False

# Flag to turn spiral-vortex LES model on/off
lesFlag = False

# Stretching parameter for tangent-hyperbolic grid
beta = 1.0e-5

# N should be of the form 2^n
# Then there will be 2^n + 2 points in total, including 2 ghost points
sLst = [2**x for x in range(12)]

# Dimensions of computational domain
Lx, Ly, Lz = 1.0, 1.0, 1.0

# Depth of each V-cycle in multigrid
VDepth = 4

# Number of V-cycles to be computed
vcCnt = 5

# Number of iterations during pre-smoothing
preSm = 5

# Number of iterations during post-smoothing
pstSm = 5

# Tolerance value for iterative solver
tolerance = 1.0e-3

# Flag to enable/disable restart of solver from previous solution file
restartFlag = False

# Time step
dt = 0.01

# CFL condition
cflNo = 0.5

# Final time
tMax = 0.1

# Number of iterations at which output is sent to standard I/O
opInt = 1

# File writing interval
fwInt = 100.0

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
#print(xCord)
xCord = 0.5*(1.0 - np.tanh(beta*(1.0 - 2*xCord))/np.tanh(beta))
#print(xCord)
yCord = np.linspace(0, Ly + hx, Ny + 2, endpoint=True) - hy/2
yCord = 0.5*(1.0 - np.tanh(beta*(1.0 - 2*yCord))/np.tanh(beta))
zCord = np.linspace(0, Lz + hx, Nz + 2, endpoint=True) - hz/2
zCord = 0.5*(1.0 - np.tanh(beta*(1.0 - 2*zCord))/np.tanh(beta))

hx2, hy2, hz2 = hx*hx, hy*hy, hz*hz

idx2, idy2, idz2 = 1.0/hx2, 1.0/hy2, 1.0/hz2

# Get array of grid sizes are tuples corresponding to each level of V-Cycle
N = [(sLst[x[0]], sLst[x[1]], sLst[x[2]]) for x in [sInd - y for y in range(VDepth + 1)]]

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
    print('\n# Grid', Nx, Ny, Nz)
    print('# No. of Processors =', nprocs)

############### Flow Parameters ###############

nu, kappa = np.sqrt(Pr/Ra), 1.0/np.sqrt(Ra*Pr)

if rootRank:
    print('#', 'Ra=', Ra, 'Pr=', Pr, 'Ta=', Ta)


############# Fields Initialization ###########

def initFields():
    global tMax
    global rootRank
    global restartFlag
    global U, V, W, P, T
    global Hx, Hy, Hz, Ht, Pp

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
    #T[:, :, :] = 1 - zCord[:]

    time = 0.0
    if restartFlag:
        time = readSoln(U, V, W, P, T)

    if time > tMax:
        if rootRank:
            print("ERROR: Restart file starts beyond the maximum time limit set. Aborting")
        quit()

    # Impose BCs
    imposeUBCs(U)
    imposeVBCs(V)
    imposeWBCs(W)
    imposePBCs(P)
    imposeTBCs(T)

    return time


################### File I/O ##################


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

        uDset = f.create_dataset("Vx", dShape, dtype = 'f')
        vDset = f.create_dataset("Vy", dShape, dtype = 'f')
        wDset = f.create_dataset("Vz", dShape, dtype = 'f')
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
            dDset = f.create_dataset("Vx", data = dFull)

        dFull = comm.gather(V[1:-1, 1:-1, 1:-1], root=0)
        if rootRank:
            dFull = np.concatenate(dFull)
            dDset = f.create_dataset("Vy", data = dFull)

        dFull = comm.gather(W[1:-1, 1:-1, 1:-1], root=0)
        if rootRank:
            dFull = np.concatenate(dFull)
            dDset = f.create_dataset("Vz", data = dFull)

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


def readSoln(U, V, W, P, T):
    global mpiH5Py
    global gSt, gEn
    global rootRank

    fName = "restartFile.h5"
    if rootRank:
        print("#Reading from file: ", fName)        

    if mpiH5Py:
        if rootRank:
            print("Parallel file read is not implemented")
        quit()
        '''
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
        '''

    else:
        f = hp.File(fName, "r")

        U[1:-1, 1:-1, 1:-1] = np.array(f["Vx"])[gSt:gEn, :, :]
        V[1:-1, 1:-1, 1:-1] = np.array(f["Vy"])[gSt:gEn, :, :]
        W[1:-1, 1:-1, 1:-1] = np.array(f["Vz"])[gSt:gEn, :, :]
        P[1:-1, 1:-1, 1:-1] = np.array(f["P"])[gSt:gEn, :, :]
        T[1:-1, 1:-1, 1:-1] = np.array(f["T"])[gSt:gEn, :, :]
        time = np.array(f["Time"])

        f.close()

    return time


def getDiv(U, V, W):
    global nuFlag
    global i2hx, i2hy, i2hz
    global xi_x, et_y, zt_z

    if nuFlag:
        divMat = ((U[xp1, y0, z0] - U[xm1, y0, z0]) * xi_x[0] * i2hx[0] +
                  (V[x0, yp1, z0] - V[x0, ym1, z0]) * et_y[0] * i2hy[0] +
                  (W[x0, y0, zp1] - W[x0, y0, zm1]) * zt_z[0] * i2hz[0])
    else:
        divMat = ((U[xp1, y0, z0] - U[xm1, y0, z0]) * i2hx[0] +
                  (V[x0, yp1, z0] - V[x0, ym1, z0]) * i2hy[0] +
                  (W[x0, y0, zp1] - W[x0, y0, zm1]) * i2hz[0])
    
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


############### Nonlinear and Diffusion Calculations ###############


def computeNLinDiff_X():
    global Hx, nu
    global nuFlag
    global U, V, W
    global ihx2, i2hx, ihy2, i2hy, ihz2, i2hz
    global xi_x, xixx, xix2, et_y, etyy, ety2, zt_z, ztzz, ztz2

    if nuFlag:
        Hx[x0, y0, z0] = ((xix2[0] * ihx2[0] * (U[xp1, y0, z0] - 2.0*U[x0, y0, z0] + U[xm1, y0, z0]) + 
                           xixx[0] * i2hx[0] * (U[xp1, y0, z0] - U[xm1, y0, z0]) + 
                           ety2[0] * ihy2[0] * (U[x0, yp1, z0] - 2.0*U[x0, y0, z0] + U[x0, ym1, z0]) + 
                           etyy[0] * i2hy[0] * (U[x0, yp1, z0] - U[x0, ym1, z0]) + 
                           ztz2[0] * ihz2[0] * (U[x0, y0, zp1] - 2.0*U[x0, y0, z0] + U[x0, y0, zm1]) +
                           ztzz[0] * i2hz[0] * (U[x0, y0, zp1] - U[x0, y0, zm1]))*0.5*nu -
                            U[x0, y0, z0] * i2hx[0] * xi_x[0] * (U[xp1, y0, z0] - U[xm1, y0, z0]) -
                            V[x0, y0, z0] * i2hy[0] * et_y[0] * (U[x0, yp1, z0] - U[x0, ym1, z0]) - 
                            W[x0, y0, z0] * i2hz[0] * zt_z[0] * (U[x0, y0, zp1] - U[x0, y0, zm1]))
    else:
        Hx[x0, y0, z0] = (((U[xp1, y0, z0] - 2.0*U[x0, y0, z0] + U[xm1, y0, z0])*ihx2[0] + 
                           (U[x0, yp1, z0] - 2.0*U[x0, y0, z0] + U[x0, ym1, z0])*ihy2[0] + 
                           (U[x0, y0, zp1] - 2.0*U[x0, y0, z0] + U[x0, y0, zm1])*ihz2[0])*0.5*nu -
                            U[x0, y0, z0]*(U[xp1, y0, z0] - U[xm1, y0, z0])*i2hx[0] -
                            V[x0, y0, z0]*(U[x0, yp1, z0] - U[x0, ym1, z0])*i2hy[0] - 
                            W[x0, y0, z0]*(U[x0, y0, zp1] - U[x0, y0, zm1])*i2hz[0])


def computeNLinDiff_Y():
    global Hy, nu
    global nuFlag
    global U, V, W
    global ihx2, i2hx, ihy2, i2hy, ihz2, i2hz
    global xi_x, xixx, xix2, et_y, etyy, ety2, zt_z, ztzz, ztz2

    if nuFlag:
        Hy[x0, y0, z0] = ((xix2[0] * ihx2[0] * (V[xp1, y0, z0] - 2.0*V[x0, y0, z0] + V[xm1, y0, z0]) + 
                           xixx[0] * i2hx[0] * (V[xp1, y0, z0] - V[xm1, y0, z0]) + 
                           ety2[0] * ihy2[0] * (V[x0, yp1, z0] - 2.0*V[x0, y0, z0] + V[x0, ym1, z0]) + 
                           etyy[0] * i2hy[0] * (V[x0, yp1, z0] - V[x0, ym1, z0]) + 
                           ztz2[0] * ihz2[0] * (V[x0, y0, zp1] - 2.0*V[x0, y0, z0] + V[x0, y0, zm1]) +
                           ztzz[0] * i2hz[0] * (V[x0, y0, zp1] - V[x0, y0, zm1]))*0.5*nu -
                            U[x0, y0, z0] * i2hx[0] * xi_x[0] * (V[xp1, y0, z0] - V[xm1, y0, z0]) -
                            V[x0, y0, z0] * i2hy[0] * et_y[0] * (V[x0, yp1, z0] - V[x0, ym1, z0]) - 
                            W[x0, y0, z0] * i2hz[0] * zt_z[0] * (V[x0, y0, zp1] - V[x0, y0, zm1]))
    else:
        Hy[x0, y0, z0] = (((V[xp1, y0, z0] - 2.0*V[x0, y0, z0] + V[xm1, y0, z0])*ihx2[0] + 
                           (V[x0, yp1, z0] - 2.0*V[x0, y0, z0] + V[x0, ym1, z0])*ihy2[0] + 
                           (V[x0, y0, zp1] - 2.0*V[x0, y0, z0] + V[x0, y0, zm1])*ihz2[0])*0.5*nu -
                            U[x0, y0, z0]*(V[xp1, y0, z0] - V[xm1, y0, z0])*i2hx[0] -
                            V[x0, y0, z0]*(V[x0, yp1, z0] - V[x0, ym1, z0])*i2hy[0] - 
                            W[x0, y0, z0]*(V[x0, y0, zp1] - V[x0, y0, zm1])*i2hz[0])


def computeNLinDiff_Z():
    global Hz, nu
    global nuFlag
    global U, V, W
    global ihx2, i2hx, ihy2, i2hy, ihz2, i2hz
    global xi_x, xixx, xix2, et_y, etyy, ety2, zt_z, ztzz, ztz2

    if nuFlag:
        Hz[x0, y0, z0] = ((xix2[0] * ihx2[0] * (W[xp1, y0, z0] - 2.0*W[x0, y0, z0] + W[xm1, y0, z0]) + 
                           xixx[0] * i2hx[0] * (W[xp1, y0, z0] - W[xm1, y0, z0]) + 
                           ety2[0] * ihy2[0] * (W[x0, yp1, z0] - 2.0*W[x0, y0, z0] + W[x0, ym1, z0]) + 
                           etyy[0] * i2hy[0] * (W[x0, yp1, z0] - W[x0, ym1, z0]) + 
                           ztz2[0] * ihz2[0] * (W[x0, y0, zp1] - 2.0*W[x0, y0, z0] + W[x0, y0, zm1]) +
                           ztzz[0] * i2hz[0] * (W[x0, y0, zp1] - W[x0, y0, zm1]))*0.5*nu -
                            U[x0, y0, z0] * i2hx[0] * xi_x[0] * (W[xp1, y0, z0] - W[xm1, y0, z0]) -
                            V[x0, y0, z0] * i2hy[0] * et_y[0] * (W[x0, yp1, z0] - W[x0, ym1, z0]) - 
                            W[x0, y0, z0] * i2hz[0] * zt_z[0] * (W[x0, y0, zp1] - W[x0, y0, zm1]))
    else:
        Hz[x0, y0, z0] = (((W[xp1, y0, z0] - 2.0*W[x0, y0, z0] + W[xm1, y0, z0])*ihx2[0] + 
                           (W[x0, yp1, z0] - 2.0*W[x0, y0, z0] + W[x0, ym1, z0])*ihy2[0] + 
                           (W[x0, y0, zp1] - 2.0*W[x0, y0, z0] + W[x0, y0, zm1])*ihz2[0])*0.5*nu -
                            U[x0, y0, z0]*(W[xp1, y0, z0] - W[xm1, y0, z0])*i2hx[0] -
                            V[x0, y0, z0]*(W[x0, yp1, z0] - W[x0, ym1, z0])*i2hy[0] - 
                            W[x0, y0, z0]*(W[x0, y0, zp1] - W[x0, y0, zm1])*i2hz[0])


def computeNLinDiff_T():
    global Ht
    global kappa
    global nuFlag
    global U, V, W, T
    global ihx2, i2hx, ihy2, i2hy, ihz2, i2hz
    global xi_x, xixx, xix2, et_y, etyy, ety2, zt_z, ztzz, ztz2

    if nuFlag:
        Ht[x0, y0, z0] = ((xix2[0] * ihx2[0] * (T[xp1, y0, z0] - 2.0*T[x0, y0, z0] + T[xm1, y0, z0]) + 
                           xixx[0] * i2hx[0] * (T[xp1, y0, z0] - T[xm1, y0, z0]) + 
                           ety2[0] * ihy2[0] * (T[x0, yp1, z0] - 2.0*T[x0, y0, z0] + T[x0, ym1, z0]) + 
                           etyy[0] * i2hy[0] * (T[x0, yp1, z0] - T[x0, ym1, z0]) + 
                           ztz2[0] * ihz2[0] * (T[x0, y0, zp1] - 2.0*T[x0, y0, z0] + T[x0, y0, zm1]) +
                           ztzz[0] * i2hz[0] * (T[x0, y0, zp1] - T[x0, y0, zm1]))*0.5*kappa -
                            U[x0, y0, z0] * i2hx[0] * xi_x[0] * (T[xp1, y0, z0] - T[xm1, y0, z0]) -
                            V[x0, y0, z0] * i2hy[0] * et_y[0] * (T[x0, yp1, z0] - T[x0, ym1, z0]) - 
                            W[x0, y0, z0] * i2hz[0] * zt_z[0] * (T[x0, y0, zp1] - T[x0, y0, zm1]))
    else:
        Ht[x0, y0, z0] = (((T[xp1, y0, z0] - 2.0*T[x0, y0, z0] + T[xm1, y0, z0])*ihx2[0] + 
                           (T[x0, yp1, z0] - 2.0*T[x0, y0, z0] + T[x0, ym1, z0])*ihy2[0] + 
                           (T[x0, y0, zp1] - 2.0*T[x0, y0, z0] + T[x0, y0, zm1])*ihz2[0])*0.5*kappa -
                            U[x0, y0, z0]*(T[xp1, y0, z0] - T[xm1, y0, z0])*i2hx[0]-
                            V[x0, y0, z0]*(T[x0, yp1, z0] - T[x0, ym1, z0])*i2hy[0] - 
                            W[x0, y0, z0]*(T[x0, y0, zp1] - T[x0, y0, zm1])*i2hz[0])


############### Iterative Solvers ###############


def uJacobi(rho):
    global dt, nu
    global nuFlag
    global ihx2, i2hx, ihy2, i2hy, ihz2, i2hz
    global xixx, xix2, etyy, ety2, ztzz, ztz2

    jCnt = 0
    while True:
        if nuFlag:
            U[x0, y0, z0] = (rho[x0, y0, z0] + 0.5*nu*dt*(
                                xix2[0] * ihx2[0] * (U[xp1, y0, z0] + U[xm1, y0, z0]) +
                                xixx[0] * i2hx[0] * (U[xp1, y0, z0] - U[xm1, y0, z0]) +
                                ety2[0] * ihy2[0] * (U[x0, yp1, z0] + U[x0, ym1, z0]) +
                                etyy[0] * i2hy[0] * (U[x0, yp1, z0] - U[x0, ym1, z0]) +
                                ztz2[0] * ihz2[0] * (U[x0, y0, zp1] + U[x0, y0, zm1]) +
                                ztzz[0] * i2hz[0] * (U[x0, y0, zp1] - U[x0, y0, zm1]))
                            ) / (1 + nu*dt*(ihx2[0]*xix2[0] + ihy2[0]*ety2[0] + ihz2[0]*ztz2[0]))
        else:
            U[x0, y0, z0] = (rho[x0, y0, z0] + 0.5*nu*dt*(
                                ihx2[0] * (U[xp1, y0, z0] + U[xm1, y0, z0]) +
                                ihy2[0] * (U[x0, yp1, z0] + U[x0, ym1, z0]) +
                                ihz2[0] * (U[x0, y0, zp1] + U[x0, y0, zm1]))) / (1 + nu*dt*(ihx2[0] + ihy2[0] + ihz2[0]))

        imposeUBCs(U)
        
        if nuFlag:
            locmaxErr = np.amax(np.fabs(rho[x0, y0, z0] - (U[x0, y0, z0] - 0.5*nu*dt*(
                               xix2[0] * ihx2[0] * (U[xp1, y0, z0] - 2.0*U[x0, y0, z0] + U[xm1, y0, z0]) + 
                               xixx[0] * i2hx[0] * (U[xp1, y0, z0] - U[xm1, y0, z0]) + 
                               ety2[0] * ihy2[0] * (U[x0, yp1, z0] - 2.0*U[x0, y0, z0] + U[x0, ym1, z0]) + 
                               etyy[0] * i2hy[0] * (U[x0, yp1, z0] - U[x0, ym1, z0]) + 
                               ztz2[0] * ihz2[0] * (U[x0, y0, zp1] - 2.0*U[x0, y0, z0] + U[x0, y0, zm1]) +
                               ztzz[0] * i2hz[0] * (U[x0, y0, zp1] - U[x0, y0, zm1])))))
        else:
            locmaxErr = np.amax(np.fabs(rho[x0, y0, z0] - (U[x0, y0, z0] - 0.5*nu*dt*(
                                (U[xp1, y0, z0] - 2.0*U[x0, y0, z0] + U[xm1, y0, z0])*ihx2[0] +
                                (U[x0, yp1, z0] - 2.0*U[x0, y0, z0] + U[x0, ym1, z0])*ihy2[0] +
                                (U[x0, y0, zp1] - 2.0*U[x0, y0, z0] + U[x0, y0, zm1])*ihz2[0]))))

        totmaxErr = comm.allreduce(locmaxErr, op=MPI.MAX)

        jCnt += 1

        if totmaxErr < VpTolerance:
            #if rootRank: print(jCnt)
            break
        
        if jCnt > maxCount:
            print("ERROR: Jacobi not converging in U. Aborting")
            quit()

    return U[x0, y0, z0]        


def vJacobi(rho):
    global dt, nu
    global nuFlag
    global ihx2, i2hx, ihy2, i2hy, ihz2, i2hz
    global xixx, xix2, etyy, ety2, ztzz, ztz2

    jCnt = 0
    while True:
        if nuFlag:
            V[x0, y0, z0] = (rho[x0, y0, z0] + 0.5*nu*dt*(
                                xix2[0] * ihx2[0] * (V[xp1, y0, z0] + V[xm1, y0, z0]) +
                                xixx[0] * i2hx[0] * (V[xp1, y0, z0] - V[xm1, y0, z0]) +
                                ety2[0] * ihy2[0] * (V[x0, yp1, z0] + V[x0, ym1, z0]) +
                                etyy[0] * i2hy[0] * (V[x0, yp1, z0] - V[x0, ym1, z0]) +
                                ztz2[0] * ihz2[0] * (V[x0, y0, zp1] + V[x0, y0, zm1]) +
                                ztzz[0] * i2hz[0] * (V[x0, y0, zp1] - V[x0, y0, zm1]))
                            ) / (1 + nu*dt*(ihx2[0]*xix2[0] + ihy2[0]*ety2[0] + ihz2[0]*ztz2[0]))
        else:
            V[x0, y0, z0] = (rho[x0, y0, z0] + 0.5*nu*dt*(
                                ihx2[0] * (V[xp1, y0, z0] + V[xm1, y0, z0]) +
                                ihy2[0] * (V[x0, yp1, z0] + V[x0, ym1, z0]) +
                                ihz2[0] * (V[x0, y0, zp1] + V[x0, y0, zm1]))) / (1 + nu*dt*(ihx2[0] + ihy2[0] + ihz2[0]))

        imposeVBCs(V)

        if nuFlag:
            locmaxErr = np.amax(np.fabs(rho[x0, y0, z0] - (V[x0, y0, z0] - 0.5*nu*dt*(
                               xix2[0] * ihx2[0] * (V[xp1, y0, z0] - 2.0*V[x0, y0, z0] + V[xm1, y0, z0]) + 
                               xixx[0] * i2hx[0] * (V[xp1, y0, z0] - V[xm1, y0, z0]) + 
                               ety2[0] * ihy2[0] * (V[x0, yp1, z0] - 2.0*V[x0, y0, z0] + V[x0, ym1, z0]) + 
                               etyy[0] * i2hy[0] * (V[x0, yp1, z0] - V[x0, ym1, z0]) + 
                               ztz2[0] * ihz2[0] * (V[x0, y0, zp1] - 2.0*V[x0, y0, z0] + V[x0, y0, zm1]) +
                               ztzz[0] * i2hz[0] * (V[x0, y0, zp1] - V[x0, y0, zm1])))))
        else:
            locmaxErr = np.amax(np.fabs(rho[x0, y0, z0] - (V[x0, y0, z0] - 0.5*nu*dt*(
                                (V[xp1, y0, z0] - 2.0*V[x0, y0, z0] + V[xm1, y0, z0])*ihx2[0] +
                                (V[x0, yp1, z0] - 2.0*V[x0, y0, z0] + V[x0, ym1, z0])*ihy2[0] +
                                (V[x0, y0, zp1] - 2.0*V[x0, y0, z0] + V[x0, y0, zm1])*ihz2[0]))))

        totmaxErr = comm.allreduce(locmaxErr, op=MPI.MAX)

        jCnt += 1

        if totmaxErr < VpTolerance:
            #if rootRank: print(jCnt)
            break
    
        if jCnt > maxCount:
            print("ERROR: Jacobi not converging in V. Aborting")
            quit()
    
    return V[x0, y0, z0]


def wJacobi(rho):
    global dt, nu
    global nuFlag
    global ihx2, i2hx, ihy2, i2hy, ihz2, i2hz
    global xixx, xix2, etyy, ety2, ztzz, ztz2

    jCnt = 0
    while True:
        if nuFlag:
            W[x0, y0, z0] = (rho[x0, y0, z0] + 0.5*nu*dt*(
                                xix2[0] * ihx2[0] * (W[xp1, y0, z0] + W[xm1, y0, z0]) +
                                xixx[0] * i2hx[0] * (W[xp1, y0, z0] - W[xm1, y0, z0]) +
                                ety2[0] * ihy2[0] * (W[x0, yp1, z0] + W[x0, ym1, z0]) +
                                etyy[0] * i2hy[0] * (W[x0, yp1, z0] - W[x0, ym1, z0]) +
                                ztz2[0] * ihz2[0] * (W[x0, y0, zp1] + W[x0, y0, zm1]) +
                                ztzz[0] * i2hz[0] * (W[x0, y0, zp1] - W[x0, y0, zm1]))
                            ) / (1 + nu*dt*(ihx2[0]*xix2[0] + ihy2[0]*ety2[0] + ihz2[0]*ztz2[0]))
        else:
            W[x0, y0, z0] = (rho[x0, y0, z0] + 0.5*nu*dt*(
                                ihx2[0] * (W[xp1, y0, z0] + W[xm1, y0, z0]) +
                                ihy2[0] * (W[x0, yp1, z0] + W[x0, ym1, z0]) +
                                ihz2[0] * (W[x0, y0, zp1] + W[x0, y0, zm1]))) / (1 + nu*dt*(ihx2[0] + ihy2[0] + ihz2[0]))

        imposeWBCs(W)

        if nuFlag:
            locmaxErr = np.amax(np.fabs(rho[x0, y0, z0] - (W[x0, y0, z0] - 0.5*nu*dt*(
                               xix2[0] * ihx2[0] * (W[xp1, y0, z0] - 2.0*W[x0, y0, z0] + W[xm1, y0, z0]) + 
                               xixx[0] * i2hx[0] * (W[xp1, y0, z0] - W[xm1, y0, z0]) + 
                               ety2[0] * ihy2[0] * (W[x0, yp1, z0] - 2.0*W[x0, y0, z0] + W[x0, ym1, z0]) + 
                               etyy[0] * i2hy[0] * (W[x0, yp1, z0] - W[x0, ym1, z0]) + 
                               ztz2[0] * ihz2[0] * (W[x0, y0, zp1] - 2.0*W[x0, y0, z0] + W[x0, y0, zm1]) +
                               ztzz[0] * i2hz[0] * (W[x0, y0, zp1] - W[x0, y0, zm1])))))
        else:
            locmaxErr = np.amax(np.fabs(rho[x0, y0, z0] - (W[x0, y0, z0] - 0.5*nu*dt*(
                                (W[xp1, y0, z0] - 2.0*W[x0, y0, z0] + W[xm1, y0, z0])*ihx2[0] +
                                (W[x0, yp1, z0] - 2.0*W[x0, y0, z0] + W[x0, ym1, z0])*ihy2[0] +
                                (W[x0, y0, zp1] - 2.0*W[x0, y0, z0] + W[x0, y0, zm1])*ihz2[0]))))
    
        totmaxErr = comm.allreduce(locmaxErr, op=MPI.MAX)

        jCnt += 1

        if totmaxErr < VpTolerance:
            #if rootRank: print(jCnt)
            break

        if jCnt > maxCount:
            print("ERROR: Jacobi not converging in W. Aborting")
            quit()
    
    return W[x0, y0, z0]       


def TJacobi(rho):
    global dt
    global kappa
    global nuFlag
    global ihx2, i2hx, ihy2, i2hy, ihz2, i2hz
    global xixx, xix2, etyy, ety2, ztzz, ztz2

    jCnt = 0
    while True:
        if nuFlag:
            T[x0, y0, z0] = (rho[x0, y0, z0] + 0.5*kappa*dt*(
                                xix2[0] * ihx2[0] * (T[xp1, y0, z0] + T[xm1, y0, z0]) +
                                xixx[0] * i2hx[0] * (T[xp1, y0, z0] - T[xm1, y0, z0]) +
                                ety2[0] * ihy2[0] * (T[x0, yp1, z0] + T[x0, ym1, z0]) +
                                etyy[0] * i2hy[0] * (T[x0, yp1, z0] - T[x0, ym1, z0]) +
                                ztz2[0] * ihz2[0] * (T[x0, y0, zp1] + T[x0, y0, zm1]) +
                                ztzz[0] * i2hz[0] * (T[x0, y0, zp1] - T[x0, y0, zm1]))
                            ) / (1 + kappa*dt*(ihx2[0]*xix2[0] + ihy2[0]*ety2[0] + ihz2[0]*ztz2[0]))
        else:
            T[x0, y0, z0] = (rho[x0, y0, z0] + 0.5*kappa*dt*(
                                ihx2[0] * (T[xp1, y0, z0] + T[xm1, y0, z0]) +
                                ihy2[0] * (T[x0, yp1, z0] + T[x0, ym1, z0]) +
                                ihz2[0] * (T[x0, y0, zp1] + T[x0, y0, zm1]))) / (1 + kappa*dt*(ihx2[0] + ihy2[0] + ihz2[0]))

        imposeTBCs(T)

        if nuFlag:
            locmaxErr = np.amax(np.fabs(rho[x0, y0, z0] - (T[x0, y0, z0] - 0.5*kappa*dt*(
                               xix2[0] * ihx2[0] * (T[xp1, y0, z0] - 2.0*T[x0, y0, z0] + T[xm1, y0, z0]) + 
                               xixx[0] * i2hx[0] * (T[xp1, y0, z0] - T[xm1, y0, z0]) + 
                               ety2[0] * ihy2[0] * (T[x0, yp1, z0] - 2.0*T[x0, y0, z0] + T[x0, ym1, z0]) + 
                               etyy[0] * i2hy[0] * (T[x0, yp1, z0] - T[x0, ym1, z0]) + 
                               ztz2[0] * ihz2[0] * (T[x0, y0, zp1] - 2.0*T[x0, y0, z0] + T[x0, y0, zm1]) +
                               ztzz[0] * i2hz[0] * (T[x0, y0, zp1] - T[x0, y0, zm1])))))
        else:
            locmaxErr = np.amax(np.fabs(rho[x0, y0, z0] - (T[x0, y0, z0] - 0.5*kappa*dt*(
                                (T[xp1, y0, z0] - 2.0*T[x0, y0, z0] + T[xm1, y0, z0])*ihx2[0] +
                                (T[x0, yp1, z0] - 2.0*T[x0, y0, z0] + T[x0, ym1, z0])*ihy2[0] +
                                (T[x0, y0, zp1] - 2.0*T[x0, y0, z0] + T[x0, y0, zm1])*ihz2[0]))))
    
        totmaxErr = comm.allreduce(locmaxErr, op=MPI.MAX)

        jCnt += 1

        if totmaxErr < VpTolerance:
            #if rootRank: print(jCnt)
            break
    
        if jCnt > maxCount:
            print("ERROR: Jacobi not converging in T. Aborting")
            quit()
    
    return T[x0, y0, z0]       




def PpJacobi(rho):
    global dt
    global kappa
    global nuFlag
    global ihx2, i2hx, ihy2, i2hy, ihz2, i2hz
    global xixx, xix2, etyy, ety2, ztzz, ztz2

    jCnt = 0
    while True:

        imposePpBCs(pData[0])

        if nuFlag:
            # For non-uniform grid
            pData[0][1:-1, 1:-1, 1:-1] = (xix2[0] * ihx2[0] * (pData[0][2:, 1:-1, 1:-1] + pData[0][:-2, 1:-1, 1:-1]) +
                                             xixx[0] * i2hx[0] * (pData[0][2:, 1:-1, 1:-1] - pData[0][:-2, 1:-1, 1:-1]) +
                                             ety2[0] * ihy2[0] * (pData[0][1:-1, 2:, 1:-1] + pData[0][1:-1, :-2, 1:-1]) +
                                             etyy[0] * i2hy[0] * (pData[0][1:-1, 2:, 1:-1] - pData[0][1:-1, :-2, 1:-1]) +
                                             ztz2[0] * ihz2[0] * (pData[0][1:-1, 1:-1, 2:] + pData[0][1:-1, 1:-1, :-2]) +
                                             ztzz[0] * i2hz[0] * (pData[0][1:-1, 1:-1, 2:] - pData[0][1:-1, 1:-1, :-2]) -
                                               rho[1:-1, 1:-1, 1:-1]) / (2.0*(ihx2[0]*xix2[0] +
                                                                             ihy2[0]*ety2[0] + ihz2[0]*ztz2[0]))
        else:
            pData[0][1:-1, 1:-1, 1:-1] = (hyhz[0]*(pData[0][2:, 1:-1, 1:-1] + pData[0][:-2, 1:-1, 1:-1]) +
                                                   hzhx[0]*(pData[0][1:-1, 2:, 1:-1] + pData[0][1:-1, :-2, 1:-1]) +
                                                   hxhy[0]*(pData[0][1:-1, 1:-1, 2:] + pData[0][1:-1, 1:-1, :-2]) -
                                                 hxhyhz[0]*rho[1:-1, 1:-1, 1:-1]) * gsFactor[0]


        locmaxErr = np.amax(np.abs(rho[1:-1, 1:-1, 1:-1] -((
                        (pData[0][:-2, 1:-1, 1:-1] - 2.0*pData[0][1:-1, 1:-1, 1:-1] + pData[0][2:, 1:-1, 1:-1])*ihx2[0] +
                        (pData[0][1:-1, :-2, 1:-1] - 2.0*pData[0][1:-1, 1:-1, 1:-1] + pData[0][1:-1, 2:, 1:-1])*ihy2[0] +
                        (pData[0][1:-1, 1:-1, :-2] - 2.0*pData[0][1:-1, 1:-1, 1:-1] + pData[0][1:-1, 1:-1, 2:])*ihz2[0]))))
    
        totmaxErr = comm.allreduce(locmaxErr, op=MPI.MAX)

        #print(jCnt, totmaxErr)

        jCnt += 1

        if totmaxErr < tolerance:
            if rootRank: print(jCnt)
            break
    
        if jCnt > maxCount:
            print("ERROR: Jacobi not converging in Pp. Aborting")
            quit()
    
    return pData[0]       



####### Stretch Spiral Vortex LES Model ########


def computeSGS():
    global N
    global nuFlag
    global U, V, W
    global Hx, Hy, Hz
    global i2hx, i2hy, i2hz
    global xi_x, et_y, zt_z

    n = N[0]

    Sxx = np.zeros(n)
    Syy = np.zeros(n)
    Szz = np.zeros(n)
    Sxy = np.zeros(n)
    Syz = np.zeros(n)
    Szx = np.zeros(n)

    if nuFlag:
        Sxx = (U[xp1, y0, z0] - U[xm1, y0, z0]) * xi_x[0] * i2hx[0]
        Syy = (V[x0, yp1, z0] - V[x0, ym1, z0]) * et_y[0] * i2hy[0]
        Szz = (W[x0, y0, zp1] - W[x0, y0, zm1]) * zt_z[0] * i2hz[0]

        Sxy = ((U[x0, yp1, z0] - U[x0, ym1, z0]) * et_y[0] * i2hy[0] + (V[xp1, y0, z0] - V[xm1, y0, z0]) * xi_x[0] * i2hx[0])/2.0
        Syz = ((V[x0, y0, zp1] - V[x0, y0, zm1]) * zt_z[0] * i2hz[0] + (W[x0, yp1, z0] - W[x0, ym1, z0]) * et_y[0] * i2hy[0])/2.0
        Szx = ((W[xp1, y0, z0] - W[xm1, y0, z0]) * xi_x[0] * i2hx[0] + (U[x0, y0, zp1] - U[x0, y0, zm1]) * zt_z[0] * i2hz[0])/2.0
    else:
        Sxx = (U[xp1, y0, z0] - U[xm1, y0, z0]) * i2hx[0]
        Syy = (V[x0, yp1, z0] - V[x0, ym1, z0]) * i2hy[0]
        Szz = (W[x0, y0, zp1] - W[x0, y0, zm1]) * i2hz[0]

        Sxy = ((U[x0, yp1, z0] - U[x0, ym1, z0]) * i2hy[0] + (V[xp1, y0, z0] - V[xm1, y0, z0]) * i2hx[0])/2.0
        Syz = ((V[x0, y0, zp1] - V[x0, y0, zm1]) * i2hz[0] + (W[x0, yp1, z0] - W[x0, ym1, z0]) * i2hy[0])/2.0
        Szx = ((W[xp1, y0, z0] - W[xm1, y0, z0]) * i2hx[0] + (U[x0, y0, zp1] - U[x0, y0, zm1]) * i2hz[0])/2.0

    a = -(Sxx + Syy + Szz)
    b = Sxx*Syy - Sxy*Sxy + Syy*Szz - Syz*Syz + Szz*Sxx - Szx*Szx
    c = -(Sxx*(Syy*Szz - Syz*Syz) - Sxy*(Sxy*Szz - Syz*Szx) + Szx*(Sxy*Syz - Syy*Szx))

    q = (3.0*b - a*a)/9.0
    r = (9.0*a*b - 27.0*c - 2.0*a*a*a)/54.0

    if np.any(q >= 0.0):
        print("q is non-negative in eigenvalue calculation. Aborting")
        quit()

    costheta = r/np.sqrt(-q*q*q)
    theta = np.zeros_like(costheta)

    theta[costheta < -1.0] = np.pi
    theta[(costheta > -1.0) & (costheta < 1.0)] = np.arccos(costheta[(costheta > -1.0) & (costheta < 1.0)])

    # Array of eigenvalues as a 4D array
    eigvals = np.zeros([n[0], n[1], n[2], 3])
    eigvals[:,:,:,0] = 2.0*np.sqrt(-q)*np.cos((theta            )/3.0) - a/3.0
    eigvals[:,:,:,1] = 2.0*np.sqrt(-q)*np.cos((theta + 2.0*np.pi)/3.0) - a/3.0
    eigvals[:,:,:,2] = 2.0*np.sqrt(-q)*np.cos((theta + 4.0*np.pi)/3.0) - a/3.0

    # Sort array along last dimension - i.e. sort eigenvalues
    eigvals.sort()

    # Choose the largest eigenvalues
    eigvals = eigvals[:,:,:,-1]

    det = np.zeros([n[0], n[1], n[2], 3])
    det[:,:,:,0] = (Syy - eigvals) * (Szz - eigvals) - Syz*Syz
    det[:,:,:,1] = (Szz - eigvals) * (Sxx - eigvals) - Szx*Szx
    det[:,:,:,2] = (Sxx - eigvals) * (Syy - eigvals) - Sxy*Sxy

    fdet = np.fabs(det)

    e = np.zeros_like(det)
    a = -Sxy*(Szz - eigvals) + Syz*Szx
    b = -Szx*(Syy - eigvals) + Sxy*Syz
    c = -Syz*(Sxx - eigvals) + Sxy*Szx

    miArray = np.argmax(fdet, axis=3)

    for i in range(n[0]):
        for j in range(n[1]):
            for k in range(n[2]):
                if miArray[i, j, k] == 0:
                    e[i, j, k, :] = np.array([1.0, a[i, j, k]/det[i, j, k, 0], b[i, j, k]/det[i, j, k, 0]])
                elif miArray[i, j, k] == 1:
                    e[i, j, k, :] = np.array([a[i, j, k]/det[i, j, k, 1], 1.0, c[i, j, k]/det[i, j, k, 1]])
                elif miArray[i, j, k] == 2:
                    e[i, j, k, :] = np.array([b[i, j, k]/det[i, j, k, 2], c[i, j, k]/det[i, j, k, 2], 1.0])

    e /= np.linalg.norm(e, axis=3)[:,:,:,np.newaxis]

    a = Sxx*e[:,:,:,0]**2.0 + Syy*e[:,:,:,1]**2.0 + Szz*e[:,:,:,2]**2.0 + 2.0*(
        Sxy*e[:,:,:,0]*e[:,:,:,1] + Syz*e[:,:,:,1]*e[:,:,:,2] + Szx*e[:,:,:,2]*e[:,:,:,0])

    lv = np.sqrt(2.0*nu/(3.0*np.abs(a)))

    q.fill(0.0)
    r.fill(0.0)

    # Temporary hack  - works for uniform grids only
    hDel = xCord[1] - xCord[0]
    for i in range(-1, 2):
        dx = xCord[xSt+i:xEn+i, np.newaxis, np.newaxis] - xCord[xSt:xEn, np.newaxis, np.newaxis]
        for j in range(-1, 2):
            dy = yCord[ySt+j:yEn+j, np.newaxis] - yCord[ySt:yEn, np.newaxis]
            for k in range(-1, 2):
                dz = zCord[zSt+k:zEn+k] - zCord[zSt:zEn]
                if (i or j or k):
                    a = U[xSt+i:xEn+i, ySt+j:yEn+j, zSt+k:zEn+k] - U[xSt:xEn, ySt:yEn, zSt:zEn]
                    b = V[xSt+i:xEn+i, ySt+j:yEn+j, zSt+k:zEn+k] - V[xSt:xEn, ySt:yEn, zSt:zEn]
                    c = W[xSt+i:xEn+i, ySt+j:yEn+j, zSt+k:zEn+k] - W[xSt:xEn, ySt:yEn, zSt:zEn]
                    q += a*a + b*b + c*c

                    dx2 = dx*dx + dy*dy + dz*dz
                    dxe = dx*e[:,:,:,0] + dy*e[:,:,:,1] + dz*e[:,:,:,2]

                    # Structure function integral
                    d = np.sqrt(dx2 - dxe*dxe)/hDel
                    d2 = d*d
                    r[d < 0.873469] += 7.4022*d2[d < 0.873469] - 1.82642*d2[d < 0.873469]*d2[d < 0.873469]
                    r[d >= 0.873469] += (12.2946*np.power(d[d >= 0.873469], 2.0/3.0) - 6.0 -
                                        0.573159*np.power(d[d >= 0.873469], -1.5)*np.sin(3.14159*d[d >= 0.873469] - 0.785398))

    q /= 26.0
    r /= 26.0

    prefac = q/r
    kc = np.pi/hDel

    q = (kc*lv)**2.0

    # Sub-grid kinetic energy integral
    r[q < 2.42806] = ((3.0 +   2.5107*q[q < 2.42806] +  0.330357*(q[q < 2.42806]**2.0) +  0.0295481*(q[q < 2.42806]**3.0))/
                      (1.0 + 0.336901*q[q < 2.42806] + 0.0416684*(q[q < 2.42806]**2.0) + 0.00187191*(q[q < 2.42806]**3.0)) -
                     4.06235*np.power(q[q < 2.42806], 1.0/3.0))/2.0

    r[q >= 2.42806] = ((1.26429 + 0.835714*q[q >= 2.42806] + 0.0964286*(q[q >= 2.42806]**2.0))*np.exp(-q[q >= 2.42806])/
                       (2.0     +      4.5*q[q >= 2.42806] +  1.928572*(q[q >= 2.42806]**2.0) + 0.1928572*(q[q >= 2.42806]**3.0)))

    K = prefac*r

    Sxx = (1.0 - e[:,:,:,0]*e[:,:,:,0])*K
    Syy = (1.0 - e[:,:,:,1]*e[:,:,:,1])*K
    Szz = (1.0 - e[:,:,:,2]*e[:,:,:,2])*K
    Sxy = (      e[:,:,:,0]*e[:,:,:,1])*K
    Syz = (      e[:,:,:,1]*e[:,:,:,2])*K
    Szx = (      e[:,:,:,2]*e[:,:,:,0])*K

    Sxx = np.pad(Sxx, 1)
    Syy = np.pad(Syy, 1)
    Szz = np.pad(Szz, 1)
    Sxy = np.pad(Sxy, 1)
    Syz = np.pad(Syz, 1)
    Szx = np.pad(Szx, 1)

    data_transfer(Sxx)
    data_transfer(Syy)
    data_transfer(Szz)
    data_transfer(Sxy)
    data_transfer(Syz)
    data_transfer(Szx)

    if nuFlag:
        U[x0, y0, z0] += (Sxx[xp1, y0, z0] - Sxx[xm1, y0, z0])*xi_x[0]*i2hx[0] + (Sxy[x0, yp1, z0] - Sxy[x0, ym1, z0])*et_y[0]*i2hy[0] + (Szx[x0, y0, zp1] - Szx[x0, y0, zm1])*zt_z[0]*i2hz[0]
        V[x0, y0, z0] += (Sxy[xp1, y0, z0] - Sxy[xm1, y0, z0])*xi_x[0]*i2hx[0] + (Syy[x0, yp1, z0] - Syy[x0, ym1, z0])*et_y[0]*i2hy[0] + (Syz[x0, y0, zp1] - Syz[x0, y0, zm1])*zt_z[0]*i2hz[0]
        W[x0, y0, z0] += (Szx[xp1, y0, z0] - Szx[xm1, y0, z0])*xi_x[0]*i2hx[0] + (Syz[x0, yp1, z0] - Syz[x0, ym1, z0])*et_y[0]*i2hy[0] + (Szz[x0, y0, zp1] - Szz[x0, y0, zm1])*zt_z[0]*i2hz[0]
    else:
        U[x0, y0, z0] += (Sxx[xp1, y0, z0] - Sxx[xm1, y0, z0])*i2hx[0] + (Sxy[x0, yp1, z0] - Sxy[x0, ym1, z0])*i2hy[0] + (Szx[x0, y0, zp1] - Szx[x0, y0, zm1])*i2hz[0]
        V[x0, y0, z0] += (Sxy[xp1, y0, z0] - Sxy[xm1, y0, z0])*i2hx[0] + (Syy[x0, yp1, z0] - Syy[x0, ym1, z0])*i2hy[0] + (Syz[x0, y0, zp1] - Syz[x0, y0, zm1])*i2hz[0]
        W[x0, y0, z0] += (Szx[xp1, y0, z0] - Szx[xm1, y0, z0])*i2hx[0] + (Syz[x0, yp1, z0] - Syz[x0, ym1, z0])*i2hy[0] + (Szz[x0, y0, zp1] - Szz[x0, y0, zm1])*i2hz[0]

    #print(Sxx.shape)
    #print(Sxx[0, 2, 21])
    #print(np.min(K))
    #print(np.unravel_index(np.argmax(K, axis=None), K.shape))
    #print(Sxx[0, 2, 21])
    #quit()




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
        #if rootRank: print(i, totmaxRes)
        if totmaxRes < tolerance:
            #if rootRank: print(i+1)
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
    global nuFlag
    global rData, pData
    global hyhz, hzhx, hxhy, hxhyhz, gsFactor
    global ihx2, i2hx, ihy2, i2hy, ihz2, i2hz
    global xixx, xix2, etyy, ety2, ztzz, ztz2

    n = N[vLev]
    for iCnt in range(sCount):
        imposePpBCs(pData[vLev])

        if nuFlag:
            # For non-uniform grid
            
            pData[vLev][1:-1, 1:-1, 1:-1] = (xix2[vLev] * ihx2[vLev] * (pData[vLev][2:, 1:-1, 1:-1] + pData[vLev][:-2, 1:-1, 1:-1]) +
                                             xixx[vLev] * i2hx[vLev] * (pData[vLev][2:, 1:-1, 1:-1] - pData[vLev][:-2, 1:-1, 1:-1]) +
                                             ety2[vLev] * ihy2[vLev] * (pData[vLev][1:-1, 2:, 1:-1] + pData[vLev][1:-1, :-2, 1:-1]) +
                                             etyy[vLev] * i2hy[vLev] * (pData[vLev][1:-1, 2:, 1:-1] - pData[vLev][1:-1, :-2, 1:-1]) +
                                             ztz2[vLev] * ihz2[vLev] * (pData[vLev][1:-1, 1:-1, 2:] + pData[vLev][1:-1, 1:-1, :-2]) +
                                             ztzz[vLev] * i2hz[vLev] * (pData[vLev][1:-1, 1:-1, 2:] - pData[vLev][1:-1, 1:-1, :-2]) -
                                               rData[vLev][1:-1, 1:-1, 1:-1]) / (2.0*(ihx2[vLev]*xix2[vLev] +
                                                                             ihy2[vLev]*ety2[vLev] + ihz2[vLev]*ztz2[vLev]))

        else:
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
    global vLev
    global iTemp, rData

    pLev = vLev
    vLev += 1

    rData[vLev][1:-1, 1:-1, 1:-1] = (iTemp[pLev][1:-1:2, 1:-1:2, 1:-1:2] + iTemp[pLev][2::2, 2::2, 2::2] +
                                     iTemp[pLev][1:-1:2, 1:-1:2, 2::2] + iTemp[pLev][2::2, 2::2, 1:-1:2] +
                                     iTemp[pLev][1:-1:2, 2::2, 1:-1:2] + iTemp[pLev][2::2, 1:-1:2, 2::2] +
                                     iTemp[pLev][2::2, 1:-1:2, 1:-1:2] + iTemp[pLev][1:-1:2, 2::2, 2::2])/8


# Solves at coarsest level using the Gauss-Seidel iterative solver
def solve():
    global nuFlag
    global N, vLev
    global tolerance
    global maxCountPp
    global pData, rData
    global hyhz, hzhx, hxhy, hxhyhz, gsFactor
    global ihx2, i2hx, ihy2, i2hy, ihz2, i2hz
    global xixx, xix2, etyy, ety2, ztzz, ztz2

    n = N[vLev]

    jCnt = 0
    while True:
        imposePpBCs(pData[vLev])

        if nuFlag:
            # For non-uniform grid
            pData[vLev][1:-1, 1:-1, 1:-1] = (xix2[vLev] * ihx2[vLev] * (pData[vLev][2:, 1:-1, 1:-1] + pData[vLev][:-2, 1:-1, 1:-1]) +
                                             xixx[vLev] * i2hx[vLev] * (pData[vLev][2:, 1:-1, 1:-1] - pData[vLev][:-2, 1:-1, 1:-1]) +
                                             ety2[vLev] * ihy2[vLev] * (pData[vLev][1:-1, 2:, 1:-1] + pData[vLev][1:-1, :-2, 1:-1]) +
                                             etyy[vLev] * i2hy[vLev] * (pData[vLev][1:-1, 2:, 1:-1] - pData[vLev][1:-1, :-2, 1:-1]) +
                                             ztz2[vLev] * ihz2[vLev] * (pData[vLev][1:-1, 1:-1, 2:] + pData[vLev][1:-1, 1:-1, :-2]) +
                                             ztzz[vLev] * i2hz[vLev] * (pData[vLev][1:-1, 1:-1, 2:] - pData[vLev][1:-1, 1:-1, :-2]) -
                                               rData[vLev][1:-1, 1:-1, 1:-1]) / (2.0*(ihx2[vLev]*xix2[vLev] +
                                                                             ihy2[vLev]*ety2[vLev] + ihz2[vLev]*ztz2[vLev]))
        else:
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

        #print(jCnt, totmaxErr)

        if totmaxErr < tolerance:
            break

        jCnt += 1
        if jCnt > maxCountPp:
            #print(maxCountPp, totmaxErr)
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
    global nuFlag
    global ihx2, i2hx, ihy2, i2hy, ihz2, i2hz
    global xixx, xix2, etyy, ety2, ztzz, ztz2

    if nuFlag:
        # For non-uniform grid
        laplacian = (xix2[vLev] * ihx2[vLev] * (function[2:, 1:-1, 1:-1] - 2.0*function[1:-1, 1:-1, 1:-1] + function[:-2, 1:-1, 1:-1]) + \
                     xixx[vLev] * i2hx[vLev] * (function[2:, 1:-1, 1:-1] - function[:-2, 1:-1, 1:-1]) +
                     ety2[vLev] * ihy2[vLev] * (function[1:-1, 2:, 1:-1] - 2.0*function[1:-1, 1:-1, 1:-1] + function[1:-1, :-2, 1:-1]) + \
                     etyy[vLev] * i2hy[vLev] * (function[1:-1, 2:, 1:-1] - function[1:-1, :-2, 1:-1]) +
                     ztz2[vLev] * ihz2[vLev] * (function[1:-1, 1:-1, 2:] - 2.0*function[1:-1, 1:-1, 1:-1] + function[1:-1, 1:-1, :-2]) + \
                     ztzz[vLev] * i2hz[vLev] * (function[1:-1, 1:-1, 2:] - function[1:-1, 1:-1, :-2]))
    else:
        # For uniform grid
        laplacian = ((function[:-2, 1:-1, 1:-1] - 2.0*function[1:-1, 1:-1, 1:-1] + function[2:, 1:-1, 1:-1]) * ihx2[vLev] + 
                     (function[1:-1, :-2, 1:-1] - 2.0*function[1:-1, 1:-1, 1:-1] + function[1:-1, 2:, 1:-1]) * ihy2[vLev] +
                     (function[1:-1, 1:-1, :-2] - 2.0*function[1:-1, 1:-1, 1:-1] + function[1:-1, 1:-1, 2:]) * ihz2[vLev])

    return np.pad(laplacian, 1)


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
    maxSolCount = 10*N[-1][0]*N[-1][1]*N[-1][2]

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


############################## GRID INITIALIZATION ##############################


# Initialize the grid. This is relevant only for non-uniform grids
def initGrid():
    global N
    global nuFlag
    global VDepth
    global hx, hy, hz
    global hyhz, hzhx, hxhy, hxhyhz, gsFactor
    global mghx, xPts, i2hx, ihx2, xi_x, xixx, xix2
    global mghy, yPts, i2hy, ihy2, et_y, etyy, ety2
    global mghz, zPts, i2hz, ihz2, zt_z, ztzz, ztz2

    hx0 = 1.0/(N[0][0])
    hy0 = 1.0/(N[0][1])
    hz0 = 1.0/(N[0][2])

    # Old coefficients used in old sections of code
    thx2 = [hx*(2**x) for x in range(VDepth+1)]
    thy2 = [hy*(2**x) for x in range(VDepth+1)]
    thz2 = [hz*(2**x) for x in range(VDepth+1)]

    mghx2 = [x*x for x in thx2]
    mghy2 = [x*x for x in thy2]
    mghz2 = [x*x for x in thz2]

    hyhz = [mghy2[i]*mghz2[i] for i in range(VDepth + 1)]
    hzhx = [mghx2[i]*mghz2[i] for i in range(VDepth + 1)]
    hxhy = [mghx2[i]*mghy2[i] for i in range(VDepth + 1)]
    hxhyhz = [mghx2[i]*mghy2[i]*mghz2[i] for i in range(VDepth + 1)]

    gsFactor = [1.0/(2.0*(hyhz[i] + hzhx[i] + hxhy[i])) for i in range(VDepth + 1)]

    # New coefficients used in new sections of code
    mghx = np.zeros(VDepth+1)
    mghy = np.zeros(VDepth+1)
    mghz = np.zeros(VDepth+1)

    ihx2 = np.zeros(VDepth+1)
    ihy2 = np.zeros(VDepth+1)
    ihz2 = np.zeros(VDepth+1)

    i2hx = np.zeros(VDepth+1)
    i2hy = np.zeros(VDepth+1)
    i2hz = np.zeros(VDepth+1)

    for i in range(VDepth+1):
        mghx[i] = hx0*(2**i)
        mghy[i] = hy0*(2**i)
        mghz[i] = hz0*(2**i)

        ihx2[i] = 1.0/(mghx[i]*mghx[i])
        ihy2[i] = 1.0/(mghy[i]*mghy[i])
        ihz2[i] = 1.0/(mghz[i]*mghz[i])

        i2hx[i] = 1.0/(2.0*mghx[i])
        i2hy[i] = 1.0/(2.0*mghy[i])
        i2hz[i] = 1.0/(2.0*mghz[i])

    # Uniform grid default values
    vPts = [np.linspace(-0.5, 0.5, n[0]+1) for n in N]
    xPts = [(x[1:] + x[:-1])/2.0 for x in vPts]
    xi_x = [np.ones_like(i) for i in xPts]
    xix2 = [np.ones_like(i) for i in xPts]
    xixx = [np.zeros_like(i) for i in xPts]

    vPts = [np.linspace(-0.5, 0.5, n[1]+1) for n in N]
    yPts = [(y[1:] + y[:-1])/2.0 for y in vPts]
    et_y = [np.ones_like(i) for i in yPts]
    ety2 = [np.ones_like(i) for i in yPts]
    etyy = [np.zeros_like(i) for i in yPts]

    vPts = [np.linspace(-0.5, 0.5, n[2]+1) for n in N]
    zPts = [(z[1:] + z[:-1])/2.0 for z in vPts]
    zt_z = [np.ones_like(i) for i in zPts]
    ztz2 = [np.ones_like(i) for i in zPts]
    ztzz = [np.zeros_like(i) for i in zPts]

    # Overwrite above arrays with values for tangent-hyperbolic grid is nuFlag is enabled.
    if nuFlag:
        for i in range(VDepth+1):
            n = N[i]

            vPts = np.linspace(0.0, 1.0, n[0]+1)
            xi = (vPts[1:] + vPts[:-1])/2.0
            xPts[i] = np.array([(1.0 - np.tanh(beta*(1.0 - 2.0*i))/np.tanh(beta))/2.0 for i in xi])
            xi_x[i] = np.array([np.tanh(beta)/(beta*(1.0 - ((1.0 - 2.0*k)*np.tanh(beta))**2.0)) for k in xPts[i]])
            xixx[i] = np.array([-4.0*(np.tanh(beta)**3.0)*(1.0 - 2.0*k)/(beta*(1.0 - (np.tanh(beta)*(1.0 - 2.0*k))**2.0)**2.0) for k in xPts[i]])
            xix2[i] = np.array([k*k for k in xi_x[i]])
            xPts[i] -= 0.5

            vPts = np.linspace(0.0, 1.0, n[1]+1)
            et = (vPts[1:] + vPts[:-1])/2.0
            yPts[i] = np.array([(1.0 - np.tanh(beta*(1.0 - 2.0*i))/np.tanh(beta))/2.0 for i in et])
            et_y[i] = np.array([np.tanh(beta)/(beta*(1.0 - ((1.0 - 2.0*k)*np.tanh(beta))**2.0)) for k in yPts[i]])
            etyy[i] = np.array([-4.0*(np.tanh(beta)**3.0)*(1.0 - 2.0*k)/(beta*(1.0 - (np.tanh(beta)*(1.0 - 2.0*k))**2.0)**2.0) for k in yPts[i]])
            ety2[i] = np.array([k*k for k in et_y[i]])
            yPts[i] -= 0.5

            vPts = np.linspace(0.0, 1.0, n[2]+1)
            zt = (vPts[1:] + vPts[:-1])/2.0
            zPts[i] = np.array([(1.0 - np.tanh(beta*(1.0 - 2.0*i))/np.tanh(beta))/2.0 for i in zt])
            zt_z[i] = np.array([np.tanh(beta)/(beta*(1.0 - ((1.0 - 2.0*k)*np.tanh(beta))**2.0)) for k in zPts[i]])
            ztzz[i] = np.array([-4.0*(np.tanh(beta)**3.0)*(1.0 - 2.0*k)/(beta*(1.0 - (np.tanh(beta)*(1.0 - 2.0*k))**2.0)**2.0) for k in zPts[i]])
            ztz2[i] = np.array([k*k for k in zt_z[i]])
            zPts[i] -= 0.5

    # Reshape arrays to make it easier to multiply with 3D arrays
    xi_x = [x[int(rank*len(x)/nprocs):int((rank + 1)*len(x)/nprocs), np.newaxis, np.newaxis] for x in xi_x]
    xixx = [x[int(rank*len(x)/nprocs):int((rank + 1)*len(x)/nprocs), np.newaxis, np.newaxis] for x in xixx]
    xix2 = [x[int(rank*len(x)/nprocs):int((rank + 1)*len(x)/nprocs), np.newaxis, np.newaxis] for x in xix2]

    et_y = [x[:, np.newaxis] for x in et_y]
    etyy = [x[:, np.newaxis] for x in etyy]
    ety2 = [x[:, np.newaxis] for x in ety2]


############### Main Solver ###############


def main():
    global nuFlag
    global lesFlag
    global dt, cflNo
    global U, V, W, P, T
    global i2hx, i2hy, i2hz
    global xi_x, et_y, zt_z
    global Hx, Hy, Hz, Ht, Pp

    iCnt = 0
    time = 0
    fwTime = 0.0
    dtnew = 1.0e10

    initGrid()
    time = initFields()
    initMGArrays()

    rhs = np.zeros([xSize, Ny+2, Nz+2])
    gradP = np.zeros([xSize, Ny+2, Nz+2])

    t1 = datetime.now()

    # Write output at t = 0
    writeSoln(U, V, W, P, T, 0.0)
    fwTime = fwTime + fwInt

    locU = np.sum(np.sqrt(U[x0, y0, z0]**2.0 + V[x0, y0, z0]**2.0 + W[x0, y0, z0]**2.0))
    locWT = np.sum(W[x0, y0, z0]*T[x0, y0, z0])

    globU = comm.reduce(locU, op=MPI.SUM, root=0)
    totalWT = comm.reduce(locWT, op=MPI.SUM, root=0)

    maxDiv = getDiv(U, V, W)

    if rootRank:
        print('\n# time \t\t Re \t\t Nu \t\t Divergence')
        Re = globU/(nu*Nx*Ny*Nz)
        Nu = 1.0 + totalWT/(kappa*Nx*Ny*Nz)
        print("%f \t %f \t %f \t %.2e" %(time, Re, Nu, maxDiv))           

    while True:
        if iCnt > 1:
            dtnew = cflNo/comm.allreduce(np.amax((abs(U)/hx) + (abs(V)/hy) + (abs(W)/hz)), op=MPI.MAX)
            
        dt = min(dtnew, dt)

        computeNLinDiff_X()
        computeNLinDiff_Y()
        computeNLinDiff_Z()
        computeNLinDiff_T()  

        if lesFlag:
            computeSGS()

        if nuFlag:
            gradP[x0, y0, z0] = (P[xp1, y0, z0] - P[xm1, y0, z0]) * xi_x[0] * i2hx[0]
        else:
            gradP[x0, y0, z0] = (P[xp1, y0, z0] - P[xm1, y0, z0]) * i2hx[0]

        Hx = U + dt*(Hx - np.sqrt((Ta*Pr)/Ra)*(-V) - gradP)
        uJacobi(Hx)

        if nuFlag:
            gradP[x0, y0, z0] = (P[x0, yp1, z0] - P[x0, ym1, z0]) * et_y[0] * i2hy[0]
        else:
            gradP[x0, y0, z0] = (P[x0, yp1, z0] - P[x0, ym1, z0]) * i2hy[0]

        Hy = V + dt*(Hy - np.sqrt((Ta*Pr)/Ra)*(U) - gradP)
        vJacobi(Hy)

        if nuFlag:
            gradP[x0, y0, z0] = (P[x0, y0, zp1] - P[x0, y0, zm1]) * zt_z[0] * i2hz[0]
        else:
            gradP[x0, y0, z0] = (P[x0, y0, zp1] - P[x0, y0, zm1]) * i2hz[0]

        Hz = W + dt*(Hz + T - gradP)
        wJacobi(Hz)

        Ht = T + dt*Ht
        TJacobi(Ht)   

        rhs.fill(0.0)
        if nuFlag:
            rhs[x0, y0, z0] = ((U[xp1, y0, z0] - U[xm1, y0, z0]) * xi_x[0] * i2hx[0] +
                               (V[x0, yp1, z0] - V[x0, ym1, z0]) * et_y[0] * i2hy[0] +
                               (W[x0, y0, zp1] - W[x0, y0, zm1]) * zt_z[0] * i2hz[0])/dt
        else:
            rhs[x0, y0, z0] = ((U[xp1, y0, z0] - U[xm1, y0, z0]) * i2hx[0] +
                               (V[x0, yp1, z0] - V[x0, ym1, z0]) * i2hy[0] +
                               (W[x0, y0, zp1] - W[x0, y0, zm1]) * i2hz[0])/dt

        Pp = multigrid(rhs)
        #Pp = PpJacobi(rhs)

        P[x0, y0, z0] = P[x0, y0, z0] + Pp[x0, y0, z0]

        if nuFlag:
            U[x0, y0, z0] = U[x0, y0, z0] - dt * (Pp[xp1, y0, z0] - Pp[xm1, y0, z0]) * xi_x[0] * i2hx[0]
            V[x0, y0, z0] = V[x0, y0, z0] - dt * (Pp[x0, yp1, z0] - Pp[x0, ym1, z0]) * et_y[0] * i2hy[0]
            W[x0, y0, z0] = W[x0, y0, z0] - dt * (Pp[x0, y0, zp1] - Pp[x0, y0, zm1]) * zt_z[0] * i2hz[0]
        else:
            U[x0, y0, z0] = U[x0, y0, z0] - dt * (Pp[xp1, y0, z0] - Pp[xm1, y0, z0]) * i2hx[0]
            V[x0, y0, z0] = V[x0, y0, z0] - dt * (Pp[x0, yp1, z0] - Pp[x0, ym1, z0]) * i2hy[0]
            W[x0, y0, z0] = W[x0, y0, z0] - dt * (Pp[x0, y0, zp1] - Pp[x0, y0, zm1]) * i2hz[0]

        imposeUBCs(U)
        imposeVBCs(V)
        imposeWBCs(W)
        imposePBCs(P)
        imposeTBCs(T)

        iCnt = iCnt + 1
        time = time + dt

        if iCnt % opInt == 0:
            uSqr = U[x0, y0, z0]**2.0 + V[x0, y0, z0]**2.0 + W[x0, y0, z0]**2.0
            uSqr = comm.gather(uSqr, root=0)
            #locU = (integrate.simps(integrate.simps(integrate.simps(uSqr, zCord[z0]), yCord[y0]), xCord[x0]))/(Lx*Ly*Lz)
            #globU = comm.reduce(locU, op=MPI.SUM, root=0)

            wT = W[x0, y0, z0]*T[x0, y0, z0]
            wT = comm.gather(wT, root=0)
            #locWT = (integrate.simps(integrate.simps(integrate.simps(wT, zCord[z0]), yCord[y0]), xCord[x0]))/(Lx*Ly*Lz)
            #totalWT = comm.reduce(locWT, op=MPI.SUM, root=0)

            maxDiv = getDiv(U, V, W)

            if rootRank:
                uSqr = np.concatenate(uSqr)
                globU = (integrate.simps(integrate.simps(integrate.simps(uSqr, zCord[1:-1]), yCord[1:-1]), xCord[1:-1]))/(Lx*Ly*Lz)
                Re = np.sqrt(globU)/nu

                wT = np.concatenate(wT)
                totalWT = (integrate.simps(integrate.simps(integrate.simps(wT, zCord[1:-1]), yCord[1:-1]), xCord[1:-1]))/(Lx*Ly*Lz)
                Nu = 1.0 + totalWT/kappa
                print("%f \t %f \t %f \t %.2e" %(time, Re, Nu, maxDiv))     

        if abs(fwTime - time) < 0.5*dt:
            writeSoln(U, V, W, P, T, time)
            fwTime = fwTime + fwInt

        if time + dt/2.0 > tMax:
            break   

    t2 = datetime.now()

    if rootRank:
        print("Time taken for simulation =", t2-t1)

main()
