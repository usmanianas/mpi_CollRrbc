
import numpy as np
import pylab as plt
from matplotlib import cm
import h5py as hp
from mpi4py import MPI
import time
import random 


#### Grid Parameters ###########################
Lx, Ly, Lz = 1.0, 1.0, 1.0

Nx, Ny, Nz = 64, 64, 64

hx, hy, hz = Lx/(Nx-1), Ly/(Ny-1), Lz/(Nz-1)

x = np.linspace(0, 1, Nx, endpoint=True)        
y = np.linspace(0, 1, Ny, endpoint=True)
z = np.linspace(0, 1, Nz, endpoint=True)    

hx2, hy2, hz2 = hx*hx, hy*hy, hz*hz

hx2hy2, hy2hz2, hz2hx2 = hx2*hy2, hy2*hz2, hz2*hx2

hx2hy2hz2 = hx2*hy2*hz2

idx2, idy2, idz2 = 1.0/hx2, 1.0/hy2, 1.0/hz2
#############################################################



comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

bn = int((Nx/nprocs)*rank)
en = int(bn + (Nx/nprocs) - 1)
if rank == 0:
    bn = 1

print(rank, bn, en)



#### Flow Parameters #############
Ra = 1.0e5

Pr = 0.786

Ta = 0.0e7

if rank == 0:
    print("#", "Ra=", Ra, "Pr=", Pr, "Ta=", Ta)

#Ro = np.sqrt(Ra/(Ta*Pr))

nu, kappa = np.sqrt(Pr/Ra), 1.0/np.sqrt(Ra*Pr)

#print(nu, kappa)

#########################################################




#########Simulation Parameters #########################
dt = 0.01

tMax = 1000

# Number of iterations after which output must be printed to standard I/O
opInt = 1

# File writing interval
fwInt = 2

# Tolerance value in Jacobi iterations
VpTolerance = 1.0e-5

# Tolerance value in Poisson iterations
PoissonTolerance = 1.0e-3

gssor = 1.6

maxCount = 1e4
#################################################



P = np.ones([Nx, Ny, Nz])

T = np.zeros([Nx, Ny, Nz])

T[:, :, 0:Nz] = 1 - z[0:Nz]

U = np.zeros([Nx, Ny, Nz])

V = np.zeros([Nx, Ny, Nz])

W = np.zeros([Nx, Ny, Nz])


Hx = np.zeros_like(U)
Hy = np.zeros_like(V)
Hz = np.zeros_like(W)
Ht = np.zeros_like(T)   
Pp = np.zeros_like(P)

Hx.fill(0.0)
Hy.fill(0.0)
Hz.fill(0.0)
Ht.fill(0.0)

time = 0
fwTime = 0.0
iCnt = 1


def writeSoln(U, V, W, P, T, time):

    fName = "Soln_" + "{0:09.5f}.h5".format(time)
    print("#Writing solution file: ", fName)        
    f = hp.File(fName, "w")

    dset = f.create_dataset("U", data = U)
    dset = f.create_dataset("V", data = V)
    dset = f.create_dataset("W", data = W)
    dset = f.create_dataset("T", data = T)
    dset = f.create_dataset("P", data = P)

    f.close()

#writeSoln(U, V, W, P, T, time)



def getDiv(U, V, W):

    divMat = np.zeros([Nx, Ny, Nz])

    divMat[1:Nx-1, 1:Ny-1, bn:en] = ((U[2:Nx, 1:Ny-1, bn:en] - U[0:Nx-2, 1:Ny-1, bn:en])*0.5/hx +
                                (V[1:Nx-1, 2:Ny, bn:en] - V[1:Nx-1, 0:Ny-2, bn:en])*0.5/hy +
                                (W[1:Nx-1, 1:Ny-1, bn+1:en+1] - W[1:Nx-1, 1:Ny-1, bn-1:en-1])*0.5/hz)
    
    #return np.unravel_index(divNyat.argmax(), divMat.shape), np.mean(divMat)
    return np.max(divMat)







def computeNLinDiff_X(U, V, W):
    global Hx
    global Nz, Ny, Nx, Nx, Ny, Nz

    Hx[1:Nx-1, 1:Ny-1, bn:en] = (((U[2:Nx, 1:Ny-1, bn:en] - 2.0*U[1:Nx-1, 1:Ny-1, bn:en] + U[0:Nx-2, 1:Ny-1, bn:en])/hx2 + 
                                (U[1:Nx-1, 2:Ny, bn:en] - 2.0*U[1:Nx-1, 1:Ny-1, bn:en] + U[1:Nx-1, 0:Ny-2, bn:en])/hy2 + 
                                (U[1:Nx-1, 1:Ny-1, bn+1:en+1] - 2.0*U[1:Nx-1, 1:Ny-1, bn:en] + U[1:Nx-1, 1:Ny-1, bn-1:en-1])/hz2)*0.5*nu -
                              U[1:Nx-1, 1:Ny-1, bn:en]*(U[2:Nx, 1:Ny-1, bn:en] - U[0:Nx-2, 1:Ny-1, bn:en])/(2.0*hx) -
                              V[1:Nx-1, 1:Ny-1, bn:en]*(U[1:Nx-1, 2:Ny, bn:en] - U[1:Nx-1, 0:Ny-2, bn:en])/(2.0*hy) - 
                              W[1:Nx-1, 1:Ny-1, bn:en]*(U[1:Nx-1, 1:Ny-1, bn+1:en+1] - U[1:Nx-1, 1:Ny-1, bn-1:en-1])/(2.0*hz))

    return Hx

def computeNLinDiff_Y(U, V, W):
    global Hy
    global Nz, Ny, Nx, Nx, Ny, Nz

    Hy[1:Nx-1, 1:Ny-1, bn:en] = (((V[2:Nx, 1:Ny-1, bn:en] - 2.0*V[1:Nx-1, 1:Ny-1, bn:en] + V[0:Nx-2, 1:Ny-1, bn:en])/hx2 + 
                                (V[1:Nx-1, 2:Ny, bn:en] - 2.0*V[1:Nx-1, 1:Ny-1, bn:en] + V[1:Nx-1, 0:Ny-2, bn:en])/hy2 + 
                                (V[1:Nx-1, 1:Ny-1, bn+1:en+1] - 2.0*V[1:Nx-1, 1:Ny-1, bn:en] + V[1:Nx-1, 1:Ny-1, bn-1:en-1])/hz2)*0.5*nu -
                              U[1:Nx-1, 1:Ny-1, bn:en]*(V[2:Nx, 1:Ny-1, bn:en] - V[0:Nx-2, 1:Ny-1, bn:en])/(2.0*hx) -
                              V[1:Nx-1, 1:Ny-1, bn:en]*(V[1:Nx-1, 2:Ny, bn:en] - V[1:Nx-1, 0:Ny-2, bn:en])/(2.0*hy) - 
                              W[1:Nx-1, 1:Ny-1, bn:en]*(V[1:Nx-1, 1:Ny-1, bn+1:en+1] - V[1:Nx-1, 1:Ny-1, bn-1:en-1])/(2.0*hz))

    return Hy


def computeNLinDiff_Z(U, V, W):
    global Hz
    global Nz, Ny, Nx, Nx, Ny, Nz

    Hz[1:Nx-1, 1:Ny-1, bn:en] = (((W[2:Nx, 1:Ny-1, bn:en] - 2.0*W[1:Nx-1, 1:Ny-1, bn:en] + W[0:Nx-2, 1:Ny-1, bn:en])/hx2 + 
                                (W[1:Nx-1, 2:Ny, bn:en] - 2.0*W[1:Nx-1, 1:Ny-1, bn:en] + W[1:Nx-1, 0:Ny-2, bn:en])/hy2 + 
                                (W[1:Nx-1, 1:Ny-1, bn+1:en+1] - 2.0*W[1:Nx-1, 1:Ny-1, bn:en] + W[1:Nx-1, 1:Ny-1, bn-1:en-1])/hz2)*0.5*nu -
                              U[1:Nx-1, 1:Ny-1, bn:en]*(W[2:Nx, 1:Ny-1, bn:en] - W[0:Nx-2, 1:Ny-1, bn:en])/(2.0*hx) -
                              V[1:Nx-1, 1:Ny-1, bn:en]*(W[1:Nx-1, 2:Ny, bn:en] - W[1:Nx-1, 0:Ny-2, bn:en])/(2.0*hy) - 
                              W[1:Nx-1, 1:Ny-1, bn:en]*(W[1:Nx-1, 1:Ny-1, bn+1:en+1] - W[1:Nx-1, 1:Ny-1, bn-1:en-1])/(2.0*hz))


    return Hz


def computeNLinDiff_T(U, V, W, T):
    global Ht
    global Nz, Ny, Nx

    Ht[1:Nx-1, 1:Ny-1, bn:en] = (((T[2:Nx, 1:Ny-1, bn:en] - 2.0*T[1:Nx-1, 1:Ny-1, bn:en] + T[0:Nx-2, 1:Ny-1, bn:en])/hx2 + 
                                (T[1:Nx-1, 2:Ny, bn:en] - 2.0*T[1:Nx-1, 1:Ny-1, bn:en] + T[1:Nx-1, 0:Ny-2, bn:en])/hy2 + 
                                (T[1:Nx-1, 1:Ny-1, bn+1:en+1] - 2.0*T[1:Nx-1, 1:Ny-1, bn:en] + T[1:Nx-1, 1:Ny-1, bn-1:en-1])/hz2)*0.5*kappa -
                              U[1:Nx-1, 1:Ny-1, bn:en]*(T[2:Nx, 1:Ny-1, bn:en] - T[0:Nx-2, 1:Ny-1, bn:en])/(2.0*hx)-
                              V[1:Nx-1, 1:Ny-1, bn:en]*(T[1:Nx-1, 2:Ny, bn:en] - T[1:Nx-1, 0:Ny-2, bn:en])/(2.0*hy) - 
                              W[1:Nx-1, 1:Ny-1, bn:en]*(T[1:Nx-1, 1:Ny-1, bn+1:en+1] - T[1:Nx-1, 1:Ny-1, bn-1:en-1])/(2.0*hz))

    return Ht


#Jacobi iterative solver for U
def uJacobi(rho):
    global hx2, hy2, hz2, hy2hz2, hz2hx2, hx2hy2, hx2hy2hz2, nu, dt, VpTolerance, maxCount
    global U
    global Nx, Ny, Nz, Nx, Ny, Nz

    temp_sol = np.zeros_like(rho)

    jCnt = 0
    while True:
        temp_sol[1:Nx-1, 1:Ny-1, bn:en] = ((hy2hz2*(U[0:Nx-2, 1:Ny-1, bn:en] + U[2:Nx, 1:Ny-1, bn:en]) +
                                        hz2hx2*(U[1:Nx-1, 0:Ny-2, bn:en] + U[1:Nx-1, 2:Ny, bn:en]) +
                                        hx2hy2*(U[1:Nx-1, 1:Ny-1, bn-1:en-1] + U[1:Nx-1, 1:Ny-1, bn+1:en+1]))*
                                nu*dt/(hx2hy2hz2*2.0) + rho[1:Nx-1, 1:Ny-1, bn:en])/ \
                            (1.0 + nu*dt*(hy2hz2 + hz2hx2 + hx2hy2)/(hx2hy2hz2))
        
        U = temp_sol

        imposeUBCs(U)
        
        maxErr = np.amax(np.fabs(rho[1:Nx-1, 1:Ny-1, bn:en] - (U[1:Nx-1, 1:Ny-1, bn:en] - 0.5*nu*dt*(
                            (U[0:Nx-2, 1:Ny-1, bn:en] - 2.0*U[1:Nx-1, 1:Ny-1, bn:en] + U[2:Nx, 1:Ny-1, bn:en])/hx2 +
                            (U[1:Nx-1, 0:Ny-2, bn:en] - 2.0*U[1:Nx-1, 1:Ny-1, bn:en] + U[1:Nx-1, 2:Ny, bn:en])/hy2 +
                            (U[1:Nx-1, 1:Ny-1, bn-1:en-1] - 2.0*U[1:Nx-1, 1:Ny-1, bn:en] + U[1:Nx-1, 1:Ny-1, bn+1:en+1])/hz2))))
        
            #if maxErr < tolerance:
        if maxErr < VpTolerance:
            #print(jCnt)
            break
        
        jCnt += 1
        if jCnt > maxCount:
                print("ERROR: Jacobi not converging in U. Aborting")
                print("Maximum error: ", maxErr)
                quit()

    return U        


#Jacobi iterative solver for V
def vJacobi(rho):
    global hx2, hy2, hz2, hy2hz2, hz2hx2, hx2hy2, hx2hy2hz2, nu, dt, VpTolerance, maxCount  
    global V
    global Nx, Ny, Nz, Nx, Ny, Nz
    
    temp_sol = np.zeros_like(rho)
    
    jCnt = 0
    while True:
        temp_sol[1:Nx-1, 1:Ny-1, bn:en] = ((hy2hz2*(V[0:Nx-2, 1:Ny-1, bn:en] + V[2:Nx, 1:Ny-1, bn:en]) +
                                    hz2hx2*(V[1:Nx-1, 0:Ny-2, bn:en] + V[1:Nx-1, 2:Ny, bn:en]) +
                                    hx2hy2*(V[1:Nx-1, 1:Ny-1, bn-1:en-1] + V[1:Nx-1, 1:Ny-1, bn+1:en+1]))*
                                nu*dt/(hx2hy2hz2*2.0) + rho[1:Nx-1, 1:Ny-1, bn:en])/ \
                        (1.0 + nu*dt*(hy2hz2 + hz2hx2 + hx2hy2)/(hx2hy2hz2))
    
        V = temp_sol

        imposeVBCs(V)


        maxErr = np.amax(np.fabs(rho[1:Nx-1, 1:Ny-1, bn:en] - (V[1:Nx-1, 1:Ny-1, bn:en] - 0.5*nu*dt*(
                        (V[0:Nx-2, 1:Ny-1, bn:en] - 2.0*V[1:Nx-1, 1:Ny-1, bn:en] + V[2:Nx, 1:Ny-1, bn:en])/hx2 +
                        (V[1:Nx-1, 0:Ny-2, bn:en] - 2.0*V[1:Nx-1, 1:Ny-1, bn:en] + V[1:Nx-1, 2:Ny, bn:en])/hy2 +
                        (V[1:Nx-1, 1:Ny-1, bn-1:en-1] - 2.0*V[1:Nx-1, 1:Ny-1, bn:en] + V[1:Nx-1, 1:Ny-1, bn+1:en+1])/hz2))))
    
        #if maxErr < tolerance:
        if maxErr < VpTolerance:
            #print(jCnt)
            break
    
        jCnt += 1
        if jCnt > maxCount:
            print("ERROR: Jacobi not converging in V. Aborting")
            print("Maximum error: ", maxErr)
            quit()
    
    return V


#Jacobi iterative solver for W
def wJacobi(rho):
    global hx2, hy2, hz2, hy2hz2, hz2hx2, hx2hy2, hx2hy2hz2, nu, dt, VpTolerance, maxCount  
    global W
    global Nx, Ny, Nz, Nx, Ny, Nz
    
    temp_sol = np.zeros_like(rho)
    
    jCnt = 0
    while True:
        temp_sol[1:Nx-1, 1:Ny-1, bn:en] = ((hy2hz2*(W[0:Nx-2, 1:Ny-1, bn:en] + W[2:Nx, 1:Ny-1, bn:en]) +
                                    hz2hx2*(W[1:Nx-1, 0:Ny-2, bn:en] + W[1:Nx-1, 2:Ny, bn:en]) +
                                    hx2hy2*(W[1:Nx-1, 1:Ny-1, bn-1:en-1] + W[1:Nx-1, 1:Ny-1, bn+1:en+1]))*
                                nu*dt/(hx2hy2hz2*2.0) + rho[1:Nx-1, 1:Ny-1, bn:en])/ \
                        (1.0 + nu*dt*(hy2hz2 + hz2hx2 + hx2hy2)/(hx2hy2hz2))
        
        W = temp_sol
    
        imposeWBCs(W)


        maxErr = np.amax(np.fabs(rho[1:Nx-1, 1:Ny-1, bn:en] - (W[1:Nx-1, 1:Ny-1, bn:en] - 0.5*nu*dt*(
                        (W[0:Nx-2, 1:Ny-1, bn:en] - 2.0*W[1:Nx-1, 1:Ny-1, bn:en] + W[2:Nx, 1:Ny-1, bn:en])/hx2 +
                        (W[1:Nx-1, 0:Ny-2, bn:en] - 2.0*W[1:Nx-1, 1:Ny-1, bn:en] + W[1:Nx-1, 2:Ny, bn:en])/hy2 +
                        (W[1:Nx-1, 1:Ny-1, bn-1:en-1] - 2.0*W[1:Nx-1, 1:Ny-1, bn:en] + W[1:Nx-1, 1:Ny-1, bn+1:en+1])/hz2))))
    
        #if maxErr < tolerance:
        if maxErr < 1e-5:
            #print(jCnt)
            break
    
        jCnt += 1
        if jCnt > maxCount:
            print("ERROR: Jacobi not converging in W. Aborting")
            print("Maximum error: ", maxErr)
            quit()
    
    return W       


#Jacobi iterative solver for T
def TJacobi(rho):
    global hx2, hy2, hz2, hy2hz2, hz2hx2, hx2hy2, hx2hy2hz2, nu, dt, VpTolerance, maxCount  
    global T
    global Nx, Ny, Nz, Nx, Ny, Nz
    
    temp_sol = np.zeros_like(rho)
    
    jCnt = 0
    while True:
        temp_sol[1:Nx-1, 1:Ny-1, bn:en] = ((hy2hz2*(T[0:Nx-2, 1:Ny-1, bn:en] + T[2:Nx, 1:Ny-1, bn:en]) +
                                    hz2hx2*(T[1:Nx-1, 0:Ny-2, bn:en] + T[1:Nx-1, 2:Ny, bn:en]) +
                                    hx2hy2*(T[1:Nx-1, 1:Ny-1, bn-1:en-1] + T[1:Nx-1, 1:Ny-1, bn+1:en+1]))*
                                kappa*dt/(hx2hy2hz2*2.0) + rho[1:Nx-1, 1:Ny-1, bn:en])/ \
                        (1.0 + kappa*dt*(hy2hz2 + hz2hx2 + hx2hy2)/(hx2hy2hz2))
    
        T = temp_sol

        imposeTBCs(T)

        maxErr = np.amax(np.fabs(rho[1:Nx-1, 1:Ny-1, bn:en] - (T[1:Nx-1, 1:Ny-1, bn:en] - 0.5*kappa*dt*(
                        (T[0:Nx-2, 1:Ny-1, bn:en] - 2.0*T[1:Nx-1, 1:Ny-1, bn:en] + T[2:Nx, 1:Ny-1, bn:en])/hx2 +
                        (T[1:Nx-1, 0:Ny-2, bn:en] - 2.0*T[1:Nx-1, 1:Ny-1, bn:en] + T[1:Nx-1, 2:Ny, bn:en])/hy2 +
                        (T[1:Nx-1, 1:Ny-1, bn-1:en-1] - 2.0*T[1:Nx-1, 1:Ny-1, bn:en] + T[1:Nx-1, 1:Ny-1, bn+1:en+1])/hz2))))
    
        #if maxErr < tolerance:
        if maxErr < VpTolerance:
            #print(jCnt)
            break
    
        jCnt += 1
        if jCnt > maxCount:
            print("ERROR: Jacobi not converging in T. Aborting")
            print("Maximum error: ", maxErr)
            quit()
    
    return T       



def PoissonSolver(rho):
    global hx2, hy2, hz2, hy2hz2, hz2hx2, hx2hy2, hx2hy2hz2, nu, dt, PoissonTolerance, maxCount 
    global Nz, Ny, Nx    
    
    
    Pp = np.zeros([Nx, Ny, Nz])
    #Pp = np.random.rand(Nx, Ny, Nz)
    #Ppp = np.zeros([Nx, Ny, Nz])
        
    jCnt = 0
    
    while True:

    
        '''
        for i in range(1,Nx-1):
            for j in range(1,Ny-1):
                for k in range(1,Nz-1):
                    Pp[i,j,k] = (1.0/(-2.0*(idx2 + idy2 + idz2))) * (rho[i, j, k] - 
                                       idx2*(Pp[i+1, j, k] + Pp[i-1, j, k]) -
                                       idy2*(Pp[i, j+1, k] + Pp[i, j-1, k]) -
                                       idz2*(Pp[i, j, k+1] + Pp[i, j, k-1]))

        Pp[1:Nx-1, 1:Ny-1, bn:en] = (1.0-gssor)*Ppp[1:Nx-1, 1:Ny-1, bn:en] + gssor * Pp[1:Nx-1, 1:Ny-1, bn:en]            
        '''
    
           
        
        Pp[1:Nx-1, 1:Ny-1, bn:en] = (1.0/(-2.0*(idx2 + idy2 + idz2))) * (rho[1:Nx-1, 1:Ny-1, bn:en] - 
                                       idx2*(Pp[0:Nx-2, 1:Ny-1, bn:en] + Pp[2:Nx, 1:Ny-1, bn:en]) -
                                       idy2*(Pp[1:Nx-1, 0:Ny-2, bn:en] + Pp[1:Nx-1, 2:Ny, bn:en]) -
                                       idz2*(Pp[1:Nx-1, 1:Ny-1, bn-1:en-1] + Pp[1:Nx-1, 1:Ny-1, bn+1:en+1]))   


        #Pp[1:Nx-1, 1:Ny-1, bn:en] = (1.0-gssor)*Ppp[1:Nx-1, 1:Ny-1, bn:en] + gssor*Pp[1:Nx-1, 1:Ny-1, bn:en]                                                                   
           
        #Ppp = Pp.copy()

        #imposePBCs(Pp)
    
        maxErr = np.amax(np.fabs(rho[1:Nx-1, 1:Ny-1, bn:en] -((
                        (Pp[0:Nx-2, 1:Ny-1, bn:en] - 2.0*Pp[1:Nx-1, 1:Ny-1, bn:en] + Pp[2:Nx, 1:Ny-1, bn:en])/hx2 +
                        (Pp[1:Nx-1, 0:Ny-2, bn:en] - 2.0*Pp[1:Nx-1, 1:Ny-1, bn:en] + Pp[1:Nx-1, 2:Ny, bn:en])/hy2 +
                        (Pp[1:Nx-1, 1:Ny-1, bn-1:en-1] - 2.0*Pp[1:Nx-1, 1:Ny-1, bn:en] + Pp[1:Nx-1, 1:Ny-1, bn+1:en+1])/hz2))))
    
    
        #if (jCnt % 100 == 0):
        #    print(maxErr)
    
        if maxErr < PoissonTolerance:
            #print(jCnt)
            #print("Poisson solver converged")
            break
    
        jCnt += 1
        if jCnt > maxCount:
            print("ERROR: Poisson solver not converging. Aborting")
            quit()
    
    return Pp     



def imposeUBCs(U):
    U[0, :, :], U[-1, :, :] = 0.0, 0.0
    U[:, 0, :], U[:, -1, :] = 0.0, 0.0
    U[:, :, 0], U[:, :, -1] = 0.0, 0.0

def imposeVBCs(V):
    V[0, :, :], V[-1, :, :] = 0.0, 0.0  
    V[:, 0, :], V[:, -1, :] = 0.0, 0.0  
    V[:, :, 0], V[:, :, -1] = 0.0, 0.0

def imposeWBCs(W):
    W[0, :, :], W[-1, :, :] = 0.0, 0.0, 
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



while True:

    if iCnt % opInt == 0:
        locE, locWT, totalE, totalWT = np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)  
        #Re = np.mean(np.sqrt(U[1:Nx-1, 1:Ny-1, bn:en]**2.0 + V[1:Nx-1, 1:Ny-1, bn:en]**2.0 + W[1:Nx-1, 1:Ny-1, bn:en]**2.0))/nu
        locE[0] = np.sum(0.5*U[1:Nx-1, 1:Ny-1, bn:en]**2.0 + V[1:Nx-1, 1:Ny-1, bn:en]**2.0 + W[1:Nx-1, 1:Ny-1, bn:en]**2.0)
        #print(locE)
        comm.Reduce(locE, totalE, op=MPI.SUM, root=0)
        
        locWT[0] = np.sum(W[1:Nx-1, 1:Ny-1, bn:en]*T[1:Nx-1, 1:Ny-1, bn:en])
        comm.Reduce(locWT, totalWT, op=MPI.SUM, root=0)

        #Nu = 1.0 + np.mean(W[1:Nx-1, 1:Ny-1, bn:en]*T[1:Nx-1, 1:Ny-1, bn:en])/kappa

        maxDiv = getDiv(U, V, W)

        if rank == 0:
            Re = np.sqrt(2)*totalE/(Nx*Ny*Nz)/nu
            Nu = 1.0 + (totalWT/(Nx*Ny*Nz))/kappa
            print("%f    %f    %f    %f" %(time, Re, Nu, maxDiv))           



        



    Hx = computeNLinDiff_X(U, V, W)
    Hy = computeNLinDiff_Y(U, V, W)
    Hz = computeNLinDiff_Z(U, V, W)
    Ht = computeNLinDiff_T(U, V, W, T)  

    Hx[1:Nx-1, 1:Ny-1, bn:en] = U[1:Nx-1, 1:Ny-1, bn:en] + dt*(Hx[1:Nx-1, 1:Ny-1, bn:en] - np.sqrt((Ta*Pr)/Ra)*(-V[1:Nx-1, 1:Ny-1, bn:en]) - (P[2:Nx, 1:Ny-1, bn:en] - P[0:Nx-2, 1:Ny-1, bn:en])/(2.0*hx))
    uJacobi(Hx)

    Hy[1:Nx-1, 1:Ny-1, bn:en] = V[1:Nx-1, 1:Ny-1, bn:en] + dt*(Hy[1:Nx-1, 1:Ny-1, bn:en] - np.sqrt((Ta*Pr)/Ra)*(U[1:Nx-1, 1:Ny-1, bn:en]) - (P[1:Nx-1, 2:Ny, bn:en] - P[1:Nx-1, 0:Ny-2, bn:en])/(2.0*hy))
    vJacobi(Hy)

    Hz[1:Nx-1, 1:Ny-1, bn:en] = W[1:Nx-1, 1:Ny-1, bn:en] + dt*(Hz[1:Nx-1, 1:Ny-1, bn:en] - ((P[1:Nx-1, 1:Ny-1, bn+1:en+1] - P[1:Nx-1, 1:Ny-1, bn-1:en-1])/(2.0*hz)) + T[1:Nx-1, 1:Ny-1, bn:en])
    wJacobi(Hz)

    Ht[1:Nx-1, 1:Ny-1, bn:en] = T[1:Nx-1, 1:Ny-1, bn:en] + dt*Ht[1:Nx-1, 1:Ny-1, bn:en]
    TJacobi(Ht)   

    rhs = np.zeros([Nx, Ny, Nz])
    rhs[1:Nx-1, 1:Ny-1, bn:en] = ((U[2:Nx, 1:Ny-1, bn:en] - U[0:Nx-2, 1:Ny-1, bn:en])/(2.0*hx) +
                                (V[1:Nx-1, 2:Ny, bn:en] - V[1:Nx-1, 0:Ny-2, bn:en])/(2.0*hy) +
                                (W[1:Nx-1, 1:Ny-1, bn+1:en+1] - W[1:Nx-1, 1:Ny-1, bn-1:en-1])/(2.0*hz))/dt

    Pp = PoissonSolver(rhs)

    P = P + Pp

    U[1:Nx-1, 1:Ny-1, bn:en] = U[1:Nx-1, 1:Ny-1, bn:en] - dt*(Pp[2:Nx, 1:Ny-1, bn:en] - Pp[0:Nx-2, 1:Ny-1, bn:en])/(2.0*hx)
    V[1:Nx-1, 1:Ny-1, bn:en] = V[1:Nx-1, 1:Ny-1, bn:en] - dt*(Pp[1:Nx-1, 2:Ny, bn:en] - Pp[1:Nx-1, 0:Ny-2, bn:en])/(2.0*hy)
    W[1:Nx-1, 1:Ny-1, bn:en] = W[1:Nx-1, 1:Ny-1, bn:en] - dt*(Pp[1:Nx-1, 1:Ny-1, bn+1:en+1] - Pp[1:Nx-1, 1:Ny-1, bn-1:en-1])/(2.0*hz)

    imposeUBCs(U)                               
    imposeVBCs(V)                               
    imposeWBCs(W)                               
    imposePBCs(P)                               
    imposeTBCs(T)       

    #if abs(fwTime - time) < 0.5*dt:
    if abs(time - tMax)<1e-5:
        writeSoln(U, V, W, P, T, time)
        Z, Y = np.meshgrid(y,z)
        plt.contourf(Y, Z, T[int(Nx/2), :, :], 500, cmap=cm.coolwarm)
        clb = plt.colorbar()
        plt.quiver(Y[0:Nx,0:Ny], Z[0:Nx,0:Ny], V[int(Nx/2),:, :], W[int(Nx/2), :, :])
        plt.axis('scaled')
        clb.ax.set_title(r'$T$', fontsize = 20)
        plt.show()
        fwTime = fwTime + fwInt                                 

    if time > tMax:
        break   

    time = time + dt

    iCnt = iCnt + 1






