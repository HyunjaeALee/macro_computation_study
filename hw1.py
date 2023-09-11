from math import sqrt, exp
import numpy as np
from scipy.stats import norm, lognorm

def rw_recur(Nz,p):
    if Nz==2:
        Π = np.array([[p, 1-p], [1-p, p]])
    else:
        Π = np.zeros((Nz,Nz))
        Pi = rw_recur(Nz-1,p)
        Π[:-1,:-1] = p * Pi
        Π[:-1,1:] += (1-p) * Pi
        Π[1:,:-1] += (1-p) * Pi
        Π[1:,1:] += p * Pi
        Π[1:-1,:] /= 2
    return Π

def stationary(Pi): # Pi is row-stochastic
    vals, vects = np.linalg.eig(Pi.T)
    s_d = vects[:,np.isclose(vals,1)][:,0]
    return s_d / sum(s_d)

def bisection(x,y,func,tol):
    w = (x+y)*.5
    fx, fy = func(x), func(y)
    if fx*fy>=0:
        print("no solution from bisection search")
        return x
    while y-x>tol:
        fw = func(w)
        if fw==0:
            return w
        elif fx*fw>0:
            x, fx, w = w, fw, (w+y)*.5
        else:
            y, fy, w = w, fw, (x+w)*.5
    return w

class tauchen:

    def __init__(self,Nz,ρ,σ,m):
        ψ = σ*m/sqrt(1-ρ**2)
        grid = 2*ψ/(Nz-1)
        self.states = np.array([i*grid-ψ for i in range(Nz)])
        Y = np.tile(self.states,(Nz,1))
        Y = (Y - ρ*Y.T) / σ
        g = grid*.5/σ
        P1, P2 = norm.cdf(Y+g), norm.cdf(Y-g)
        P1[:,-1], P2[:,0] = 1, 0
        self.Π = P1 - P2

class rouwenhorst:

    def __init__(self,Nz,ρ,σ,m): # m is only to match with Tauchen
        self.Π = rw_recur(Nz,(ρ+1)*.5)
        ψ = sqrt((Nz-1)/(1-ρ**2))*σ
        grid = 2*ψ/(Nz-1)
        self.states = np.array([i*grid-ψ for i in range(Nz)])

class linear:
    # linear interpolation

    def __init__(self,inp,out):
        self.func = np.array([(out[1:] - out[:-1]) / (inp[1:] - inp[:-1]),
                             out[:-1],
                             inp[:-1]]).T

    def __call__(self,x):
        if x < self.func[0,-1]:
            return self.func[0,1]
        g, o, i = self.func[self.func[:,-1] <= x,:][-1]
        return o + (x-i)*g

class quadratic:
    # shape-preserving quadratic interpolation

    def __init__(self,inp,out):
        N = len(inp)
        inp0, inp1, out0, out1 = inp[:-1], inp[1:], out[:-1], out[1:]
        idif, odif = inp1 - inp0, out1 - out0
        dbar, norms = odif / idif, np.sqrt(idif**2 + odif**2)
        nd = norms * dbar
        d, z, a0, a1 = np.zeros(N), np.zeros(N-1), np.zeros(N-1), np.zeros(N-1)
        d[1:-1] = (nd[1:] + nd[:-1]) / (norms[1:] + norms[:-1])
        d[0], d[-1] = (3*dbar[0]-d[1])*.5, (3*dbar[-1]-d[-2])*.5
        d0, d1 = d[:-1], d[1:]
        ddb0, ddb1, dmd = d0-dbar, d1-dbar, d1-d0
        case1 = ddb0 * ddb1 >= 0
        case2 = (~case1) & (abs(ddb0) > abs(ddb1))
        case3 = ~(case1 | case2)
        z[case1] = (inp0[case1]+inp1[case1]) * .5
        z[case2] = inp0[case2] + idif[case2]*ddb1[case2]/dmd[case2]
        z[case3] = inp1[case3] + idif[case3]*ddb0[case3]/dmd[case3]
        zmi0, zmi1 = z - inp0, z - inp1
        a0 = (dmd*zmi1*.5 - d0*idif + odif) / (idif * zmi0)
        a1 = ((d1*(zmi1-idif) - d0*zmi0)*.5 + odif) / (idif * zmi1)
        self.func = np.array([a0, a1, d0, d1, z, out0, out1, inp0, inp1]).T

    def __call__(self,x):
        if x < self.func[0,-2]:
            return self.func[0,5]
        a1, a2, d0, d1, z, y0, y1, x0, x1 = self.func[self.func[:,-2] <= x,:][-1]
        if x <= z:
            return a1*(x-x0)**2 + d0*(x-x0) + y0
        else:
            return a2*(x-x1)**2 + d1*(x-x1) + y1

class H92:
    # labor is a numeraire (w=1)
    # log of productivity follows AR(1)

    def __init__(self,
                 π=lambda p,s,n,c: p*(s*n**(2/3)-c)-n,
                 n=lambda p,s: np.maximum((8/27)*(p*s)**3,0),
                 β=0.96, cf=16, sbar=1.2, ρ=.9, σ=.2,
                 ma=rouwenhorst, Ns=300, m=3):
        # π : profit function
        # n : optimal n
        # β : beta
        # cf : fixed cost for each period
        # sbar, ρ, σ : AR(1) parameters
        # ma : method for Markov approximation, tauchen or rouwenhorst
        # Nz : number of grids for state space
        # m : number of standard deviations to approximate out to (only for Tauchen)
        self.π = lambda p,s: π(p,s,n(p,s),cf)
        self.n = n
        self.β = β
        self.cf = cf
        mc = ma(Ns,ρ,σ,m)
        self.exbar = exp(sbar)
        self.S = np.exp(mc.states + sbar)
        self.P = mc.Π
        self.s1 = σ/(1-ρ**2) # standard deviation of log of initial shock
        self.ss = stationary(mc.Π)
        self.Ns = Ns

    def steady_state(self, p=1, tol=.00001, interp=quadratic):
        # interp : interpolation method, linear or quadratic
        pi = self.π(p,self.S)
        v0 = np.zeros(self.Ns)
        d = 1
        iter = 0
        while d > tol:
            v1 = pi + self.β * np.maximum(self.P @ v0, 0)
            d = np.max(abs(v0-v1))
            v0 = v1
            iter += 1
        print("Iteration : ", iter)
        self.v0 = v0
        self.value = interp(self.S, v0)
        self.ce = lognorm.expect(self.value,args=(self.s1,),scale=self.exbar)
        x = self.S[self.P @ v0 >= 0][0]
        self.rate = sum(self.ss[self.S>=x])
