# EX 1
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(7)


# Solves the minimization problem
class SolveMinProbl:

    def __int__(self, y=np.ones((3, 1)), A=np.eye(3)):  # inizialization
        self.matr = A
        self.Np = y.shape[0]
        self.Nf = A.shape[1]
        self.vect = y
        self.sol = np.zeros((self.Nf, 1), dtype=float)
        return

    def plot_w(self, title='Solution'):
        w = self.sol
        n = np.arange(self.Nf)
        plt.figure()
        plt.plot(n, w)
        plt.xlabel('n')
        plt.ylabel('w(n)')
        plt.title('w(n)')
        plt.title(title)
        plt.grid()
        str = title+'.pdf'
        plt.savefig(str, bbox_inches='tight')
        plt.close()
        return

    def print_result(self, title):
        print(title, ' :')
        print('the optimum weight vector is: ')
        print(self.sol)
        return

    def plot_err(self, title='Square error', logy=0, logx=0):

        err = self.err
        plt.figure()
        if(logy == 0) & (logx == 0):
            plt.plot(err[:, 0], err[:, 1])
        if(logy == 1) & (logx == 0):
            plt.semilogy(err[:, 0], err[:, 1])
        if(logy == 0) & (logx == 1):
            plt.semilogx(err[:, 0], err[:, 1])
        if(logy == 1) & (logx == 1):
            plt.loglog(err[:, 0], err[:, 1])
        plt.xlabel('n')
        plt.ylabel('e(n)')
        plt.title(title)
        plt.margins(0.01, 0.1)
        plt.grid()
        str = title+'.pdf'
        plt.savefig(str, bbox_inches='tight')
        plt.close()
        return

# Solves Linear Least square estimation


class SolveLLS(SolveMinProbl):

    def __init__(self, y, A):
        self.matr = A
        self.vect = y
        self.Np = A.shape[0]
        self.Nf = A.shape[1]
        self.sol = np.zeros((self.Nf, 1), dtype=float)
        return

    def run(self):
        A = self.matr
        y = self.vect
        w = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)), A.T), y)
        self.sol = w
        self.min = np.linalg.norm(np.dot(A, w) - y)
        return

# Solves the gradient


class SolveGrad(SolveMinProbl):

    def __init__(self, y, A):
        self.matr = A
        self.vect = y
        self.Np = A.shape[0]
        self.Nf = A.shape[1]
        self.sol = np.zeros((self.Nf, 1), dtype=float)
        return

    def run(self, gamma=1e-3, Nit=100):
        self.err = np.zeros((Nit, 2), dtype=float)
        A = self.matr
        y = self.vect
        w = np.random.rand(self.Nf, 1)
        for it in range(Nit):
            grad = 2 * np.dot(A.T, (np.dot(A, w) - y))
            w = w - gamma * grad
            self.err[it, 0] = it
            self.err[it, 1] = np.linalg.norm(np.dot(A, w) - y)
        self.sol = w
        self.min = self.err[it, 1]


# Solves the Steepest Descent
class SolveSteepestDescent(SolveMinProbl):

    def __init__(self, y, A):
        self.matr = A
        self.vect = y
        self.Np = A.shape[0]
        self.Nf = A.shape[1]
        self.err = []
        self.sol = np.zeros((self.Nf, 1), dtype=float)
        return

    def run(self, Nit=500):
        self.err = np.zeros((Nit, 2), dtype=float)
        A = self.matr
        y = self.vect
        w = np.random.rand(self.Nf, 1)
        H = 4 * (np.dot(A.T, A))

        for it in range(Nit):
            grad = 2 * np.dot(A.T, (np.dot(A, w) - y))
            gamma = ((np.linalg.norm(grad))**2)/(np.dot(np.dot(grad.T, H), grad))
            w = w - gamma * grad
            self.err[it, 0] = it
            self.err[it, 1] = np.linalg.norm(np.dot(A, w) - y)
            print(self.err[it, 1])
        self.sol = w
        self.min = self.err[it, 1]

# Solves the Stochastic gradient


class SolveStochasticGradient(SolveMinProbl):

    def __init__(self, y, A):
        self.matr = A
        self.vect = y
        self.Np = A.shape[0]
        self.Nf = A.shape[1]
        self.err = []
        self.sol = np.zeros((self.Nf, 1), dtype=float)
        return

    def run(self, gamma=1e-3, Nit=500):
        self.err = np.zeros((Nit, 2), dtype=float)
        A = self.matr
        y = self.vect
        w = np.random.rand(self.Nf, 1)
        for it in range(Nit):
            for i in range(Np):
                grad_i = 2*(np.dot(A[i, :], w)-y[i])*A[i, :].reshape(len(A[i, :]), 1)
                w = w - gamma * grad_i
            self.err[it, 0] = it
            self.err[it, 1] = np.linalg.norm(np.dot(A, w) - y)
        self.sol = w
        self.min = self.err[it, 1]


# Solves the Minibatches
class SolveMinibatches(SolveMinProbl):

    def __init__(self, y, A):
        self.matr = A
        self.vect = y
        self.Np = A.shape[0]
        self.Nf = A.shape[1]
        self.err = []
        self.sol = np.zeros((self.Nf, 1), dtype=float)
        return

    def run(self, gamma=1e-3, Nit=500, K=10):
        self.err = np.zeros((Nit, 2), dtype=float)
        A = self.matr
        y = self.vect
        w = np.random.rand(self.Nf, 1)
        K_1 = int(self.Np/K)

        for it in range(Nit):
            k = 0
            for i in range(0, K_1):
                X = A[range(k, k+K_1), :]
                Y = y[range(k, k+K_1), :]
                grad_i = 2*(np.dot(np.dot(X.T, X), w)-np.dot(X.T, Y))
                w = w-gamma*grad_i
                k = k+K_1
            self.err[it, 0] = it
            self.err[it, 1] = np.linalg.norm(np.dot(A, w)-y)
        self.sol = w
        self.min = self.err[it, 1]


#### MAIN ####
if __name__ == "__main__":

    Np = 80
    Nf = 4

    A = np.random.randn(Np, Nf)
    y = np.random.randn(Np, 1)

    # LLS
    m = SolveLLS(y, A)
    m.run()
    m.print_result('LLS')
    m.plot_w('LLS')

    # Gradient
    Nit = 500
    gamma = 1e-2
    logx = 0
    logy = 1
    g = SolveGrad(y, A)
    g.run(gamma, Nit)
    g.print_result('Gradient algo.')
    g.plot_err('Gradient algo: square error', logy, logx)
    g.plot_w('Gradient opt result')

    # SteepestDescent
    Nit = 20
    s = SolveSteepestDescent(y, A)
    s.run(Nit)
    s.print_result('Steepest Descent algo')
    s.plot_err('Steepest Descent', logy, logx)
    s.plot_w('SteepestDescent opt result')

    # Stochastic Gradient
    stg = SolveStochasticGradient(y, A)
    stg.run()
    stg.print_result('Stochastic Gradient algo')
    stg.plot_err('Stochastic Gradient', logy, logx)
    stg.plot_w('Stochastic Gradient opt result')

    # Minibatches
    mini = SolveMinibatches(y, A)
    mini.run()
    mini.print_result('Minibatches Gradient algo')
    mini.plot_err('Minibatches Gradient', logy, logx)
    mini.plot_w('Minibatches opt result')
