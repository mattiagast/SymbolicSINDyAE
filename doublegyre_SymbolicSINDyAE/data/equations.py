import numpy as np
import scipy.integrate
import sympy
import abc

# X0 = sympy.Symbol('X0', positive=True)
# X1 = sympy.Symbol('X1', positive=True)
# X2 = sympy.Symbol('X2', positive=True)
# X3 = sympy.Symbol('X3', positive=True)
# C = sympy.Symbol('C', positive=True)
#
# VarDict = {
#     'X0': X0,
#     'X1': X1,
#     'X2': X2,
#     'X3': X3,
#     'C': C,
# }

def get_ode(ode_name, param):
    if ode_name == 'SineWave':
        ode = SineWave(param)
    elif ode_name == 'VdpODE':
        ode = VdpODE(param)
    elif ode_name == 'MMODE':
        ode = MMODE(param)
    elif ode_name == 'GompertzODE':
        ode = GompertzODE(param)
    elif ode_name == 'GompertzODE_par_a':
        ode = GompertzODE_par_a(param)
    elif ode_name == 'GompertzODE_par_b':
        ode = GompertzODE_par_b(param)
    elif ode_name == 'GompertzODE_par_ab':
        ode = GompertzODE_par_ab(param)
    elif ode_name == 'LogisticODE':
        ode = LogisticODE(param)
    elif ode_name == 'HillODE':
        ode = HillODE(param)
    elif ode_name == 'HillODE_par_k':
        ode = HillODE_par_k(param)
    elif ode_name == 'SirODE':
        ode = SirODE(param)
    elif ode_name == 'LvODE':
        ode = LvODE(param)
    elif ode_name == 'TVLvODE':
        ode = TVLvODE(param)
    elif ode_name == 'TVLvODE2':
        ode = TVLvODE2(param)
    elif ode_name == 'TVLvODE2_d':
        ode = TVLvODE2_d(param)
    elif ode_name == 'TVLvODE3':
        ode = TVLvODE3(param)
    elif ode_name == 'TVSelkovODE':
        ode = TVSelkovODE(param)
    elif ode_name == 'ThetaModel':
        ode = ThetaModel(param)
    elif ode_name == 'LIF':
        ode = LIF(param)
    elif ode_name == 'HopfODE':
        ode = HopfODE(param)
    elif ode_name == 'SelkovODE':
        ode = SelkovODE(param)
    elif ode_name == 'OscillatingSelkovODE_d':
        ode = OscillatingSelkovODE_d(param)
    elif ode_name == 'Brusselator':
        ode = Brusselator(param)
    elif ode_name == 'FHN':
        ode = FHN(param)
    elif ode_name == 'Lorenz':
        ode = Lorenz(param)
    elif ode_name == 'FracODE':
        ode = FracODE(param)
    elif ode_name == 'FracODE_par_rho':
        ode = FracODE_par_rho(param)
    elif ode_name == 'Rossler':
        ode = Rossler(param)
    elif ode_name == 'OscilVdpODE':
        ode = OscilVdpODE(param)
    elif ode_name == 'OscilVdpODE_par_w':
        ode = OscilVdpODE_par_w(param)
    elif ode_name == 'OscilVdpODE_par_w_mu':
        ode = OscilVdpODE_par_w_mu(param)
    elif ode_name == 'OscilVdpODE_par_w_a':
        ode = OscilVdpODE_par_w_a(param)
    elif ode_name == 'TVLogODE':
        ode = TVLogODE(param)  
    elif ode_name == 'ReactionDiffusion':
        ode = ReactionDiffusion(param) 
    elif ode_name == 'TVVdpODE':
        ode = TVVdpODE(param)
    elif ode_name == 'TVVdpODE2':
        ode = TVVdpODE2(param)
    elif ode_name == 'DoubleGyre':
        ode = DoubleGyre(param)
    else:
        raise ValueError('{} is not a supported ode.'.format(ode_name))
    return ode

def get_var_pos():
    X0 = sympy.Symbol('X0', positive=True)
    X1 = sympy.Symbol('X1', positive=True)
    X2 = sympy.Symbol('X2', positive=True)
    X3 = sympy.Symbol('X3', positive=True)
    C = sympy.Symbol('C', positive=True)
    t = sympy.Symbol('t', positive=True)

    VarDict = {
        'X0': X0,
        'X1': X1,
        'X2': X2,
        'X3': X3,
        'C': C,
        't': t,
    }
    return VarDict

def get_var_real():
    X0 = sympy.Symbol('X0', real=True)
    X1 = sympy.Symbol('X1', real=True)
    X2 = sympy.Symbol('X2', real=True)
    X3 = sympy.Symbol('X3', real=True)
    C = sympy.Symbol('C', positive=True)
    VarDict = {
        'X0': X0,
        'X1': X1,
        'X2': X2,
        'X3': X3,
        'C': C,
    }
    return VarDict


class ODE(metaclass=abc.ABCMeta):
    def __init__(self, dim_x, param=None):
        self.dim_x = dim_x
        self.has_coef = param is not None
        self.param = param if self.has_coef else self.get_default_param()
        self.T = 5
        self.init_high = 0.1
        self.init_low = 0
        self.std_base = 1.
        self.positive = True

    @abc.abstractmethod
    def get_default_param(self):
        pass

    @abc.abstractmethod
    def _dx_dt(self, *args):
        pass

    @staticmethod
    def get_var_dict():
        return get_var_pos()

    @abc.abstractmethod
    def get_expression(self):
        pass

    def dx_dt(self, t, x):
        arg_list = list()
        for i in range(self.dim_x):
            arg_list.append(x[i])
        return self._dx_dt(*arg_list)

    def dx_dt_batch(self, t, x):
        arg_list = list()
        for i in range(self.dim_x):
            arg_list.append(x[:, :, i])
        return np.stack(self._dx_dt(*arg_list), axis=-1)

    @abc.abstractmethod
    def functional_theta(self, theta):
        pass


class ODESolver:
    def __init__(self, ode, T, freq, return_list=False, integrator='lsoda'):
        self.ode = ode
        self.integrator = integrator
        # if T > freq:
        #     assert T % freq == 0
        self.T = T
        self.freq = freq
        n_step = int(T * freq)
        self.n_step = n_step
        self.dt = 1 / freq
        self.t = np.arange(0, T + self.dt, self.dt)
        self.return_list = return_list

    def solve_one(self, init):
        ode = scipy.integrate.ode(self.ode.dx_dt).set_integrator(self.integrator)
        ode.set_initial_value(init, 0)

        res_list = [init]

        while ode.successful() and ode.t < self.T:
            res = ode.integrate(ode.t + self.dt)
            res_list.append(res)
        res = np.stack(res_list, axis=-1)
        # D, T
        return res

    def solve(self, init_list):
        res_list = []
        for init in init_list:
            res_list.append(self.solve_one(init))
        # B, D, T -> T, B, D
        # print({arr.shape for arr in res_list})
        if not self.return_list:
            res = np.stack(res_list, axis=0).transpose((2, 0, 1))
            res_t = res.shape[0]
            true_t = len(self.t)
            if res_t > true_t:
                res = res[(res_t - true_t):, ...]
            return res
        else:
            return res_list
        

# solver handling a SR _Program object instead of an equations.ODE object
class ODESolver_p:
    def __init__(self, f, T, freq, return_list=False, integrator='lsoda'):
        self.f = f
        self.integrator = integrator
        # if T > freq:
        #     assert T % freq == 0
        self.T = T
        self.freq = freq
        n_step = int(T * freq)
        self.n_step = n_step
        self.dt = 1 / freq
        self.t = np.arange(0, T + self.dt, self.dt)
        self.return_list = return_list

    def solve_one(self, init):
        ode = scipy.integrate.ode(self.f).set_integrator(self.integrator)
        ode.set_initial_value(init, 0)

        res_list = [init]

        while ode.successful() and ode.t < self.T:
            res = ode.integrate(ode.t + self.dt)
            res_list.append(res)
        res = np.stack(res_list, axis=-1)
        # D, T
        return res

    def solve(self, init_list):
        res_list = []
        for init in init_list:
            res_list.append(self.solve_one(init))
        # B, D, T -> T, B, D
        # print({arr.shape for arr in res_list})
        if not self.return_list:
            res = np.stack(res_list, axis=0).transpose((2, 0, 1))
            res_t = res.shape[0]
            true_t = len(self.t)
            if res_t > true_t:
                res = res[(res_t - true_t):, ...]
            return res
        else:
            return res_list


class InferredODE(ODE):
    def __init__(self, dim_x, param=None, f_hat_list=None, T=None):
        super().__init__(dim_x, param)
        assert len(f_hat_list) == dim_x
        self.f_hat_list = f_hat_list
        self.T = T

    def get_default_param(self):
        return 1., 1.

    def _dx_dt(self, X):
        return None

    def get_expression(self):
        return None

    def functional_theta(self, theta):
        return None

    def dx_dt(self, t, x):
        return [f(x[None, :]) for f in self.f_hat_list]



class LinearODE(ODE):
    def __init__(self, dim_x, param=None):
        super().__init__(dim_x, param)
        self.beta = self.param[0]
        assert self.beta.shape[0] == self.beta.shape[1]
        assert self.beta.shape[0] == dim_x

    def get_default_param(self):
        return [np.array([[0, 1], [-1, 0]])]

    def _dx_dt(self, t, x):
        pass

    def dx_dt(self, t, x):
        x = np.array(x)
        return np.matmul(x, self.beta)

    def dx_dt_batch(self, t, x):
        # x: T, B, D
        # beta: D, D
        # out: T, B, D
        return np.matmul(x, self.beta)

    def functional_theta(self, theta):
        new_ode = LinearODE(self.dim_x, theta)
        return new_ode.dx_dt_batch

    def get_expression(self):
        raise NotImplementedError


class SineWave(LinearODE):
    def __init__(self, param=None):
        # param: [1.0]
        dim_x = 2
        if param is not None:
            param = [np.array([[0, 1], [-1 * param[0], 0]])]

        super().__init__(dim_x, param)
        self.has_coef = param is not None

    def get_default_param(self):
        return [np.array([[0, 1], [-1, 0]])]

    @staticmethod
    def get_var_dict():
        return get_var_real()

    def get_expression(self):
        var_dict = self.get_var_dict()
        X0 = var_dict['X0']
        X1 = var_dict['X1']
        C = var_dict['C']

        if self.has_coef:
            eq1 = X1
            eq2 = -1 * C * X0
        else:
            eq1 = X1
            eq2 = -1 * X0
        return [eq1, eq2]


class VdpODE(ODE):
    """
    van der pol equation
    https://en.wikipedia.org/wiki/Van_der_Pol_oscillator#Forced_Van_der_Pol_oscillator
    """

    def __init__(self, param=None):
        super().__init__(2, param)
        self.mu = self.param[0]
        self.init_high = 1.
        if not self.has_coef:
            self.T = 25
        
        self.name = 'VdpODE'

    def get_default_param(self):
        return [1.]

    def _dx_dt(self, x0, x1):
        dx = 1 / self.mu * (x1 - 1.0 / 3.0 * x0 ** 3 + x0)
        dy = -x0
        return [dx, dy]

    @staticmethod
    def get_var_dict():
        return get_var_real()

    def get_expression(self):
        var_dict = self.get_var_dict()
        X0 = var_dict['X0']
        X1 = var_dict['X1']
        C = var_dict['C']

        if self.has_coef:
            eq1 = C * (X1 - C * X0 * X0 * X0 + X0)
        else:
            eq1 = X1 - C * X0 * X0 * X0 + X0
        eq2 = -1 * X0
        return [eq1, eq2]

    def functional_theta(self, theta):
        assert len(theta) in [1]
        new_ode = VdpODE(theta[0])
        return new_ode.dx_dt_batch


class MMODE(ODE):
    """
    Michaelis–Menten kinetics
    https://https://en.wikipedia.org/wiki/Michaelis%E2%80%93Menten_kinetics
    """

    def __init__(self, param=None):
        super().__init__(4, param)
        self.k_f, self.k_r, self.k_cat = self.param
        self.init_high = 1.

    def _dx_dt(self, x0, x1, x2, x3):
        dedt = -1 * self.k_f * x0 * x1 + self.k_r * x2 + self.k_cat * x2
        dsdt = -1 * self.k_f * x0 * x1 + self.k_r * x2
        desdt = self.k_f * x0 * x1 - self.k_r * x2 - self.k_cat * x2
        dpdt = self.k_cat * x2
        return [dedt, dsdt, desdt, dpdt]

    def get_default_param(self):
        return 1., 1., 1.

    def get_expression(self):
        var_dict = self.get_var_dict()
        X0 = var_dict['X0']
        X1 = var_dict['X1']
        X2 = var_dict['X2']
        C = var_dict['C']

        if self.has_coef:
            eq1 = -1 * C * X0 * X1 + C * X2
            eq2 = -1 * C * X0 * X1 + C * X2
            eq3 = X0 * X1 - C * X2
            eq4 = C * X2
        else:
            eq1 = -1 * X0 * X1 + C * X2
            eq2 = -1 * X0 * X1 + X2
            eq3 = X0 * X1 - C * X2
            eq4 = X2
        return eq1, eq2, eq3, eq4

    def functional_theta(self, theta):
        assert len(theta) == 3
        new_ode = MMODE(theta)
        return new_ode.dx_dt_batch


class GompertzODE(ODE):
    """
    Gompertz
    https://en.wikipedia.org/wiki/Gompertz_function
    """

    def __init__(self, param=None):
        super().__init__(1, param)
        self.a, self.b = self.param
        self.init_high = 0.01
        if self.has_coef:
            self.T = 5
        else:
            self.T = 5
        self.name = 'GompertzODE'
        self.std_base = 0.23405202469895242

    def _dx_dt(self, X):
        dxdt = -1 * self.a * X * np.log(self.b * X)
        return [dxdt]

    def get_default_param(self):
        return 1., 1.

    def get_expression(self):
        var_dict = self.get_var_dict()
        X0 = var_dict['X0']
        C = var_dict['C']
        if self.has_coef:
            # eq1 = -1 * C * X0 * sympy.log(X0) - C * X0
            eq1 = -1 * C * X0 * sympy.log(C * X0)
        else:
            eq1 = -1 * X0 * sympy.log(X0)
        return [eq1]

    def functional_theta(self, theta):
        assert len(theta) == 2
        new_ode = GompertzODE(theta)
        return new_ode.dx_dt_batch
    

class GompertzODE_par_a(ODE):
    """
    Gompertz parametrize on a
    https://en.wikipedia.org/wiki/Gompertz_function
    dim_k = 1
    """

    def __init__(self, param=None):
        super().__init__(2, param)
        self.b = self.param[0]

        # In this case x:[0., 0.01] and a:[1., 2.]
        self.init_high = [0.01, 2.]
        self.init_low = [0.0, 1.]

        self.has_coef = True
        self.T = 5
        
        self.name = 'GompertzODE_par_a'
        self.std_base = 0.23405202469895242

    def _dx_dt(self, X, a):
        dxdt = -1 * a * X * np.log(self.b * X)
        dadt = 0 * a * X
        return [dxdt, dadt]

    def get_default_param(self):
        return 1.5

    def get_expression(self):
        var_dict = self.get_var_dict()
        X0 = var_dict['X0']
        X1 = var_dict['X1']
        C = var_dict['C']
        if self.has_coef:
            # eq1 = -1 * C * X0 * sympy.log(X0) - C * X0
            eq1 = -1 * X1 * X0 * sympy.log(C * X0)
            eq2 = 0 * X0 * X1
        else:
            eq1 = -1 * X1 * X0 * sympy.log(X0)
            eq2 = 0 * X0 * X1
        return [eq1, eq2]

    def functional_theta(self, theta):
        assert len(theta) == 1
        new_ode = GompertzODE_par_a(theta)
        return new_ode.dx_dt_batch


class GompertzODE_par_b(ODE):
    """
    Gompertz parametrize on b
    https://en.wikipedia.org/wiki/Gompertz_function
    dim_k = 1
    """

    def __init__(self, param=None):
        super().__init__(2, param)
        self.a = self.param[0]

        # In this case x:[0., 0.01] and b:[1., 2.]
        self.init_high = [0.01, 2.]
        self.init_low = [0.0, 1.]

        self.has_coef = True
        self.T = 5
        
        self.name = 'GompertzODE_par_b'
        self.std_base = 0.23405202469895242

    def _dx_dt(self, X, b):
        dxdt = -1 * self.a * X * np.log(b * X)
        dbdt = 0 * b * X
        return [dxdt, dbdt]

    def get_default_param(self):
        return 1.5

    def get_expression(self):
        var_dict = self.get_var_dict()
        X0 = var_dict['X0']
        X1 = var_dict['X1']
        C = var_dict['C']
        if self.has_coef:
            # eq1 = -1 * C * X0 * sympy.log(X0) - C * X0
            eq1 = -1 * C * X0 * sympy.log(X1 * X0)
            eq2 = 0 * X0 * X1
        else:
            eq1 = -1 * X0 * sympy.log(X1 * X0)
            eq2 = 0 * X0 * X1
        return [eq1, eq2]

    def functional_theta(self, theta):
        assert len(theta) == 1
        new_ode = GompertzODE_par_b(theta)
        return new_ode.dx_dt_batch
    

class GompertzODE_par_ab(ODE):
    """
    Gompertz parametrize on a and b
    https://en.wikipedia.org/wiki/Gompertz_function
    dim_k = 2
    """

    def __init__(self, param=None):
        super().__init__(3, param)

        # In this case x:[0., 0.01], a:[1., 2.] and b:[1., 2.]
        self.init_high = [0.01, 2., 2.]
        self.init_low = [0.0, 1., 1.]

        self.has_coef = True
        self.T = 8 # 5
        
        self.name = 'GompertzODE_par_ab'
        self.std_base = 0.23405202469895242

    def _dx_dt(self, X, a, b):
        dxdt = -1 * a * X * np.log(b * X)
        dadt = 0 * a * b * X
        dbdt = 0 * a * b * X
        return [dxdt, dadt, dbdt]

    def get_default_param(self):
        return None

    def get_expression(self):
        var_dict = self.get_var_dict()
        X0 = var_dict['X0']
        X1 = var_dict['X1']
        X2 = var_dict['X2']
        C = var_dict['C']
        if self.has_coef:
            # eq1 = -1 * C * X0 * sympy.log(X0) - C * X0
            eq1 = -1 * X1 * X0 * sympy.log(X2 * X0)
            eq2 = 0 * X0 * X1 * X2
            eq3 = 0 * X0 * X1 * X2
        else:
            eq1 = -1 * X1 * X0 * sympy.log(X2 * X0)
            eq2 = 0 * X0 * X1 * X2
            eq3 = 0 * X0 * X1 * X2
        return [eq1, eq2, eq3]

    def functional_theta(self, theta):
        assert len(theta) == 0
        new_ode = GompertzODE_par_ab(theta)
        return new_ode.dx_dt_batch


class LogisticODE(ODE):
    """
    Logistic
    https://en.wikipedia.org/wiki/Logistic_function#Logistic_differential_equation
    """

    def __init__(self, param=None):
        super().__init__(1, param)
        self.a, self.k = self.param
        self.T = 10
        self.has_coef = True
        self.name = 'LogisticODE'
        self.std_base = 0.31972985438694346

    def _dx_dt(self, X):
        dxdt = self.a * (1 - np.power(X, self.k)) * X
        return [dxdt]

    def get_default_param(self):
        return 1., 0.5

    def get_expression(self):
        var_dict = self.get_var_dict()
        X0 = var_dict['X0']
        C = var_dict['C']
        if self.has_coef:
            eq1 = X0 - X0 ** C
        else:
            eq1 = (1 - X0) * X0
        return [eq1]

    def functional_theta(self, theta):
        assert len(theta) == 2
        new_ode = LogisticODE(theta)
        return new_ode.dx_dt_batch


class HillODE(ODE):
    """
    Hill_equation_(biochemistry)
    https://en.wikipedia.org/wiki/Hill_equation_(biochemistry)
    """

    def __init__(self, param=None):
        super().__init__(2, param)
        self.n, self.k, self.ka, self.ky = self.param

        self.init_low = [0., 0.]
        self.init_high = [10., 10.]

        self.name = "HillODE"
        self.T = 15
        self.std_base = 0.5831910543183362

    def _dx_dt(self, X, Y):
        dxdt = self.k * np.power(Y, self.n) / (self.ka + np.power(Y, self.n))
        dydt = -1 * self.ky * Y
        return [dxdt, dydt]

    def get_default_param(self):
        return 2.8, 1.0, 0.5, 0.1

    def get_expression(self):
        var_dict = self.get_var_dict()
        X0 = var_dict['X0']
        X1 = var_dict['X1']
        C = var_dict['C']
        if self.has_coef:
            eq1 = C * sympy.Pow(X1, C) / (C + sympy.Pow(X1, C))
            eq2 = -1 * C * X1
        else:
            eq1 = X1 / (C + X1)
            eq2 = -1 * X1
        return [eq1, eq2]

    def functional_theta(self, theta):
        assert len(theta) == 4
        new_ode = HillODE(theta)
        return new_ode.dx_dt_batch

    
class HillODE_par_k(ODE):
    """
    Hill_equation_(biochemistry)
    https://en.wikipedia.org/wiki/Hill_equation_(biochemistry)
    dim_k = 1
    """

    def __init__(self, param=None):
        super().__init__(3, param)
        self.n, self.ka, self.ky = self.param

        self.init_low = [0., 0., 2]
        self.init_high = [10., 10., 1]

        self.name = "HillODE_par_k"
        self.T = 15
        self.std_base = 0.5831910543183362

    def _dx_dt(self, X, Y, k):
        dxdt = k * np.power(Y, self.n) / (self.ka + np.power(Y, self.n))
        dydt = -1 * self.ky * Y
        dkdt = 0
        return [dxdt, dydt, dkdt]

    def get_default_param(self):
        return 2.8, 0.5, 0.1

    def get_expression(self):
        var_dict = self.get_var_dict()
        X0 = var_dict['X0']
        X1 = var_dict['X1']
        X2 = var_dict['X2']
        C = var_dict['C']
        if self.has_coef:
            eq1 = X2 * sympy.Pow(X1, C) / (C + sympy.Pow(X1, C))
            eq2 = -1 * C * X1
            eq3 = 0
        else:
            eq1 = X2 * X1 / (C + X1)
            eq2 = -1 * X1
            eq3 = 0
        return [eq1, eq2, eq3]

    def functional_theta(self, theta):
        assert len(theta) == 3
        new_ode = HillODE_par_k(theta)
        return new_ode.dx_dt_batch


class SirODE(ODE):
    """
    SIR model
    https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology
    """

    def __init__(self, param=None):
        super().__init__(3, param)
        self.b, self.k = self.param
        self.init_high = [100., 0.01, 0.1]

    def _dx_dt(self, x0, x1, x2):
        dsdt = -1 * self.b * x0 * x1
        didt = self.b * x0 * x1 - self.k * x1
        drdt = self.k * x1
        return [dsdt, didt, drdt]

    def get_default_param(self):
        return 1., 1.

    def get_expression(self):
        var_dict = self.get_var_dict()
        X0 = var_dict['X0']
        X1 = var_dict['X1']
        C = var_dict['C']
        if self.has_coef:
            eq1 = -1 * C * X0 * X1
            eq2 = C * X0 * X1 - C * X1
            eq3 = C * X1
        else:
            eq1 = -1 * X0 * X1
            eq2 = X0 * X1 - X1
            eq3 = X1
        return [eq1, eq2, eq3]

    def functional_theta(self, theta):
        assert len(theta) == 2
        new_ode = SirODE(theta)
        return new_ode.dx_dt_batch


class LvODE(ODE):
    """
    Lotka-Volterra equations
    https://en.wikipedia.org/wiki/Lotka-Volterra_equations
    """

    def __init__(self, param=None):
        super().__init__(2, param)
        self.a, self.c, self.gamma = self.param
        self.init_high = [1., 1.]
        self.init_low = [0., 0.]
        self.T = 15
        self.name = 'LvODE'

    def _dx_dt(self, X, Y):
        dxdt = self.a * X - self.a * X * Y
        dydt = -1. * self.c * Y + self.gamma * X * Y
        return [dxdt, dydt]

    def get_default_param(self):
        return 1., 1., 1

    def get_expression(self):
        var_dict = self.get_var_dict()
        X0 = var_dict['X0']
        X1 = var_dict['X1']
        C = var_dict['C']
        if self.has_coef:
            eq1 = C * X0 - C * X0 * X1
            eq2 = -1 * C * X1 + C * X0 * X1
        else:
            eq1 = X0 - X0 * X1
            eq2 = -1 * X1 + X0 * X1
        return [eq1, eq2]

    def functional_theta(self, theta):
        assert len(theta) == 3
        new_ode = LvODE(theta)
        return new_ode.dx_dt_batch
    

class TVLvODE(ODE):
    """
    Time varying Lotka-Volterra equations inspired by EKF-SINDy test-case
    https://en.wikipedia.org/wiki/Lotka-Volterra_equations
    EKF-SINDy: https://arxiv.org/abs/2411.04842
    """

    def __init__(self, param=None):
        super().__init__(3, param)
        self.a, self.b, self.c, self.d = self.param
        self.init_high = [20., 20., 0.]
        self.init_low = [5., 5., 0.]
        self.T = 150
        self.std_base = 8.743084594979782 # NOTE: calcolato come RMSE di dX_list su 200 curve noise-free
        self.name = 'TVLvODE'

    def _dx_dt(self, X, Y, t):
        res = ((self.b + 0.01) if t > 50 else self.b) * X * Y      # Gestire l'impulso
        dxdt = (self.a + 0.2*np.sin(0.04*t))* X + res
        dydt = self.c * Y + (self.d + 0.01/150 * t) * X * Y
        dtdt = 1
        return [dxdt, dydt, dtdt]

    def get_default_param(self):
        return 1., -0.1, -1.5, 0.075

    def get_expression(self):
        var_dict = self.get_var_dict()
        X0 = var_dict['X0']
        X1 = var_dict['X1']
        X2 = var_dict['X2']
        C = var_dict['C']
        if self.has_coef:
            eq1 = (C + C*sympy.sin(C*X2)) * X0 + (C + C*sympy.Heaviside(X2-C)) * X0 * X1
            eq2 = C * X1 + (C + C*X2) * X0 * X1
            eq3 = 1
        else:
            eq1 = (C + C*sympy.sin(C*X2)) * X0 + (C + C*sympy.Heaviside(X2-C)) * X0 * X1
            eq2 = C * X1 + (C + C*X2) * X0 * X1
            eq3 =  1
        return [eq1, eq2, eq3]

    def functional_theta(self, theta):
        assert len(theta) == 4
        new_ode = TVLvODE(theta)
        return new_ode.dx_dt_batch
    

class TVLvODE2(ODE):
    """
    Time varying Lotka-Volterra equations inspired by EKF-SINDy test-case modified for the problem
    https://en.wikipedia.org/wiki/Lotka-Volterra_equations
    EKF-SINDy: https://arxiv.org/abs/2411.04842
    """

    def __init__(self, param=None):
        super().__init__(3, param)
        self.a, self.b, self.c, self.d = self.param
        self.init_high = [20., 20., 0.]     
        self.init_low = [5., 5., 0.]
        self.T = 50 # 100
        self.std_base = 9.000675159676526 # NOTE: calcolato come RMSE di dX_list su 200 curve noise-free
        self.name = 'TVLvODE2'

    def _dx_dt(self, X, Y, t):
        imp = ((self.b + 0.01) if t > 50 else self.b) * X * Y      # Gestire l'impulso
        dxdt = (self.a + 0.4*np.sin(5*t))* X + imp
        dydt = self.c * Y + self.d * X * Y
        dtdt = 1
        return [dxdt, dydt, dtdt]

    def get_default_param(self):
        return 1., -0.1, -1.5, 0.075

    def get_expression(self):
        var_dict = self.get_var_dict()
        X0 = var_dict['X0']
        X1 = var_dict['X1']
        X2 = var_dict['X2']
        C = var_dict['C']
        if self.has_coef:
            eq1 = (C + C*sympy.sin(C*X2)) * X0 + (C + C*sympy.Heaviside(X2-C)) * X0 * X1
            eq2 = C * X1 + C * X0 * X1
            eq3 = 1
        else:
            eq1 = (C + C*sympy.sin(C*X2)) * X0 + (C + C*sympy.Heaviside(X2-C)) * X0 * X1
            eq2 = C * X1 + C * X0 * X1
            eq3 =  1
        return [eq1, eq2, eq3]

    def functional_theta(self, theta):
        assert len(theta) == 4
        new_ode = TVLvODE2(theta)
        return new_ode.dx_dt_batch
    

class TVLvODE2_d(ODE):
    """
    Time varying Lotka-Volterra equations inspired by EKF-SINDy test-case modfied for the problem
    https://en.wikipedia.org/wiki/Lotka-Volterra_equations
    EKF-SINDy: https://arxiv.org/abs/2411.04842
    """

    def __init__(self, param=None):
        super().__init__(3, param)
        self.a, self.b, self.c, self.d = self.param
        self.init_high = [20., 20., 0.]    
        self.init_low = [5., 5., 0.]
        self.T = 100
        self.std_base = 9.738077440682273 # NOTE: calcolato come RMSE di dX_list su 200 curve noise-free
        self.name = 'TVLvODE2_d'

    def _dx_dt(self, X, Y, t):
        imp = ((self.b + 0.01) if t > 50 else self.b) * X * Y      # Gestire l'impulso
        dxdt = (self.a + 0.4*np.sin(5*t) * (1 / (1 + np.exp(-1 * (t - 20)))) )* X + imp
        dydt = self.c * Y + self.d * X * Y
        dtdt = 1
        return [dxdt, dydt, dtdt]

    def get_default_param(self):
        return 1., -0.1, -1.5, 0.075

    def get_expression(self):
        var_dict = self.get_var_dict()
        X0 = var_dict['X0']
        X1 = var_dict['X1']
        X2 = var_dict['X2']
        C = var_dict['C']
        if self.has_coef:
            eq1 = (C + C*sympy.sin(C*X2)) * X0 + (C + C*sympy.Heaviside(X2-C)) * X0 * X1
            eq2 = C * X1 + C * X0 * X1
            eq3 = 1
        else:
            eq1 = (C + C*sympy.sin(C*X2)) * X0 + (C + C*sympy.Heaviside(X2-C)) * X0 * X1
            eq2 = C * X1 + C * X0 * X1
            eq3 =  1
        return [eq1, eq2, eq3]

    def functional_theta(self, theta):
        assert len(theta) == 4
        new_ode = TVLvODE2_d(theta)
        return new_ode.dx_dt_batch


class TVLvODE3(ODE):
    """
    Time varying Lotka-Volterra equations inspired by EKF-SINDy test-case
    https://en.wikipedia.org/wiki/Lotka-Volterra_equations
    EKF-SINDy: https://arxiv.org/abs/2411.04842
    """

    def __init__(self, param=None):
        super().__init__(3, param)
        self.a, self.b, self.c, self.d = self.param
        self.init_high = [24., 14., 0.]    
        self.init_low = [16., 6., 0.]
        self.T = 50
        self.std_base = 5.441674054927272  # TODO: definire per il caso specifico finale (non considerare la variabilità del tempo)
        self.name = 'TVLvODE3'

    def _dx_dt(self, X, Y, t):
        imp = ((self.b + 0.01) if t > 50 else self.b) * X * Y      # Gestire l'impulso
        dxdt = (self.a + 0.4*np.sin(4*t)* (1 / (1 + np.exp(-1 * (t - 20)))) ) * X + imp
        dydt = self.c * Y + self.d * X * Y
        dtdt = 1
        return [dxdt, dydt, dtdt]

    def get_default_param(self):
        return 1., -0.1, -1.5, 0.075

    def get_expression(self):
        var_dict = self.get_var_dict()
        X0 = var_dict['X0']
        X1 = var_dict['X1']
        X2 = var_dict['X2']
        C = var_dict['C']
        if self.has_coef:
            eq1 = (C + C*sympy.sin(C*X2)) * X0 + (C + C*sympy.Heaviside(X2-C)) * X0 * X1
            eq2 = C * X1 + (C + C*X2) * X0 * X1
            eq3 = 1
        else:
            eq1 = (C + C*sympy.sin(C*X2)) * X0 + (C + C*sympy.Heaviside(X2-C)) * X0 * X1
            eq2 = C * X1 + (C + C*X2) * X0 * X1
            eq3 =  1
        return [eq1, eq2, eq3]

    def functional_theta(self, theta):
        assert len(theta) == 4
        new_ode = TVLvODE3(theta)
        return new_ode.dx_dt_batch
    

class TVSelkovODE(ODE):
    """
    Time varying Selkov model inspired by EKF-SINDy test-case
    https://services.math.duke.edu/ode-book/xppaut/ch4-selkov.pdf
    EKF-SINDy: https://arxiv.org/abs/2411.04842
    """
    def __init__(self, param=None):
        super().__init__(3, param)
        self.rho1, self.rho2, self.sigma = self.param
        self.has_coef = True
        self.init_high = [0.1, 0.1, 0.]       # TODO: definire per il caso specifico finale
        self.init_low = [0., 0., 0.]
        self.name = 'TVSelkovODE'
        self.std_base = 0.5641061           # TODO: definire per il caso specifico finale
        self.T = 100 # 300 !!!!

    def get_default_param(self):
        return 0.95, 0.60, 0.01

    def _dx_dt(self, x, y, t):

        sig = -(self.rho1-self.rho2)/(1+np.exp(-0.5*(t-40)))
        dxdt = self.rho1 + sig - self.sigma * x - x * y * y
        dydt = -1 * y + self.sigma * x + x * y * y 
        dtdt = 1
        return [dxdt, dydt, dtdt]

    def get_expression(self):
        var_dict = self.get_var_dict()
        X0 = var_dict['X0']
        X1 = var_dict['X1']
        X2 = var_dict['X2']
        C = var_dict['C']

        if self.has_coef:
            eq1 = C - C * X0 - X0 * X1 * X1
            eq2 = -1 * X1 + C * X0 + X0 * X1 * X1
            eq3 = 1
        else:
            eq1 = 1 - X0 - X0 * X1 * X1
            eq2 = -1 * X1 + X0 + X0 * X1 * X1
            eq3 = 1
        return [eq1, eq2, eq3]

    def functional_theta(self, theta):
        assert len(theta) == 3
        new_ode = TVSelkovODE(theta)
        return new_ode.dx_dt_batch


class ThetaModel(ODE):
    """
    Theta model
    https://en.wikipedia.org/wiki/Theta_model#General_equations
    """
    def __init__(self, param=None):
        super().__init__(2, param)
        self.a, self.b, self.k = self.param
        self.init_high = 1.
        self.T = 10

    def get_default_param(self):
        return [1., 1, 0]

    def _dx_dt(self, x, y):
        # dxdt = 1 - np.cos(self.a * x) + (1 + np.cos(self.a * x)) * self.k
        dxdt = 1 - np.cos(self.a * x) + (1 + np.cos(self.a * x)) * np.sin(self.b * y) * self.k
        dydt = 1
        return [dxdt, dydt]

    def get_expression(self):
        var_dict = self.get_var_dict()
        X0 = var_dict['X0']
        X1 = var_dict['X1']
        C = var_dict['C']
        if self.has_coef:
            eq1 = C - sympy.cos(C * X0) + (1 + sympy.cos(C * X0)) * sympy.sin(C * X1) * C
        else:
            eq1 = 1 - sympy.cos(X0)
        eq2 = 1
        return [eq1, eq2]

    def functional_theta(self, theta):
        assert len(theta) == 1
        new_ode = ThetaModel(theta)
        return new_ode.dx_dt_batch


class LIF(ODE):
    """
    Leaky integrate and fire
    https://en.wikipedia.org/wiki/Biological_neuron_model#Leaky_integrate-and-fire
    """
    def __init__(self, param=None):
        super().__init__(2, param)
        self.c, self.r = self.param
        print(self.param)

    def get_default_param(self):
        return 1., 1.

    def _dx_dt(self, x, y):
        dxdt = self.c * y - self.r * x
        dydt = 0
        return [dxdt, dydt]

    def get_expression(self):
        var_dict = self.get_var_dict()
        X0 = var_dict['X0']
        X1 = var_dict['X1']
        C = var_dict['C']
        if self.has_coef:
            eq1 = C * X1 - C * X0
        else:
            eq1 = X1 - X0
        eq2 = 0
        return [eq1, eq2]

    def functional_theta(self, theta):
        assert len(theta) == 2
        new_ode = LIF(theta)
        return new_ode.dx_dt_batch


class HopfODE(ODE):
    """
    2D Hopf normal form
    """
    def __init__(self, param=None):
        super().__init__(2, param)
        self.mu, self.omega, self.A = self.param
        self.T = 10

    def get_default_param(self):
        return 1., 1., 0.

    def _dx_dt(self, x, y):
        dxdt = self.mu * x + self.omega * y - self.A * (x ** 2 + y ** 2)
        dydt = -1 * self.omega * x + self.mu * y - self.A * (x ** 2 + y ** 2)
        return [dxdt, dydt]

    def get_expression(self):
        var_dict = self.get_var_dict()
        X0 = var_dict['X0']
        X1 = var_dict['X1']
        C = var_dict['C']
        if self.has_coef:
            eq1 = C * X0 + C * X1 - C * (X0 ** 2 + X0 ** 2)
            eq2 = -1 * C * X0 + C * X1 - C * (X0 ** 2 + X0 ** 2)
        else:
            eq1 = X0 + X1 - (X0 ** 2 + X0 ** 2)
            eq2 = -1 * X0 + X1 - (X0 ** 2 + X0 ** 2)
        return [eq1, eq2]

    def functional_theta(self, theta):
        assert len(theta) == 3
        new_ode = HopfODE(theta)
        return new_ode.dx_dt_batch


class SelkovODE(ODE):
    """
    Selkov model for Glycolysis
    https://services.math.duke.edu/ode-book/xppaut/ch4-selkov.pdf
    """
    def __init__(self, param=None):
        super().__init__(2, param)
        self.rho, self.sigma = self.param
        self.has_coef = True
        self.name = 'SelkovODE'
        self.std_base = 0.5641061
        self.T = 15

    def get_default_param(self):
        return 0.75, 0.1

    def _dx_dt(self, x, y):
        dxdt = self.rho - self.sigma * x - x * y * y
        dydt = -1 * y + self.sigma * x + x * y * y
        return [dxdt, dydt]

    def get_expression(self):
        var_dict = self.get_var_dict()
        X0 = var_dict['X0']
        X1 = var_dict['X1']
        C = var_dict['C']
        if self.has_coef:
            eq1 = C - C * X0 - X0 * X1 * X1
            eq2 = -1 * X1 + C * X0 + X0 * X1 * X1
        else:
            eq1 = 1 - X0 - X0 * X1 * X1
            eq2 = -1 * X1 + X0 + X0 * X1 * X1
        return [eq1, eq2]

    def functional_theta(self, theta):
        assert len(theta) == 2
        new_ode = SelkovODE(theta)
        return new_ode.dx_dt_batch



class FracODE(ODE):
    def __init__(self, param=None):
        super().__init__(2, param)
        self.rho = self.param[0]
        self.has_coef = True
        self.name = 'FracODE'
        self.std_base = 2.5388
        self.T = 3
        self.init_high = [1., 1.]
        self.init_low = [0.8, 0.8]

    def get_default_param(self):
        return 1.

    def _dx_dt(self, x, y):
        dxdt = x ** 2 / (y + self.rho)
        dydt = 1
        return [dxdt, dydt]

    def get_expression(self):
        var_dict = self.get_var_dict()
        X0 = var_dict['X0']
        X1 = var_dict['X1']
        C = var_dict['C']
        if self.has_coef:
            eq1 = X0 ** C / (X1 + C), X0 ** 2 / (X1 + C), X0 ** 2 / (X1 + 1)
            eq2 = C
        else:
            eq1 = X0 ** 2 / X1
            eq2 = C
        return [eq1, eq2]

    def functional_theta(self, theta):
        assert len(theta) == 2
        new_ode = FracODE(theta)
        return new_ode.dx_dt_batch
    

class FracODE_par_rho(ODE):
    """
    dim_k = 1
    """
    def __init__(self, param=None):
        super().__init__(3, param)
        self.has_coef = True
        self.name = 'FracODE_par_rho'
        self.std_base = 2.5388
        self.T = 3
        self.init_high = [1., 1., 2.]
        self.init_low = [0.8, 0.8, 1.]

    def get_default_param(self):
        return None

    def _dx_dt(self, x, y, r):
        dxdt = x ** 2 / (y + r)
        dydt = 1
        drdt = 0 * x * y * r
        return [dxdt, dydt, drdt]

    def get_expression(self):
        var_dict = self.get_var_dict()
        X0 = var_dict['X0']
        X1 = var_dict['X1']
        X2 = var_dict['X2']
        C = var_dict['C']
        if self.has_coef:
            eq1 = X0 ** C / (X1 + X2), X0 ** 2 / (X1 + X2), X0 ** 2 / (X1 + X2)
            eq2 = C
            eq3 = 0 * X0 * X1 * X2
        else:
            eq1 = X0 ** 2 / (X1 + X2)
            eq2 = C
            eq3 = 0 * X0 * X1 * X2
        return [eq1, eq2, eq3]

    def functional_theta(self, theta):
        assert len(theta) == 0
        new_ode = FracODE_par_rho(theta)
        return new_ode.dx_dt_batch


class Brusselator(ODE):
    """
    Brusselator
    https://www.bibliotecapleyades.net/archivos_pdf/brusselator.pdf
    """
    def __init__(self, param=None):
        super().__init__(2, param)
        self.a, self.b = self.param
        self.T = 15
        self.init_high = 1.

    def get_default_param(self):
        return 1., 1.

    def _dx_dt(self, x, y):
        dxdt = 1. - (self.b + 1) * x + self.a * x * x * y
        dydt = self.b * x - self.a * x * x * y
        return [dxdt, dydt]

    def get_expression(self):
        var_dict = self.get_var_dict()
        X0 = var_dict['X0']
        X1 = var_dict['X1']
        C = var_dict['C']
        if self.has_coef:
            eq1 = C - C * X0 + C * X0 * X0 * X1
            eq2 = C * X0 - C * X0 * X0 * X1
        else:
            eq1 = C - C * X0 + X0 * X0 * X1
            eq2 = X0 - X0 * X0 * X1
        return [eq1, eq2]

    def functional_theta(self, theta):
        assert len(theta) == 2
        new_ode = Brusselator(theta)
        return new_ode.dx_dt_batch


class FHN(ODE):
    """
        FitzHugh–Nagumo model
        https://en.wikipedia.org/wiki/FitzHugh%E2%80%93Nagumo_model
    """
    def __init__(self, param=None):
        super().__init__(3, param)
        self.a, self.b, self.eps, self.I = self.param
        self.init_high = [0.5, 0.5, 0]
        self.init_low = [0.,0., 0]
        self.T = 150 # 100 # 25

        self.name = "FHN"
        self.std_base = 1.

    def get_default_param(self):
        return 1., 0., 1., 0.

    def _dx_dt(self, x, y, t):

        A = 3
        B = 0.04
        if t<60:
            I_inp = 0
        else:
            I_inp = 0.4
        dxdt = x - 1. / 3. * x * x * x - y + I_inp
        dydt = self.eps * (x + self.a - self.b * y)
        dtdt = 1.
        return [dxdt, dydt, dtdt]

    def get_expression(self):
        var_dict = self.get_var_dict()
        X0 = var_dict['X0']
        X1 = var_dict['X1']
        C = var_dict['C']
        if self.has_coef:
            eq1 = X0 - C * X0 * X0 * X0 - X1
            eq2 = X0 + C - C * X1
        else:
            eq1 = X0 - C * X0 * X0 * X0 - X1
            eq2 = X0 + C
        return [eq1, eq2]

    def functional_theta(self, theta):
        assert len(theta) == 4
        new_ode = FHN(theta)
        return new_ode.dx_dt_batch


class Lorenz(ODE):
    """
        Lorenz System
        https://en.wikipedia.org/wiki/Lorenz_system
    """
    def __init__(self, param=None):
        super().__init__(3, param)
        self.sigma, self.rho, self.beta = self.param
        self.T = 10
        self.has_coef = True
        self.init_high = 10
        self.std_base = 8.55515291
        self.name = 'Lorenz'
        self.positive = False

    def get_default_param(self):
        return 10, 28, 8/3

    def _dx_dt(self, x, y, z):
        dxdt = self.sigma * (y - x)
        dydt = x * (self.rho - z) - y
        dzdt = x * y - self.beta * z
        return [dxdt, dydt, dzdt]

    def get_expression(self):
        var_dict = self.get_var_dict()
        X0 = var_dict['X0']
        X1 = var_dict['X1']
        X2 = var_dict['X2']
        C = var_dict['C']

        if self.has_coef:
            eq1 = C * (X1 - X0)
            eq2 = (X0 * (C - X2) - X1, X0*(C - X2) + X0 - X1,
                   C*X0 - C*X1 - X0*X2, -C*X1 + X0*(C - X2) + X0, -C*X1 + X0*(C - X2) + C*X0)
            eq3 = X0 * X1 - C * X2
        else:
            eq1 = X1 - X0
            eq2 = -1. * X0 * X2 - X1
            eq3 = X0 * X1 - X2
        return [eq1, eq2, eq3]

    def functional_theta(self, theta):
        assert len(theta) == 3
        new_ode = Lorenz(theta)
        return new_ode.dx_dt_batch


class RealODEPlaceHolder:
    def __init__(self):
        self.name = 'real'
        self.std_base = 0.281234968452
        self.positive = False

    @staticmethod
    def get_var_dict():
        X0 = sympy.Symbol('X0', real=True)
        X1 = sympy.Symbol('X1', positive=True)
        C = sympy.Symbol('C', positive=True)

        VarDict = {
            'X0': X0,
            'X1': X1,
            'C': C,
        }
        return VarDict
    


class Rossler(ODE):
    """
        Rössler System
    """
    def __init__(self, param=None):
        super().__init__(3, param)
        self.a, self.b, self.c = self.param
        self.T = 10
        self.has_coef = True
        self.init_high = 10
        self.std_base = 8. # TODO: come si calcola std_base (?) 
        self.name = 'Rossler'
        self.positive = False

    def get_default_param(self):
        return 0.2, 0.2, 5.7

    def _dx_dt(self, x, y, z):
        dxdt = - y - z
        dydt = x + self.a * y
        dzdt = self.b + z * (x - self.c)
        return [dxdt, dydt, dzdt]

    def get_expression(self):
        var_dict = self.get_var_dict()
        X0 = var_dict['X0']
        X1 = var_dict['X1']
        X2 = var_dict['X2']
        C = var_dict['C']

        if self.has_coef:
            eq1 = - X1 - X2
            eq2 = X0 + C*X1
            eq3 = C + X2 * (X0 - C)
        else:
            eq1 = - X1 - X2
            eq2 = X0 + X1
            eq3 = 1. + X2 * (X0 - 1.)
        return [eq1, eq2, eq3]

    def functional_theta(self, theta):
        assert len(theta) == 3
        new_ode = Rossler(theta)
        return new_ode.dx_dt_batch
    

class OscilVdpODE(ODE):
    """
    Modified Van der Pol system with a oscillating forcing term.
    Fictitius ODE to test Simbolic-SINDy
    """

    def __init__(self, param=None):
        super().__init__(2, param)
        self.mu, self.A, self.omega = self.param

        self.init_high = [1., 1.]
        self.init_low = [0., 0.]

        self.has_coef = True
        self.T = 10
        self.positive = False
        
        self.name = 'OscilVdpODE'
        self.std_base = 1.366500494994911

    def _dx_dt(self, X, Y):
        dxdt = self.mu * (1 - Y**2) * X - Y +  self.A * np.sin(self.omega * Y**2)
        dydt = X 
        return [dxdt, dydt]

    def get_default_param(self):
        return 1., 1., 1.

    def get_expression(self):
        var_dict = self.get_var_dict()
        X0 = var_dict['X0']
        X1 = var_dict['X1']
        C = var_dict['C']
        if self.has_coef:
            eq1 = C * (1 - X1**2) * X0 - X1 + C * sympy.sin(C* X1**2) 
            eq2 = X0
        else:
            eq1 = (1 - X1**2) * X0 - X1 + sympy.sin(X1**2) 
            eq2 = X0 
        return [eq1, eq2]

    def functional_theta(self, theta):
        assert len(theta) == 3
        new_ode = OscilVdpODE(theta)
        return new_ode.dx_dt_batch
    

class OscilVdpODE_par_w(ODE):
    """
    Modified Van der Pol system with a oscillating forcing term.
    Fictitius ODE to test Simbolic-SINDy
    dim_k = 1
    """

    def __init__(self, param=None):
        super().__init__(3, param)
        self.mu, self.A = self.param

        self.init_high = [1., 1., np.pi]
        self.init_low = [0., 0., 1.]

        self.has_coef = True
        self.T = 10
        self.positive = False
        
        self.name = 'OscilVdpODE_par_w'
        self.std_base = 1.366500494994911

    def _dx_dt(self, X, Y, Z):
        dxdt = self.mu * (1 - Y**2) * X - Y +  self.A * np.sin(Z * Y**2)
        dydt = X 
        dzdt = 0
        return [dxdt, dydt, dzdt]

    def get_default_param(self):
        return 1., 1.

    def get_expression(self):
        var_dict = self.get_var_dict()
        X0 = var_dict['X0']
        X1 = var_dict['X1']
        X2 = var_dict['X2']
        C = var_dict['C']
        if self.has_coef:
            eq1 = C * (1 - X1**2) * X0 - X1 + C * sympy.sin( X2* X1**2) 
            eq2 = X0
            eq3 = 0
        else:
            eq1 = (1 - X1**2) * X0 - X1 + sympy.sin(X2 * X1**2) 
            eq2 = X0 
            eq3 = 0
        return [eq1, eq2, eq3]

    def functional_theta(self, theta):
        assert len(theta) == 2
        new_ode = OscilVdpODE_par_w(theta)
        return new_ode.dx_dt_batch
    

class OscilVdpODE_par_w_mu(ODE):
    """
    Modified Van der Pol system with a oscillating forcing term.
    Fictitius ODE to test Simbolic-SINDy
    dim_k = 2
    """

    def __init__(self, param=None):
        super().__init__(4, param)
        self.A = self.param[0]

        self.init_high = [1., 1., np.pi, 2]
        self.init_low = [0., 0., 1., 0.5]

        self.has_coef = True
        self.T = 10
        self.positive = False
        
        self.name = 'OscilVdpODE_par_w_mu'
        self.std_base = 1.366500494994911

    def _dx_dt(self, X, Y, Z, m):
        dxdt = m * (1 - Y**2) * X - Y +  self.A * np.sin(Z * Y**2)
        dydt = X 
        dzdt = 0
        dmdt = 0
        return [dxdt, dydt, dzdt, dmdt]

    def get_default_param(self):
        return 1.

    def get_expression(self):
        var_dict = self.get_var_dict()
        X0 = var_dict['X0']
        X1 = var_dict['X1']
        X2 = var_dict['X2']
        X3 = var_dict['X3']
        C = var_dict['C']
        if self.has_coef:
            eq1 = X3 * (1 - X1**2) * X0 - X1 + C * sympy.sin( X2* X1**2) 
            eq2 = X0
            eq3 = 0
            eq4 = 0
        else:
            eq1 = X3 * (1 - X1**2) * X0 - X1 + sympy.sin(X2 * X1**2) 
            eq2 = X0 
            eq3 = 0
            eq4 = 0
        return [eq1, eq2, eq3, eq4]

    def functional_theta(self, theta):
        assert len(theta) == 1
        new_ode = OscilVdpODE_par_w_mu(theta)
        return new_ode.dx_dt_batch
    

class OscilVdpODE_par_w_a(ODE):
    """
    Modified Van der Pol system with a oscillating forcing term.
    Fictitius ODE to test Simbolic-SINDy
    dim_k = 2
    """

    def __init__(self, param=None):
        super().__init__(4, param)
        self.mu = self.param[0]

        self.init_high = [1., 1., np.pi, 1]
        self.init_low = [0., 0., 1., 0]

        self.has_coef = True
        self.T = 10
        self.positive = False
        
        self.name = 'OscilVdpODE_par_w_mu'
        self.std_base = 1.366500494994911

    def _dx_dt(self, X, Y, Z, A):
        dxdt = self.mu * (1 - Y**2) * X - Y +  A * np.sin(Z * Y**2)
        dydt = X 
        dzdt = 0
        dAdt = 0
        return [dxdt, dydt, dzdt, dAdt]

    def get_default_param(self):
        return 1.

    def get_expression(self):
        var_dict = self.get_var_dict()
        X0 = var_dict['X0']
        X1 = var_dict['X1']
        X2 = var_dict['X2']
        X3 = var_dict['X3']
        C = var_dict['C']
        if self.has_coef:
            eq1 = C * (1 - X1**2) * X0 - X1 + X3 * sympy.sin( X2* X1**2) 
            eq2 = X0
            eq3 = 0
            eq4 = 0
        else:
            eq1 = (1 - X1**2) * X0 - X1 + X3 * sympy.sin(X2 * X1**2) 
            eq2 = X0 
            eq3 = 0
            eq4 = 0
        return [eq1, eq2, eq3, eq4]

    def functional_theta(self, theta):
        assert len(theta) == 1
        new_ode = OscilVdpODE_par_w_a(theta)
        return new_ode.dx_dt_batch
    

class OscillatingSelkovODE_d(ODE):
    """
    Selkov model for Glycolysis
    https://services.math.duke.edu/ode-book/xppaut/ch4-selkov.pdf
    """
    def __init__(self, param=None):
        super().__init__(3, param)
        self.rho, self.sigma = self.param # aggiungere parametri del termine sinusoidale
        self.has_coef = True
        self.name = 'OscillatingSelkovODE_d'
        self.std_base = 0.5641061
        self.T = 50 # 30 !!! 
        self.init_high = [0.1, 0.1, 0.]
        self.init_low = [0., 0., 0.]

    def get_default_param(self):
        return 0.75, 0.1 # 

    def _dx_dt(self, x, y, t):
        dxdt = self.rho - self.sigma * x - x * y * y + 1.0*np.sin(1.0 * np.pi * t) * (1 / (1 + np.exp(-1 * (t - 30))))
        dydt = -1 * y + self.sigma * x + x * y * y 
        dtdt = 1
        return [dxdt, dydt, dtdt]

    def get_expression(self):
        var_dict = self.get_var_dict()
        X0 = var_dict['X0']
        X1 = var_dict['X1']
        X2 = var_dict['X2']
        C = var_dict['C']
        if self.has_coef:
            eq1 = C - C * X0 - X0 * X1 * X1 + C * sympy.sin(C * X2) * (1 / (1 + sympy.exp(-1 * (X2 - C))))
            eq2 = -1 * X1 + C * X0 + X0 * X1 * X1
            eq3 = 1
        else:
            eq1 = 1 - X0 - X0 * X1 * X1 + sympy.sin(X2) * (1 / (1 + sympy.exp(-1 * X2)))
            eq2 = -1 * X1 + X0 + X0 * X1 * X1
            eq3 = 1
        return [eq1, eq2, eq3]

    def functional_theta(self, theta):
        assert len(theta) == 2
        new_ode = OscillatingSelkovODE_d(theta)
        return new_ode.dx_dt_batch
    

class TVLogODE(ODE):
    """
    Time varying oscillator with logaritmic forcing term
    """
    def __init__(self, param=None):
        super().__init__(3, param)
        self.k, self.c, self.A = self.param
        self.has_coef = True
        self.init_high = [1, 1, 0.]       # TODO: definire per il caso specifico finale
        self.init_low = [0., 0., 0.]
        self.name = 'TVLogODE'
        self.std_base = 0.5641061           # TODO: definire per il caso specifico finale
        self.T = 50 # 300 !!!!

    def get_default_param(self):
        return 1, 0.3, 0.15

    def _dx_dt(self, x, y, t):
        dxdt = y
        dydt = - self.k * x - self.c * y + self.A * np.log(1 + t**2) 
        dtdt = 1
        return [dxdt, dydt, dtdt]

    def get_expression(self):
        var_dict = self.get_var_dict()
        X0 = var_dict['X0']
        X1 = var_dict['X1']
        X2 = var_dict['X2']
        C = var_dict['C']

        if self.has_coef:
            eq1 = X1
            eq2 = -C * X0 - C * X1 + sympy.log(1+X2**2)
            eq3 = 1
        else:
            eq1 = X1
            eq2 = -C * X0 - C * X1 + sympy.log(1+X2**2)
            eq3 = 1
        return [eq1, eq2, eq3]

    def functional_theta(self, theta):
        assert len(theta) == 3
        new_ode = TVLogODE(theta)
        return new_ode.dx_dt_batch


class TVVdpODE(ODE):
    """
    Time varying van der Pol equations
    """

    def __init__(self, param=None):
        super().__init__(3, param)
        self.mu, self.A, self.w0 = self.param
        self.init_high = [2., 2., 0.]    
        self.init_low = [0., 0., 0.]
        self.T = 150
        self.std_base = 5.269730472894847 # NOTE: calcolato come RMSE di dX_list su 200 curve noise-free
        self.name = 'TVVdpODE'
        self.positive=False

    def _dx_dt(self, X, Y, t):
        dxdt = self.mu * (1 - Y**2) * X - Y + self.A * np.cos((self.w0 * t) * t)
        dydt = X
        dtdt = 1
        return [dxdt, dydt, dtdt]

    def get_default_param(self):
        return 5., 10., 0.02

    def get_expression(self):
        var_dict = self.get_var_dict()
        X0 = var_dict['X0']
        X1 = var_dict['X1']
        X2 = var_dict['X2']
        C = var_dict['C']
        if self.has_coef:
            eq1 = (C + C*sympy.sin(C*X2)) * X0 + (C + C*sympy.Heaviside(X2-C)) * X0 * X1
            eq2 = C * X1 + C * X0 * X1
            eq3 = 1
        else:
            eq1 = (C + C*sympy.sin(C*X2)) * X0 + (C + C*sympy.Heaviside(X2-C)) * X0 * X1
            eq2 = C * X1 + C * X0 * X1
            eq3 =  1
        return [eq1, eq2, eq3]

    def functional_theta(self, theta):
        assert len(theta) == 3
        new_ode = TVVdpODE(theta)
        return new_ode.dx_dt_batch
    

class TVVdpODE2(ODE):
    """
    Time varying van der Pol equations
    """

    def __init__(self, param=None):
        super().__init__(3, param)
        self.mu, self.A0, self.w = self.param
        self.init_high = [2., 2., 0.]    
        self.init_low = [0., 0., 0.]
        self.T = 80
        self.std_base = 9.165338768841108 # NOTE: calcolato come RMSE di dX_list su 200 curve noise-free
        self.name = 'TVVdpODE2'
        self.positive=False

    def _dx_dt(self, X, Y, t):
        dxdt = self.mu * (1 - Y**2) * X - Y + self.A0 * t * np.cos(self.w * t)
        dydt = X
        dtdt = 1
        return [dxdt, dydt, dtdt]

    def get_default_param(self):
        return 5., 0.5, 4

    def get_expression(self):
        var_dict = self.get_var_dict()
        X0 = var_dict['X0']
        X1 = var_dict['X1']
        X2 = var_dict['X2']
        C = var_dict['C']
        if self.has_coef:
            eq1 = (C + C*sympy.sin(C*X2)) * X0 + (C + C*sympy.Heaviside(X2-C)) * X0 * X1
            eq2 = C * X1 + C * X0 * X1
            eq3 = 1
        else:
            eq1 = (C + C*sympy.sin(C*X2)) * X0 + (C + C*sympy.Heaviside(X2-C)) * X0 * X1
            eq2 = C * X1 + C * X0 * X1
            eq3 =  1
        return [eq1, eq2, eq3]

    def functional_theta(self, theta):
        assert len(theta) == 3
        new_ode = TVVdpODE2(theta)
        return new_ode.dx_dt_batch




# class Vortex(ODE):
#     """
#     Vortex shedding
#     """
#
#     def __init__(self, param=None):
#         super().__init__(3, param)
#         self.mu, self.omega, self.A, self.lam = self.param
#
#     def get_default_param(self):
#         return 1., 1., 1., 1.
#
#     def _dx_dt(self, x, y, z):
#         dxdt = self.mu * x - self.omega * y + self.A * x * z
#         dydt = self.omega * x + self.mu * y + self.A * y * z
#         dzdt = x * x + y * y - z
#         return [dxdt, dydt, dzdt]
#
#     def get_expression(self):
#         if self.has_coef:
#             eq1 = C * X0 - C * X1 + C * X0 * X2
#             eq2 = C * X0 + C * X1 + C * X1 * X2
#             eq3 = C * (X0 * X0 + X1 * X1 - X2)
#         else:
#             eq1 = X0 - X1 + X0 * X2
#             eq2 = X0 + X1 + X1 * X2
#             eq3 = X0 * X0 + X1 * X1 - X2
#         return [eq1, eq2, eq3]
#
#     def functional_theta(self, theta):
#         assert len(theta) == 3
#         new_ode = Vortex(theta)
#         return new_ode.dx_dt_batch


class ReactionDiffusion(ODE):
    """
    Reaction-Diffusion class for the HD test.
    Note: this ODE is a fictitius ODe aiming to store the training information 
    """

    def __init__(self, param=None):
        super().__init__(2, param)
        self.a, self.b = self.param
        self.init_high = [0.01, 0.01]
        self.init_low = [0., 0.]

        self.positive = False

        self.T = 20

        self.name = 'ReactionDiffusion'
        self.std_base = 1.0

    def _dx_dt(self, X, Y):
        dxdt = 1 * X * Y
        dydt = 1 * X * Y
        return [dxdt, dydt]

    def get_default_param(self):
        return 1., 1.

    def get_expression(self):
        var_dict = self.get_var_dict()
        X0 = var_dict['X0']
        X1 = var_dict['X1']
        C = var_dict['C']
        if self.has_coef:
            eq1 = C + 0*X0
            eq2 = C + 0*X1
        else:
            eq1 = C + 0*X0
            eq2 = C + 0*X1
        return [eq1, eq2]

    def functional_theta(self, theta):
        assert len(theta) == 2
        new_ode = ReactionDiffusion(theta)
        return new_ode.dx_dt_batch


class DoubleGyre(ODE):
    """
    Double Gyre class for the HD test.
    Note: this ODE is a fictitius ODE aiming to store the training information 
    """

    def __init__(self, param=None):
        super().__init__(2, param)
        self.a, self.b = self.param
        self.init_high = [1., 1.]
        self.init_low = [0., 0.]

        self.positive = False

        self.T = 10

        self.name = 'DoubleGyre'
        self.std_base = 1.0

    def _dx_dt(self, X, Y):
        dxdt = 1 + 0*X
        dydt = 1 + 0*Y
        return [dxdt, dydt]

    def get_default_param(self):
        return 1., 1.

    def get_expression(self):
        var_dict = self.get_var_dict()
        X0 = var_dict['X0']
        X1 = var_dict['X1']
        C = var_dict['C']
        if self.has_coef:
            eq1 = C + 0*X0
            eq2 = C + 0*X1
        else:
            eq1 = C + 0*X0
            eq2 = C + 0*X1
        return [eq1, eq2]

    def functional_theta(self, theta):
        assert len(theta) == 2
        new_ode = DoubleGyre(theta)
        return new_ode.dx_dt_batch