import numpy as np
import matplotlib.pyplot as plt
import csv

class ODEnet:

    def __init__(self, hyper_parameters, structure, time_step=0.1):
        self.HP = hyper_parameters
        self.ts = time_step

        self.structure = structure
        self.wb = self.init_weights()

        self.Z = []
        self.A = []

    def init_weights(self):
        wb = []
        for t in range(self.r_f2i(self.HP['T'] * (1 / self.ts))):
            wb.append([])
            wb[t].append(np.zeros((self.structure[t + 1], self.structure[t])))
            wb[t].append(np.zeros(self.structure[t + 1]))
        wb.append([])
        wb[-1].append(np.zeros((self.structure[-1], self.structure[-1])))
        wb[-1].append(np.zeros(self.structure[-1]))
        return wb

    def update_parameter(self, p, step):
        self.HP[p] += step

    def update_weights(self, wb):
        self.wb = wb.copy()

    def reset_weights(self):
        self.wb = self.init_weights()

    def f(self, t, z0, wb):
        e1, e2 = self.HP['E1'], self.HP['E2']
        t = self.t2t(t)
        w, b = wb[t][0], wb[t][1]
        dz = e1 * self.activation(np.dot(w, z0) + b) - e2 * z0
        return dz

    def a_adjoint(self, t, a0, parameters):
        e1, e2 = self.HP['E1'], self.HP['E2']
        t = self.t2t(t)
        z, w, b = parameters[0][t], parameters[1][t][0], parameters[1][t][1]
        da = -np.dot(a0, (
                    w * e1 * self.der_activation(np.dot(w, z) + b).reshape(len(z), -1) - e2 * np.ones(len(z)).reshape(
                len(z), -1)))
        return da

    def w_adjoint(self, t, dw0, parameters):
        t = self.t2t(t)
        z, a, w, b = parameters[0][t], parameters[1][t], parameters[2][t][0], parameters[2][t][1]
        der_f = self.der_activation(np.dot(w, z) + b)
        db = -(a * der_f).flatten()
        der_f = np.array([der_f for _ in range(len(der_f))]).T * z
        dw = -(a.reshape(-1, 1) * der_f).flatten()
        return np.array([*dw, *db])

    def augmented_f(self, t, za0, params):
        hp, wb = params[0], params[1]
        e1, e2 = hp['E1'], hp['E2']
        t = self.t2t(t)
        w, b = wb[t][0], wb[t][1]
        z0, a0 = za0[:len(za0) // 2], za0[len(za0) // 2:]
        dz = e1 * self.activation(np.dot(w, z0) + b) - e2 * z0
        da = -np.dot(a0, (w * e1 * self.der_activation(np.dot(w, z0) + b).reshape(len(z0), -1) - e2 * np.ones(
            len(z0)).reshape(len(z0), -1)))
        return np.array([*dz, *da])

    def bc(self, bc_params, za):
        z0_true, z1_true = bc_params[0], bc_params[1]
        z0 = za[0][:len(za[0]) // 2]
        z1 = za[-1][:len(za[-1]) // 2]
        a1 = za[-1][len(za[-1]) // 2:]

        z0 = z0 - z0_true
        a1 = a1 - z1 + z1_true
        return np.concatenate((z0, a1))

     def activation(self, x):
         return x
     def der_activation(self, x):
         return np.ones_like(x)
    def activation(self, x):
        return np.tanh(x)

    def der_activation(self, x):
        return np.ones(len(x)) - np.tanh(x) ** 2

    def time_steps(self, t0, t1):
        T = []
        if t0 > t1:
            while t0 > t1:
                T.append(t0)
                t0 = np.round(t0 - self.ts, 1)
            T.append(t0)
        else:
            while t0 < t1:
                T.append(t0)
                t0 = np.round(t0 + self.ts, 1)
            T.append(t0)
        return T

    def r_f2i(self, x):
        return int(np.round(x))

    def t2t(self, x):
        return int(np.round(x / self.ts))

    def change_weights(self, dwb):
        dwb.reverse()
        for t in range(len(self.wb)):
            dw = np.array(dwb[t][:-len(self.wb[t][1])])
            db = np.array(dwb[t][-len(self.wb[t][1]):])
            dw = dw.reshape(self.wb[t][0].shape)
            db = db.reshape(self.wb[t][1].shape)
            self.wb[t][0] -= self.HP['LR'] * dw
            self.wb[t][1] -= self.HP['LR'] * db

    def forward(self, z0):
        time_steps = self.time_steps(0, self.HP['T'])
        self.Z = euler(time_steps, z0, self.f, self.wb)

    def back_propagation(self, zs):
        a0 = -self.Z[-1] + zs
        time_steps = self.time_steps(self.HP['T'], 0)
        time_steps.reverse()
        self.A = euler(time_steps, a0, self.a_adjoint, (self.Z, self.wb))
        dwb0 = np.zeros(len(zs) ** 2 + len(zs))
        dwb = euler(time_steps, dwb0, self.w_adjoint, (self.Z, self.A, self.wb))
        self.change_weights(dwb)

    def augmented_propagation(self, za0, z0, zs):
        time_steps = self.time_steps(0, self.HP['T'])
        newton_steps, za, det, sv = shooting(time_steps, za0, self.augmented_f, [self.HP, self.wb], self.bc, [z0, zs],
                                             runge_kutta)
        self.save_za(za)
        time_steps.reverse()
        dwb0 = np.zeros(len(zs) ** 2 + len(zs))
        dwb = runge_kutta(time_steps, dwb0, self.w_adjoint, (self.Z, self.A, self.wb))
        self.change_weights(dwb)
        return newton_steps, det, sv

    def matrix_analyses(self, za0, z0, zs):
        time_steps = self.time_steps(0, self.HP['T'])
        aug_F = augmented_frechet_matrix(time_steps, za0, self.augmented_f, [self.HP, self.wb], 'E2', self.bc, [z0, zs],
                                         runge_kutta)
        with open("dets.txt", 'a') as f:
            print(analyse_point(aug_F), file=f)

    def save_za(self, za):
        z = []
        a = []
        for l in za:
            z.append(l[:len(l) // 2])
            a.append(l[len(l) // 2:])
        self.Z = z
        self.A = a

    def plot_z(self, z0, zs, epoch, parameter=False, determinant=False):
        time_steps = self.time_steps(0, self.HP['T'])
        z = np.array(self.Z).T
        colors = ['b', 'g', 'r', 'm', 'c', 'silver', 'y', 'lightcoral', 'lime', 'gold']
        for i in range(len(zs)):
            plt.scatter(time_steps[0], z0[i], c=colors[i], marker='o')
            plt.scatter(time_steps[-1], zs[i], c=colors[i], marker='o')
            plt.plot(time_steps, z[i], c=colors[i])
            if parameter and determinant:
                plt.title(
                    f"epoch {epoch} : {parameter} = {np.round(self.HP[parameter], 4)} : determinant = {np.round(determinant, 4)}")
            else:
                plt.title(f"epoch {epoch}")
        plt.ylim(-1.1, 1.1)
        plt.show()


def create_csv(file, fieldnames):
    with open(file, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def write_csv(file, info):
    fieldnames = info.keys()
    with open(file, 'a') as f:
        csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
        csv_writer.writerow(info)


import numpy as np
from numpy.linalg import det, inv, svd


def interpolation(Xs, Ys, x_next):
    a0 = Ys[0]
    if len(Ys) == 1: return a0
    a1 = (Ys[1] - Ys[0]) / (Xs[1] - Xs[0])
    if len(Ys) == 2: return a0 + a1 * (x_next - Xs[0])

    yx1x2 = (Ys[2] - Ys[1]) / (Xs[2] - Xs[1])
    a2 = (yx1x2 - a1) / (Xs[2] - Xs[0])
    return a0 + a1 * (x_next - Xs[0]) + a2 * (x_next - Xs[0]) * (x_next - Xs[1])


def continuation_parameter(Xs, Ys, x_cur, y_cur, step):
    x_next = x_cur + step
    if len(Ys) < 3:
        Ys.append(y_cur)
        Xs.append(x_cur)
    else:
        Ys[0], Ys[1], Ys[2] = Ys[1], Ys[2], y_cur
        Xs[0], Xs[1], Xs[2] = Xs[1], Xs[2], x_cur
    y_next = interpolation(Xs, Ys, x_next)
    return y_next


def interpolate_wb(ps, wbs, p_cur, wb, step):
    if len(wbs) < 3:
        wbs.append(wb)
        ps.append(p_cur)
    else:
        wbs[0], wbs[1], wbs[2] = wbs[1], wbs[2], wb.copy()
        ps[0], ps[1], ps[2] = ps[1], ps[2], p_cur

    for t in range(len(wb)):
        if len(wbs) == 1:
            cur_ws = [wbs[0][t][0]]
            cur_bs = [wbs[0][t][1]]
        if len(wbs) == 2:
            cur_ws = [wbs[0][t][0], wbs[1][t][0]]
            cur_bs = [wbs[0][t][1], wbs[1][t][1]]
        if len(wbs) == 3:
            cur_ws = [wbs[0][t][0], wbs[1][t][0], wbs[2][t][0]]
            cur_bs = [wbs[0][t][1], wbs[1][t][1], wbs[2][t][1]]
        wb[t][0] = interpolation(ps, cur_ws, p_cur + step)
        wb[t][1] = interpolation(ps, cur_bs, p_cur + step)
    return wb


def euler(time_steps, y0, system, params):
    ys = [y0]
    for t in range(len(time_steps) - 1):
        dt = time_steps[t + 1] - time_steps[t]
        t0 = time_steps[t]
        y0 = y0 + dt * system(t0, y0, params)
        ys.append(y0)
        ys[t + 1] = y0
    return ys


def runge_kutta(time_steps, y0, system, params):
    ys = [y0]
    for t in range(len(time_steps) - 1):
        dt = time_steps[t + 1] - time_steps[t]
        t0 = time_steps[t]
        t1 = time_steps[t + 1]
        k1 = system(t0, y0, params)
        k2 = system(t0 / 2, y0 + dt / 2 * k1, params)
        k3 = system(t0 / 2, y0 + dt / 2 * k2, params)
        k4 = system(t1, y0 + dt * k3, params)
        y0 = y0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        ys.append(y0)
        ys[t + 1] = y0
    return ys


def shooting(time_steps, y_approx, system, params, bc, bc_params, solver):
    eps = 10 ** (-4)
    t_left = time_steps[len(time_steps) // 2::-1]
    t_right = time_steps[len(time_steps) // 2:]
    newton_steps = 0
    F = np.zeros(len(y_approx) * len(y_approx)).reshape(len(y_approx), len(y_approx))
    while (True):
        ys = np.concatenate((solver(t_left, y_approx, system, params)[::-1],
                             solver(t_right, y_approx, system, params)[1:]))
        rs = bc(bc_params, ys)
        if (np.abs(rs) < eps).all():
            break

        F = np.zeros(len(y_approx) * len(y_approx)).reshape(len(y_approx), len(y_approx))
        for i in range(len(y_approx)):
            yi_approx = y_approx.copy()
            yi_approx[i] += eps

            yis = np.concatenate((solver(t_left, yi_approx, system, params)[::-1],
                                  solver(t_right, yi_approx, system, params)[1:]))
            rsi = bc(bc_params, yis)

            columni = (rsi - rs) / eps
            for j in range(len(F)):
                F[j][i] = columni[j]
        newton_steps += 1
        y_approx = y_approx - np.dot(inv(F), rs)
    ys = np.concatenate((solver(t_left, y_approx, system, params)[::-1],
                         solver(t_right, y_approx, system, params)[1:]))
    return newton_steps, ys, det(F), svd(F)[1]

def make_filename(params, cur_param):
    filename = f"MODEL~{cur_param}~"
    for k, i in params.items():
        filename += (str(k) + "=" + str(np.round(i, 5)) + ";")
    return filename


def make_fieldnames(params, n_sv):
    fieldnames = list(params.keys())
    fieldnames.append('det')
    fieldnames.extend([f'sv{i}' for i in range(n_sv)])
    return fieldnames


def make_info(fieldnames, model, det, sv):
    data = [*model.HP.values(), det, *sv]
    info = {}
    for i in range(len(fieldnames)):
        info[fieldnames[i]] = data[i]
    return info


def plot_model(model, parameter, z0, zs):
    z = np.array(model.Z).T
    a = np.array(model.A).T
    colors = ['red', 'green']
    t = np.linspace(0, model.HP['T'], len(z[0]))
    plt.title(f"Z  {parameter} = {model.HP[parameter]}")
    for i in range(len(z)):
        plt.plot(t, z[i], label=f"Z{i}", color=colors[i])
        plt.scatter([0, t[-1]], [z0[i], zs[i]], color=colors[i])
    plt.show()
    plt.title(f"A  {parameter} = {model.HP[parameter]}")
    for i in range(len(a)):
        plt.plot(t, a[i], label=f"A{i}")
    plt.show()


def plot_wb(model, parameter):
    t = np.linspace(0, model.HP['T'], len(model.Z))
    ws = [[] for i in range(model.HP['N'] ** 2)]
    bs = [[] for i in range(model.HP['N'])]
    for l in model.wb:
        w = l[0]
        b = l[1]
        for i, wi in enumerate(np.nditer(w)):
            ws[i].append(wi)
        for bi in range(len(b)):
            bs[bi].append(b[bi])
    wb = [*ws, *bs]
    for i in range(len(wb)):
        if i < model.HP['N'] ** 2:
            plt.title(f"w{i + 1}~{parameter}={model.HP[parameter]}")
        else:
            title = 'b'
            plt.title(f"b{i - 3}~{parameter}={model.HP[parameter]}")
        if i == 4:
            plt.plot(t, wb[i])
            plt.show()


def gen_model(hyper_parameters):
    nn = ODEnet(hyper_parameters, [hyper_parameters['N'] for i in range(int(HP['T'] * 10) + 1)])
    return nn


def normal_net_training(model, epochs, z0, zs):
    p = epochs // 10
    for epoch in range(epochs):
        if epoch % p == 0 and epoch != 0:
            #             print(f'{epoch}: {nn.Z[-1]}')
            model.plot_z(z0, zs, epoch)
        model.forward(z0)
        model.back_propagation(zs)
    model.forward(z0)


def boundary_net_training(model, precision, za0, z0, zs, parameter):
    epoch = 0
    while True:
        if epoch % 10 == 0 and epoch != 0:
            model.plot_z(z0, zs, epoch, parameter, det)
        newton_steps, det, sv = model.augmented_propagation(za0, z0, zs)
        za0 = np.concatenate((model.Z[len(model.Z) // 2], model.A[len(model.A) // 2]))
        if all(np.abs(model.Z[-1][i] - zs[i]) < precision for i in range(len(zs))):
            model.matrix_analyses(za0, z0, zs)
            break
        epoch += 1
    return newton_steps, det, sv


def bifurcation_analysis(model, parameter, step, until, precision=0.01):
    z0 = np.random.uniform(-1, 1, model.HP['N'])
    zs = np.random.uniform(-1, 1, model.HP['N'])
    za0 = np.random.uniform(-0.5, 0.5, model.HP['N'] * 2)

    zas = []
    ps = []

    fieldnames = make_fieldnames(model.HP, len(za0))
    file = make_filename(model.HP, parameter)
    create_csv(file, fieldnames)
    while True:
        newton_steps, det, sv = boundary_net_training(model, precision, za0, z0, zs, parameter)
        write_csv(file, make_info(fieldnames, model, det, sv))
        za0 = continuation_parameter(ps, zas, model.HP[parameter], za0, step)
        model.update_parameter(parameter, step)
        model.reset_weights()
        if model.HP['E2'] < until or det > 1000:
            break
    plt.show()


def analyse_points(model, precision, parameter, p_values):
    z0 = np.random.uniform(-1, 1, model.HP['N'])
    zs = np.random.uniform(-1, 1, model.HP['N'])
    za0 = np.random.uniform(-0.5, 0.5, model.HP['N'] * 2)
    for i in range(len(p_values)):
        model.HP[parameter] = p_values[i]
        boundary_net_training(model, precision, za0, z0, zs, parameter)


def augmented_frechet_matrix(time_steps, y_approx, system, params, cur_hp, bc, bc_params, solver):
    eps = 10 ** (-4)
    t_left = time_steps[len(time_steps) // 2::-1]
    t_right = time_steps[len(time_steps) // 2:]

    ys = np.concatenate((solver(t_left, y_approx, system, params)[::-1],
                         solver(t_right, y_approx, system, params)[1:]))
    rs = bc(bc_params, ys)

    aug_F = np.zeros(len(y_approx) * len(y_approx) + len(y_approx)).reshape(len(y_approx), len(y_approx) + 1)
    for i in range(len(y_approx)):
        yi_approx = y_approx.copy()
        yi_approx[i] += eps

        yis = np.concatenate((solver(t_left, yi_approx, system, params)[::-1],
                              solver(t_right, yi_approx, system, params)[1:]))
        rsi = bc(bc_params, yis)

        columni = (rsi - rs) / eps
        aug_F[:, i] = columni

    params[0][cur_hp] += eps
    yps = np.concatenate((solver(t_left, y_approx, system, params)[::-1],
                          solver(t_right, y_approx, system, params)[1:]))
    params[0][cur_hp] -= eps
    rsp = bc(bc_params, yps)
    columnp = (rsp - rs) / eps
    aug_F[:, -1] = columnp

    return aug_F


def analyse_point(aug_F):
    dets = []
    for col in range(len(aug_F[0])):
        cols = [i for i in range(len(aug_F[0])) if i != col]
        dets.append(det(aug_F[:, cols]))
    return np.round(dets, 4)









