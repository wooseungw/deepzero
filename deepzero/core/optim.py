class Optimizer:
    def step(self):
        pass

    def zero_grad(self):
        pass


class SGD(Optimizer):
    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr

    def step(self):
        for param in self.params:
            param.data -= self.lr * param.grad.data

    def zero_grad(self):
        for param in self.params:
            param.grad.data = 0


class Adam(Optimizer):
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = [0] * len(params)
        self.v = [0] * len(params)
        self.t = 0

    def step(self):
        self.t += 1
        for i, param in enumerate(self.params):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * param.grad.data
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (param.grad.data ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            param.data -= self.lr * m_hat / (v_hat ** 0.5 + self.epsilon)

    def zero_grad(self):
        for param in self.params:
            param.grad.data = 0