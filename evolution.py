import numpy as np
import matplotlib.pyplot as plt

# Adam optimizer
class AdamOptimizer(object):
    def __init__(self, size: int, alpha: float=0.01, beta1: float=0.9, beta2: float=0.99, epsilon: float=1e-8, decay: float=1, mode: str="minimize") -> None:
        self.size = size # Size of vector
        self.alpha_initial, self.alpha = alpha, alpha  # Learning rate
        self.beta1_initial, self.beta1 = beta1, beta1  # Learning parameter 1
        self.beta2_initial, self.beta2 = beta2, beta2  # Learning parameter 2
        self.epsilon = epsilon  # Offset to ensure no zero divisions
        self.decay = decay  # Learning decay over time
        
        # Optimize for maximum or minimum value
        if mode == "minimize":
            self.direction = -1
        elif mode == "maximize":
            self.direction = 1
        else:
            raise ValueError(f"Unknown mode {mode}", "mode parameter can only be 'minimize' or 'maximize'")
        
        # Continuous parameters
        self.m = np.zeros(shape=(size))
        self.v = np.zeros(shape=(size))
    
    def optimize(self, gradient, t) -> np.ndarray:
        # Create gradient parameters
        self.m = self.beta1*self.m + (1 - self.beta1)*gradient
        self.v = self.beta2*self.v + (1 - self.beta2)*(gradient ** 2)
        mhat = self.m / (1 - self.beta1)
        vhat = self.v / (1 - self.beta2)

        # Apply decay
        decay_amount = self.decay * t
        self.beta1 = self.beta1_initial ** (decay_amount)
        self.beta2 = self.beta2_initial ** (decay_amount)
        self.alpha = self.alpha_initial / np.sqrt(decay_amount)

        return self.direction * self.alpha * mhat / (np.sqrt(vhat) + self.epsilon)
    
# An Evolution-Strategies solver roughly based on PGPE
class ESSolver(object):
    def __init__(self, parameter_count: int, alpha: float=0.27, alpha_deviation: float=0.2, seed: int=0) -> None:
        pass

if __name__ == "__main__":
    def f(x): return x**6 + x**5 - x**4 - 3*x
    def fp(x): return 6*x**5 + 5*x**4 - 4*x**3 - 3

    adam = AdamOptimizer(
        size=1,
        alpha=0.3,
        beta1=0.9,
        beta2=0.99,
        epsilon=1e-8,
        decay=1.3,
        mode="minimize"
    )

    x = -1.5

    xt = []
    fxt = []

    for i in range(1, 100):
        shift = adam.optimize(fp(x), i)
        x += shift
        xt.append(x + 0)
        fxt.append(f(x))

    plt.plot(xt)
    plt.plot(fxt)
    plt.legend(["x", "f(x)"])
    plt.show()