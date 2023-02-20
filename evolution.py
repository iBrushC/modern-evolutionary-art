import multiprocess as mp
import ctypes as c
import numpy as np

# Adam optimizer
class AdamOptimizer(object):
    def __init__(self, size: int, alpha: float=0.01, beta1: float=0.9, beta2: float=0.99, epsilon: float=1e-8, decay: float=1, decay_alpha: bool=True, mode: str="minimize") -> None:
        self.size = size # Size of vector
        self.alpha_initial, self.alpha = alpha, alpha  # Learning rate
        self.beta1_initial, self.beta1 = beta1, beta1  # Learning parameter 1
        self.beta2_initial, self.beta2 = beta2, beta2  # Learning parameter 2
        self.epsilon = epsilon  # Offset to ensure no zero divisions
        self.decay = decay  # Learning decay over time
        self.decay_alpha = decay_alpha # Whether or not to decay the learning rate over time
        
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
        if (self.decay_alpha): self.alpha = self.alpha_initial / (np.sqrt(decay_amount))

        gradient = self.direction * self.alpha * mhat / (np.sqrt(vhat) + self.epsilon)
        return gradient

# An Evolution-Strategies solver roughly based on PGPE
class ESSolver(object):
    def __init__(self, 
                 parameter_count: int,              # How many parameters are in the solution, or how long the solution vector is.
                 population_count: int,             # How many population samples are made in each step. In other words, how many directions are sampled.

                 center: np.ndarray=None,           # The center, or mu. This will normally be initialized with random uniform variables.
                 center_bounds: tuple=(0, 1),       # The bounds of the center, or what each parameter in the center solution vector will be constrained by.
                 center_alpha: float=0.27,          # The learning rate for the center.

                 sigma: np.ndarray=None,            # The standard deviation of the population sample, or how spread out the populatuion will look.
                 sigma_bounds: tuple=(0.1, 0.3),    # The bounds of the standard deviation, or what each standard devation value will be constrained by.
                 sigma_alpha: float=0.2,            # The learning rate of the standard deviation.

                 seed: int=0,                       # The random seed that will be used to calculate all random values
                 
                 optimizer: object=None,            # The optimizer for the gradient descent
                 error_function=None,               # The function used to get the error of each solution in the population
                ) -> None:
        
        self.parameter_count = parameter_count
        self.population_count = population_count
        self.center = center
        self.center_bounds = center_bounds
        self.center_alpha = center_alpha
        self.sigma = sigma
        self.sigma_bounds = sigma_bounds
        self.sigma_alpha = sigma_alpha
        self.optimizer = optimizer
        self.error_function = error_function

        np.random.seed(seed)

    # Randomizes values using normally distributed random variables about zero
    def randomize_center(self, min=0, max=1) -> None:
        self.center = np.random.uniform(low=min, high=max, size=self.parameter_count)

    # Computes the current gradients
    def compute_gradients(self) -> tuple:
        # Two lists need to be used because the baseline for the change of the standard deviation relies on the average of the errors collected over the population
        # To greatly save time, everything is stored in lists so it only has to be computed once.
        epsilon_list = np.zeros(shape=(self.population_count, self.parameter_count))
        r_positive_list = np.zeros(shape=(self.population_count))
        r_negative_list = np.zeros(shape=(self.population_count))

        # Computes the population error
        for i in range(self.population_count):
            epsilon = np.random.normal(loc=0, scale=self.sigma ** 2, size=self.parameter_count)
            epsilon_list[i] = epsilon

            r_positive = self.error_function(self.center + epsilon)
            r_negative = self.error_function(self.center - epsilon)
            r_positive_list[i] = r_positive
            r_negative_list[i] = r_negative

        baseline = (np.mean(r_positive_list) + np.mean(r_negative_list)) / 2

        # Computes the gradients for the center and standard deviation
        center_gradient_total = np.zeros(self.parameter_count)
        sigma_gradient_total = np.zeros(self.parameter_count)

        sigma_squared = self.sigma ** 2

        for i in range(self.population_count):
            epsilon = epsilon_list[i]
            r_positive = r_positive_list[i]
            r_negative = r_negative_list[i]

            center_directionality = (r_positive - r_negative)  # Used to "flip" the direction of the solution vector depending on which end of the mirrored population has a higher value
            center_gradient = epsilon * center_directionality * 0.5
            center_gradient_total += center_gradient

            standard_error = ((epsilon ** 2) - sigma_squared) / self.sigma
            standard_directionality = ((r_positive + r_negative) / 2) - baseline
            sigma_gradient = standard_error * standard_directionality
            sigma_gradient_total += sigma_gradient

        center_gradient_total /= self.population_count
        sigma_gradient_total /= self.population_count

        return (center_gradient_total, sigma_gradient_total)
    
    # Performs gradient descent / ascent over a set number of cycles
    def climb(self, cycles: int=100, log_every: int=100, update_callback=None):
        errors = []
        for i in range(cycles):
            center_gradient, sigma_gradient = self.compute_gradients()

            self.center += self.center_alpha * self.optimizer.optimize(center_gradient, i+1)
            # self.center -= self.center_alpha * center_gradient
            self.center = np.clip(self.center, self.center_bounds[0], self.center_bounds[1])

            self.sigma += self.sigma_alpha * self.optimizer.optimize(sigma_gradient, i+1)
            # self.sigma -= self.sigma_alpha * sigma_gradient
            self.sigma = np.clip(self.sigma, self.sigma_bounds[0], self.sigma_bounds[1])

            error = self.error_function(self.center)
            errors.append(error)

            if (i % log_every == 0):
                print(f"Error of cycle {i}: {error}")

            if (update_callback is not None):
                update_callback(self.center)

        return errors
    
