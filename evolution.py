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
    def randomize_center(self, low=0, high=1) -> None:
        self.center = np.random.uniform(low=low, high=high, size=self.parameter_count)

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
    # Offset is needed so that there isn't a massive spike when switching between modes of ascent
    def climb(self, cycles: int=100, offset: int=0, log_every: int=100):
        errors = []
        for i in range(offset, cycles + offset):
            center_gradient, sigma_gradient = self.compute_gradients()

            self.center += self.center_alpha * self.optimizer.optimize(center_gradient, i+1)
            # self.center -= self.center_alpha * center_gradient * 10000
            self.center = np.clip(self.center, self.center_bounds[0], self.center_bounds[1])

            self.sigma += self.sigma_alpha * self.optimizer.optimize(sigma_gradient, i+1)
            # self.sigma -= self.sigma_alpha * sigma_gradient * 10000
            self.sigma = np.clip(self.sigma, self.sigma_bounds[0], self.sigma_bounds[1])

            error = self.error_function(self.center)
            errors.append(error)

            if ((i+1) % log_every == 0):
                print(f"Error of cycle {i+1}: {error}")

        return errors
    
# Genetic algorithm loose implimentation
class GeneticSolver(object):
    def __init__(
        self, 
        parameter_count: int, 
        population_count: int, 

        population_bounds: tuple=(0, 1),

        mutation_probability: float=0.1,
        mutation_strength: float=0.5,
        survival_amount: float=0.25,
        decay: float=0.5,
        use_crossbreeding: bool=False,

        error_function=None,
        mode: str="minimize",
        seed: int=0,
    ):
        self.parameter_count = parameter_count
        self.population_count = population_count
        self.population = np.zeros(shape=(population_count, parameter_count))
        self.best = np.zeros(parameter_count)
        self.population_bounds = population_bounds
        self.mutation_probability = mutation_probability
        self.mutation_strength_initial, self.mutation_strength = mutation_strength, mutation_strength
        self.survivor_count = int(survival_amount * population_count)
        self.use_crossbreeding = use_crossbreeding
        self.error_function = error_function
        
        if mode == "minimize":
            self.reverse = False
        elif mode == "maximize":
            self.reverse = True
        else:
            raise ValueError(f"Unknown mode {mode}", "mode parameter can only be 'minimize' or 'maximize'")

        np.random.seed(seed)

    def randomize_population(self, low: float=0, high: float=1):
        self.population = np.random.uniform(low=low, high=high, size=(self.population_count, self.parameter_count))

    def cull_fill_population(self):
        scored_population = [(self.error_function(gene), gene) for gene in self.population]
        scored_population.sort(reverse=self.reverse, key=lambda x: x[0])
        self.best = scored_population[0][1]
        survivors = [ranked_gene[1] for ranked_gene in scored_population[:self.survivor_count]]
        new_population = [*survivors]

        for i in range(self.population_count - self.survivor_count):
            mutations = (
                (np.random.rand(self.parameter_count) < self.mutation_probability) *
                ((np.random.rand(self.parameter_count) - 0.5) * self.mutation_strength * 2)
            )
            if (self.use_crossbreeding):
                parent1 = survivors[np.random.randint(low=0, high=self.survivor_count)]
                parent2 = survivors[np.random.randint(low=0, high=self.survivor_count)]
                inheritance = (np.random.rand(self.parameter_count) < 0.5)
                new_gene = (parent1 * inheritance) + (parent2 * (1 - inheritance))
                new_gene += mutations
                new_gene = np.clip(new_gene, a_min=self.population_bounds[0], a_max=self.population_bounds[1])
                new_population.append(new_gene)
            else:
                parent = survivors[np.random.randint(low=0, high=self.survivor_count)]
                new_gene = parent + mutations
                new_gene = np.clip(new_gene, a_min=self.population_bounds[0], a_max=self.population_bounds[1])
                new_population.append(new_gene)
        
        self.population = new_population.copy()

        return scored_population[0][0]

    def climb(self, cycles: int=100, log_every: int=10):
        errors = []
        for i in range(cycles):
            error = self.cull_fill_population()
            errors.append(error)

            if ((i+1) % log_every == 0):
                print(f"Error of cycle {i+1}: {error}")

        return errors