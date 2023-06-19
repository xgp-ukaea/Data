
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
from numpy import array, zeros, argsort, where, isfinite
from numpy import sqrt, exp, dot
from numpy import load, ceil, savez
from numpy.random import normal, random, choice, randint, seed
from scipy.optimize import fmin_l_bfgs_b
from multiprocessing import Process, Pipe, Event
from itertools import chain

from copy import copy
from time import time




class LogWrapper(object):
    def __init__(self, func):
        self.func = func

    def __call__(self, theta):
        return self.func(exp(theta))

    def gradient(self, theta):
        p = exp(theta)
        return p*self.func.gradient(p)

    def cost(self, theta):
        return -self.func(exp(theta))

    def cost_gradient(self, theta):
        p = exp(theta)
        return -p*self.func.gradient(p)




def bfgs_worker(posterior, bounds, maxiter, connection, end):
    # main loop persists until shutdown event is triggered
    while not end.is_set():
        # poll the pipe until there is something to read
        while not end.is_set():
            if connection.poll(timeout = 0.05):
                theta_list = connection.recv()
                break

        # if read loop was broken because of shutdown event
        # then break the main loop as well
        if end.is_set(): break

        # run L-BFGS-B for each starting position in theta_list
        results = []
        for theta0 in theta_list:
            if hasattr(posterior, 'cost_gradient'):
                x_opt, fmin, D = fmin_l_bfgs_b(posterior.cost, theta0, fprime=posterior.cost_gradient, bounds = bounds, maxiter=maxiter)
            else:
                x_opt, fmin, D = fmin_l_bfgs_b(posterior.cost, theta0, bounds = bounds, approx_grad = True)

            # store results in a dictionary
            result = {
                'probability': -fmin,
                'solution' : x_opt,
                'flag' : D['warnflag'],
                'evaluations' : D['funcalls'],
                'iterations' : D['nit']
            }

            results.append(result)

        # send the results
        connection.send(results)



class EGAResult(object):
    def __init__(self):
        # storage for the settings
        self.generations = None
        self.population_size = None
        self.threads_used = None
        self.perturbation_size = None
        self.mutation_probability = None

        # storage for the results
        self.optimal_theta = None
        self.max_log_prob = None

        # storage for convergence history
        self.log_prob_history = None
        self.solution_history = None
        self.optimisation_times = None
        self.evolution_times = None
        self.generation_times = None
        self.population_log_probs = None

        # BGFS information
        self.iteration_history = None
        self.evaluation_history = None
        self.flag_history = None

    def save(self, filename):
        savez(filename,
              generations = self.generations,
              population_size = self.population_size,
              threads_used = self.threads_used,
              perturbation_size = self.perturbation_size,
              mutation_probability = self.mutation_probability,
              optimal_theta = self.optimal_theta,
              max_log_prob = self.max_log_prob,
              log_prob_history = self.log_prob_history,
              solution_history = self.solution_history,
              optimisation_times = self.optimisation_times,
              evolution_times = self.evolution_times,
              generation_times = self.generation_times,
              population_log_probs = self.population_log_probs,
              iteration_history = self.iteration_history,
              evaluation_history = self.evaluation_history,
              flag_history = self.flag_history)

    def __str__(self):
        m, s = divmod(sum(self.generation_times), 60)
        h, m = divmod(m, 60)
        total_time = "%d:%02d:%02d" % (h, m, s)
        info = '\n'
        info += ' # # # [ Evolutionary gradient ascent results ] # # # \n'
        info += ' #  max log probability: {} \n'.format(self.max_log_prob)
        info += ' #   total time elapsed: {} \n'.format(total_time)
        info += ' #    total generations: {} \n'.format(self.generations)
        info += ' #      population size: {} \n'.format(self.population_size)
        info += ' #    perturbation size: {} \n'.format(self.perturbation_size)
        info += ' # mutation probability: {} \n'.format(self.mutation_probability)
        return info




def evolutionary_gradient_ascent(posterior=None, initial_population = None, generations=20, threads=1,
                                 perturbation=0.075, operator_probs = None, bounds = None, maxiter = 1000):

    # initialise all the processes
    shutdown_evt = Event()
    processes = []
    connections = []
    for i in range(threads):
        parent_ctn, child_ctn = Pipe()
        p = Process(target=bfgs_worker, args=(posterior, bounds, maxiter, child_ctn, shutdown_evt))
        p.start()
        processes.append(p)
        connections.append(parent_ctn)

    # set up the genetic algorithm
    pop = Population(n_params = len(initial_population[0]),
                     n_pop = len(initial_population),
                     perturbation = perturbation,
                     operator_probs = operator_probs,
                     bounds = bounds)

    # initialise the first generation
    pop.children = initial_population

    # create command-line arguments by dividing up population between threads
    batchsize = int(ceil(pop.popsize / threads))
    L = [i for i in range(pop.popsize)]
    index_groups = [L[i:i + batchsize] for i in range(0, pop.popsize, batchsize)]


    times = []
    iteration_history = []
    evaluation_history = []
    flag_history = []
    for iteration in range(generations):
        # chop up the population for each thread
        theta_groups = [[pop.children[i] for i in inds] for inds in index_groups]

        t1 = time()
        # send the starting positions to each process
        for starts, ctn in zip(theta_groups, connections):
            ctn.send(starts)

        # wait until results from all processes are received
        results = [ ctn.recv() for ctn in connections ]
        # join the results from each process into one list
        results = [ r for r in chain(*results) ]
        t2 = time()

        # unpack the results
        solutions = [ D['solution'] for D in results ]
        probabilities = [ D['probability'] for D in results ]
        gen_iterations = [ D['iterations'] for D in results ]
        gen_evaluations = [ D['evaluations'] for D in results ]
        gen_flags = [ D['flag'] for D in results ]

        iteration_history.append(gen_iterations)
        evaluation_history.append(gen_evaluations)
        flag_history.append(gen_flags)

        pop.give_fitness(probabilities, solutions)
        t3 = time()

        times.append((t1,t2,t3))
        msg =  '\n # generation ' + str(iteration)
        msg += '\n # best prob: ' + str(pop.elite_fitnesses[-1])
        m, s = divmod(t3 - t1, 60)
        h, m = divmod(m, 60)
        time_left = "%d:%02d:%02d" % (h, m, s)
        msg += '\n # time taken this gen: {} ({:1.1f}% in ascent)'.format( time_left, 100*(t2-t1)/(t3-t1))
        print(msg)

    # trigger the shutdown event and terminate the processes
    shutdown_evt.set()
    for p in processes: p.join()

    # build the results dictionary:
    result = {
        'generations' : generations,
        'population_size' : pop.popsize,
        'threads_used' : threads,
        'perturbation_size' : perturbation,
        'operator_probabilities' : pop.operator_probs,
        'optimal_theta' : pop.elite_adults[-1],
        'max_log_prob' : pop.elite_fitnesses[-1],
        'log_prob_history' : pop.best_fitness_history,
        'solution_history' : pop.best_adult_history,
        'optimisation_times' : [ b-a for a,b,c in times ],
        'evolution_times' : [ c-b for a,b,c in times ],
        'generation_times' : [ c-a for a,b,c in times ],
        'population_log_probs' : pop.fitnesses,
        'iteration_history' : iteration_history,
        'evaluation_history' : evaluation_history,
        'flag_history' : flag_history
    }
    return result




def unzip(v):
    return [[x[i] for x in v] for i in range(len(v[0]))]

class Population(object):
    """
    Population management and generation class for evolutionary algorithms.

    :param initial_population:
    :param initial_fitnesses:
    :param scale_lengths:
    :param perturbation:
    """
    def __init__(self, n_params = None, n_pop = 20, scale_lengths = None, perturbation = 0.1,
                 operator_probs = (0.35,0.4,0.25), bounds = None, record_ancestors = True):

        if scale_lengths is None:
            self.L = 1
        else:
            self.L = array(scale_lengths)

        # unpack bounds into two arrays
        if bounds is not None:
            self.lwr_bounds = array([b[0] for b in bounds])
            self.upr_bounds = array([b[1] for b in bounds])
        else:
            self.lwr_bounds = None
            self.upr_bounds = None

        if operator_probs is None:
            self.operator_probs = array([0.35,0.4,0.25])
        else:
            self.operator_probs = array(operator_probs)

        # settings
        self.n = n_params
        self.popsize = n_pop
        self.p = 6./self.popsize
        self.mutation_str = perturbation
        self.fitness_only = False
        self.record_ancestors = record_ancestors

        # storage
        self.fitnesses = []
        self.ancestors = []
        self.children = []

        self.n_elites = 3
        self.elite_adults = []
        self.elite_fitnesses = []

        self.breeding_adults = []
        self.breeding_fitnesses = []

        self.best_adult_history = []
        self.best_fitness_history = []

    def get_genes(self):
        return self.adults

    def give_fitness(self, fitnesses, adults):
        # include elites in population
        adults.extend(self.elite_adults)
        fitnesses.extend(self.elite_fitnesses)

        # sort the population by fitness
        fitnesses, adults = unzip(sorted(zip(fitnesses, adults), key=lambda x : x[0]))

        # update elites
        self.elite_adults = adults[-self.n_elites:]
        self.elite_fitnesses = fitnesses[-self.n_elites:]

        # results of this generation
        self.best_adult_history.append(self.elite_adults[-1])
        self.best_fitness_history.append(self.elite_fitnesses[-1])
        self.fitnesses.append(fitnesses)
        if self.record_ancestors: self.ancestors.append(copy(adults))

        # specify the breeding population
        self.adults = adults
        self.adult_ranks = self.rank_prob(fitnesses)

        # breed until population is of correct size
        self.children.clear()
        while len(self.children) < self.popsize:
            self.breed()

    def breed(self):
        # pick one of the two breeding methods
        operator = choice([self.crossover, self.merge, self.mutation], p = self.operator_probs)
        operator()

    def rank_prob(self, data):
        ranks = argsort(-1*array(data))
        probs = array([ self.p*(1 - self.p)**r for r in ranks ])
        return probs / probs.sum()

    def mutation(self):
        if hasattr(self.mutation_str, '__len__' ):
            strength = choice(self.mutation_str)
        else:
            strength = self.mutation_str

        child = self.select_member() + (strength * self.L) * normal(size=self.n)

        # apply bounds if they are specified
        if self.lwr_bounds is not None:
            lwr_bools = child < self.lwr_bounds
            upr_bools = child > self.lwr_bounds

            if any(lwr_bools):
                lwr_inds = where(lwr_bools)
                child[lwr_inds] = self.lwr_bounds[lwr_inds]

            if any(upr_bools):
                upr_inds = where(upr_bools)
                child[upr_inds] = self.lwr_bounds[upr_inds]
        self.children.append(child)

    def crossover(self):
        g1, g2 = self.select_pair()

        # create a random ordered list containing
        booles = zeros(self.n)
        booles[0::2] = True
        booles[1::2] = False
        booles = booles[ argsort(random(size=self.n)) ]

        child = g1.copy()
        inds = where(booles)
        child[inds] = g2[inds]

        self.children.append(child)

    def merge(self):
        g1, g2 = self.select_pair()
        w = 0.1 + random()*0.8
        child = g1*w + (1-w)*g2
        self.children.append(child)

    def select_member(self):
        weights = self.get_weights()
        i = choice(len(self.adults), p = weights)
        return self.adults[i]

    def select_pair(self):
        weights = self.get_weights()
        i, j = choice(len(self.adults), p=weights, size=2, replace=False)
        return self.adults[i], self.adults[j]

    def get_weights(self):
        if (len(self.children) is 0) or self.fitness_only:
            weights = self.adult_ranks
        else:
            weights = self.adult_ranks * self.rank_prob(self.get_diversity())
            weights /= weights.sum()
        return weights

    def get_diversity(self):
        div = []
        for adult in self.adults:
            dist = sum([ self.distance(adult, child) for child in self.children ])
            div.append(dist)
        return div

    def distance(self, g1, g2):
        z = (array(g1) - array(g2)) / self.L
        return sqrt(dot(z,z))




class DiffEv(object):
    def __init__(self, posterior = None, initial_population = None, initial_probabilities = None,
                 differential_weight = 0.3, crossover_prob = 0.85, show_status = True):

        # settings
        self.F = differential_weight
        self.CR = crossover_prob
        self.max_tries = 200
        self.show_status = show_status

        self.N = len(initial_population)
        self.L = len(initial_population[0])

        self.posterior = posterior
        self.population = initial_population
        self.probabilities = initial_probabilities
        self.max_prob = max(self.probabilities)

    def run_for(self, seconds = 0, minutes = 0, hours = 0, days = 0):
        # first find the runtime in seconds:
        run_time = ((days*24. + hours)*60. + minutes)*60. + seconds
        start_time = time()
        end_time = start_time + run_time

        while time() < end_time:
            self.take_step()

            # display the progress status message
            if self.show_status:
                seconds_remaining = end_time - time()
                m, s = divmod(seconds_remaining, 60)
                h, m = divmod(m, 60)
                time_left = "%d:%02d:%02d" % (h, m, s)
                msg = '\r   [ best probability: {}, time remaining: {} ]'.format(self.max_prob, time_left)
                sys.stdout.write(msg)
                sys.stdout.flush()

        # this is a little ugly...
        if self.show_status:
            sys.stdout.write('\r   [ best probability: {}, run complete ]           '.format(self.max_prob))
            sys.stdout.flush()
            sys.stdout.write('\n')

    def take_step(self):
        for i in range(self.max_tries):
            flag, P = self.breed()
            if flag:
                if P > self.max_prob: self.max_prob = P
                break

    def breed(self):
        ix, ia, ib, ic = self.pick_indices()
        X, A, B, C = self.population[ix], self.population[ia], self.population[ib], self.population[ic]
        inds = where(random(size=self.L) < self.CR)
        Y = copy(X)
        Y[inds] = A[inds] + self.F*(B[inds] - C[inds])
        P_Y = self.posterior(Y)

        if P_Y > self.probabilities[ix]:
            self.population[ix] = Y
            self.probabilities[ix] = P_Y
            return True, P_Y
        else:
            return False, P_Y

    def pick_indices(self):
        return choice(self.N, 4, replace=False)

    def get_dict(self):
        items = [
            ('F', self.F),
            ('CR', self.CR),
            ('population', self.population),
            ('probabilities', self.probabilities),
            ('max_prob', self.max_prob) ]

        D = {} # build the dict
        for key, value in items:
            D[key] = value
        return D

    def save(self, filename):
        D = self.get_dict()
        savez(filename, **D)

    @classmethod
    def load(cls, filename, posterior = None):
        data = load(filename)
        pop = data['population']
        C = cls(posterior = posterior,
                initial_population = [ pop[i,:] for i in range(pop.shape[0])],
                initial_probabilities = data['probabilities'],
                differential_weight = data['F'],
                crossover_prob = data['CR'])
        return C




def diff_evo_worker(proc_seed, initial_population, initial_probabilities, posterior, hyperpars, runtime, pipe):
    seed(proc_seed)
    DE = DiffEv(posterior = posterior,
                initial_population = initial_population,
                initial_probabilities = initial_probabilities,
                differential_weight = hyperpars[0],
                crossover_prob = hyperpars[1],
                show_status = False)

    DE.run_for(seconds = runtime)
    max_ind = array(DE.probabilities).argmax()
    pipe.send( [DE.probabilities[max_ind], DE.population[max_ind]] )




def parallel_evolution(initial_population = None, popsize = 30, posterior = None, threads = None, runtime = None):

    L = len(initial_population)
    if L < popsize: popsize = L
    initial_probabilities = [ posterior(t) for t in initial_population ]

    # randomly sample the hyperparameters
    cross_probs = random(size = threads)*0.5 + 0.5
    diff_weights = random(size = threads)*1.2 + 0.1
    hyperpars = [ k for k in zip(cross_probs, diff_weights) ]

    # generate random seeds for each process
    seeds = randint(3, high=100, size=threads).cumsum()

    # loop to generate the separate processes
    processes = []
    connections = []
    for proc_seed, hyprs in zip(seeds, hyperpars):
        # create a pipe to communicate with the process
        parent_ctn, child_ctn = Pipe()
        connections.append(parent_ctn)

        # sub-sample the population
        inds = choice(L, popsize, replace=False)
        proc_pop = [initial_population[k] for k in inds]
        proc_probs = [initial_probabilities[k] for k in inds]

        # collect all the arguments
        args = [ proc_seed, proc_pop, proc_probs,
                 posterior, hyprs, runtime, child_ctn ]

        # create the process
        p = Process(target = diff_evo_worker, args = args)
        p.start()
        processes.append(p)

    # wait for results to come in from each process
    results = [ c.recv() for c in connections ]
    # terminate the processes now we have the results
    for p in processes: p.join()
    # unpack the results
    probabilities, parameters = unzip(results)

    return parameters, probabilities, cross_probs, diff_weights