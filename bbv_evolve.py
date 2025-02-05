import random
from deap import base, creator, tools
import matplotlib.pyplot as plt
import multiprocessing
from timeit import default_timer as timer
import datetime
import pickle


from utils import *

# DEAP
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

def prior():
    return list((np.random.rand(1))) + list((-np.random.rand(1))) + list((np.random.rand(10)))

# Structure initializers
toolbox.register("individual", tools.initIterate,  creator.Individual, prior)

# define the population to be a list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

#----------
# Operator registration
#----------

# register the goal / fitness function
toolbox.register("evaluate", evaluate)

# register the crossover operator
toolbox.register("mate", tools.cxTwoPoint)

# register a mutation operator with a probability to
# flip each attribute/gene of 0.25
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.25, indpb=0.25)

# Function to put constraints on parameters
def checkBounds():
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                for i in [0, 2, 8]:
                    if child[i]< 0:
                        child[i] = 0
                for i in [1, 9, 10]:
                    if child[i] > 0:
                        child[i] = 0
                for i in [4, 6]:
                    if child[i] < 0.1:
                        child[i] = 0.1
            return offspring
        return wrapper
    return decorator

# Decorate functions with constraints
toolbox.decorate("mutate", checkBounds())
toolbox.decorate("mate", checkBounds())

# operator for selecting individuals for breeding the next generation
toolbox.register("select", tools.selNSGA2_spdr)

def main(tot_pop, num_gens, checkpoint):
    if checkpoint:
        with open('checkpoint.pkl', "rb") as cp_file:
            cp = pickle.load(cp_file)
        logbook = cp['logbook']
        pop = cp['population']
        gen = cp['generation']
        pareto = cp["pareto"]
        hof = cp["hof"]
        random.setstate(cp["rndstate"])
        initial_state = cp['init_state']
        print("Continuing evolution from generation", gen)
    else:
        pop = toolbox.population(n=tot_pop)
        gen = 0
        pareto = tools.ParetoFront()
        logbook = tools.Logbook()
        hof = tools.HallOfFame(2*tot_pop*num_gens)
        initial_state = random.getstate()
        print("Start of evolution")


    # statistics
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("med", np.median, axis = 0)
    stats.register("med_std", median_abs_deviation, axis=0)
    stats.register("med_std_sq", median_abs_deviation2)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    stats2 = tools.Statistics(key=lambda ind: ind)
    stats2.register("avg2", np.mean, axis=0)
    stats2.register("std2", np.std, axis=0)
    stats2.register("med2", np.median, axis = 0)
    stats2.register("med_std2", median_abs_deviation, axis=0)
    stats2.register("min2", np.min, axis=0)
    stats2.register("max2", np.max, axis=0)

    # CXPB  is the probability with which two individuals are crossed
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = 0.5, 0.5

    # Evaluate the entire population
    jobs = toolbox.map(toolbox.evaluate, pop)
    fitnesses = jobs.get()
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
        
    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, tot_pop)
    
    print("  Evaluated %i individuals" % len(pop))

    # Variable keeping track of the number of generations
    FREQ = 10

    # Begin the evolution
    while gen < num_gens:
        # record statistics
        record = stats.compile(pop)
        record2 = stats2.compile(pop)
        logbook.record(gen=gen, **record)
        logbook.record(gen=gen, **record2)

        # A new generation
        gen = gen + 1
        print("-- Generation %i --" % gen)

        # Select the next generation individuals
        offspring = tools.selTournamentDCD(unique_check(pop), tot_pop)
        
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        jobs = toolbox.map(toolbox.evaluate, invalid_ind)
        fitnesses = jobs.get()
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print("  Evaluated %i individuals" % len(invalid_ind))

        # The population is entirely replaced by the offspring
        pareto.update(pop+offspring)
        hof.update(pop+offspring)
        pop = toolbox.select(unique_check(pop+offspring),tot_pop)

        if gen % FREQ == 0:
        # Fill the dictionary using the dict(key=value[, ...]) constructor
            cp = dict(population=pop, generation=gen, rndstate=random.getstate(), pareto = pareto, hof = hof, logbook = logbook, init_state = initial_state)
            with open("checkpoint.pkl", "wb") as cp_file:
                pickle.dump(cp, cp_file)

    print("-- End of (successful) evolution --")

    # Store best ones at the end
    notes = "Jan 16, 2023: (aushra) param stats, optimize bin1, adjust wC across 3, abs, one sided sum"
    cp = dict(population=pop, generation=gen, rndstate=random.getstate(), pareto = pareto, hof = hof, logbook = logbook, init_state = initial_state, message = notes)
    with open("checkpoint.pkl", "wb") as cp_file:
        pickle.dump(cp, cp_file)

    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

if __name__ == '__main__':
    args = parse_commandline()
    print("Number of cpus Python thinks you have : ", multiprocessing.cpu_count())
    pool = multiprocessing.Pool(args.nproc)

    toolbox.register('map', pool.map_async)
    tic = timer()
    main(tot_pop=112, num_gens=500, checkpoint=False)
    pool.close()
    print(str(datetime.timedelta(seconds=int(timer()-tic))))