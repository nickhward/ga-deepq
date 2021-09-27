from mchgenalg import GeneticAlgorithm
import mchgenalg
import numpy as np
import os

timesEvaluated = 0
bestepochs = -1

# First, define function that will be used to evaluate the fitness
def fitness_function(genome):

    global timesEvaluated
    timesEvaluated += 1

    print("Fitness function invoked "+str(timesEvaluated)+" times")

    #setting parameter values using genome
    prioritized_replay_alpha = decode_function(genome[0:10])
    if prioritized_replay_alpha > 1:
        prioritized_replay_alpha = 1
    gamma = decode_function(genome[11:21])
    if gamma > 1:
        gamma = 1
    prioritized_replay_beta0 = decode_function(genome[22:33])
    if prioritized_replay_beta0 > 1:
        prioritized_replay_beta0 = 1
    exploration_fraction = decode_function(genome[34:44])
    if exploration_fraction > 1:
        exploration_fraction = 1
    exploration_final_eps = decode_function(genome[45:55])
    if exploration_final_eps > 0.2:
        exploration_final_eps = 0.2
    exploration_initial_eps = decode_function(genome[56:66])
    if exploration_initial_eps > 1:
        exploration_initial_eps = 1
    elif exploration_initial_eps < 0.6:
        exploration_initial_eps = 0.6
    epochs_default = 50 #80
    env = 'FetchSlide-v1'
    logdir ='/tmp/openaiGA'
    num_cpu = 4

    query = "python3 -m train_with_ga.py --prioritized_replay_alpha="+ str(prioritized_replay_alpha) + " --gamma=" + str(gamma) + " --prioritized_replay_beta0=" + str(prioritized_replay_beta0) + " --exploration_fraction=" + str(exploration_fraction) + " --exploration_final_eps=" + str(exploration_final_eps) + " --exploration_initial_eps=" + str(exploration_initial_eps)

    print(query)
    #calling training to calculate number of epochs required to reach close to maximum success rate
    os.system(query)
    #epochs = train.launch(env, logdir, epochs_default, num_cpu, 0, 'future', 5, 1, polyak, gamma)
    #env, logdir, n_epochs, num_cpu, seed, replay_strategy, policy_save_interval, clip_return   

    file = open('epochs.txt', 'r')

    #one run is expected to converge before epochs_efault
    #if it does not converge, either add condition here, or make number of epochs as dynamic

    epochs = int(file.read())

    if epochs == None:
        epochs = epochs_default

    global bestepochs
    if bestepochs == -1:
        bestepochs = epochs
    if epochs < bestepochs:
        bestepochs = epochs
        with open('BestParameters.txt', 'a') as output:
            output.write("Epochs taken to converge : " + str(bestepochs) + "\n")
            output.write("prioritized_replay_alpha = " + str(prioritized_replay_alpha) + "\n")
            output.write("Gamma = " + str(gamma) + "\n")
            output.write("prioritized_replay_beta0 = " + str(prioritized_replay_beta0) + "\n")
            output.write("exploration_fraction = " + str(exploration_fraction) + "\n")
            output.write("exploration_final_eps = " + str(exploration_final_eps) + "\n")
            output.write("exploration_initial_eps = " + str(exploration_initial_eps) + "\n")
            output.write("\n")
            output.write("=================================================")
            output.write("\n")

    print('EPOCHS taken to converge:' + str(epochs))

    print("Best epochs so far : "+str(bestepochs))
    return 1/epochs

def decode_function(genome_partial):

    prod = 0
    for i,e in reversed(list(enumerate(genome_partial))):
        if e == False:
            prod += 0
        else:
            prod += 2**abs(i-len(genome_partial)+1)
    return prod/1000

# Configure the algorithm:
population_size = 30
genome_length = 66
ga = GeneticAlgorithm(fitness_function)
ga.generate_binary_population(size=population_size, genome_length=genome_length)

# How many pairs of individuals should be picked to mate
ga.number_of_pairs = 5

# Selective pressure from interval [1.0, 2.0]
# the lower value, the less will the fitness play role
ga.selective_pressure = 1.5
ga.mutation_rate = 0.1

# If two parents have the same genotype, ignore them and generate TWO random parents
# This helps preventing premature convergence
ga.allow_random_parent = True # default True
# Use single point crossover instead of uniform crossover
ga.single_point_cross_over = False # default False

# Run 100 iteration of the algorithm
# You can call the method several times and adjust some parameters
# (e.g. number_of_pairs, selective_pressure, mutation_rate,
# allow_random_parent, single_point_cross_over)
ga.run(30) # default 1000

best_genome, best_fitness = ga.get_best_genome()

print("BEST CHROMOSOME IS")
print(best_genome)
print("It's decoded value is")
print("prioritized_replay_alpha = " + str(decode_function(best_genome[0:10])))
print("Gamma = " + str(decode_function(best_genome[11:22])))
print("prioritized_replay_beta0 = " + str(decode_function(best_genome[23:33])))
print("exploration_fraction = " + str(decode_function(best_genome[34:44])))
print("exploration_final_eps = " + str(decode_function(best_genome[45:55])))
print("exploration_initial_eps = " + str(decode_function(best_genome[56:66])))

# If you want, you can have a look at the population:
population = ga.population

# and the fitness of each element:
fitness_vector = ga.get_fitness_vector()