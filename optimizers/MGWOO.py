import random
import numpy
import math
from solution import solution
import time
from math import cos, pi,sin


def git(i, t):
    return cos(i*t/2) * sin(i*t)

def pit(i, t):
    return sin(i*t/2) * cos(i*t)

def rv():
    return 2 * random.random() - 1

def TF(l, Max_iter):
    t = l/Max_iter
    return 1 - t

def MGWO(objf, lb, ub, dim, SearchAgents_no, Max_iter,trainInput,trainOutput,net):
    SearchAgents_no = 20
    Alpha_pos = numpy.zeros(dim)
    Alpha_score = float("inf")
    Beta_pos = numpy.zeros(dim)
    Beta_score = float("inf")
    Delta_pos = numpy.zeros(dim)
    Delta_score = float("inf")

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    Positions = numpy.zeros((SearchAgents_no, dim))
    for i in range(dim):
        Positions[:, i] = (
            numpy.random.uniform(0, 1, SearchAgents_no) * (ub[i] - lb[i]) + lb[i]
        )

    Convergence_curve = numpy.zeros(Max_iter)
    s = solution()

    print('MGWO is optimizing "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    for l in range(0, Max_iter):
        for i in range(0, SearchAgents_no):
            for j in range(dim):
                Positions[i, j] = numpy.clip(Positions[i, j], lb[j], ub[j])
            fitness = objf(Positions[i, :], trainInput, trainOutput, net)

            # Update Alpha, Beta, and Delta
            if fitness < Alpha_score:
                Delta_score = Beta_score
                Delta_pos = Beta_pos.copy()
                Beta_score = Alpha_score
                Beta_pos = Alpha_pos.copy()
                Alpha_score = fitness
                Alpha_pos = Positions[i, :].copy()

            elif fitness > Alpha_score and fitness < Beta_score:
                Delta_score = Beta_score
                Delta_pos = Beta_pos.copy()
                Beta_score = fitness
                Beta_pos = Positions[i, :].copy()

            elif fitness > Alpha_score and fitness > Beta_score and fitness < Delta_score:
                Delta_score = fitness
                Delta_pos = Positions[i, :].copy()

        a = 2 - l * ((2) / Max_iter)

        # Update the position of agents
        for i in range(0, SearchAgents_no):
            new_position = Positions[i,:].copy()
            for j in range(0, dim):

                # GWO Update Rule
                A1 = 2 * a * random.random() - a
                C1 = 2 * random.random()
                D_alpha = abs(C1 * Alpha_pos[j] - new_position)
                X1 = Alpha_pos[j] - A1 * D_alpha

                A2 = 2 * a * random.random() - a
                C2 = 2 * random.random()
                D_beta = abs(C2 * Beta_pos[j] - new_position)
                X2 = Beta_pos[j] - A2 * D_beta

                A3 = 2 * a * random.random() - a
                C3 = 2 * random.random()
                D_delta = abs(C3 * Delta_pos[j] - new_position)
                X3 = Delta_pos[j] - A3 * D_delta

                # New Modifications
                F = random.randint(1, 2)
                if F == 1:
                    new_position[j] += git(i, l) * rv() * (Alpha_pos[j] - new_position[j])
                else:
                    new_position[j] += pit(i, l) * rv() * (Alpha_pos[j] - new_position[j])

                xr = random.choice(Positions)
                alpha = random.random()
                L = random.random() - 0.5
                U = random.random() + 0.5
                new_position[j] += xr[j] * cos(2 * pi * alpha) * (L + random.random() * (U - L))
                new_position[j] += TF(l, Max_iter) * rv() * 2 * random.random() * (xr[j] - new_position[j])
                new_position[j] += rv() * (xr[j] - new_position[j])

                # Check if new position is better
            new_fitness = objf(new_position, trainInput, trainOutput, net)
            if new_fitness < fitness:
                Positions[i,:] = new_position

        Convergence_curve[l] = Alpha_score

        if l % 1 == 0:
            print(["At iteration " + str(l) + " the best fitness is " + str(Alpha_score)])

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = Convergence_curve
    s.optimizer = "MGWO"
    s.objfname = objf.__name__
    s.bestIndividual = Alpha_pos

    return s
