import numpy as np
import pandas as pd
 
num_customers = 10
D = 60
K = 0.25
delta = 0.0025
rt = 0.01
rd = 0.009
beta = rd*D

def intial_sol(df,num_customers):
    global K
    global D
    while True:
        sol = np.random.random(size = num_customers)
        for ind in range(len(sol)):
            if sol[ind] >= 0.5:
                sol[ind] = 1
            else:
                sol[ind] = 0
        if np.sum(np.multiply(np.asarray(df['Loan Size']),sol)) <= (1-K)*D:
            break
    return sol

def loan_revenue(df,sol):
    loan_rev = np.multiply(np.asarray(df['Interest']),np.asarray(df['Loan Size'])) - np.asarray(df['Loss'])
    loan_rev = np.multiply(loan_rev,sol)
    loan_rev = sum(loan_rev)
    return loan_rev

def loan_cost(df,sol):
    global delta
    lcost = np.multiply(np.asarray(df['Loan Size']),delta)
    lcost = np.multiply(lcost,sol)
    lcost = sum(lcost)
    return lcost

def trans_cost(df,sol):
    global D
    global K
    global rt

    T = ((1-K)*D)*np.ones(len(df)) - np.asarray(df['Loan Size'])
    tcost = np.multiply(T,rt)
    tcost = np.multiply(tcost,sol)
    tcost = sum(tcost)
    return tcost

def fitness(df,sol):
    global D
    global K
    global rt
    global rd
    global beta

    fit = loan_revenue(df,sol) + trans_cost(df,sol) - beta - sum(np.multiply(np.asarray(df['Loss']),sol))
    if np.sum(np.multiply(np.asarray(df['Loan Size']),sol)) > (1-K)*D:
        # print(fit)
        fit = fit - ((np.sum(np.multiply(np.asarray(df['Loan Size']),sol)) - (1-K)*D))/10
        # print(fit)
    return fit

def check_feasible(df,sol):
    global D
    global K
    if np.sum(np.multiply(np.asarray(df['Loan Size']),sol)) > (1-K)*D:
        return False
    else:
        return True

def ga(df):
    global D
    global K
    global rt
    global rd
    global beta

    pop_size = 50
    gen_size = 100
    crossover_ratio = 0.8
    mutation_ratio = 0.1
    reproduction_ratio = 0.194

    sols = []
    for i in range(pop_size):
        sol = intial_sol(df,len(df))
        sols.append(sol)
    sols = np.asarray(sols)

    best_sol_fit = -100
    best_sol = []

    for gen in range(gen_size):
        sols_new = []
        fitness_pop = []

        for i in range(pop_size):
            fit = fitness(df, sols[i])
            fitness_pop.append(fit)
            # r = np.random.random()
        #     if r <= reproduction_ratio:
        #         sols_new.append(sols[i])

        #Crossover
        cumm_prob = [0 for i in range(pop_size)]
        sum_fit = sum(fitness_pop)
        cumm_prob[0] = fitness_pop[0]/sum_fit
        for i in range(1,pop_size):
            cumm_prob[i] = cumm_prob[i-1] + fitness_pop[i]/sum_fit
        for count in range(int(pop_size/2)):
            r = np.random.random()
            for i in range(pop_size):
                if r < cumm_prob[0]:
                    p1 = sols[0]
                    break
                elif r < cumm_prob[i] and r >= cumm_prob[i-1]:
                    p1 = sols[i]
            r = np.random.random()
            for i in range(pop_size):
                if r < cumm_prob[0]:
                    p2 = sols[0]
                    break
                elif r < cumm_prob[i] and r >= cumm_prob[i-1]:
                    p2 = sols[i]
            r = np.random.random()
            if r < crossover_ratio:
                # print('crossover')
                pt = np.random.randint(1,len(p1))
                c1 = []
                c2 = []
                for j in range(pt):
                    c1.append(p1[j])
                    c2.append(p2[j])
                for j in range(pt,len(p1)):
                    c1.append(p2[j])
                    c2.append(p1[j])
                # print(p1,p2)
                # print(c1,c2)
            else:
                c1 = p1
                c2 = p2
            sols_new.append(c1)
            sols_new.append(c2)
        
        #Mutation
        for i in range(pop_size):
            r = np.random.random()
            if r < mutation_ratio:
                p = sols[i]
                pt1 = np.random.randint(0,len(p))
                pt2 = np.random.randint(0,len(p))
                while pt2 == pt1:
                    pt2 = np.random.randint(0,len(p))
                val = p[pt1] 
                p[pt1] = p[pt2]
                p[pt2] = val
                sols_new[i] = p
        sols = sols_new
        fitness_pop = []
        for i in range(pop_size):
            fit = fitness(df, sols[i])
            fitness_pop.append(fit)
        best_sol_ind = np.argmax(fitness_pop)
        # print(fitness_pop[best_sol_ind])
        if best_sol_fit <= fitness_pop[best_sol_ind] and check_feasible(df,sols[best_sol_ind]) == True:
            best_sol_fit = fitness_pop[best_sol_ind]
            best_sol = np.asarray(sols[best_sol_ind])
        print(best_sol_fit)
    return best_sol

def pso(df):
    pop_size = 20
    max_iter = 100
    min_val = -1
    max_val = 2
    v_max = 1
    v_min = -1
    w = 0.9
    c1 = 0.2
    c2 = 0.1

    sols = []
    for i in range(pop_size):
        sol = intial_sol(df, len(df))
        sol1 = {'solution' : sol, 'fitness' : fitness(df,sol), 'pbest' : sol, 'vel' : [0 for i in range(len(df))]}
        sols.append(sol1)
    fit = []
    for sol in sols:
        fit_sol = sol['fitness']
        fit.append(fit_sol)
    max_ind = np.argmax(fit)
    gbest = sols[max_ind]['solution']

    for iter in range(max_iter):
        if iter%30 == 0:
            sols = []
        for i in range(pop_size):
            sol = intial_sol(df, len(df))
            sol1 = {'solution' : sol, 'fitness' : fitness(df,sol), 'pbest' : sol, 'vel' : [0 for i in range(len(df))]}
            sols.append(sol1)
        fit = []
        for sol in sols:
            fit_sol = fitness(df,sol['pbest'])
            fit.append(fit_sol)
        max_ind = np.argmax(fit)
        # print(max_ind)
        max_sol = sols[max_ind]
        if fitness(df,gbest)<fitness(df,max_sol['pbest']) and check_feasible(df,sols[max_ind]['pbest']):
            gbest = sols[max_ind]['pbest']
        
        for sol in sols:
            if iter%20 == 0:
                sol['vel'] = [0 for i in range(len(df))]
            # print(sol['solution'])
            u1 = np.random.rand(len(df))
            u2 = np.random.rand(len(df))
            sol['vel'] = np.multiply(w,sol['vel']) + u1*c1*(np.subtract(sol['pbest'],sol['solution'])) + u2*c2*(np.subtract(gbest,sol['solution']))
            for i in range(len(sol['solution'])):
                if sol['vel'][i] > v_max:
                    sol['vel'][i] = v_max
                if sol['vel'][i] < v_min:
                    sol['vel'][i] = v_min
            # print(sol['vel'])
            for i in range(len(sol['solution'])):
                if sol['solution'][i] + sol['vel'][i] <= min_val:
                    sol['solution'][i] = min_val
                elif sol['solution'][i] + sol['vel'][i] >= max_val:
                    sol['solution'][i] = max_val
                else:
                    sol['solution'][i] = sol['solution'][i] + sol['vel'][i]
                if sol['solution'][i] < 0.5:
                    sol['solution'][i] = 0
                else:
                    sol['solution'][i] = 1
            sol1 = sol['solution']
            sol['fitness'] = fitness(df,sol1)
            if sol['fitness'] >= fitness(df,sol['pbest']) and check_feasible(df,sol['pbest']) == True:
                sol['pbest'] = sol['solution']
            # print('sol',sol['solution'],sol['fitness'])
        print(fitness(df,gbest))
    return gbest

def pso_for_ga(df,pop_size,sols_ga):
    max_iter = 5
    min_val = -1
    max_val = 2
    v_max = 1
    v_min = -1
    w = 0.9
    c1 = 0.2
    c2 = 0.1

    sols = []
    for i in range(len(sols_ga)):
        sol = sols_ga[i]
        sol1 = {'solution' : sol, 'fitness' : fitness(df,sol), 'pbest' : sol, 'vel' : [0 for i in range(len(df))]}
        sols.append(sol1)
    fit = []
    # print('len_sols', len(sols))
    for sol in sols:
        fit_sol = sol['fitness']
        fit.append(fit_sol)
    max_ind = np.argmax(fit)
    gbest = sols[max_ind]['solution']

    for iter in range(max_iter):
        for sol in sols:
            fit_sol = fitness(df,sol['pbest'])
            fit.append(fit_sol)
        max_ind = np.argmax(fit)
        # print(max_ind)
        max_sol = sols[max_ind]
        if fitness(df,gbest)<fitness(df,max_sol['pbest']) and check_feasible(df,sols[max_ind]['pbest']):
            gbest = sols[max_ind]['pbest']
        
        for sol in sols:
            if iter%20 == 0:
                sol['vel'] = [0 for i in range(len(df))]
            # print(sol['solution'])
            u1 = np.random.rand(len(df))
            u2 = np.random.rand(len(df))
            sol['vel'] = np.multiply(w,sol['vel']) + u1*c1*(np.subtract(sol['pbest'],sol['solution'])) + u2*c2*(np.subtract(gbest,sol['solution']))
            for i in range(len(sol['solution'])):
                if sol['vel'][i] > v_max:
                    sol['vel'][i] = v_max
                if sol['vel'][i] < v_min:
                    sol['vel'][i] = v_min
            # print(sol['vel'])
            for i in range(len(sol['solution'])):
                if sol['solution'][i] + sol['vel'][i] <= min_val:
                    sol['solution'][i] = min_val
                elif sol['solution'][i] + sol['vel'][i] >= max_val:
                    sol['solution'][i] = max_val
                else:
                    sol['solution'][i] = sol['solution'][i] + sol['vel'][i]
                if sol['solution'][i] < 0.5:
                    sol['solution'][i] = 0
                else:
                    sol['solution'][i] = 1
            sol1 = sol['solution']
            sol['fitness'] = fitness(df,sol1)
            if sol['fitness'] >= fitness(df,sol['pbest']) and check_feasible(df,sol['pbest']) == True:
                sol['pbest'] = sol['solution']
            # print('sol',sol['solution'],sol['fitness'])
        # print(gbest,fitness(df,gbest))
    return sols

def ga_pso(df):
    global D
    global K
    global rt
    global rd
    global beta

    pop_size = 30
    gen_size = 100
    crossover_ratio = 0.6
    mutation_ratio = 0.1
    pso_prob = 0.3

    sols = []
    for i in range(pop_size):
        sol = intial_sol(df,len(df))
        sols.append(sol)
    sols = np.asarray(sols)

    best_sol_fit = -100
    best_sol = []

    for gen in range(gen_size):
        sols_new = []
        fitness_pop = []

        for i in range(pop_size):
            fit = fitness(df, sols[i])
            fitness_pop.append(fit)
            # r = np.random.random()
        #     if r <= reproduction_ratio:
        #         sols_new.append(sols[i])
        #Crossover
        cumm_prob = [0 for i in range(pop_size)]
        sum_fit = sum(fitness_pop)
        cumm_prob[0] = fitness_pop[0]/sum_fit
        for i in range(1,pop_size):
            cumm_prob[i] = cumm_prob[i-1] + fitness_pop[i]/sum_fit
        for count in range(int(pop_size/2)):
            r = np.random.random()
            for i in range(pop_size):
                if r < cumm_prob[0]:
                    p1 = sols[0]
                    break
                elif r < cumm_prob[i] and r >= cumm_prob[i-1]:
                    p1 = sols[i]
            r = np.random.random()
            for i in range(pop_size):
                if r < cumm_prob[0]:
                    p2 = sols[0]
                    break
                elif r < cumm_prob[i] and r >= cumm_prob[i-1]:
                    p2 = sols[i]
            r = np.random.random()
            if r < crossover_ratio:
                # print('crossover')
                pt = np.random.randint(1,len(p1))
                c1 = []
                c2 = []
                for j in range(pt):
                    c1.append(p1[j])
                    c2.append(p2[j])
                for j in range(pt,len(p1)):
                    c1.append(p2[j])
                    c2.append(p1[j])
                # print(p1,p2)
                # print(c1,c2)
            else:
                c1 = p1
                c2 = p2
            sols_new.append(c1)
            sols_new.append(c2)
        #Mutation
        for i in range(pop_size):
            r = np.random.random()
            if r < mutation_ratio:
                p = sols[i]
                pt1 = np.random.randint(0,len(p))
                pt2 = np.random.randint(0,len(p))
                while pt2 == pt1:
                    pt2 = np.random.randint(0,len(p))
                val = p[pt1] 
                p[pt1] = p[pt2]
                p[pt2] = val
                sols_new[i] = p
        #PSO
        r = np.random.random()
        if r < pso_prob:
            # print(len(sols_new))
            sol_pso = pso_for_ga(df, pop_size, sols_new)
            # print(len(sol_pso))
            for i in range(len(sol_pso)):
                sol_pso_val = sol_pso[i]
                sol_val = sol_pso_val['pbest']
                sols_new[i] = sol_val
            
        sols = sols_new
        fitness_pop = []
        for i in range(pop_size):
            fit = fitness(df, sols[i])
            fitness_pop.append(fit)
        best_sol_ind = np.argmax(fitness_pop)
        # print(fitness_pop[best_sol_ind])
        if best_sol_fit <= fitness_pop[best_sol_ind] and check_feasible(df,sols[best_sol_ind]) == True:
            best_sol_fit = fitness_pop[best_sol_ind]
            best_sol = np.asarray(sols[best_sol_ind])
        # print(best_sol_fit)
    return best_sol
    


if __name__ == "__main__":
    df = pd.read_csv('data.csv')
    for i in range(1):
        best_sol = ga(df)
        print(fitness(df, best_sol))

    # print(check_feasible(df,best_sol))
    # best_sol,gbest = pso(df)
    # print(best_sol,gbest)
    # print(check_feasible(df,best_sol))
    # best_sol = ga_pso(df)
    # print(best_sol, fitness(df,best_sol)