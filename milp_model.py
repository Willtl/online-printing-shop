from collections import namedtuple
from docplex.mp.model import Model
from docplex.mp.progress import ProgressListener
from docplex.util.environment import get_environment
from operator import itemgetter
import math
import json 
import sys 

instance = str(sys.argv[1])
terminal_output = False

running_time = 7200.0

with open('instances/' + instance + '.json') as data_file:
        data = json.load(data_file)

print('running ' + instance)

# number of resources
m = len(data["resources"])
# number of operations
n = sum(len(j["topology"]) for j in data["jobs"])
n_op = n

# composed of (i,j) such that i precedes j in the job route
A = [(o["id"] - 1, p - 1) for j in data["jobs"] for o in j["topology"] for p in o["sucessors"]]
# the ith element stores the resources capable of processing the ith operation
M = [[m - 1 for m in o["resources"]] for j in data["jobs"] for o in j["topology"]]
# the ith element sotes all the processing times related to each resource capable of processing i
P = [o["time"] for j in data["jobs"] for o in j["topology"]]
# available periods of each resource
U = [k["availability"] for k in data["resources"]]
# the ith element stores the overlap for the ith operation
overlap = [o['overlap'] for j in data["jobs"] for o in j["topology"]]
# release time of the ith operation (release >= 0)
release = [o['release'] for j in data["jobs"] for o in j["topology"]]
# fixed starting time of the ith operation (-1 means not fixed)
fstarting = [o['starting'] for j in data["jobs"] for o in j["topology"]]
# the job id for the ith operatiom
job = [j["id"] for j in data["jobs"] for o in j["topology"]]

# Pre-process unavailable periods
avs = [[[U[k][2 * v], U[k][2 * v + 1]] for v in range(len(U[k]) // 2)] for k in range(m)]
unavs = []
id_unav = 0
for k in range(m):
    unavs_in_k = []
    if U[k][0] > 0:
        unavs_in_k.append((0, U[k][0], id_unav))
        id_unav += 1
    for v in range(2, len(U[k]), 2):
        unavs_in_k.append((U[k][v - 1], U[k][v], id_unav))
        id_unav += 1
    unavs.append(unavs_in_k)  

linear_unavs = [unav for unav_in_k in unavs for unav in unav_in_k]  

## Data used for the setup time calculation
# kth element indicates the time to setup resource k when size changes
# (pairs, first pair indicates size bigger to smaller, second indicates smaller to bigger)
kSize = [k['setup_size'] for k in data["resources"]]
# kth element indicates the time to setup resource k when color changes
kColor = [k['setup_color'] for k in data["resources"]]
# kth element indicates the time to setup resource k when varnish changes
kVarnish = [k['setup_varnish'] for k in data["resources"]]
# the size of the ith operation (coefficient)
size = [o['size'] for j in data["jobs"] for o in j["topology"]]
# the color of the ith operation (integer value)
color = [o['color'] for j in data["jobs"] for o in j["topology"]]
# the varnish of the ith operation (integer value)
varnish = [o['varnish'] for j in data["jobs"] for o in j["topology"]]

# Transitive reduction
B = []
for k in range(m):
    processed_by_k = []
    for i in range(n_op):
        if k in M[i]:
            processed_by_k.append(i)
    B.append(processed_by_k)


class Listener(ProgressListener):
    def __init__(self, _time):
        ProgressListener.__init__(self)
        self.time = _time 
        self.history_makespan = []
        self.history_time = []

    def notify_progress(self, data):
        if data.has_incumbent:
            if not self.history_makespan:
                self.history_makespan.append(data.current_objective)
                self.history_time.append(float("%.3f" % data.time))
            else:
                size = len(self.history_makespan)
                if self.history_makespan[size - 1] != data.current_objective:
                    self.history_makespan.append(data.current_objective)
                    self.history_time.append(float("%.3f" % data.time))

            abs_gap = data.current_objective - data.best_bound
            rel_gap = abs(data.current_objective - data.best_bound) / (0.00000000001 + abs(data.current_objective))

            # print('Abs ' + str(abs_gap))
            # print('Rel ' + str(rel_gap))
            
            if abs_gap <= 1 - 0.000001:
                self.abort()
            
            if rel_gap <= 0.0:
                self.abort()   
        
        if data.time >= self.time:
            self.abort()  


def main(): 
    build_model()


def build_model():
    maxFull = 0
    for k in range(m):
        maxFull = max(maxFull, get_full_setup(k))

    maxSetup = 0
    for i in range(n_op):
        for j in range(n_op):
            for k in range(m):
                maxSetup = max(maxSetup, get_setup_time(i, j, k))
    bigM1 = max(maxFull, maxSetup)  

    rightSum = 0
    for i in range(n_op):
        max_value = 0
        for g in range(len(M[i])):
            max_value = max(max_value, P[i][g] + get_full_setup(M[i][g]))
        rightSum += max_value + get_full_setup(k) 
       
    latestUnav = 0
    for k in range(m):
        for unav in unavs[k]:
            latestUnav = max(latestUnav, unav[1])
    bigM2 = latestUnav + rightSum
    bigM3 = latestUnav

    mdl = Model(name='IJPE')

    mdl.parameters.threads     = 1
    mdl.parameters.mip.tolerances.mipgap    = float(0.0)
    mdl.parameters.mip.tolerances.absmipgap = float(0.0)

    # Variables 
    Cmax    = mdl.continuous_var(name='CMAX') 
    start   = {i : mdl.continuous_var() for i in range(n_op)} 
    comp    = {i : mdl.continuous_var() for i in range(n_op)}  
    cbar    = {i : mdl.continuous_var() for i in range(n_op)}   
    p       = {i : mdl.continuous_var() for i in range(n_op)} 
    pbar    = {i : mdl.continuous_var() for i in range(n_op)} 
    u       = {i : mdl.continuous_var() for i in range(n_op)}  
    ubar    = {i : mdl.continuous_var() for i in range(n_op)}  

    Y       = {(i, j, k) : mdl.binary_var() for k in range(m) for i in B[k] for j in B[k] if i != j}
    X       = {(i, k)    : mdl.binary_var() for i in range(n_op) for k in range(m)} 
    v       = {(i, k, l) : mdl.binary_var() for i in range(n_op) for k in M[i] for l in range(len(unavs[k]))} 
    w       = {(i, k, l) : mdl.binary_var() for i in range(n_op) for k in M[i] for l in range(len(unavs[k]))} 
    wbar    = {(i, k, l) : mdl.binary_var() for i in range(n_op) for k in M[i] for l in range(len(unavs[k]))} 

    hatxi     = {(j, k) : mdl.continuous_var() for k in range(m) for j in B[k]}  
    barxi     = {(j, k) : mdl.continuous_var() for k in range(m) for j in B[k]}  
    xi        = {j      : mdl.continuous_var() for j in range(n_op)} 

    mdl.add_kpi(Cmax, publish_name="MakeSpan:") 


    for k in range(m):
        for j in B[k]:
            lsum  = mdl.sum(Y[i, j, k] * get_setup_time(i, j, k) for i in B[k] if i != j)
            rsum = mdl.sum(Y[i, j, k] for i in B[k] if i != j)
            mdl.add_constraint(hatxi[j, k] == lsum + (1 - rsum) * get_full_setup(k))
 
    for k in range(m):
        for j in B[k]:
            mdl.add_constraint(0 <= barxi[j, k])
            mdl.add_constraint(barxi[j, k] <= bigM1 * X[j, k])
            mdl.add_constraint(hatxi[j, k] - bigM1 * (1 - X[j, k]) <= barxi[j, k])
            mdl.add_constraint(barxi[j, k] <= hatxi[j, k])
            
    for j in range(n_op):
        xi[j] = mdl.sum(barxi[j, k] for k in M[j])

    for k in range(m):
        for i in B[k]:
            for j in B[k]:
                if i != j:
                    mdl.add_constraint(Y[i, j, k] <= X[i, k])
                    mdl.add_constraint(Y[i, j, k] <= X[j, k])

    for k in range(m):
        sumy = mdl.sum(Y[i, j, k] for i in B[k] for j in B[k] if i != j)
        sumx = mdl.sum(X[i, k] for i in B[k])
        mdl.add_constraint(sumy == sumx - 1)
        
    for k in range(m):
        for i in B[k]:
            sumyj = mdl.sum(Y[i, j, k] for j in B[k] if j != i)
            mdl.add_constraint(sumyj <= 1)

    for k in range(m):
        for j in B[k]:
            sumyi = mdl.sum(Y[i, j, k] for i in B[k] if i != j)
            mdl.add_constraint(sumyi <= 1)
       
    for i in range(n_op):  
        mdl.add_constraint(comp[i] <= Cmax) 

    for i in range(n_op):
        mdl_sum = mdl.sum(X[i, k] for k in M[i])
        mdl.add_constraint(mdl_sum == 1)  
   
    for element in A:
        i = element[0]
        j = element[1] 
        mdl.add_constraint(cbar[i] <= start[j])
        mdl.add_constraint(comp[i] <= comp[j])
    

    for i in range(n_op):
        for j in range(n_op):
            if i != j:
                has_one = False # True if exist machine k which proc. i and j
                for machine in M[i]:
                    if machine in M[j]:
                        has_one = True

                if has_one: 
                    sumy = mdl.sum(Y[i, j, k] for k in range(m) if k in M[i] and k in M[j])
                    mdl.add_constraint(comp[i] - (1 - sumy) * bigM2 <= start[j] - xi[j])

    for i in range(n_op):
        # Release time
        mdl.add_constraint(start[i] >= release[i]) 
        # Full setup time
        mdl.add_constraint(start[i] >= xi[i]) 
        # Fixed Operations
        if fstarting[i] > -1:
            mdl.add_constraint(start[i] == fstarting[i])  

    for i in range(n_op):
        proc_sum = mdl.sum(X[i, M[i][g]] * P[i][g] for g in range(len(M[i])))
        mdl.add_constraint(p[i] == proc_sum)               

        proc_ov_sum = mdl.sum(X[i, M[i][g]] * math.ceil(overlap[i] * P[i][g]) for g in range(len(M[i])))
        mdl.add_constraint(pbar[i] == proc_ov_sum)           
    
    for i in range(n_op):
        for k in M[i]:
            for l in range(len(unavs[k])):
                # v_ikl
                mdl.add_constraint(v[i, k, l] <= X[i, k])
                mdl.add_constraint(start[i] <= unavs[k][l][0] - 1 + bigM2 * v[i, k, l] + bigM2 * (1 - X[i, k]))
                mdl.add_constraint(start[i] - xi[i] >= unavs[k][l][1] - bigM3 * (1 - v[i, k, l]) - bigM3 * (1 - X[i, k]))
                # w_ikl
                mdl.add_constraint(w[i, k, l] <= X[i, k])
                mdl.add_constraint(comp[i] <= unavs[k][l][0] + bigM2 * w[i, k, l] + bigM2 * (1 - X[i, k]))
                mdl.add_constraint(comp[i] >= unavs[k][l][1] + 1 - bigM3 * (1 - w[i, k, l]) - bigM3 * (1 - X[i, k]))
                # wbar_ikl
                mdl.add_constraint(wbar[i, k, l] <= X[i, k])
                mdl.add_constraint(cbar[i] <= unavs[k][l][0] + bigM2 * wbar[i, k, l] + bigM2 * (1 - X[i, k]))
                mdl.add_constraint(cbar[i] >= unavs[k][l][1] + 1 - bigM3 * (1 - wbar[i, k, l]) - bigM3 * (1 - X[i, k]))
                    
    for i in range(n_op):
        unav_sum = mdl.sum((w[i, k, l] - v[i, k, l]) * (unavs[k][l][1] - unavs[k][l][0]) for k in M[i] for l in range(len(unavs[k])))
        mdl.add_constraint(u[i] == unav_sum)

    for i in range(n_op):
        unav_sum = mdl.sum((wbar[i, k, l] - v[i, k, l]) * (unavs[k][l][1] - unavs[k][l][0]) for k in M[i] for l in range(len(unavs[k])))
        mdl.add_constraint(ubar[i] == unav_sum)

    for i in range(n_op):
        mdl.add_constraint(start[i] <= cbar[i])
        mdl.add_constraint(cbar[i] <= comp[i])
        mdl.add_constraint(start[i] + p[i] + u[i] == comp[i])
        mdl.add_constraint(start[i] + pbar[i] + ubar[i] == cbar[i])

    mdl.minimize(Cmax)
    # mdl.print_information()

    # Add progress listener
    listener = Listener(running_time)
    mdl.add_progress_listener(listener) 

    numb_binary = mdl.number_of_binary_variables
    numb_constraints = mdl.number_of_constraints
    numb_continuous = mdl.number_of_continuous_variables

    mdl.solve(log_output = terminal_output)    
    cpx = mdl.get_engine().get_cplex()   
    
    details = mdl.get_solve_details() 
      
    if details.status == 'aborted, no integer solution':
        print('NO SOLUTION')
        lower_bound = cpx.solution.MIP.get_best_objective()
        mip_iterations = details.nb_iterations
        nb_nodes = details.nb_nodes_processed
        time = details.time 

        store_details(lower_bound, -1, -1, mip_iterations, nb_nodes, time, [], [])
        quit() 

    lower_bound = cpx.solution.MIP.get_best_objective()
    makespan = Cmax.solution_value
    mip_relative_gap = details.mip_relative_gap
    mip_iterations = details.nb_iterations
    nb_nodes = details.nb_nodes_processed
    time = details.time

    history_makespan = listener.history_makespan
    history_time = listener.history_time
 
    if len(history_makespan) == 0:
        history_makespan.append(makespan)
        history_time.append(float("%.3f" % time))
    elif history_makespan[len(history_makespan) - 1] != makespan:
        print("missing")
        history_makespan.append(makespan)
        history_time.append(float("%.3f" % time))

    store_details(lower_bound, makespan, mip_relative_gap, mip_iterations, nb_nodes, time, history_makespan, history_time)
          
    if mdl.solve_details.status == 'integer optimal solution':
        print('Optimal')  
        print("Cmax: ", Cmax.solution_value)  
    else:
        print(mdl.solve_details.status) 
        print('lower_bound ', cpx.solution.MIP.get_best_objective())    
        print('upper_bound ', Cmax.solution_value)     
    print(time)
    return mdl
    
    
def get_full_setup(k):
    setup_time = 0
    if kSize[k][0] > kSize[k][1]:
        setup_time += kSize[k][0]
    else:
        setup_time += kSize[k][1]

    setup_time += kColor[k]
    setup_time += kVarnish[k]

    return setup_time


def get_setup_time(i, j, k):
    setup_time = 0
    if size[i] > size[j]:
        setup_time += kSize[k][0]
    elif size[i] < size[j]:
        setup_time += kSize[k][1]

    if color[i] != color[j]:
        setup_time += kColor[k]

    if varnish[i] != varnish[j]:
        setup_time += kVarnish[k]

    return setup_time


def store_details(lower_bound, makespan, mip_relative_gap, mip_iterations, nb_nodes, time, history_makespan, history_time):
    output = {}
    output['lower_bound'] = float("%.3f" % lower_bound)
    output['makespan'] = float("%.3f" % makespan)
    output['mip_relative_gap'] = float("%.3f" % mip_relative_gap)
    output['mip_iterations'] = mip_iterations
    output['nb_nodes'] = nb_nodes
    output['time'] = float("%.3f" % time)
    output['history_makespan'] = history_makespan
    output['history_time'] = history_time

    with open('output/MIP/' + instance + '.json', 'w') as outfile:
        json.dump(output, outfile, sort_keys=True, indent=4)
 
 
if __name__ == '__main__':
    main()
