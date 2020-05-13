import sys

instance    = str(sys.argv[1])
RandSeed    = 1
TimeLimit   = 7200
UseCumul    = True  # If True, formulation adds the redundant cumulative function
UseTwoSteps = False  # If True, uses a two step approach, otherwise only uses the full model

history_makespan    = []
history_time        = []

import json

with open('settings/instances/' + instance + '.json') as data_file:
    data = json.load(data_file)

# number of resources
m = len(data["resources"])
# number of operations
n = sum(len(j["topology"]) for j in data["jobs"])
# composed of (i,j) such that i precedes j in the job route
A = [(o["id"] - 1, p - 1) for j in data["jobs"] for o in j["topology"] for p in o["sucessors"]]
# the ith element stores the resources capable of processing the ith operation
M = [[m - 1 for m in o["resources"]] for j in data["jobs"] for o in j["topology"]]
# the ith element sotes all the processing times related to each resource capable of processing i
P = [o["time"] for j in data["jobs"] for o in j["topology"]]
# available periods of each resource
U = [k["availability"] for k in data["resources"]]
# the ith element stores the overlap for the ith operation
OVERLAP = [o["overlap"] for j in data["jobs"] for o in j["topology"]]
# the processing time for precedence constraint taking into acount the overlap
POV = [ [ (999+(int(1000*o["overlap"])*t))//1000 for t in o["time"]] for j in data["jobs"] for o in j["topology"]]
# release time of the ith operation (release >= 0)
RELEASE = [o['release'] for j in data["jobs"] for o in j["topology"]]
# fixed starting time of the ith operation (-1 means not fixed)
F_STARTING = [o['starting'] for j in data["jobs"] for o in j["topology"]]
# job of operation
JOB = [j["id"] for j in data["jobs"] for o in j["topology"]]

## Data used for the setup time calculation
# kth element indicates the time to setup resource k when size changes
# (pairs, first pair indicates size bigger to smaller, second indicates smaller to bigger)
K_SIZE    = [k['setup_size'] for k in data["resources"]]
# kth element indicates the time to setup resource k when color changes
K_COLOR   = [k['setup_color'] for k in data["resources"]]
# kth element indicates the time to setup resource k when varnish changes
K_VARNISH = [k['setup_varnish'] for k in data["resources"]]
# the size of the ith operation (coefficient)
SIZE      = [o['size'] for j in data["jobs"] for o in j["topology"]]
# the color of the ith operation (integer value)
COLOR     = [o['color'] for j in data["jobs"] for o in j["topology"]]
# the varnish of the ith operation (integer value)
VARNISH   = [o['varnish'] for j in data["jobs"] for o in j["topology"]]

# Pre-process unavailable periods
avs = [ [ [U[k][2*v], U[k][2*v + 1]] for v in range(len(U[k])//2) ] for k in range(m)]

# PHL: START TRANSITION MATRIX DEFINITION
# WE COMPRESS THE 3 INFORMATION SIZE-COLOR-VARNINSH INTO A SINGLE TYPE
VECTOR = list(set([(SIZE[i],COLOR[i],VARNISH[i]) for i in range(n)]))
NTYPES = len(VECTOR)+1
TYPE   = { VECTOR[i]:i for i in range(NTYPES-1)}
FIRST   = NTYPES-1
MAXTT  = max([max(K_SIZE[k])+K_COLOR[k]+K_VARNISH[k] for k in range(m)]) + 1


def store_output(output_data):
    path = 'output/exp2/cpmodel4/' + instance + '.json'

    data = {}
    try:
        with open(path) as data_file:
            data = json.load(data_file)
    except FileNotFoundError:
        print('File not found')
    
    model = 'CPO'
    if model not in data:
        data[model] = []

    cpo_values = data[model]
    cpo_values.append(output_data)
    data[model] = cpo_values

    with open(path, 'w') as outfile:
        json.dump(data, outfile, indent=4, sort_keys=False)


def get_initial_setup(k):
    setup_time = 0
    if K_SIZE[k][0] > K_SIZE[k][1]:
        setup_time += K_SIZE[k][0]
    else:
        setup_time += K_SIZE[k][1]
    setup_time += K_COLOR[k]
    setup_time += K_VARNISH[k]
    i=0
    while avs[k][i][1]-avs[k][i][0] < setup_time:
        i = i+1
    return [avs[k][i][0],avs[k][i][0]+setup_time]


def get_setup_time(ti, tj, k):
    if ti==FIRST or tj==FIRST:
        return get_initial_setup(k)[1]-get_initial_setup(k)[0]
    vi = VECTOR[ti]
    vj = VECTOR[tj]
    setup_time = 0
    if vi[0] > vj[0]:
        setup_time += K_SIZE[k][0]
    elif vi[0] < vj[0]:
        setup_time += K_SIZE[k][1]
    if vi[1] != vj[1]:
        setup_time += K_COLOR[k]
    if vi[2] != vj[2]:
        setup_time += K_VARNISH[k]
    return setup_time


def get_type(i):
    return TYPE[(SIZE[i],COLOR[i],VARNISH[i])]

MATRIX  = [[[ get_setup_time(ti, tj, k) for tj in range(NTYPES)] for ti in range(NTYPES)] for k in range(m)]
IMATRIX = [[[ get_setup_time(tj, ti, k) for tj in range(NTYPES)] for ti in range(NTYPES)] for k in range(m)]

# PHL: END TRANSITION MATRIX DEFINITION

# CP OPTIMIZER MODEL
from docplex.cp.model import *

# context.solver.agent = 'local'
# context.solver.local.execfile = '/opt/ibm/ILOG/CPLEX_Studio129/cpoptimizer/bin/x86-64_linux/cpoptimizer'

# The ith value in calendar defines the intensity function for the ith resource
H = 10000000000
calendar = [CpoStepFunction(steps=[(0,100)]) for k in range(m)]
for k in range(m):
    if len(avs[k])>0:
        calendar[k].set_value(0, avs[k][0][0], 0)
        for j in range(len(avs[k])-1):
            calendar[k].set_value(avs[k][j][1], avs[k][j+1][0], 0)

def CreateModel(disjSetups =True, lowerBound =0):
    model = CpoModel()

    # Decision variables
    op     = [interval_var(name="O" + str(i + 1)) for i in range(n)] # EQ-B1
    opm    = [[interval_var(optional=True, size=P[i][j], intensity=calendar[M[i][j]],
                            name="O" + str(i + 1) + "M" + str(M[i][j] + 1)) for j in range(len(M[i]))] for i in range(n)]  # EQ-B2

    # These variables are used only in case overlap != 1
    opov   = [interval_var(name="O" + str(i + 1) + "OV") for i in range(n)]         # EQ-1
    opmov  = [[interval_var(optional=True, size=POV[i][j], intensity=calendar[M[i][j]],
                            name="O" + str(i + 1) + "M" + str(M[i][j] + 1) + "OV") for j in range(len(M[i]))] for i in range(n)] # EQ-2

    setup = None
    if disjSetups:
        setup  = [[interval_var(optional=True,
                                name="O" + str(i + 1) + "Setup" + str(M[i][j] + 1)) for j in range(len(M[i]))] for i in range(n)]  # EQ-10
        cover  = [[interval_var(optional=True, size=[P[i][j],H], intensity=calendar[M[i][j]],
                                name="O" + str(i + 1) + "Cover" + str(M[i][j] + 1)) for j in range(len(M[i]))] for i in range(n)]  # EQ-11
        seqc   = [sequence_var([cover[i][j] for i in range(n) for j in range(len(M[i])) if M[i][j] == k],
                               [get_type(i) for i in range(n) for j in range(len(M[i])) if M[i][j] == k],
                               name="Cover" + str(k+1)) for k in range(m)] # EQ-13

    # PHL: SEQUENCE VARIABLE FOR MACHINE, WITH TRANSITION TYPES
    seq    = [sequence_var([opm[i][j]   for i in range(n) for j in range(len(M[i])) if M[i][j] == k],
                           [get_type(i) for i in range(n) for j in range(len(M[i])) if M[i][j] == k],
                           name="M" + str(k+1)) for k in range(m)]   # EQ-12

    # Minimize makespan
    makespan = max(end_of(op[i]) for i in range(n))
    model.add(minimize(makespan))
    model.add(makespan >= lowerBound)

    # PHL: RELEASE TIME AND STARTING TIME
    for i in range(n):
        if 0<=F_STARTING[i]:
            model.add(start_of(op[i],F_STARTING[i])==F_STARTING[i])
        else:
            model.add(start_of(op[i],RELEASE[i])>=RELEASE[i])

    if disjSetups:
        # PHL: HANDLING OF NON-BREAKABLE TRANSITION TIMES
        for i in range(n):
            for j in range(len(M[i])):
                model.add(span(cover[i][j], [setup[i][j],opm[i][j]])) # EQ-13
                model.add(end_at_start(setup[i][j], opm[i][j])) # EQ-17
                model.add(presence_of(cover[i][j]) == presence_of(opm[i][j])) # EQ-14
                model.add(presence_of(setup[i][j]) == presence_of(opm[i][j])) # EQ-15
                model.add(size_of(setup[i][j], IMATRIX[M[i][j]][get_type(i)][FIRST]) ==
                          element(IMATRIX[M[i][j]][get_type(i)], type_of_prev(seqc[M[i][j]],cover[i][j],FIRST,FIRST))) # EQ-16
                model.add(forbid_extent(setup[i][j], calendar[M[i][j]])) # EQ-20
        model.add([no_overlap(seqc[k]) for k in range(m)]) # EQ-19

    # Avoid overlapping between interval variables at the same resource
    model.add([no_overlap(seq[k], MATRIX[k]) for k in range(m) ]) # EQ-18

    # For each operation, choose one interval variables respective to the selected resource
    model.add([alternative(op[i], [opm[i][j] for j in range(len(M[i]))]) for i in range(n)]) # EQ-B3

    model.add([alternative(opov[i], [opmov[i][j] for j in range(len(M[i]))]) for i in range(n) if OVERLAP[i]<1]) # EQ-4
    model.add([start_at_start(op[i], opov[i]) for i in range(n) if OVERLAP[i]<1]) # EQ-5
    for i in range(n):
        if OVERLAP[i]<1:
            for j in range(len(M[i])):
                model.add(presence_of(opm[i][j])==presence_of(opmov[i][j])) # EQ-3

    # Precedence constraints
    for s in A:
        if OVERLAP[s[0]]==1:
            model.add(end_before_start(op[s[0]], op[s[1]])) # (EQ-6)
        else:
            model.add(end_before_start(opov[s[0]], op[s[1]])) # EQ-6
            model.add(end_before_end(op[s[0]], op[s[1]])) # (EQ-6)

    # Redundant constraint
    if UseCumul:
        model.add(sum(pulse(op[i], 1) for i in range(n)) <= m)

    for i in range(n):
        for j in range(len(M[i])):
            # Avoid starting at unavailable period
            model.add(forbid_start(opm[i][j], calendar[M[i][j]]))  # EQ-21
            # Avoid ending at unavailable period
            model.add(forbid_end(opm[i][j], calendar[M[i][j]]))  # EQ-22
            # PHL: MINIMAL START TIME ON MACHINE
            model.add(start_of(opm[i][j], get_initial_setup(M[i][j])[1]) >= get_initial_setup(M[i][j])[1])

    return model, opm, setup

# Display convergence curve

def log_convergence(time, val):
    print(str(time) + '\t' + str(val))
    history_makespan.append(val)
    history_time.append(time)

# Convergence of first step

TraceLog = False
ShowNumberVariables = True
ShowNumberOfBranches = True
TrackConvergence = True
ExportSolution = False

import math

stored = []
N1 = 10 # Keep solutions every TLimit / N1
N2 = 5  # Keep N2 last solutions
def run_model1(cposols, opm0, tlimit) :
    lastsol = None
    bestobj = H
    # Store all improving solutions
    for sol in cposols:
        t = sol.get_solver_infos().get_solve_time()
        v = sol.get_objective_values()[0]
        lb = sol.get_objective_bounds()[0]
        stored.append((t,v,sol))
        lastsol = sol
    status = cposols.get_last_result().get_solve_status()
    time = cposols.get_last_result().get_solver_infos().get_solve_time()
    branches = cposols.get_last_result().get_solver_infos()['NumberOfBranches']
    if status == 'Infeasible':
        return lastsol, bestobj, time, status
    elif status == 'Optimal':
        lb = v

    if TrackConvergence:
        # Run heuristic on selected solutions
        model,opm,setup = CreateModel(disjSetups=True)
        nb = len(stored)
        for s in range(nb):
            if s==0 or s>=nb-N2 or math.floor(stored[s][0]*N1/tlimit)!=math.floor(stored[s+1][0]*N1/tlimit):
                if TraceLog: print('Keeping solution ' + str(s) + ': t=' + str(stored[s][0]) + '\t obj=' + str(stored[s][1]))
                sol = stored[s][2]
                sp = CpoModelSolution()  # Starting point solution for simulation of heuristic
                for i in range(n):
                    for j in range(len(M[i])):
                        if sol.get_var_solution(opm0[i][j]).is_present():
                            sp.add_interval_var_solution(opm0[i][j], presence = True,
                                                         start = sol.get_var_solution(opm0[i][j]).start,
                                                         end   = sol.get_var_solution(opm0[i][j]).end)
                        else:
                            sp.add_interval_var_solution(opm0[i][j], presence = False)
                model.set_starting_point(sp)
                cposol = model.solve(TemporalRelaxation='Off', SolutionLimit=1, trace_log=False, Workers=1, RandomSeed=RandSeed)
                ht = cposol.get_solver_infos().get_solve_time()
                t = stored[s][0]
                vr = stored[s][1]
                v  = cposol.get_objective_values()[0]
                if v <= bestobj:
                    if v < bestobj :
                        log_convergence(t, v)
                    bestobj=v
                    lastsol = sol
                if TraceLog: print(str(t) + '\t' + str(vr) + ' ---(' + str(ht) + ')--> ' + str(bestobj))

    return lastsol, lb, time, status, branches

# Convergence of second step
def run_model2(solver, currTime) :
    bestsol = None
    for sol in solver:
        t = currTime + sol.get_solver_infos().get_solve_time()
        v = sol.get_objective_values()[0]
        log_convergence(t, v)
        bestsol = sol
    return bestsol 

# CP OPTIMIZER AUTOMATIC SEARCH

# Use TimeLimit0 = 0 for only running the full model
TimeLimit0 = 0
if UseTwoSteps:
    TimeLimit0 = 2 * TimeLimit / 3

def solve():
    print("######### SOLVING " + instance)
    TimeLimit1 = TimeLimit
    if 0<TimeLimit0:
        model0,opm0,setup0 = CreateModel(disjSetups=False)
        solver0 = model0.start_search(LogPeriod=100000000, Workers=1, SearchType='Restart', TimeLimit=TimeLimit0, trace_log=TraceLog, RandomSeed=RandSeed)
        cposol0, lb0, currTime, status0, branches0 = run_model1(solver0, opm0, TimeLimit0)
        if status0 == 'Infeasible':
            print("######### INFEASIBLE")
            return

        model,opm,setup = CreateModel(disjSetups=True, lowerBound=lb0)
    else:
        currTime = 0
        branches0 = 0
        model,opm,setup = CreateModel(disjSetups=True, lowerBound=0)

    if 0<TimeLimit0:
        if status0 != 'Unknown':
            # Starting point solution for second step
            sp = CpoModelSolution()
            for i in range(n):
                for j in range(len(M[i])):
                    if cposol0.get_var_solution(opm0[i][j]).is_present():
                        sp.add_interval_var_solution(opm0[i][j], presence = True,
                                                     start = cposol0.get_var_solution(opm0[i][j]).start,
                                                     end   = cposol0.get_var_solution(opm0[i][j]).end)
                    else:
                        sp.add_interval_var_solution(opm0[i][j], presence = False)
            model.set_starting_point(sp)


    last_lower_bound = None
    TimeLimit1 = TimeLimit - currTime  
    solver = model.start_search(LogPeriod=100000000, Workers=1, TemporalRelaxation='Off', TimeLimit=TimeLimit1, trace_log=TraceLog, RandomSeed=RandSeed)
    cposol = run_model2(solver, currTime)
    status = solver.get_last_result().get_solve_status()
    branches2 = solver.get_last_result().get_solver_infos()['NumberOfBranches']
    vars2 = solver.get_last_result().get_solver_infos()['NumberOfVariables']
    csts2 = solver.get_last_result().get_solver_infos()['NumberOfConstraints']
    time2 = solver.get_last_result().get_solver_infos().get_solve_time()

    timeInfo = ' ' + str(currTime + time2)
    branchInfo = ''
    if ShowNumberOfBranches:
        branchInfo = " BRANCHES[ " + str(branches0) + " " + str(branches2) + " ]"
    modelInfo = ''
    if ShowNumberVariables:
        modelInfo = " VARS-CTS[ " + str(vars2) + " " + str(csts2) + " ]"

    if status == 'Infeasible':
        print("######### INFEASIBLE " + timeInfo + branchInfo + modelInfo)
        return
    elif status == 'Optimal':
        opt = cposol.get_objective_values()[0]
        last_lower_bound = opt
        print("######### " + str(opt) + " " + str(opt) + " * " + timeInfo + branchInfo + modelInfo)
    elif status == 'Feasible':
        lb = cposol.get_objective_bounds()[0]
        last_lower_bound = lb
        ub = cposol.get_objective_values()[0]
        print("######### " + str(lb) + " " + str(ub) + "   " + timeInfo + branchInfo + modelInfo)
    else:
        print("######### UNKNOWN")
        return;

    mks = cposol.get_objective_values()[0]   

    output = {
        'best_makespan': mks, 
        'lower_bound': last_lower_bound, 
        'history_makespan': history_makespan, 
        'history_time': history_time, 
        'branches1': branches0, 
        'branches2': branches2,
        'nb_variables': vars2,
        'nb_constraints': csts2
    }
    store_output(output)

    
    # EXPORT SOLUTION

    sol = []
    # Adding operations
    if ExportSolution:
        for i in range(n):
            for j in range(len(M[i])):
                if cposol.get_var_solution(opm[i][j]).is_present():
                    sol.append(dict(id       = i,
                                    id_job   = JOB[i],
                                    resource = M[i][j],
                                    start    = cposol.get_var_solution(opm[i][j]).start,
                                    end      = cposol.get_var_solution(opm[i][j]).end))
        # Adding setups
        for i in range(n):
            for j in range(len(M[i])):
                if cposol.get_var_solution(opm[i][j]).is_present():
                    s = cposol.get_var_solution(setup[i][j]).start
                    e = cposol.get_var_solution(setup[i][j]).end
                    if s<e and e-s<MAXTT: # if e-s=MAXTT it is the last setup, we do not show it
                        sol.append(dict(id       = i,
                                        id_job   = -2,
                                        resource = M[i][j],
                                        start    = cposol.get_var_solution(setup[i][j]).start,
                                        end      = cposol.get_var_solution(setup[i][j]).end))
        # Adding unavailability
        for k in range(m):
            s = 0
            for a in avs[k]:
                e = a[0]
                if e>s:
                    sol.append(dict(id       = -1,
                                    id_job   = -1,
                                    resource = k,
                                    start    = s,
                                    end      = e))
                s = a[1]

        with open('visualization/gantt.json', 'w') as outfile:
            json.dump(sol, outfile)


solve()
