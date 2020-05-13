import random
import json
import math

def GetRandomDag():
    all_nodes_connected = False
    while not all_nodes_connected:
        nodes = 0
        node_counter = 0
        adjacency = []
        rank_list = []

        i = 0
        numb_of_operations = random.randint(min_numb_op_per_job, max_numb_op_per_job)
        while nodes < numb_of_operations:
            # New nodes of 'higher' rank than all nodes generated till now
            new_nodes = 0
            if i == 0:
                new_nodes = random.randint(1, numb_start_end)
            elif nodes + max(nodes_per_rank) > numb_of_operations:
                new_nodes = nodes + max(nodes_per_rank) - numb_of_operations
            else:
                new_nodes = random.choice(nodes_per_rank)

            list = []
            for j in range(new_nodes):
                list.append(node_counter)
                node_counter += 1
            rank_list.append(list)

            # Edges from old nodes ('nodes') to new ones ('new_nodes')
            if i > 0:
                for j in rank_list[i - 1]:
                    one_connection = random.randint(nodes, nodes + new_nodes - 1)
                    adjacency.append((j, one_connection))
                    for k in range(new_nodes):
                        if k + nodes != one_connection and random.random() < connection_chance:
                            adjacency.append((j, k + nodes))
            i += 1
            nodes += new_nodes

        all_nodes_connected = True
        for i in range(nodes):
            node_is_connected = False
            for edge in adjacency:
                if i == edge[0] or i == edge[1]:
                    node_is_connected = True

            if not node_is_connected:
                # print('a node is not connected')
                all_nodes_connected = False

    # You may vizualize the DAG by pasting the output from the code below at "Model code" in http://dagitty.net/dags.html
    # export = []
    # for i in range(nodes):
    #     print(i)
    # print()
    # for edge in adjacency:
    #     print(str(edge[0]) + ' ' + str(edge[1]))
    #     export.append([str(edge[0] + 1), str(edge[1] + 1)])
    # print()
    # print(json.dumps(export))
    # quit()
    return nodes, adjacency


for instance_id in range(30):
    multiplier = instance_id + 1
    instance_name = 'test' + str(multiplier)
    lin = multiplier / 30

    # Parameters SOPS
    total_numb_of_jobs = 1 + math.ceil(lin * 3)
    min_numb_op_per_job = 2
    max_numb_op_per_job = 3 + math.ceil(lin * 2)
    min_numb_of_machines = 2
    max_numb_of_machines = 3 + math.ceil(lin * 2)

    # Parameters MOPS
    # total_numb_of_jobs = 4 + math.ceil(lin * 6)
    # min_numb_op_per_job = 6
    # max_numb_op_per_job = 7 + math.ceil(lin * 5)
    # min_numb_of_machines = 6
    # max_numb_of_machines = 7 + math.ceil(lin * 13)

    print(total_numb_of_jobs, min_numb_op_per_job, max_numb_op_per_job, min_numb_of_machines, max_numb_of_machines)

    fixed_operation_chance = 0.01
    release_operation_chance = 0.025
    overlap_operation_chance = 0.1

    # Machine parameters
    numb_of_machines = random.randint(min_numb_of_machines, max_numb_of_machines)
    max_numb_unavs = 4

    # Digraph parameters
    nodes_per_rank = [1, 2, 3, 4]       # Number/chance of nodes per rank (min 1, max 4) -> SOPS
    numb_start_end = 3                  # Max number of nodes at starting ranks
    connection_chance = 0.85            # Chance of adding an arc

    resources = []
    jobs = []
    instance = {}
    # Create machines
    for i in range(numb_of_machines):
        resource = {}
        resource['id'] = i + 1
        # Time to setup when size change
        resource['setup_size'] = [random.randint(1, 6), random.randint(1, 6)]
        # Time to color change
        resource['setup_color'] = random.randint(1, 6);
        # Time to varnish change
        resource['setup_varnish'] = random.randint(1, 6);
        resources.append(resource)

    current_numb_operations = 0
    job_id_counter = 1
    amount_proc_k = [0 for k in range(numb_of_machines)]
    while len(jobs) < total_numb_of_jobs:
        job = {}
        job['id'] = job_id_counter
        job['rid'] = job_id_counter
        job['priority'] = 0
        job['duedate'] = 0

        # Define job topology and create operations
        topology = []
        nodes, adjacency = GetRandomDag()
        for i in range(nodes):
            current_numb_operations += 1
            operation = {}
            operation['id'] = i + 1
            operation['rid'] = current_numb_operations
            operation['connection'] = 0
            operation['starting'] = -1
            # Define release time
            if random.random() <= release_operation_chance:
                operation['release'] = random.randint(1, 99)
            else:
                operation['release'] = 0
            # Define overlap coefficient
            if random.random() <= overlap_operation_chance:
                operation["overlap"] = round(random.uniform(0.5, 0.99), 2)
            else:
                operation["overlap"] = 1.0
            # Size, color, and varnish type for setup calculation
            operation['size'] = random.randint(1, 10)
            operation['color'] = random.randint(1, 4)
            operation['varnish'] = random.randint(1, 6)

            min_p_res = math.ceil(0.3 * numb_of_machines)
            max_p_res = math.ceil(0.7 * numb_of_machines)
            numb_of_processing_resources = random.randint(min_p_res, max_p_res)
            subset = []
            for j in range(1, numb_of_machines + 1):
                subset.append(j)
            shuffled = sorted(subset, key=lambda k: random.random())
            operation['resources'] = [shuffled[j] for j in range(numb_of_processing_resources)]

            # Define processing time for each resource
            p = random.randint(1, 99)
            index = random.randint(0, numb_of_processing_resources - 1)
            time = []
            for j in range(numb_of_processing_resources):
                time.append(0)
            time[index] = p
            for j in range(numb_of_processing_resources):
                if time[j] == 0:
                    time[j] = random.randint(p, min(3 * p, 99))
            operation['time'] = time

            for k in range(len(operation['resources'])):
                amount_proc_k[operation['resources'][k] - 1] += operation['time'][k]


            successors = []
            # Define successors
            for edge in adjacency:
                if edge[0] == i:
                    successors.append(edge[1] + 1)
            operation['sucessors'] = successors
            if 'starting' not in operation:
                print(operation)
                quit()
            topology.append(operation)

        job['topology'] = topology
        job_id_counter += 1

        jobs.append(job)

    operation_id = 1
    # Remap the id of each operation
    for i in range(len(jobs)):
        map_ids = {}

        # Set new operation id and store old
        for j in range(len(jobs[i]['topology'])):
            map_ids[jobs[i]['topology'][j]['id']] = operation_id
            jobs[i]['topology'][j]['id'] = operation_id
            operation_id += 1

        # Loop sucessors and update new ids
        for j in range(len(jobs[i]['topology'])):
            for k in range(len(jobs[i]['topology'][j]['sucessors'])):
                old_value = jobs[i]['topology'][j]['sucessors'][k]
                jobs[i]['topology'][j]['sucessors'][k] = map_ids[old_value]

    # Add unavailabilities based on the amount of processing per machine
    for i in range(numb_of_machines):
        numb_of_unavs = random.randint(1, max_numb_unavs)
        amount = amount_proc_k[i]
        availability_size = 1 + math.ceil(amount / numb_of_unavs + 1)
        last_time = availability_size
        vec = [0, last_time]

        for j in range(numb_of_unavs):
            unav_size = 1 + math.ceil(availability_size / random.randint(2, 10))
            value1 = last_time + unav_size
            value2 = value1 + availability_size
            last_time = value2
            vec.append(value1)
            vec.append(value2)

        resources[i]['availability'] = vec

    # 0 at index k if there is no fixed at machine k
    fixed_at_k = [0 for k in range(numb_of_machines)]
    numb_of_fixed = 0
    for i in range(len(jobs)):
        if random.random() <= fixed_operation_chance and numb_of_fixed < numb_of_machines:
            random_resource = random.randint(0, numb_of_machines-1)
            while fixed_at_k[random_resource] != 0:
                random_resource = random.randint(0, numb_of_machines-1)

            if fixed_at_k[random_resource] == 0:
                end_availability = resources[random_resource]['availability'][1]
                time = random.randint(1, 99)
                if end_availability - 20 > time:
                    start = random.randint(20, end_availability - time - 1)

                    jobs[i]['topology'][0]['resources'] = [random_resource + 1]
                    jobs[i]['topology'][0]['time'] = [time]
                    jobs[i]['topology'][0]['starting'] = start
                    jobs[i]['topology'][0]['release'] = 0

                    fixed_at_k[random_resource] = 1
            numb_of_fixed += 1


    instance['resources'] = resources
    instance['jobs'] = jobs

    with open('instances/' + instance_name + '.json', 'w') as outfile:
        json.dump(instance, outfile, indent=4, sort_keys=False)
