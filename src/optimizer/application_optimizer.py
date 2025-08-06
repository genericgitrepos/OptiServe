import itertools
import warnings
import networkx as nx
import numpy as np
import pandas as pd
import random
import copy

from tqdm import tqdm
from typing import Dict, List, Union
from queue import PriorityQueue

from src.modeling.application_performance_modeling import ApplicationPerformanceModeling

warnings.filterwarnings("ignore")


class ApplicationOptimizer:
    def __init__(self,
                 Appworkflow: ApplicationPerformanceModeling,
                 mem_list: Dict[str, List],
                 model_list: Dict[str, List]):

        self.App = Appworkflow
        
        self.mem_list = mem_list
        self.model_list = model_list
        
        # Normalized accuracy of each model for each function
        self.model_accuracy_list = {}
        
        for node in self.App.workflow_graph.nodes:
            if node in ['Start', 'End']:
                continue
            
            models_variants = self.App.workflow_graph.nodes[node]['models_list']
            
            self.model_accuracy_list[node] = [i / len(models_variants) for i in range(1, len(models_variants) + 1)]    
        
        self.minimal_mem_configuration, \
            self.maximal_mem_configuration, \
                self.minimal_model_configuration, \
                    self.maximal_model_configuration, \
                        self.maximal_cost, \
                            self.minimal_avg_rt, \
                                self.minimal_cost, \
                                    self.maximal_avg_rt = self.get_optimization_boundary()

        self.update_BCR()

        self.all_simple_paths = [path for path in
                                 nx.all_simple_paths(self.App.delooped_graph, self.App.start_point, self.App.end_point)]

        self.simple_paths_num = len(self.all_simple_paths)

        self.CPcounter = 0


    # Update mem, model and rt attributes of each node in the workflow
    def update_mem_rt(self, G: ApplicationPerformanceModeling, mem_dict, model_dict):
        for node in mem_dict:
            G.nodes[node]['mem'] = mem_dict[node]
            G.nodes[node]['rt'] = G.nodes[node]['perf_profile'][model_dict[node]][mem_dict[node]]


    # Update mem and rt attributes of each node in the workflow
    def update_App_workflow_mem_rt(self,
                                   App: ApplicationPerformanceModeling,
                                   mem_dict,
                                   model_dict):
        self.update_mem_rt(App.workflow_graph, mem_dict, model_dict)
        App.update_rt()


    def get_perf_cost_table(self, file, start_iterations=1, end_iterations=None):
        rows = []
        self.App.update_ne()
        node_list = [item for item in self.App.workflow_graph.nodes if item not in ['Start', 'End']]
        
        all_available_mem_list = [
            sorted(self.App.workflow_graph.nodes[node]['perf_profile'][0].keys())
            for node in node_list
        ]
        
        all_available_model_list = [
            sorted(list(range(len(self.App.workflow_graph.nodes[node]['perf_profile']))))
            for node in node_list
        ]
        
        if end_iterations is not None:
            task_size = end_iterations - start_iterations + 1
        else:
<<<<<<< HEAD
            task_size = np.prod([len(item) for item in all_available_mem_list], [len(item) for item in all_available_model_list]) - start_iterations + 1
=======
            task_size = np.prod([len(item) for item in all_available_mem_list]) - start_iterations + 1
>>>>>>> f26fbc8c44671b1ca1b8bee045531cfdf38d87ac

        model_configurations = list(itertools.product(*all_available_model_list))
        mem_configurations = list(itertools.product(*all_available_mem_list))
        
<<<<<<< HEAD
        total_configurations = itertools.product(model_configurations, mem_configurations)
=======
        total_configurations = list(itertools.product(model_configurations, mem_configurations))
>>>>>>> f26fbc8c44671b1ca1b8bee045531cfdf38d87ac

        iterations_count = start_iterations - 1
        print('Get Performance Cost Table - Task Size: {}'.format(task_size))
        
        for i in range(start_iterations - 1):
            next(total_configurations)

        with tqdm(total=task_size) as pbar:
            for model_config, mem_config in total_configurations:
                iterations_count += 1
                current_model_config = dict(zip(node_list, model_config))
                current_mem_config = dict(zip(node_list, mem_config))
                
                self.update_App_workflow_mem_rt(self.App, current_mem_config, current_model_config)
                current_cost = self.App.get_avg_cost()
                
                self.App.get_simple_dag()
                current_rt = self.App.get_avg_rt()
                # Create row
                mem_config = {f'f{str(node)}_mem': current_mem_config[node] for node in current_mem_config}
                raw_model_config = {f'f{str(node)}_acc': current_model_config[node] for node in current_model_config}
                model_config = {f'f{str(node)}_acc_value': self.model_accuracy_list[node][current_model_config[node]] for node in current_model_config}

                row = mem_config.copy()
                row.update(raw_model_config)
                row.update(model_config)
                row['Cost'] = current_cost
                row['RT'] = current_rt
                row['ID'] = iterations_count
                rows.append(row)
                pbar.update()
                if end_iterations is not None and iterations_count >= end_iterations:
                    break

        # Convert list of rows to DataFrame and save
        data = pd.DataFrame(rows).set_index('ID')
        data.to_csv(file, index=True)

    def get_optimization_boundary(self):
        node_list = [node for node in self.App.workflow_graph.nodes if node not in ['Start', 'End']]
        
        minimal_mem_configuration = {}
        maximal_mem_configuration = {}
        
        minimal_model_configuration = {}
        maximal_model_configuration = {}
        
        for node in node_list:
            model_variants = self.App.workflow_graph.nodes[node]['perf_profile']
            
            minimal_mem_configuration[node] = np.inf
            maximal_mem_configuration[node] = -np.inf
            
            minimal_model_configuration[node] = 0
            maximal_model_configuration[node] = len(model_variants) - 1

            for var in model_variants:
                minimal_mem_configuration[node] = min(minimal_mem_configuration[node],
                                                    min(list(var.keys())))
                maximal_mem_configuration[node] = max(maximal_mem_configuration[node],
                                                    max(list(var.keys())))

        self.App.update_ne()

        # Calculating the maximal possible cost
        self.update_App_workflow_mem_rt(self.App, maximal_mem_configuration, maximal_model_configuration)
        maximal_cost = self.App.get_avg_cost()

        # Calculating the minimal possible average response time
        self.update_App_workflow_mem_rt(self.App, maximal_mem_configuration, minimal_model_configuration)
        self.App.get_simple_dag()
        minimal_avg_rt = self.App.get_avg_rt()

        # Calculating the minimal possible cost
        self.update_App_workflow_mem_rt(self.App, minimal_mem_configuration, minimal_model_configuration)
        minimal_cost = self.App.get_avg_cost()

        # Calculating the maximal possible average response time
        self.update_App_workflow_mem_rt(self.App, minimal_mem_configuration, maximal_model_configuration)
        self.App.get_simple_dag()
        maximal_avg_rt = self.App.get_avg_rt()

        print('Minimal Memory Configuration: {}'.format(minimal_mem_configuration))
        print('Maximal Memory Configuration: {}'.format(maximal_mem_configuration))
        print('Maximal Model Configuration: {}'.format(maximal_model_configuration))
        print('Minimal Model Configuration: {}'.format(minimal_model_configuration))
        print('Maximal Cost: {}'.format(maximal_cost))
        print('Minimal Average Response Time: {}'.format(minimal_avg_rt))
        print('Minimal Cost: {}'.format(minimal_cost))
        print('Maximal Average Response Time: {}'.format(maximal_avg_rt))
        print('Optimization Boundary Calculation Completed.')
        
        return (minimal_mem_configuration, maximal_mem_configuration, minimal_model_configuration, maximal_model_configuration, maximal_cost, minimal_avg_rt, minimal_cost,
                maximal_avg_rt)


    # Get the Benefit Cost Ratio (absolute value) of each function
    def update_BCR(self):
        node_list = [item for item in self.App.workflow_graph.nodes]
        for node in node_list:
            self.App.workflow_graph.nodes[node]['BCR'] = {}
            if node in ['Start', 'End']:
                continue
            for model_i, _ in enumerate(self.model_accuracy_list[node]):
                available_mem_list = [item for item in np.sort(list(self.App.workflow_graph.nodes[node]['perf_profile'][model_i].keys()))]
                available_rt_list = [self.App.workflow_graph.nodes[node]['perf_profile'][model_i][item] for item in available_mem_list]
                slope, intercept = np.linalg.lstsq(np.vstack([available_mem_list, np.ones(len(available_mem_list))]).T,
                                                np.array(available_rt_list), rcond=None)[0]
                self.App.workflow_graph.nodes[node]['BCR'][model_i] = np.abs(slope)


    # Find the probability refined critical path in self.App
    def find_PRCP(self, order=0, leastCritical=False):
        self.CPcounter += 1
        tp_list = self.App.get_tp(self.App.delooped_graph, self.all_simple_paths)
        rt_list = self.App.sum_rt_with_ne(self.all_simple_paths, include_start_node=True, include_end_node=True)
        prrt_list = np.multiply(tp_list, rt_list)
        if (leastCritical):
            PRCP = np.argsort(prrt_list)[order]
        else:
            PRCP = np.argsort(prrt_list)[-1 - order]
        return (self.all_simple_paths[PRCP])


    # Update the list of available memory configurations in ascending order
    def update_available_mem_list(self, BCR=False, BCRthreshold=0.1, BCRinverse=False):
        node_list = [item for item in self.App.workflow_graph.nodes]
        for node in node_list:
            self.App.workflow_graph.nodes[node]['available_mem'] = {}
            if node in ['Start', 'End']:
                continue
            for model_i, _ in enumerate(self.model_accuracy_list[node]):
                if (BCR):
                    available_mem_list = [item for item in
                                        np.sort(list(self.App.workflow_graph.nodes[node]['perf_profile'][model_i].keys()))]
                    mem_zip = [item for item in zip(available_mem_list, available_mem_list[1:])]
                    if (BCRinverse):
                        available_mem_list = [item for item in mem_zip if np.abs((item[1] - item[0]) / (
                                self.App.workflow_graph.nodes[node]['perf_profile'][model_i][item[1]] -
                                self.App.workflow_graph.nodes[node]['perf_profile'][model_i][item[0]])) > 1.0 / (
                                                self.App.workflow_graph.nodes[node]['BCR'][model_i]) * BCRthreshold]
                    else:
                        available_mem_list = [item for item in mem_zip if np.abs((self.App.workflow_graph.nodes[node][
                                                                                    'perf_profile'][model_i][item[1]] -
                                                                                self.App.workflow_graph.nodes[node][
                                                                                    'perf_profile'][model_i][item[0]]) / (
                                                                                        item[1] - item[0])) >
                                            self.App.workflow_graph.nodes[node]['BCR'][model_i] * BCRthreshold]
                    available_mem_list = list(np.sort(list(set(itertools.chain(*available_mem_list)))))
                else:
                    available_mem_list = [item for item in
                                        np.sort(list(self.App.workflow_graph.nodes[node]['perf_profile'][model_i].keys()))]
                self.App.workflow_graph.nodes[node]['available_mem'][model_i] = available_mem_list  # Sorted list
<<<<<<< HEAD


=======


>>>>>>> f26fbc8c44671b1ca1b8bee045531cfdf38d87ac
    def compute_accuracy(self, model_configuration, accuracy_formula):
        accuracy_values = [self.model_accuracy_list[node][model_configuration[node]] for node in self.App.workflow_graph.nodes if node not in ['Start', 'End']]
        
        return accuracy_formula(*accuracy_values)
    
    
    def accuracy_is_satisfied(self, model_configuration, accuracy_constraint, accuracy_formula):    
        return self.compute_accuracy(model_configuration, accuracy_formula) >= accuracy_constraint
    

    def BPBA(self, budget, accuracy_constraint, accuracy_formula, optimize_model_configuration=True, BCR=False, BCRtype="RT/M", BCRthreshold=0.1):
        '''
        Probability Refined Critical Path Algorithm - Minimal end-to-end response time under a budget constraint
        Best Performance under budget constraint

        Args:
            budget (float): the budge constraint
            BCR (bool): True - use benefit-cost ratio optimization False - not use BCR optimization
            BCRtype (string): 'RT/M' - Benefit is RT, Cost is Mem. Eliminate mem configurations which do not conform to BCR limitations.
                                         The greedy strategy is to select the config with maximal RT reduction.
                              'ERT/C' - Benefit is the reduction on end-to-end response time, Cost is increased cost.
                                             The greedy strategy is to select the config with maximal RT reduction.
                              'MAX' - Benefit is the reduction on end-to-end response time, Cost is increased cost.
                                       The greedy strategy is to select the config with maximal BCR
            BCRthreshold (float): The threshold of BCR cut off
        '''
        if BCRtype == 'rt-mem':
            BCRtype = 'RT/M'
        elif BCRtype == 'e2ert-cost':
            BCRtype = 'ERT/C'
        elif BCRtype == 'max':
            BCRtype = 'MAX'
        if (BCR and BCRtype == "RT/M"):
            self.update_available_mem_list(BCR=True, BCRthreshold=BCRthreshold, BCRinverse=False)
        else:
            self.update_available_mem_list(BCR=False)
        if (BCR):
            cost = self.minimal_cost
                    
        curr_model_configuration = copy.deepcopy(self.minimal_model_configuration)
        curr_mem_configuration = copy.deepcopy(self.minimal_mem_configuration)
        curr_accuracy = self.compute_accuracy(curr_model_configuration, accuracy_formula)
        
        # First phase is finding the model configuration that satisfies the accuracy constraint holding the budget constraint.
        # We start the minimal memory configuration and try to optimize the model configuration.
        
        self.update_App_workflow_mem_rt(self.App, curr_mem_configuration, curr_model_configuration)
        current_cost = self.minimal_cost
        surplus = budget - current_cost
    
        last_e2ert_cost_BCR = 0
        order = 0
        iterations_count = 0
        
        ml_functions = [node for node in self.App.workflow_graph.nodes if node not in ['Start', 'End'] and len(self.model_accuracy_list[node]) > 1]
        mem_list = curr_mem_configuration

        w = 100
        if optimize_model_configuration:
            while not self.accuracy_is_satisfied(curr_model_configuration, accuracy_constraint, accuracy_formula) and (round(surplus, 4) >= 0):
                iterations_count += 1
                cp = self.find_PRCP(order=order, leastCritical=False)
                min_avg_cost_increase_of_each_node = {}
                for node in cp:
                    if node not in ml_functions:
                        continue
                    avg_cost_increase_of_each_model_config = {}
                    node_curr_mem = mem_list[node]
                    model_backup = curr_model_configuration[node]
                    for model_i in list(range(len(self.model_accuracy_list[node]))):
                        if model_i <= curr_model_configuration[node]:
                            continue
                        self.update_App_workflow_mem_rt(self.App, mem_dict={node: node_curr_mem}, model_dict={node: model_i})
                        curr_model_configuration[node] = model_i
                        increased_cost = self.App.get_avg_cost() - current_cost
                        acc_after = self.compute_accuracy(curr_model_configuration, accuracy_formula)
                        
                        if (increased_cost <= surplus):
                            acc_gap = acc_after - accuracy_constraint
                            score = -increased_cost + w * min(acc_gap, 0.0)
                            increased_acc = (acc_after - curr_accuracy)
                            avg_cost_increase_of_each_model_config[model_i] = (increased_acc,
                                                                            increased_cost,
                                                                            score)
                            
                        curr_model_configuration[node] = model_backup
                        self.update_App_workflow_mem_rt(self.App, mem_dict={node: node_curr_mem}, model_dict={node: model_backup})
                    
                    if len(avg_cost_increase_of_each_model_config) != 0:
                        
                        max_BCR = np.max([item[2] for item in avg_cost_increase_of_each_model_config.values()])
                        min_cost_increase_under_MAX_BCR = np.min([item[1] for item in avg_cost_increase_of_each_model_config.values()
                                                                if item[2] == max_BCR])
                        max_increased_acc_under_MAX_cost_increase_MAX_BCR = np.max(
                            [item[0] for item in avg_cost_increase_of_each_model_config.values()
                            if item[1] == min_cost_increase_under_MAX_BCR and item[2] == max_BCR])
                        
                        reversed_dict = dict(zip(avg_cost_increase_of_each_model_config.values(),
                                                    avg_cost_increase_of_each_model_config.keys()))
                        
                        min_avg_cost_increase_of_each_node[node] = (reversed_dict[(
                                max_increased_acc_under_MAX_cost_increase_MAX_BCR, min_cost_increase_under_MAX_BCR,
                                max_BCR)],
                                                                    max_increased_acc_under_MAX_cost_increase_MAX_BCR,
                                                                    min_cost_increase_under_MAX_BCR,
                                                                    max_BCR)
                            
                if (len(min_avg_cost_increase_of_each_node) == 0):
                    if (order >= self.simple_paths_num - 1):
                        break
                    else:
                        order += 1
                        continue
                
                max_BCR = np.max([item[3] for item in min_avg_cost_increase_of_each_node.values()])
                max_increased_acc_under_MAX_cost_increase_MAX_BCR = np.max(
                    [item[1] for item in min_avg_cost_increase_of_each_node.values() if item[3] == max_BCR])
                target_node = [key for key in min_avg_cost_increase_of_each_node if
                                min_avg_cost_increase_of_each_node[key][3] == max_BCR and
                                min_avg_cost_increase_of_each_node[key][1] == max_increased_acc_under_MAX_cost_increase_MAX_BCR][0]
                
                target_model = min_avg_cost_increase_of_each_node[target_node][0]
                
                self.update_App_workflow_mem_rt(self.App,
                                                mem_dict={target_node: mem_list[target_node]},
                                                model_dict={target_node: target_model})
                curr_model_configuration[target_node] = target_model
                max_increased_acc_under_MAX_cost_increase_MAX_BCR = min_avg_cost_increase_of_each_node[target_node][1]
                min_cost_increase_under_MAX_BCR = min_avg_cost_increase_of_each_node[target_node][2]
                self.App.get_simple_dag()
                current_avg_rt = self.App.get_avg_rt()
                curr_accuracy = self.compute_accuracy(curr_model_configuration, accuracy_formula)
                surplus -= min_cost_increase_under_MAX_BCR
                
                print(min_avg_cost_increase_of_each_node)
                print(self.accuracy_is_satisfied(curr_model_configuration, accuracy_constraint, accuracy_formula), surplus, '\n\n')

        order = 0
        
        cost = self.App.get_avg_cost()
        surplus = budget - cost

        self.App.get_simple_dag()
        current_avg_rt = self.App.get_avg_rt()
        current_cost = cost
                
        while (round(surplus, 4) >= 0):
            iterations_count += 1
            cp = self.find_PRCP(order=order, leastCritical=False)
            max_avg_rt_reduction_of_each_node = {}
            mem_backup = nx.get_node_attributes(self.App.workflow_graph, 'mem')
            for node in cp:
                if node in ['Start', 'End']:
                    continue
                avg_rt_reduction_of_each_mem_config = {}
                for mem in reversed(self.App.workflow_graph.nodes[node]['available_mem'][curr_model_configuration[node]]):
                    if (mem <= mem_backup[node]):
                        break
                    self.update_App_workflow_mem_rt(self.App, mem_dict={node: mem}, model_dict={node: curr_model_configuration[node]})
                    increased_cost = self.App.get_avg_cost() - current_cost
                    if (increased_cost < surplus):
                        self.App.get_simple_dag()
                        rt_reduction = current_avg_rt - self.App.get_avg_rt()
                        if (rt_reduction > 0):
                            avg_rt_reduction_of_each_mem_config[mem] = (rt_reduction, increased_cost)
                self.update_App_workflow_mem_rt(self.App, mem_dict={node: mem_backup[node]}, model_dict={node: curr_model_configuration[node]})
                if (BCR and BCRtype == "ERT/C"):
                    avg_rt_reduction_of_each_mem_config = {item: avg_rt_reduction_of_each_mem_config[item] for item in
                                                           avg_rt_reduction_of_each_mem_config.keys() if
                                                           avg_rt_reduction_of_each_mem_config[item][0] /
                                                           avg_rt_reduction_of_each_mem_config[item][
                                                               1] > last_e2ert_cost_BCR * BCRthreshold}
                if (BCR and BCRtype == "MAX"):
                    avg_rt_reduction_of_each_mem_config = {item: (
                        avg_rt_reduction_of_each_mem_config[item][0], avg_rt_reduction_of_each_mem_config[item][1],
                        avg_rt_reduction_of_each_mem_config[item][0] / avg_rt_reduction_of_each_mem_config[item][1]) for
                        item in avg_rt_reduction_of_each_mem_config.keys()}
                if (len(avg_rt_reduction_of_each_mem_config) != 0):
                    if (BCR and BCRtype == "MAX"):
                        max_BCR = np.max([item[2] for item in avg_rt_reduction_of_each_mem_config.values()])
                        max_rt_reduction_under_MAX_BCR = np.max(
                            [item[0] for item in avg_rt_reduction_of_each_mem_config.values() if
                             item[2] == max_BCR])
                        min_increased_cost_under_MAX_rt_reduction_MAX_BCR = np.min(
                            [item[1] for item in avg_rt_reduction_of_each_mem_config.values() if
                             item[0] == max_rt_reduction_under_MAX_BCR and item[2] == max_BCR])
                        reversed_dict = dict(zip(avg_rt_reduction_of_each_mem_config.values(),
                                                 avg_rt_reduction_of_each_mem_config.keys()))
                        max_avg_rt_reduction_of_each_node[node] = (reversed_dict[(
                            max_rt_reduction_under_MAX_BCR, min_increased_cost_under_MAX_rt_reduction_MAX_BCR,
                            max_BCR)],
                                                                   max_rt_reduction_under_MAX_BCR,
                                                                   min_increased_cost_under_MAX_rt_reduction_MAX_BCR,
                                                                   max_BCR)
                    else:
                        max_rt_reduction = np.max([item[0] for item in avg_rt_reduction_of_each_mem_config.values()])
                        min_increased_cost_under_MAX_rt_reduction = np.min(
                            [item[1] for item in avg_rt_reduction_of_each_mem_config.values() if
                             item[0] == max_rt_reduction])
                        reversed_dict = dict(zip(avg_rt_reduction_of_each_mem_config.values(),
                                                 avg_rt_reduction_of_each_mem_config.keys()))
                        max_avg_rt_reduction_of_each_node[node] = (
                            reversed_dict[(max_rt_reduction, min_increased_cost_under_MAX_rt_reduction)],
                            max_rt_reduction,
                            min_increased_cost_under_MAX_rt_reduction)

            if (len(max_avg_rt_reduction_of_each_node) == 0):
                if (order >= self.simple_paths_num - 1):
                    break
                else:
                    order += 1
                    continue
            if (BCR and BCRtype == "MAX"):
                max_BCR = np.max([item[3] for item in max_avg_rt_reduction_of_each_node.values()])
                max_rt_reduction_under_MAX_BCR = np.max(
                    [item[1] for item in max_avg_rt_reduction_of_each_node.values() if item[3] == max_BCR])
                target_node = [key for key in max_avg_rt_reduction_of_each_node if
                               max_avg_rt_reduction_of_each_node[key][3] == max_BCR and
                               max_avg_rt_reduction_of_each_node[key][1] == max_rt_reduction_under_MAX_BCR][0]
                target_mem = max_avg_rt_reduction_of_each_node[target_node][0]
            else:
                max_rt_reduction = np.max([item[1] for item in max_avg_rt_reduction_of_each_node.values()])
                min_increased_cost_under_MAX_rt_reduction = np.min(
                    [item[2] for item in max_avg_rt_reduction_of_each_node.values() if item[1] == max_rt_reduction])
                target_mem = np.min([item[0] for item in max_avg_rt_reduction_of_each_node.values() if
                                     item[1] == max_rt_reduction and item[
                                         2] == min_increased_cost_under_MAX_rt_reduction])
                target_node = [key for key in max_avg_rt_reduction_of_each_node if
                               max_avg_rt_reduction_of_each_node[key] == (
                                   target_mem, max_rt_reduction, min_increased_cost_under_MAX_rt_reduction)][0]
            self.update_App_workflow_mem_rt(self.App, mem_dict={target_node: target_mem}, model_dict={target_node: curr_model_configuration[target_node]})
            max_rt_reduction = max_avg_rt_reduction_of_each_node[target_node][1]
            min_increased_cost_under_MAX_rt_reduction = max_avg_rt_reduction_of_each_node[target_node][2]
            current_avg_rt = current_avg_rt - max_rt_reduction
            surplus = surplus - min_increased_cost_under_MAX_rt_reduction
            current_cost = self.App.get_avg_cost()
            current_e2ert_cost_BCR = max_rt_reduction / min_increased_cost_under_MAX_rt_reduction
            if (current_e2ert_cost_BCR == float('Inf')):
                last_e2ert_cost_BCR = 0
            else:
                last_e2ert_cost_BCR = current_e2ert_cost_BCR
        current_mem_configuration = nx.get_node_attributes(self.App.workflow_graph, 'mem')
        del current_mem_configuration['Start']
        del current_mem_configuration['End']
        print('Optimized Memory Configuration: {}'.format(current_mem_configuration))
        print('Average end-to-end response time: {}'.format(current_avg_rt))
        print('Optimized Accuracy Configuration: {}'.format(curr_model_configuration))
        print('Average Cost: {}'.format(current_cost))
        print('PRCP_BPBC Optimization Completed.\n\n')
        return (current_avg_rt, current_cost, self.compute_accuracy(curr_model_configuration, accuracy_formula), current_mem_configuration, curr_model_configuration, iterations_count)
<<<<<<< HEAD


=======


>>>>>>> f26fbc8c44671b1ca1b8bee045531cfdf38d87ac
    def BCPA(self, rt_constraint, accuracy_constraint, accuracy_formula, optimize_model_configuration=True, BCR=False, BCRtype="RT/M", BCRthreshold=0.2):
        '''
        Probability Refined Critical Path Algorithm - Minimal cost under an end-to-end response time constraint
        Best cost under performance (end-to-end response time) constraint

        Args:
            rt_constraint (float): End-to-end response time constraint
            BCR (bool): True - use benefit-cost ratio optimization False - not use BCR optimization
            BCRtype (string): 'M/RT' - Benefit is Mem, Cost is RT. (inverse) Eliminate mem configurations which do not conform to BCR limitations
                              'C/ERT' - Benefit is the cost reduction, Cost is increased ERT.
                              'MAX' - Benefit is the cost reduction, Cost is increased ERT. The greedy strategy is to select the config with maximal BCR
            BCRthreshold (float): The threshold of BCR cut off
        '''
        if BCRtype == 'rt-mem':
            BCRtype = 'M/RT'
        elif BCRtype == 'e2ert-cost':
            BCRtype = 'C/ERT'
        elif BCRtype == 'max':
            BCRtype = 'MAX'
        if (BCR and BCRtype == "M/RT"):
            self.update_available_mem_list(BCR=True, BCRthreshold=BCRthreshold, BCRinverse=True)
        else:
            self.update_available_mem_list(BCR=False)
        
        order = 0
        iterations_count = 0
        last_e2ert_cost_BCR = 0
        
        curr_model_configuration = copy.deepcopy(self.minimal_model_configuration)
        curr_mem_configuration = copy.deepcopy(self.maximal_mem_configuration)
        curr_accuracy = self.compute_accuracy(curr_model_configuration, accuracy_formula)
        
        # First phase is finding the model configuration that satisfies the accuracy constraint holding the budget constraint.
        # We start the maximal memory configuration and try to optimize the model configuration.

        self.update_App_workflow_mem_rt(self.App, curr_mem_configuration, curr_model_configuration)
        current_avg_rt = self.minimal_avg_rt
        performance_surplus = rt_constraint - current_avg_rt
            
        ml_functions = [node for node in self.App.workflow_graph.nodes if node not in ['Start', 'End'] and len(self.model_accuracy_list[node]) > 1]
        mem_list = nx.get_node_attributes(self.App.workflow_graph, 'mem') 
        
        print('RT Constraint: {}'.format(rt_constraint))
        print('Accuracy Constraint: {}'.format(accuracy_constraint))
        print('Performance Surplus: {}'.format(performance_surplus))
        print('Current Average Response Time: {}'.format(current_avg_rt))
        w = 100

        if optimize_model_configuration:
            while not self.accuracy_is_satisfied(curr_model_configuration, accuracy_constraint, accuracy_formula) and (round(performance_surplus, 4) >= 0):
                iterations_count += 1
                cp = self.find_PRCP(order=order, leastCritical=False)
                min_avg_rt_increase_of_each_node = {}
                for node in cp:
                    if node not in ml_functions:
                        continue
                    avg_rt_increase_of_each_model_config = {}
                    node_curr_mem = mem_list[node]
                    model_backup = curr_model_configuration[node]
                    for model_i in list(range(len(self.model_accuracy_list[node]))):
                        if model_i <= curr_model_configuration[node]:
                            continue
                        self.update_App_workflow_mem_rt(self.App, mem_dict={node: node_curr_mem}, model_dict={node: model_i})
                        curr_model_configuration[node] = model_i
                        self.App.get_simple_dag()
                        increased_rt = self.App.get_avg_rt() - current_avg_rt
                        acc_after = self.compute_accuracy(curr_model_configuration, accuracy_formula)
                        
                        print(node, increased_rt, performance_surplus, curr_model_configuration)
                        if (increased_rt <= performance_surplus):
                            acc_gap = acc_after - accuracy_constraint
                            score = -increased_rt + w * min(acc_gap, 0.0)
                            increased_acc = (acc_after - curr_accuracy)
                            avg_rt_increase_of_each_model_config[model_i] = (increased_acc,
                                                                            increased_rt,
                                                                            score)
                            
                        curr_model_configuration[node] = model_backup
                        self.update_App_workflow_mem_rt(self.App, mem_dict={node: node_curr_mem}, model_dict={node: model_backup})
                    
                    if len(avg_rt_increase_of_each_model_config) != 0:
                        
                        max_BCR = np.max([item[2] for item in avg_rt_increase_of_each_model_config.values()])
                        min_rt_increase_under_MAX_BCR = np.min([item[1] for item in avg_rt_increase_of_each_model_config.values()
                                                                if item[2] == max_BCR])
                        max_increased_acc_under_MAX_rt_increase_MAX_BCR = np.max(
                            [item[0] for item in avg_rt_increase_of_each_model_config.values()
                            if item[1] == min_rt_increase_under_MAX_BCR and item[2] == max_BCR])
                        
                        reversed_dict = dict(zip(avg_rt_increase_of_each_model_config.values(),
                                                    avg_rt_increase_of_each_model_config.keys()))
                        
                        min_avg_rt_increase_of_each_node[node] = (reversed_dict[(
                                max_increased_acc_under_MAX_rt_increase_MAX_BCR, min_rt_increase_under_MAX_BCR,
                                max_BCR)],
                                                                    max_increased_acc_under_MAX_rt_increase_MAX_BCR,
                                                                    min_rt_increase_under_MAX_BCR,
                                                                    max_BCR)
                            
                if (len(min_avg_rt_increase_of_each_node) == 0):
                    if (order >= self.simple_paths_num - 1):
                        break
                    else:
                        order += 1
                        continue
                
                max_BCR = np.max([item[3] for item in min_avg_rt_increase_of_each_node.values()])
                max_increased_acc_under_MAX_rt_increase_MAX_BCR = np.max(
                    [item[1] for item in min_avg_rt_increase_of_each_node.values() if item[3] == max_BCR])
                target_node = [key for key in min_avg_rt_increase_of_each_node if
                                min_avg_rt_increase_of_each_node[key][3] == max_BCR and
                                min_avg_rt_increase_of_each_node[key][1] == max_increased_acc_under_MAX_rt_increase_MAX_BCR][0]
                
                target_model = min_avg_rt_increase_of_each_node[target_node][0]
                
                self.update_App_workflow_mem_rt(self.App,
                                                mem_dict={target_node: mem_list[target_node]},
                                                model_dict={target_node: target_model})
                curr_model_configuration[target_node] = target_model
                max_increased_acc_under_MAX_rt_increase_MAX_BCR = min_avg_rt_increase_of_each_node[target_node][1]
                min_rt_increase_under_MAX_BCR = min_avg_rt_increase_of_each_node[target_node][2]
                self.App.get_simple_dag()
                current_avg_rt = self.App.get_avg_rt()
                curr_accuracy = self.compute_accuracy(curr_model_configuration, accuracy_formula)
                performance_surplus = performance_surplus - min_rt_increase_under_MAX_BCR
                
                print(min_avg_rt_increase_of_each_node)
                print(self.accuracy_is_satisfied(curr_model_configuration, accuracy_constraint, accuracy_formula), performance_surplus, '\n\n')

        
        current_cost = self.App.get_avg_cost()
        
        self.App.get_simple_dag()
        current_avg_rt = self.App.get_avg_rt()
        performance_surplus = rt_constraint - current_avg_rt

        while (round(performance_surplus, 4) >= 0):
            iterations_count += 1
            cp = self.find_PRCP(leastCritical=True, order=order)
            max_cost_reduction_of_each_node = {}
            mem_backup = nx.get_node_attributes(self.App.workflow_graph, 'mem')
            for node in cp:
                if node in ['Start', 'End']:
                    continue
                cost_reduction_of_each_mem_config = {}
                for mem in self.App.workflow_graph.nodes[node][
                    'available_mem'][curr_model_configuration[node]]:
                    if (mem >= mem_backup[node]):
                        break
                    self.update_App_workflow_mem_rt(self.App, mem_dict={node: mem}, model_dict={node: curr_model_configuration[node]})
                    self.App.get_simple_dag()
                    temp_avg_rt = self.App.get_avg_rt()
                    increased_rt = temp_avg_rt - current_avg_rt
                    cost_reduction = current_cost - self.App.get_avg_cost()
                    if (increased_rt < performance_surplus and cost_reduction > 0):
                        cost_reduction_of_each_mem_config[mem] = (cost_reduction, increased_rt)
                self.update_App_workflow_mem_rt(self.App, mem_dict={node: mem_backup[node]}, model_dict={node: curr_model_configuration[node]})
                if (BCR and BCRtype == 'C/ERT'):
                    cost_reduction_of_each_mem_config = {item: cost_reduction_of_each_mem_config[item] for item in
                                                         cost_reduction_of_each_mem_config.keys() if
                                                         cost_reduction_of_each_mem_config[item][0] /
                                                         cost_reduction_of_each_mem_config[item][
                                                             1] > last_e2ert_cost_BCR * BCRthreshold}
                elif (BCR and BCRtype == "MAX"):
                    cost_reduction_of_each_mem_config = {item: (
                        cost_reduction_of_each_mem_config[item][0], cost_reduction_of_each_mem_config[item][1],
                        cost_reduction_of_each_mem_config[item][0] / cost_reduction_of_each_mem_config[item][1]) for
                        item in
                        cost_reduction_of_each_mem_config.keys()}
                if (len(cost_reduction_of_each_mem_config) != 0):
                    if (BCR and BCRtype == "MAX"):
                        max_BCR = np.max([item[2] for item in cost_reduction_of_each_mem_config.values()])
                        max_cost_reduction_under_MAX_BCR = np.max(
                            [item[0] for item in cost_reduction_of_each_mem_config.values() if
                             item[2] == max_BCR])
                        min_increased_rt_under_MAX_rt_reduction_MAX_BCR = np.min(
                            [item[1] for item in cost_reduction_of_each_mem_config.values() if
                             item[0] == max_cost_reduction_under_MAX_BCR and item[2] == max_BCR])
                        reversed_dict = dict(zip(cost_reduction_of_each_mem_config.values(),
                                                 cost_reduction_of_each_mem_config.keys()))
                        max_cost_reduction_of_each_node[node] = (reversed_dict[(
                            max_cost_reduction_under_MAX_BCR, min_increased_rt_under_MAX_rt_reduction_MAX_BCR,
                            max_BCR)],
                                                                 max_cost_reduction_under_MAX_BCR,
                                                                 min_increased_rt_under_MAX_rt_reduction_MAX_BCR,
                                                                 max_BCR)
                    else:
                        max_cost_reduction = np.max([item[0] for item in cost_reduction_of_each_mem_config.values()])
                        min_increased_rt_under_MAX_cost_reduction = np.min(
                            [item[1] for item in cost_reduction_of_each_mem_config.values() if
                             item[0] == max_cost_reduction])
                        reversed_dict = dict(
                            zip(cost_reduction_of_each_mem_config.values(), cost_reduction_of_each_mem_config.keys()))
                        max_cost_reduction_of_each_node[node] = (
                            reversed_dict[(max_cost_reduction, min_increased_rt_under_MAX_cost_reduction)],
                            max_cost_reduction,
                            min_increased_rt_under_MAX_cost_reduction)
            if (len(max_cost_reduction_of_each_node) == 0):
                if (order >= self.simple_paths_num - 1):
                    break
                else:
                    order += 1
                    continue
            if (BCR and BCRtype == "MAX"):
                max_BCR = np.max([item[3] for item in max_cost_reduction_of_each_node.values()])
                max_cost_reduction_under_MAX_BCR = np.max(
                    [item[1] for item in max_cost_reduction_of_each_node.values() if item[3] == max_BCR])
                target_node = [key for key in max_cost_reduction_of_each_node if
                               max_cost_reduction_of_each_node[key][3] == max_BCR and
                               max_cost_reduction_of_each_node[key][1] == max_cost_reduction_under_MAX_BCR][0]
                target_mem = max_cost_reduction_of_each_node[target_node][0]
            else:
                max_cost_reduction = np.max([item[1] for item in max_cost_reduction_of_each_node.values()])
                min_increased_rt_under_MAX_cost_reduction = np.min(
                    [item[2] for item in max_cost_reduction_of_each_node.values() if item[1] == max_cost_reduction])
                target_mem = np.min([item[0] for item in max_cost_reduction_of_each_node.values() if
                                     item[1] == max_cost_reduction and item[
                                         2] == min_increased_rt_under_MAX_cost_reduction])
                target_node = [key for key in max_cost_reduction_of_each_node if
                               max_cost_reduction_of_each_node[key] == (
                                   target_mem, max_cost_reduction, min_increased_rt_under_MAX_cost_reduction)][0]
            self.update_App_workflow_mem_rt(self.App, mem_dict={target_node: target_mem}, model_dict={target_node: curr_model_configuration[target_node]})
            max_cost_reduction = max_cost_reduction_of_each_node[target_node][1]
            min_increased_rt_under_MAX_cost_reduction = max_cost_reduction_of_each_node[target_node][2]
            current_cost = current_cost - max_cost_reduction
            performance_surplus = performance_surplus - min_increased_rt_under_MAX_cost_reduction
            current_avg_rt = current_avg_rt + min_increased_rt_under_MAX_cost_reduction
            current_e2ert_cost_BCR = max_cost_reduction / min_increased_rt_under_MAX_cost_reduction
            if (current_e2ert_cost_BCR == float('Inf')):
                last_e2ert_cost_BCR = 0
            else:
                last_e2ert_cost_BCR = current_e2ert_cost_BCR
        current_mem_configuration = nx.get_node_attributes(self.App.workflow_graph, 'mem')
        del current_mem_configuration['Start']
        del current_mem_configuration['End']
        print('Optimized Memory Configuration: {}'.format(current_mem_configuration))
        print('Average end-to-end response time: {}'.format(current_avg_rt))
        print('Optimized Accuracy Configuration: {}'.format(curr_model_configuration))
        print('Average Cost: {}'.format(current_cost))
        print('PRCPG_BCPC Optimization Completed.')
        return (current_avg_rt, current_cost, self.compute_accuracy(curr_model_configuration, accuracy_formula), current_mem_configuration, curr_model_configuration, iterations_count)
<<<<<<< HEAD

    
    def BAPB(self, rt_constraint, budget, accuracy_formula, BCR=False, BCRtype="RT/M", BCRthreshold=0.1):
        
        delta_rt = lambda new_rt, old_rt: abs(new_rt - old_rt) / (self.maximal_avg_rt - self.minimal_avg_rt) if (self.maximal_avg_rt - self.minimal_avg_rt) != 0 else 0
        delta_cost = lambda new_cost, old_cost: abs(new_cost - old_cost) / (self.maximal_cost - self.minimal_cost) if (self.maximal_cost - self.minimal_cost) != 0 else 0
        
        order = 0
        iterations_count = 0
        
        # First phase is finding the best possible RT under the budget constraint and the lowest possible model configuration.
        current_avg_rt, current_avg_cost, current_accuracy, curr_mem_configuration, curr_model_configuration, _ = self.BCPA(rt_constraint,
                                                                                                                    None, 
                                                                                                                    accuracy_formula, 
                                                                                                                    optimize_model_configuration=False, 
                                                                                                                    BCR=BCR, BCRtype=BCRtype, BCRthreshold=BCRthreshold)
        
        performance_surplus = rt_constraint - current_avg_rt
        budget_surplus = budget - current_avg_cost
        
        while (round(performance_surplus, 4) >= 0) and (round(budget_surplus, 4) >= 0):
            iterations_count += 1
            cp = self.find_PRCP(order=order, leastCritical=False)
            max_acc_increase_of_each_node = {}
            
            mem_backup = copy.deepcopy(curr_mem_configuration)
            model_backup = copy.deepcopy(curr_model_configuration)

=======

    
    def BAPB(self, rt_constraint, budget, accuracy_formula, BCR=False, BCRtype="RT/M", BCRthreshold=0.1):
        
        delta_rt = lambda new_rt, old_rt: abs(new_rt - old_rt) / (self.maximal_avg_rt - self.minimal_avg_rt) if (self.maximal_avg_rt - self.minimal_avg_rt) != 0 else 0
        delta_cost = lambda new_cost, old_cost: abs(new_cost - old_cost) / (self.maximal_cost - self.minimal_cost) if (self.maximal_cost - self.minimal_cost) != 0 else 0
        
        order = 0
        iterations_count = 0
        
        # First phase is finding the best possible RT under the budget constraint and the lowest possible model configuration.
        current_avg_rt, current_avg_cost, current_accuracy, curr_mem_configuration, curr_model_configuration, _ = self.BCPA(rt_constraint,
                                                                                                                    None, 
                                                                                                                    accuracy_formula, 
                                                                                                                    optimize_model_configuration=False, 
                                                                                                                    BCR=BCR, BCRtype=BCRtype, BCRthreshold=BCRthreshold)
        
        performance_surplus = rt_constraint - current_avg_rt
        budget_surplus = budget - current_avg_cost
        
        while (round(performance_surplus, 4) >= 0) and (round(budget_surplus, 4) >= 0):
            iterations_count += 1
            cp = self.find_PRCP(order=order, leastCritical=False)
            max_acc_increase_of_each_node = {}
            
            mem_backup = copy.deepcopy(curr_mem_configuration)
            model_backup = copy.deepcopy(curr_model_configuration)

>>>>>>> f26fbc8c44671b1ca1b8bee045531cfdf38d87ac
            for node in cp:
                if node in ['Start', 'End']:
                    continue
                
                bcr_values_for_each_change = {}

                for model_i, _ in enumerate(self.model_accuracy_list[node]):
                    if model_i <= model_backup[node]:
                        continue
                    
                    for mem in reversed(self.App.workflow_graph.nodes[node]['available_mem'][model_i]):
                        if mem <= mem_backup[node]:
                            break
                        self.update_App_workflow_mem_rt(self.App, mem_dict={node: mem}, model_dict={node: model_i})
                        curr_model_configuration[node] = model_i
                        temp_avg_cost = self.App.get_avg_cost()
                        self.App.get_simple_dag()
                        temp_avg_rt = self.App.get_avg_rt()
                        temp_accuracy = self.compute_accuracy(curr_model_configuration, accuracy_formula)
                        
                        increased_cost = temp_avg_cost - current_avg_cost
                        increased_rt = temp_avg_rt - current_avg_rt
                        increased_accuracy = temp_accuracy - current_accuracy
                        
                        if increased_accuracy < 0:
                            break
                        
                        bcr = (increased_accuracy) / (delta_cost(temp_avg_cost, current_avg_cost) + delta_rt(temp_avg_rt, current_avg_rt))
                        
                        if (increased_cost <= budget_surplus) and (increased_rt <= performance_surplus) and not np.isnan(bcr):
                            
                            bcr_values_for_each_change[(mem, model_i)] = (
                                bcr,
                                increased_accuracy,
                                increased_cost,
                                increased_rt
                            )
                            
                    self.update_App_workflow_mem_rt(self.App, mem_dict={node: mem_backup[node]}, model_dict={node: model_i})
                curr_model_configuration[node] = model_backup[node]
                self.update_App_workflow_mem_rt(self.App, mem_dict={node: mem_backup[node]}, model_dict={node: model_backup[node]})
                        
                if len(bcr_values_for_each_change) != 0:
                    max_BCR = np.max([item[0] for item in bcr_values_for_each_change.values()])
                    max_accuracy_increase_under_MAX_BCR = np.max(
                        [item[1] for item in bcr_values_for_each_change.values() if
                            item[0] == max_BCR])
                    min_increased_cost_under_MAX_BCR = np.min(
                        [item[2] for item in bcr_values_for_each_change.values() if
                            item[1] == max_accuracy_increase_under_MAX_BCR and item[0] == max_BCR])
                    min_increased_rt_under_MAX_BCR = np.min(
                        [item[3] for item in bcr_values_for_each_change.values() if
                            item[2] == min_increased_cost_under_MAX_BCR and item[1] == max_accuracy_increase_under_MAX_BCR and item[0] == max_BCR])
                    reversed_dict = dict(zip(bcr_values_for_each_change.values(),
                                                bcr_values_for_each_change.keys()))
                    max_acc_increase_of_each_node[node] = (reversed_dict[(
                        max_BCR, max_accuracy_increase_under_MAX_BCR, min_increased_cost_under_MAX_BCR,
                        min_increased_rt_under_MAX_BCR)],
                                                                max_accuracy_increase_under_MAX_BCR,
                                                                min_increased_cost_under_MAX_BCR,
                                                                min_increased_rt_under_MAX_BCR,
                                                                max_BCR)
                
            if (len(max_acc_increase_of_each_node) == 0):
                if (order >= self.simple_paths_num - 1):
                    break
                else:
                    order += 1
                    continue
            max_BCR = np.max([item[4] for item in max_acc_increase_of_each_node.values()])
            max_accuracy_increase_under_MAX_BCR = np.max(
                [item[1] for item in max_acc_increase_of_each_node.values() if item[4] == max_BCR])
            target_node = [key for key in max_acc_increase_of_each_node if
                            max_acc_increase_of_each_node[key][4] == max_BCR and
                            max_acc_increase_of_each_node[key][1] == max_accuracy_increase_under_MAX_BCR][0]
            target_mem = max_acc_increase_of_each_node[target_node][0][0]
            target_model = max_acc_increase_of_each_node[target_node][0][1]
            
            curr_mem_configuration[target_node] = target_mem
            curr_model_configuration[target_node] = target_model
            
            print('Target Node: {}, Target Mem: {}, Target Model: {}'.format(target_node, target_mem, target_model))
            
            self.update_App_workflow_mem_rt(self.App, mem_dict={target_node: target_mem}, model_dict={target_node: target_model})
            max_accuracy_increase = max_acc_increase_of_each_node[target_node][1]
            min_increased_cost_under_MAX_BCR = max_acc_increase_of_each_node[target_node][2]
            min_increased_rt_under_MAX_BCR = max_acc_increase_of_each_node[target_node][3]
            
            current_avg_cost = current_avg_cost + min_increased_cost_under_MAX_BCR
            current_avg_rt = current_avg_rt + min_increased_rt_under_MAX_BCR
            current_accuracy = current_accuracy + max_accuracy_increase
            
            budget_surplus = budget_surplus - min_increased_cost_under_MAX_BCR
            performance_surplus = performance_surplus - min_increased_rt_under_MAX_BCR
        
        current_mem_configuration = nx.get_node_attributes(self.App.workflow_graph, 'mem')
        del current_mem_configuration['Start']
        del current_mem_configuration['End']
        print('Optimized Memory Configuration: {}'.format(current_mem_configuration))
        print('Average end-to-end response time: {}'.format(current_avg_rt))
        print('Optimized Accuracy Configuration: {}'.format(curr_model_configuration))
        print('Average Cost: {}'.format(current_avg_cost))
        print('PRCPG_BAPB Optimization Completed.')
        return (current_avg_rt, current_avg_cost, self.compute_accuracy(curr_model_configuration, accuracy_formula), current_mem_configuration, curr_model_configuration, iterations_count)
            
            

    def get_opt_curve(self, filenameprefix, budget_list, performance_constraint_list, accuracy_constraint_list, accuracy_formula, BCRthreshold=0.2):
        '''
        Get the Optimization Curve and save as csv.
        '''
                
<<<<<<< HEAD
        # random.seed(42)  # For reproducibility
        # budget_list_copy = random.sample(budget_list, k=len(budget_list))
        # random.seed(24)  # For reproducibility
        # accuracy_constraint_list_copy = random.sample(accuracy_constraint_list, k=len(accuracy_constraint_list))
        # random.seed(42)  # For reproducibility
        # performance_constraint_list_copy = random.sample(performance_constraint_list, k=len(performance_constraint_list))
        
        budget_list_copy = budget_list
        accuracy_constraint_list_copy = accuracy_constraint_list
        performance_constraint_list_copy = performance_constraint_list
        
        # ----- BPBA -----
        bpba_rows = []                
        for budget, accuracy_constraint in list(itertools.product(budget_list_copy, accuracy_constraint_list_copy)):
=======
        random.seed(42)  # For reproducibility
        budget_list_copy = random.sample(budget_list, k=len(budget_list))
        random.seed(24)  # For reproducibility
        accuracy_constraint_list_copy = random.sample(accuracy_constraint_list, k=len(accuracy_constraint_list))
        random.seed(42)  # For reproducibility
        performance_constraint_list_copy = random.sample(performance_constraint_list, k=len(performance_constraint_list))
        
        
        # ----- BPBA -----
        bpba_rows = []                
        for budget, accuracy_constraint in zip(budget_list_copy, accuracy_constraint_list_copy):
>>>>>>> f26fbc8c44671b1ca1b8bee045531cfdf38d87ac
            aRow = {'Budget': budget, 'Accuracy_Constraint': accuracy_constraint, 'BCR_threshold': BCRthreshold}

            rt, cost, acc_score, mem_config, model_config, iterations = self.BPBA(budget, accuracy_constraint, accuracy_formula, BCR=False)
            aRow['BCR_disabled_RT'] = rt
            aRow['BCR_disabled_Cost'] = cost
            aRow['BCR_disabled_Config'] = mem_config
            aRow['BCR_disabled_Acc_Config'] = model_config
            aRow['BCR_disabled_Iterations'] = iterations
            aRow['BCR_disabled_Acc_Score'] = acc_score

            rt, cost, acc_score, mem_config, model_config, iterations = self.BPBA(budget, accuracy_constraint, accuracy_formula, BCR=True, BCRtype='RT/M', BCRthreshold=BCRthreshold)
            aRow['BCR_RT/M_RT'] = rt
            aRow['BCR_RT/M_Cost'] = cost
            aRow['BCR_RT/M_Config'] = mem_config
            aRow['BCR_RT/M_Acc_Config'] = model_config
            aRow['BCR_RT/M_Iterations'] = iterations
            aRow['BCR_RT/M_Acc_Score'] = acc_score

            rt, cost, acc_score, mem_config, model_config, iterations = self.BPBA(budget, accuracy_constraint, accuracy_formula, BCR=True, BCRtype='ERT/C', BCRthreshold=BCRthreshold)
            aRow['BCR_ERT/C_RT'] = rt
            aRow['BCR_ERT/C_Cost'] = cost
            aRow['BCR_ERT/C_Config'] = mem_config
            aRow['BCR_ERT/C_Acc_Config'] = model_config
            aRow['BCR_ERT/C_Iterations'] = iterations
            aRow['BCR_ERT/C_Acc_Score'] = acc_score

            rt, cost, acc_score, mem_config, model_config, iterations = self.BPBA(budget, accuracy_constraint, accuracy_formula, BCR=True, BCRtype='MAX')
            aRow['BCR_MAX_RT'] = rt
            aRow['BCR_MAX_Cost'] = cost
            aRow['BCR_MAX_Config'] = mem_config
            aRow['BCR_MAX_Acc_Config'] = model_config
            aRow['BCR_MAX_Iterations'] = iterations
            aRow['BCR_MAX_Acc_Score'] = acc_score  
            

            bpba_rows.append(aRow)

        BPBC_data = pd.DataFrame(bpba_rows)[
            ['Budget', 'Accuracy_Constraint', 'BCR_disabled_RT', 'BCR_RT/M_RT', 'BCR_ERT/C_RT', 'BCR_MAX_RT',
            'BCR_disabled_Cost', 'BCR_RT/M_Cost', 'BCR_ERT/C_Cost', 'BCR_MAX_Cost',
            'BCR_disabled_Config', 'BCR_RT/M_Config', 'BCR_ERT/C_Config', 'BCR_MAX_Config',
            'BCR_disabled_Acc_Config', 'BCR_RT/M_Acc_Config', 'BCR_ERT/C_Acc_Config', 'BCR_MAX_Acc_Config',
            'BCR_disabled_Acc_Score', 'BCR_RT/M_Acc_Score', 'BCR_ERT/C_Acc_Score', 'BCR_MAX_Acc_Score',
            'BCR_disabled_Iterations', 'BCR_RT/M_Iterations', 'BCR_ERT/C_Iterations',
            'BCR_MAX_Iterations', 'BCR_threshold']
        ]
        BPBC_data.to_csv(filenameprefix + '_BPBC.csv', index=False)

        # ----- BCPA -----
        bcpa_rows = []
<<<<<<< HEAD
        for perf_constraint, accuracy_constraint in list(itertools.product(performance_constraint_list_copy, accuracy_constraint_list_copy)):
=======
        for perf_constraint, accuracy_constraint in zip(performance_constraint_list_copy, accuracy_constraint_list_copy):
>>>>>>> f26fbc8c44671b1ca1b8bee045531cfdf38d87ac
            aRow = {'Performance_Constraint': perf_constraint, 'Accuracy_Constraint': accuracy_constraint, 'BCR_threshold': BCRthreshold}

            rt, cost, acc_score, mem_config, model_config, iterations = self.BCPA(perf_constraint, accuracy_constraint, accuracy_formula, BCR=False)
            aRow['BCR_disabled_RT'] = rt
            aRow['BCR_disabled_Cost'] = cost
            aRow['BCR_disabled_Config'] = mem_config
            aRow['BCR_disabled_Acc_Config'] = model_config
            aRow['BCR_disabled_Iterations'] = iterations
            aRow['BCR_disabled_Acc_Score'] = acc_score

            rt, cost, acc_score, mem_config, model_config, iterations = self.BCPA(perf_constraint, accuracy_constraint, accuracy_formula, BCR=True,
                                                        BCRtype='RT/M', BCRthreshold=BCRthreshold)
            aRow['BCR_M/RT_RT'] = rt
            aRow['BCR_M/RT_Cost'] = cost
            aRow['BCR_M/RT_Config'] = mem_config
            aRow['BCR_M/RT_Acc_Config'] = model_config
            aRow['BCR_M/RT_Iterations'] = iterations
            aRow['BCR_M/RT_Acc_Score'] = acc_score

            rt, cost, acc_score, mem_config, model_config, iterations = self.BCPA(perf_constraint, accuracy_constraint, accuracy_formula, BCR=True,
                                                        BCRtype='ERT/C', BCRthreshold=BCRthreshold)
            aRow['BCR_C/ERT_RT'] = rt
            aRow['BCR_C/ERT_Cost'] = cost
            aRow['BCR_C/ERT_Config'] = mem_config
            aRow['BCR_C/ERT_Acc_Config'] = model_config
            aRow['BCR_C/ERT_Iterations'] = iterations
            aRow['BCR_C/ERT_Acc_Score'] = acc_score

            rt, cost, acc_score, mem_config, model_config, iterations = self.BCPA(perf_constraint, accuracy_constraint, accuracy_formula, BCR=True, BCRtype='MAX')
            aRow['BCR_MAX_RT'] = rt
            aRow['BCR_MAX_Cost'] = cost
            aRow['BCR_MAX_Config'] = mem_config
            aRow['BCR_MAX_Acc_Config'] = model_config
            aRow['BCR_MAX_Iterations'] = iterations
            aRow['BCR_MAX_Acc_Score'] = acc_score

            bcpa_rows.append(aRow)

        BCPC_data = pd.DataFrame(bcpa_rows)[
            ['Performance_Constraint', 'Accuracy_Constraint', 'BCR_disabled_RT', 'BCR_M/RT_RT', 'BCR_C/ERT_RT', 'BCR_MAX_RT',
            'BCR_disabled_Cost', 'BCR_M/RT_Cost', 'BCR_C/ERT_Cost', 'BCR_MAX_Cost',
            'BCR_disabled_Config', 'BCR_M/RT_Config', 'BCR_C/ERT_Config', 'BCR_MAX_Config',
            'BCR_disabled_Acc_Config', 'BCR_M/RT_Acc_Config', 'BCR_C/ERT_Acc_Config', 'BCR_MAX_Acc_Config',
            'BCR_disabled_Acc_Score', 'BCR_M/RT_Acc_Score', 'BCR_C/ERT_Acc_Score', 'BCR_MAX_Acc_Score',
            'BCR_disabled_Iterations', 'BCR_M/RT_Iterations', 'BCR_C/ERT_Iterations',
            'BCR_MAX_Iterations', 'BCR_threshold']
        ]
        BCPC_data.to_csv(filenameprefix + '_BCPC.csv', index=False)
        
        
        # ----- BAPB -----
        bapb_rows = []
<<<<<<< HEAD
        for performance_constraint, budget in list(itertools.product(performance_constraint_list_copy, budget_list_copy)):
=======
        for performance_constraint, budget in zip(performance_constraint_list_copy, budget_list_copy):
>>>>>>> f26fbc8c44671b1ca1b8bee045531cfdf38d87ac
            aRow = {'Performance_Constraint': performance_constraint, 'Budget': budget, 'BCR_threshold': BCRthreshold}
        
            rt, cost, acc_score, mem_config, model_config, iterations = self.BAPB(performance_constraint, budget, accuracy_formula, BCR=True, BCRtype='ERT/C', BCRthreshold=BCRthreshold)
            aRow['BCR_disabled_RT'] = rt
            aRow['BCR_disabled_Cost'] = cost
            aRow['BCR_disabled_Config'] = mem_config
            aRow['BCR_disabled_Acc_Config'] = model_config
            aRow['BCR_disabled_Iterations'] = iterations
            aRow['BCR_disabled_Acc_Score'] = acc_score
            
            bapb_rows.append(aRow)
            
        BAPB_data = pd.DataFrame(bapb_rows)[
            ['Performance_Constraint', 'Budget', 'BCR_disabled_RT', 'BCR_disabled_Cost', 'BCR_disabled_Config',
             'BCR_disabled_Acc_Config', 'BCR_disabled_Acc_Score', 'BCR_disabled_Iterations', 'BCR_threshold']
        ]
        BAPB_data.to_csv(filenameprefix + '_BAPB.csv', index=False)