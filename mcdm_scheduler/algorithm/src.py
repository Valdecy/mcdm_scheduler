############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# MCDM Scheduler

# Citation: 
# PEREIRA, V. (2024). Project: MCDM Scheduler, GitHub repository: <https://github.com/Valdecy/MCDM_Scheduler>

############################################################################

# Required Libraries
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.style.use('ggplot')
import numpy as np

from mcdm_scheduler.util.fuzzy_ppf_ahp import ppf_ahp_method
from mcdm_scheduler.util.ga            import genetic_algorithm
from mcdm_scheduler.util.ht2fs         import ht2fs_weight_calculation

############################################################################

# MCDM Scheduler Class
class load_mcdm_scheduler():
    def __init__(self, sequences = [], due_dates = [], setup_time_matrix = [], setup_waste_matrix = [], comparison_matrix = [], crisp_inputs = [], uncertainty_ranges = [], criteria_importance = [], population_size = 5, elite = 1, mutation_rate = 0.1, generations = 100, custom_job_weights = [], custom_objective_weights = [], custom_job_sequence = [], brute_force = False, parallel = False): 
      self.job_weights                    = custom_job_weights         # Job Weights: Opitional
      self.objectives_weights             = custom_objective_weights   # Objectives Weights (Makespan, Max WeightedTardiness, Total Waste, Total Setup Time): Opitional
      self.custom_job_sequence            = custom_job_sequence
      self.sequences                      = sequences                  # Job Shop Scheduling Input: Mandatory
      self.due_dates                      = due_dates                  # Job Shop Scheduling Input: Only Relevant if Due Date    is a measure
      self.setup_time_matrix              = setup_time_matrix          # Job Shop Scheduling Input: Only Relevant if Setup Time  is a measure
      self.setup_waste_matrix             = setup_waste_matrix         # Job Shop Scheduling Input: Only Relevant if Setup Waste is a measure
      self.comparison_matrix              = comparison_matrix          # PPF-AHP             Input: Only Relevant if custom_objectives_weights = []
      self.crisp_inputs                   = crisp_inputs               # HT2FS               Input: Only Relevant if custom_job_weights        = []
      self.uncertainty_ranges             = uncertainty_ranges         # HT2FS               Input: Only Relevant if custom_job_weights        = []
      self.criteria_importance            = criteria_importance        # HT2FS               Input: Only Relevant if custom_job_weights        = []
      self.population_size                = population_size            # GA
      self.elite                          = int(elite)                 # GA
      self.mutation_rate                  = mutation_rate              # GA
      self.generations                    = generations                # GA
      self.brute_force                    = brute_force
      self.parallel                       = parallel
      self.machine_sequences, self.matrix = self.sequence_inputs()
      self.num_jobs                       = len(self.sequences)
      self.num_machines                   = self.matrix.shape[1]
    
    ###############################################################################
    
    # Plot
    def create_gantt_chart(self, schedule_matrix, size_x = 12, size_y = 8):
        total_length   = schedule_matrix.shape[1]
        fig, ax        = plt.subplots(figsize = (size_x, size_y))
        job_color_dict = {}
        idle_patches   = []
        job_colors     = itertools.cycle([
                                            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', 
                                            '#7f7f7f', '#bcbd22', '#17becf', '#bf77f6', '#ff9408', '#d1ffbd', '#c85a53', 
                                            '#3a18b1', '#ff796c', '#04d8b2', '#ffb07c', '#aaa662', '#0485d1', '#fffe7a', 
                                            '#b0dd16', '#6f7be3', '#12e193', '#82cafc', '#ac9362', '#f8481c', '#c292a1', 
                                            '#c0fa8b', '#ca7b80', '#f4d054', '#fbdd7e', '#ffff7e', '#cd7584', '#f9bc08', 
                                            '#c7c10c'
                                          ])
        for machine in range(0, self.num_machines):
            for time in range(0, total_length):
                job = schedule_matrix[machine][time]
                if (job):
                    if (job not in job_color_dict):
                        job_color_dict[job] = next(job_colors)
                    color = job_color_dict[job]
                    ax.add_patch(mpatches.Rectangle((time, self.num_machines - machine - 1), 1, 1, edgecolor = 'black', facecolor = color))
                    ax.text(time + 0.5, self.num_machines - machine - 0.5, job, ha = 'center', va = 'center', color = 'black', fontsize = 8, weight = 'bold')
                else:
                    left_hatch  = mpatches.Rectangle((time, self.num_machines - machine - 1), 1, 1, edgecolor = 'grey', facecolor = 'none', hatch = '//')
                    right_hatch = mpatches.Rectangle((time, self.num_machines - machine - 1), 1, 1, edgecolor = 'grey', facecolor = 'none', hatch = '\\')
                    ax.add_patch(left_hatch)
                    ax.add_patch(right_hatch)
                    idle_patches.append((time, machine, left_hatch, right_hatch))
        for machine in range(0, self.num_machines):
            for time in range(total_length - 1, -1, -1):
                if (schedule_matrix[machine][time] == ''):
                    for patch in idle_patches:
                        if patch[0] == time and patch[1] == machine:
                            patch[2].remove()
                            patch[3].remove()
                else:
                    break
        ax.set_xlim(0, total_length)
        ax.set_ylim(0, self.num_machines)
        ax.set_xticks(np.arange(0, total_length + 1, 1))  
        ax.set_xticklabels(np.arange(0, total_length + 1, 1))  
        ax.set_yticks(np.arange(0, self.num_machines, 1))
        ax.set_yticklabels([])
        for i in range(0, self.num_machines):
            ax.text(-0.5, self.num_machines - i - 0.5, f'Machine {i}', va = 'center', ha = 'right', fontsize = 10)
        ax.set_xlabel('Time')
        ax.set_title('Gantt Chart')
        ax.grid(True, linestyle = '--', alpha = 0.7)
        plt.show()
    
    ###############################################################################
    
    # Objectives
    def calculate_makespan(self, schedule_matrix):
        total_length = schedule_matrix.shape[1]
        makespan     = 0
        for machine in range(0, self.num_machines):
            for time in range(total_length - 1, -1, -1):
                if (schedule_matrix[machine][time] != ''):
                    makespan = max(makespan, time + 1)
                    break
        return makespan
    
    def calculate_max_weighted_tardiness(self, schedule_matrix):
        completion_times = {f'j{job}': 0 for job in range(0, self.num_jobs)}
        total_length     = schedule_matrix.shape[1]
        for machine in range(0, self.num_machines):
            for time in range(0, total_length):
                job = schedule_matrix[machine][time]
                if (job):
                    completion_times[job] = time + 1
        max_weighted_tardiness = 0
        for job in range(0, self.num_jobs):
            tardiness              = max(0, completion_times[f'j{job}'] - self.due_dates[job])
            weighted_tardiness     = self.job_weights[job] * tardiness
            max_weighted_tardiness = max(max_weighted_tardiness, weighted_tardiness)
        return max_weighted_tardiness
    
    def calculate_total_waste(self, permutation):
        total_waste = 0
        for i in range(len(permutation) - 1):
            total_waste = total_waste + self.setup_waste_matrix[permutation[i]][permutation[i+1]]
        return total_waste
    
    def calculate_total_setup_time(self, permutation):
        total_setup_time = 0
        for i in range(len(permutation) - 1):
            total_setup_time = total_setup_time + self.setup_time_matrix[permutation[i]][permutation[i+1]]
        return total_setup_time
    
    ###############################################################################
    
    # Inputs
    def sequence_inputs(self):
        num_jobs     = len(self.sequences)
        num_machines = max(max(machine for machine, _ in job) for job in self.sequences) + 1
        matrix       = np.zeros((num_jobs, num_machines), dtype = int)
        for job_id, job in enumerate(self.sequences):
            for machine, time in job:
                matrix[job_id, machine] = time
        machine_sequences = []
        for job_id, job in enumerate(self.sequences):
            machine_sequence = [machine for machine, _ in job]
            machine_sequences.append(machine_sequence)
        return machine_sequences, matrix
    
    ###############################################################################
    
    # Schedule
    def schedule_jobs(self, permutation):
        total_length      = np.sum(self.matrix)
        schedule          = [['' for _ in range(0, total_length)] for _ in range(0, self.num_machines)]
        machine_end_times = [0] * self.num_machines
        job_end_times     = [0] * self.num_jobs
        if (self.parallel == False):
            for job_id in permutation:
                operations = self.machine_sequences[job_id]
                for op_index, machine in enumerate(operations):
                    time_required = self.matrix[job_id, machine]
                    start_time    = job_end_times[job_id]
                    while any(schedule[machine][start_time:start_time + time_required]):
                        start_time = start_time + 1
                    end_time = start_time + time_required
                    for t in range(start_time, end_time):
                        schedule[machine][t] = f"j{job_id}"
                    job_end_times[job_id]      = end_time
                    machine_end_times[machine] = end_time
        else:        
            for job_id in permutation:
                earliest_start_time = float('inf')
                best_machine = None
                
                for machine, time_required in self.sequences[job_id]:
                    start_time = machine_end_times[machine]
                    while any(schedule[machine][start_time:start_time + time_required]):
                        start_time = start_time + 1
                    if (start_time < earliest_start_time):
                        earliest_start_time = start_time
                        best_machine = machine
                time_required = self.matrix[job_id, best_machine]
                end_time = earliest_start_time + time_required
                for t in range(earliest_start_time, end_time):
                    schedule[best_machine][t] = f"j{job_id}"
                machine_end_times[best_machine] = end_time
                job_end_times[job_id] = end_time
        max_time = max(len(row) for row in schedule)
        for col in range(max_time - 1, -1, -1):
            if (all(row[col] == '' for row in schedule)):
                for row in schedule:
                    row.pop()
            else:
                break
        return np.array(schedule)
    
    def brute_force_search(self):
        job_ids                 = list(range(len(self.sequences)))
        best_sequence           = None
        minimal_objective_value = float('inf')
        for permutation in itertools.permutations(job_ids):
            schedule_matrix = self.schedule_jobs(permutation)
            objective_value = 0.0 / 1.0
            if (self.objectives_weights[0] != 0):
                makespan               = self.calculate_makespan(schedule_matrix)
                objective_value        = objective_value + self.objectives_weights[0] * makespan
            if (self.objectives_weights[1] != 0):
                max_weighted_tardiness = self.calculate_max_weighted_tardiness(schedule_matrix)
                objective_value        = objective_value + self.objectives_weights[1] * max_weighted_tardiness
            if (self.objectives_weights[2] != 0):
                total_waste            = self.calculate_total_waste(permutation)
                objective_value        = objective_value + self.objectives_weights[2] * total_waste
            if (self.objectives_weights[3] != 0):
                total_setup_time       = self.calculate_total_setup_time(permutation)
                objective_value        = objective_value + self.objectives_weights[3] * total_setup_time
            if (objective_value < minimal_objective_value):
                minimal_objective_value = objective_value
                best_sequence           = permutation
        self.best_sequence = best_sequence
        return best_sequence, minimal_objective_value
    
    def target_function(self, permutation): 
        schedule_matrix = self.schedule_jobs(permutation)
        objective_value = 0.0 / 1.0
        if (self.objectives_weights[0] != 0):
            makespan               = self.calculate_makespan(schedule_matrix)
            objective_value        = objective_value + self.objectives_weights[0] * makespan
        if (self.objectives_weights[1] != 0):
            max_weighted_tardiness = self.calculate_max_weighted_tardiness(schedule_matrix)
            objective_value        = objective_value + self.objectives_weights[1] * max_weighted_tardiness
        if (self.objectives_weights[2] != 0):
            total_waste            = self.calculate_total_waste(permutation)
            objective_value        = objective_value + self.objectives_weights[2] * total_waste
        if (self.objectives_weights[3] != 0):
            total_setup_time       = self.calculate_total_setup_time(permutation)
            objective_value        = objective_value + self.objectives_weights[3] * total_setup_time
        return objective_value
    
    def run_mcdm_scheduler(self):
        if (len(self.objectives_weights) == 0):
            weights = [[] for item in self.comparison_matrix]
            rc      = [[] for item in self.comparison_matrix]
            g_mean  = []
            i       = 0
            for item in self.comparison_matrix:
                weights[i], rc[i] = ppf_ahp_method(item)
                if (len(g_mean) == 0):
                    g_mean = weights[i]
                else:
                    g_mean = [g_mean[j]*weights[i][j] for j in range(0, len(weights[i]))]
                i = i + 1
            g_mean = [g_mean[j]**(1/len(weights)) for j in range(0, len(g_mean))]   
            self.objectives_weights = g_mean
        if (len(self.job_weights) == 0):
            for k in range(0, len(self.crisp_inputs)):
                job_weights = ht2fs_weight_calculation(self.crisp_inputs[k], self.uncertainty_ranges[k], self.criteria_importance[k])
                if (len(self.job_weights) == 0):
                    self.job_weights = job_weights
                else:
                    self.job_weights = [self.job_weights[j] + job_weights[j] for j in range(0, len(self.job_weights))]
            self.job_weights = [x / sum(self.job_weights) for x in self.job_weights]
        if (self.brute_force == False and len(self.custom_job_sequence) == 0):
            job_sequence, obj_fun = genetic_algorithm(self.num_jobs, self.population_size, self.elite, self.mutation_rate, self.generations, self.target_function, False)
            schedule_matrix       = self.schedule_jobs(job_sequence) 
        if (self.brute_force == False and len(self.custom_job_sequence) != 0):
            job_sequence    = self.custom_job_sequence
            schedule_matrix = self.schedule_jobs(job_sequence) 
            obj_fun         = self.target_function(self.custom_job_sequence)
        if (self.brute_force == True and len(self.custom_job_sequence) == 0):
            job_sequence, obj_fun = self.brute_force_search()
            schedule_matrix       = self.schedule_jobs(job_sequence)
        return job_sequence, schedule_matrix, obj_fun
    
############################################################################
