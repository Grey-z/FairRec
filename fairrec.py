import math
import random
import numpy as np
import time
from itertools import filterfalse
import pandas as pd
from rec_metrics import RecMetrics
from utilis import get_rule_list, create_single_rule, create_sparse_rule_query,even


def set_dpso_params(dpso_params,test_info):
    
    """
    The implementation of initialization.

    Args:
    dpso_params: dict, containing initial params of dpso
        dpso_params['epsilon']: float, the proportion of the size of initial particles to the number of user groups.
        dpso_params['iter']:int, number of iterations
        dpso_params['v_low']:float, minimum limit on the speed of each particle
        dpso_params['v_high']:float, maximum limit on the speed of each particle
    test_info：dict, including test_data and sensitive_features.
        test_info['test_data']: pandas, data containing information about the users to be tested.
        test_info['sensitive_features']: list, sensitive attributes to be tested.
    
    Return:
    dpso_params: dict, containing initial params of fairrec.
            dpso_params['dimension']: int, number of sensitive attributes.
            dpso_params['iter']:int, number of iterations.
            dpso_params['size']:int, number of particles.
            dpso_params['bound_low']: list, the minimum value of each sensitive attribute after encoding.
            dpso_params['bound_up']: list, the maximum value of each sensitive attribute after encoding.
            dpso_params['v_low']:float, minimum limit on the speed of each particle.
            dpso_params['v_high']:float, maximum limit on the speed of each particle.
            dpso_params['value_prob_list']:list, the probability distributions of users.
    """
    
    bound_low = []
    bound_up = []
    group_total = 1
    epsilon = dpso_params['epsilon']
    test_data = test_info['test_data']
    sensitive_features = test_info['sensitive_features']
    dpso_params['dimension'] = len(sensitive_features)
    
    for feat in sensitive_features:
        max_value = test_data[feat].max()
        min_value = test_data[feat].min()
        bound_low.append(min_value)
        bound_up.append(max_value)
        group_total = group_total * (max_value - min_value + 1)
    
    dpso_params['size'] = int(group_total * epsilon)
    dpso_params['bound_low'] = bound_low
    dpso_params['bound_up'] = bound_up
    value_prob_list = []
    
    for feat in sensitive_features:
        feats_counts = test_data[feat].value_counts()
        feats_values = list(feats_counts.index)
        prob = list(feats_counts.values)
        feats_prob = [x/sum(prob) for x in prob]
        value_prob_list.append([feats_values,feats_prob])
    dpso_params['value_prob_list'] = value_prob_list
    return dpso_params

def run_fair_rec(dpso_params,test_info):
    """
    An example for the testing using fairrec.

    Args:
    dpso_params: dict, containing initial params of dpso.
        dpso_params['epsilon']: float, the proportion of the size of initial particles to the number of user groups.
        dpso_params['iter']:int, number of iterations.
        dpso_params['v_low']:float, minimum limit on the speed of each particle.
        dpso_params['v_high']:float, maximum limit on the speed of each particle.
    test_info：dict, including test_data and sensitive_features.
        test_info['test_data']: pandas, data containing information about the users to be tested.
        test_info['sensitive_features']: list, sensitive attributes to be tested.
        test_info['metric']: str, metric used in the fairness testing.
        test_info['threshold']:float, criteria for filtering groups.
        test_info['test_model']: model, recommendation model to be tested.
        test_info['train_features']:list, features for model training.
        
    Return:
    uf_score: float, the result of unfairness score.
    test_time: float, time consumed for the testing.
    """
    start_time = time.time()
    dpso_params = set_dpso_params(dpso_params,test_info)
    infobase = {}
    fair_rec = FairRec(dpso_params,test_info,infobase)
    adv_group,adv_score, disadv_group,disadv_score = fair_rec.double_ended_dpso()

    end_time = time.time()
    value = list(infobase.values())
    value[:] = filterfalse(even, value)
    adv_score = max(value)
    disadv_score = min(value)
    uf_score = adv_score - disadv_score
    test_time = end_time - start_time
    return uf_score, test_time

class FairRec:
    
    """
    The implementation of FairRec for recommender systems.
    """
    
    def __init__(self, dpso_params,test_info,infobase):
        
        """
        The implementation of initialization.

        Args:
        dpso_params: dict, containing initial params of fairrec.
            dpso_params['dimension']: int, number of sensitive attributes.
            dpso_params['iter']:int, number of iterations.
            dpso_params['size']:int, number of particles.
            dpso_params['bound_low']: list, the minimum value of each sensitive attribute after encoding.
            dpso_params['bound_up']: list, the maximum value of each sensitive attribute after encoding.
            dpso_params['v_low']:float, minimum limit on the speed of each particle.
            dpso_params['v_high']:float, maximum limit on the speed of each particle.
            dpso_params['value_prob_list']:list, the probability distributions of users.
        test_info：dict, including test_data, sensitive_features,metric,threshold,test_model and train_features.
            test_info['test_data']: pandas, data containing information about the users to be tested.
            test_info['sensitive_features']: list, sensitive attributes to be tested.
            test_info['metric']: str, metric used in the fairness testing.
            test_info['threshold']:float, criteria for filtering groups.
            test_info['test_model']: model, recommendation model to be tested.
            test_info['train_features']:list, features for model training.
        infobase: dict, an empty dict to store the results of all groups.
        """
        
        self.dimension = dpso_params['dimension']  
        self.iter = dpso_params['iter']  
        self.size = dpso_params['size']   
        self.bound = []   
        self.bound.append(dpso_params['bound_low'])
        self.bound.append(dpso_params['bound_up'])
        self.v_low = dpso_params['v_low']
        self.v_high = dpso_params['v_high']
        self.value_prob_list = dpso_params['value_prob_list']
        
        self.test_info = test_info
        self.x_max = np.zeros((self.size, self.dimension))  
        self.v_max = np.zeros((self.size, self.dimension))  
        self.p_best_max = np.zeros((self.size, self.dimension))  
        self.g_best_max = np.zeros((1, self.dimension))[0]  
        
        self.x_min = np.zeros((self.size, self.dimension))  
        self.v_min = np.zeros((self.size, self.dimension))  
        self.p_best_min = np.zeros((self.size, self.dimension))  
        self.g_best_min = np.zeros((1, self.dimension))[0]  
        self.infobase = infobase   

        # Initialize the position, velocity, Pbest of each particles in two swarms with different search targets.
        temp = -1000000
        x_temp = np.zeros((self.dimension, self.size))
        for i in range(self.dimension):
            x_temp[i] = np.random.choice(a=self.value_prob_list[i][0], size=self.size, replace=True, p=self.value_prob_list[i][1])
        self.x_max = x_temp.T
        for i in range(self.size):
            for j in range(self.dimension):
                self.v_max[i][j] = round(random.uniform(self.v_low, self.v_high))
            self.p_best_max[i] = self.x_max[i]  
            fit = self.fitness(self.p_best_max[i])
            if fit > temp:
                self.g_best_max = self.p_best_max[i]
                temp = fit
        temp = 1000000
        x_temp = np.zeros((self.dimension, self.size))
        for i in range(self.dimension):
            x_temp[i] = np.random.choice(a=self.value_prob_list[i][0], size=self.size, replace=True, p=self.value_prob_list[i][1])
        self.x_min = x_temp.T
        for i in range(self.size):
            for j in range(self.dimension):
                self.v_min[i][j] = round(random.uniform(self.v_low, self.v_high))
            self.p_best_min[i] = self.x_min[i]  
            fit = self.fitness(self.p_best_min[i])
            if (fit < temp) and (fit > -90):
                self.g_best_min = self.p_best_min[i]
                temp = fit

    def fitness(self, position):
        
        """
        Calculate fitness of each particle.
        
        Args:
        position: the position of particles.
        
        Return:
        fitness: float, the fitness of current position.
        """
        
        index = str(position)
        if index in self.infobase:
            score = self.infobase[index]
        else:
            score = self.get_score(position,self.test_info)
            self.infobase[index] = score
        fitness = score
        return fitness
    
    def get_score(self,position,test_info):
        
        """
        Calculate metrics of selected groups.
        
        Args:
        position: the position of particles.
        test_info：dict, including test_data, sensitive_features,metric,threshold,test_model and train_features.
        
        Return:
        score_target: float, the metrics of target groups.
        """
        
        metric = test_info['metric']
        sensitive_features = test_info['sensitive_features']
        test_data = test_info['test_data']      
        model = test_info['test_model']
        train_features = test_info['train_features']
        threshold = test_data['user_id'].nunique() * test_info['threshold']
        
        rule = create_single_rule(sensitive_features,position)
        rule_query = create_sparse_rule_query(rule)
        group_target = test_data.query(rule_query)
        if (group_target['user_id'].nunique()) <= (threshold):
            score_target = -100 
        else:
            test_input = {name: group_target[name].values for name in train_features}
            y_pred = model.predict(test_input)   
            group_target['predict'] = np.array(y_pred)
            if metric == 'mrr':
                score_target = RecMetrics(group_target).mrr_score()
            if metric == 'auc':
                score_target = RecMetrics(group_target).auc_score()
            if metric == 'ndcg':
                score_target = RecMetrics(group_target).ndcg_score()
            if metric == 'urd':
                score_target = RecMetrics(group_target).cate_diversity_score()
            if metric == 'urp':
                score_target = RecMetrics(group_target).popularity_score()
        return score_target
    
    def update(self):   
        
        """
        The implementation of update process.
        """
        
        c1 = 2.0  
        c2 = 2.0
        alpha = 0.09  
        c_max = self.p_best_max.mean(axis=0)
        c_min = self.p_best_min.mean(axis=0)

        for i in range(self.size):
            # update velocity of each particle.
            phi = np.random.normal()
            # we introduce the thermal motion, as the first term of the following equation.
            self.v_max[i] = alpha * np.abs(c_max - self.x_max[i]) * phi + c1 * random.uniform(0, 1) * (
                 self.p_best_max[i] - self.x_max[i]) + c2 * random.uniform(0, 1) * (self.g_best_max - self.x_max[i])
            
            self.v_min[i] = alpha * np.abs(c_min - self.x_min[i]) * phi + c1 * random.uniform(0, 1) * (
                 self.p_best_min[i] - self.x_min[i]) + c2 * random.uniform(0, 1) * (self.g_best_min - self.x_min[i])
            
            for j in range(self.dimension):
                if self.v_max[i][j] < self.v_low:
                    self.v_max[i][j] = self.v_low
                if self.v_max[i][j] > self.v_high:
                    self.v_max[i][j] = self.v_high
                
                if self.v_min[i][j] < self.v_low:
                    self.v_min[i][j] = self.v_low
                if self.v_min[i][j] > self.v_high:
                    self.v_min[i][j] = self.v_high

            # update position of each particle.
            self.x_max[i] = list(map(int,self.x_max[i] + self.v_max[i]))
            self.x_min[i] = list(map(int,self.x_min[i] + self.v_min[i]))
            
            for j in range(self.dimension):
                if self.x_max[i][j] < self.bound[0][j]:
                    self.x_max[i][j] = self.bound[0][j]
                if self.x_max[i][j] > self.bound[1][j]:
                    self.x_max[i][j] = self.bound[1][j]
                    
                if self.x_min[i][j] < self.bound[0][j]:
                    self.x_min[i][j] = self.bound[0][j]
                if self.x_min[i][j] > self.bound[1][j]:
                    self.x_min[i][j] = self.bound[1][j]
                    
            # update p_best and g_best.
            if self.fitness(self.x_max[i]) > self.fitness(self.p_best_max[i]):
                self.p_best_max[i] = self.x_max[i]
            
            if (self.fitness(self.x_min[i]) < self.fitness(self.p_best_min[i])) or (self.fitness(self.p_best_min[i]) < -90):
                if self.fitness(self.x_min[i]) > -90 :
                    self.p_best_min[i] = self.x_min[i]
                
            if self.fitness(self.x_max[i]) > self.fitness(self.g_best_max):
                self.g_best_max = self.x_max[i]
            if self.fitness(self.x_min[i]) > self.fitness(self.g_best_max):
                self.g_best_max = self.x_min[i]
                
            if self.fitness(self.x_max[i]) < self.fitness(self.g_best_min) or (self.fitness(self.g_best_min) < -90):
                if self.fitness(self.x_max[i]) > -90 :
                    self.g_best_min = self.x_max[i]
                    
            if self.fitness(self.x_min[i]) < self.fitness(self.g_best_min) or (self.fitness(self.g_best_min) < -90):
                if self.fitness(self.x_min[i]) > -90 :
                    self.g_best_min = self.x_min[i]

    def double_ended_dpso(self):
        
        """
        The implementation of double-ended discrete particle swarm optimization algorithm.
        
        Return:
        group_max: the position of most advatantaged groups.
        score_max: the metric score of most advatantaged groups.
        group_min: the position of most disadvatantaged groups.
        score_min: the metric score of most disadvatantaged groups.
        """
        
        best = []
        self.final_best_max = np.zeros(self.dimension)
        self.final_best_min = np.zeros(self.dimension)
        for gen in range(self.iter):
            self.update()
            if self.fitness(self.g_best_max) > self.fitness(self.final_best_max):
                self.final_best_max = self.g_best_max.copy()
            fitness_temp = self.fitness(self.final_best_min)
            if (self.fitness(self.g_best_min) < fitness_temp) or (fitness_temp < -90):
                self.final_best_min = self.g_best_min.copy()
            group_max = self.final_best_max
            score_max = self.fitness(self.final_best_max)
            group_min = self.final_best_min
            score_min = self.fitness(self.final_best_min)
        return group_max, score_max, group_min, score_min

