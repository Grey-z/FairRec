import numpy as np
import pandas as pd


class RecMetrics(object):
    def __init__(self, evaluate_data,k=5):
        self.users = np.array(evaluate_data['user_id'].to_list())
        self.y_true = np.array(evaluate_data['label'].to_list())
        self.y_pred = np.array(evaluate_data['predict'].to_list())
        self.item_id = np.array(evaluate_data['item_id'].to_list())
        self.item_cate = np.array(evaluate_data['cate'].to_list())
        self.item_pop = np.array(evaluate_data['popularity'].to_list())
        self.user_pop = np.array(evaluate_data['user_pop'].to_list())
        self.k = k
        self.select_cols = [self.y_true,self.y_pred,self.item_id,self.item_cate,self.item_pop,self.user_pop]
        
    def get_user_pred(self):
        """
        divide the result into different group by user id

        Args:
        y_true: array, all true labels of the data
        y_pred: array, the predicted score
        users: array, user id

        Return:
        user_pred: dict, key is user id and value is the labels and scores of each user
        """
        user_pred = {}
        for i, u in enumerate(self.users):
            if u not in user_pred:
                user_pred[u] = [[feat[i]] for feat in self.select_cols]
            else:
                for index in range(len(self.select_cols)):
                    user_pred[u][index].append(self.select_cols[index][i])
        return user_pred
    
    def get_user_topk(self):
        
        """
        sort y_pred and find topk results
        this function is used to find topk predicted scores
        and the corresponding index is applied to find the corresponding labels

        """
        user_pred = self.get_user_pred()
        for u in user_pred:
            idx = np.argsort(user_pred[u][1])[::-1][:self.k]
            for i in range(len(self.select_cols)):
                user_pred[u][i] = np.array(user_pred[u][i])[idx]
        return user_pred

    def auc_score(self):
        return roc_auc_score(self.y_true, self.y_pred)
    
    def gauc_score(self,weights=None):
        """
        Args:
        y_true: array, dim(N, ), all true labels of the data
        y_pred: array, dim(N, ), the predicted score
        users: array, dim(N, ), user id
        weight: dict, it contains weights for each group.
            if it is None, the weight is equal to the number
            of times the user is recommended
        Return:
        score: float, GAUC
        """
        assert len(self.y_true) == len(self.y_pred) and len(self.y_true) == len(self.users)

        user_pred = self.get_user_topk()
        score = 0
        num = 0
        for u in user_pred.keys():
            auc = auc_score(user_pred[u][0], user_pred[u][1])
            if weights is None:
                wg = len(user_pred[u][0])
            else:
                wg = weights[u]
            auc *= wg
            num += wg
            score += auc
        return score / num
    
    def log_loss(self):
        score = self.y_true * np.log(self.y_pred) + (1 - self.y_true) * np.log(1 - self.y_pred)
        return -score.sum() / len(self.y_true)
    
    def ndcg_score(self):
        """compute NDCG
        Args:
        user_pred: dict, computed by get_user_topk()
        """
        user_pred = self.get_user_topk()
        rank = np.arange(1, self.k+1, 1)
        idcgs = 1. / np.log2(rank + 1)
        idcg = sum(idcgs)
        score = 0
        for u in user_pred:
            dcgs = idcgs[np.where(user_pred[u][0] == 1)]
            dcg = sum(dcgs)
            score += dcg / idcg
        return score / len(user_pred.keys())
    
    def cate_diversity_score(self):
        user_pred = self.get_user_topk()
        score = 0
        for u in user_pred:
            item_list = list(user_pred[u][3]) 
            ild_score = self.ild_score(item_list)
            score += ild_score
        return score / len(user_pred.keys())
    
    def ild_score(self,item_list):
        score = 0
        for i in range(5):
            for j in range(i+1,5):
                item_a = item_list[i]
                item_b = item_list[j]
                if type(item_a) != type([]):
                    item_a = [item_a]
                if type(item_b) != type([]):
                    item_b = [item_b]
                tmp = list(set(item_a).intersection(set(item_b)))
                union = list(set(item_a).union(set(item_b)))
                score += len(tmp) / len(union)
        score = 2 * score / (self.k * (self.k-1))
        return score
        
        return ils_score
    def popularity_score(self):
        user_pred = self.get_user_topk()
        score = 0
        for u in user_pred:
            pop_list = user_pred[u][4]
            user_pop_list = user_pred[u][5]
            pop_score = np.mean(pop_list) - np.mean(user_pop_list)
            score += pop_score
        return score / len(user_pred.keys())
    
    def hit_score(self):
        user_pred = self.get_user_topk()
        score = 0
        for u in user_pred:
            if 1 in user_pred[u][0]:
                score += 1.0
        return score / len(user_pred.keys())
    
    def mrr_score(self):
        user_pred = self.get_user_topk()
        score = 0
        for u in user_pred:
            if 1 in user_pred[u][0]:
                score += 1.0 / (np.where(user_pred[u][0] == 1)[0][0] + 1)
        return score / len(user_pred.keys())
    
    def recall_score(self):
        user_pred = self.get_user_topk()
        score = 0
        for u in user_pred:
            score += sum(user_pred[u][0]) * 1. / len(user_pred[u][0])
        return score / len(user_pred.keys())
    
