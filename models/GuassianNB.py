from scipy import stats
from functools import reduce
import numpy as np

class GSNB:
    def __init__(self):
        self.pos_dists=[]
        self.neg_dists=[]

    def fit(self,X,Y):
        pos_X=X[Y==1,:]
        neg_X=X[Y!=1,:]

        self.pos_dists=list(zip(np.mean(pos_X,axis=0),np.std(pos_X,axis=0)))
        self.neg_dists=list(zip(np.mean(neg_X,axis=0),np.std(neg_X,axis=0)))

    def predict(self,X,epsilon=1e-5):

        pos_mean=list(map(lambda x:x[0], self.pos_dists))
        pos_std=list(map(lambda x:x[1], self.pos_dists))

        neg_mean=list(map(lambda x:x[0], self.neg_dists))
        neg_std=list(map(lambda x:x[1], self.neg_dists))

        results=[]
        for i in range(len(X)):
            instance=X[i,:]
            probs_pos=stats.norm.pdf(x=instance,loc=pos_mean,scale=pos_std)
            probs_pos=np.nan_to_num(probs_pos)
#             prob_pos=reduce((lambda x,y: x*y), probs_pos)
            log_prob_pos=sum(np.log(probs_pos+epsilon))
            probs_neg=stats.norm.pdf(x=instance,loc=neg_mean,scale=neg_std)
            probs_neg=np.nan_to_num(probs_neg)
#             prob_neg=reduce((lambda x,y: x*y), probs_neg)
            log_prob_neg=sum(np.log(probs_neg+epsilon))

            results.append(log_prob_pos-log_prob_neg)
#             print(log_prob_pos,log_prob_neg)
        return np.array(results)
