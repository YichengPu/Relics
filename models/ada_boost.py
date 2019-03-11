import numpy as np


def g_log(x,epsilon=1e-7):
    return np.log(x+epsilon)

def split(mat,target,data_weight):

    nrow,ncol=mat.shape
    index=np.arange(nrow)

    split_point=None
    min_error=np.inf
    for i in range(nrow):
        for j in range(ncol):
            for k in ['lt','gt']:
                if k=='lt':

                    pred_pos=index[mat[:,j] <= mat[i,j]]
                    pred_neg=index[mat[:,j] > mat[i,j]]

                    true_pos=index[target==1]
                    true_neg=index[target==0]

                    false_pos=np.intersect1d(pred_pos,true_neg)
                    false_neg=np.intersect1d(pred_neg,true_pos)

                    weights_pos=data_weight[false_pos]
                    weights_neg=data_weight[false_neg]

                    error=sum(weights_pos)+sum(weights_neg)

                    epsilon=error/sum(data_weight)
                    if error>0:
                        alpha=g_log((1-epsilon)/epsilon)
                    else:
                        alpha=None



#                     error=-weights_pos@(np.exp(target[false_pos]))-weights_neg@(np.exp(-target[false_neg]))
#                     print(i,j,k,error)

                    if error<min_error:
                        min_error=error
                        split_point=[i,j,k]
                        split_alpha=alpha
                        false_list=np.concatenate((false_pos,false_neg))
                else:

                    pred_pos=index[mat[:,j] > mat[i,j]]
                    pred_neg=index[mat[:,j] <= mat[i,j]]

                    true_pos=index[target==1]
                    true_neg=index[target==0]

                    false_pos=np.intersect1d(pred_pos,true_neg)
                    false_neg=np.intersect1d(pred_neg,true_pos)

                    weights_pos=data_weight[false_pos]
                    weights_neg=data_weight[false_neg]

                    error=sum(weights_pos)+sum(weights_neg)

                    epsilon=error/sum(data_weight)
                    if error>0:
                        alpha=g_log((1-epsilon)/epsilon)
                    else:
                        alpha=None

#                     error=-weights_pos@(np.exp(target[false_pos]))-weights_neg@(np.exp(-target[false_neg]))
#                     print(i,j,k,error)
                    if error<min_error:
                        min_error=error
                        split_point=[i,j,k]
                        split_alpha=alpha
                        false_list=np.concatenate((false_pos,false_neg))
    if min_error>0:
        data_weight[false_list]=data_weight[false_list]*np.exp(split_alpha)
    return mat[split_point[0],split_point[1]],split_point[1],split_point[2],data_weight,min_error,split_alpha

#Prediction Function
def predict(data,hs,alphas):
    scores=np.zeros(len(data))
    index=np.arange(len(data))
    for h,a in zip(hs,alphas):
        threshold,col,way=h
        if way=='lt':
            pred_pos=index[data[:,col] <= threshold]
            pred_neg=index[data[:,col] > threshold]
            scores[pred_pos] = scores[pred_pos]+a
            scores[pred_neg] = scores[pred_neg]-a
        else:
            pred_pos=index[data[:,col] >= threshold]
            pred_neg=index[data[:,col] < threshold]
            scores[pred_pos] = scores[pred_pos]+a
            scores[pred_neg] = scores[pred_neg]-a
#             print(scores[pred_pos])
    return (scores)

class ada_boost:
    def __init__(self):
        self.alphas=[]
        self.hs=[]

    def fit(self,train_x,train_y,n_iter):
        data_weight_init = np.ones(len(train_y))/len(train_y)
        data_weight=data_weight_init

        for i in range(n_iter):
            threshold,col,way,data_weight,min_error,alpha_m = split(train_x,train_y,data_weight)
            self.hs.append((threshold,col,way))
            self.alphas.append(alpha_m)

    def predict(self,test_x):
        return predict(test_x,self.hs,self.alphas)
