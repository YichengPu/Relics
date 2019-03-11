import numpy as np

#x should be [.....,1] for the bias term
def logit(x,theta):
    return 1/(1+1/np.exp(x@theta))

def get_single_loss(x,y,theta,epsilon):
    return y*np.log(max(logit(x,theta),epsilon))+(1-y)*np.log(max(1-logit(x,theta),epsilon))

def get_gradient(x,y,theta):
    return y*x-x*logit(x,theta)

def get_loss(X,Y,theta,epsilon):
    n=len(X[0])
    loss=[]
    for i in range(n):
        loss.append(get_single_loss(X[i],Y[i],theta,epsilon))
    return loss

class logistic_model:
    def __init__(self):
        self.theta=[]



    def fit(self,X,Y,lr=0.01,limit=10000,epsilon=1e-4,verbose=True):


        #initialize weights

        n=len(X[0])
        self.theta=np.random.rand(n)
        # e is scalar
        k=0
        while k<limit:
            for i in range(n):
#                 print(X[i]@self.theta)
#                 print(logit(X[i],self.theta))
                self.theta=self.theta+lr*get_gradient(X[i],Y[i],self.theta)
                k+=1
                if k>limit:
                    break
#             print(X[0]@self.theta)
#             print(logit(X[0],self.theta))
#             print(get_gradient(X[0],Y[0],self.theta))
#             print(logit(X[8],self.theta))
            if verbose:
                print(sum(get_loss(X,Y,self.theta,epsilon)))
#             print(self.theta[:5])
        return


    def predict(self,X):
        n=len(X)
        res=[]
        for i in range(n):
            ind=logit(X[i],self.theta)
            res.append(ind)
        return res
