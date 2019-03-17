import numpy as np
class svm_subgrad:
    def __init__(self):
        self.w=None
        self.loss=None
        self.lamb=None

    def fit(self,x,y,lamb,step_size,num_iter):

        self.w=np.random.rand(x.shape[1])
        self.lamb=lamb
        n=0
        while n<num_iter:
            n+=1
            margins=y*(self.w@x.T)
#             print(margins[:10])
            loss_margin=np.mean(list(map(lambda x: max(x,0),1-margins)))
            loss_regul=self.lamb*np.linalg.norm(self.w)**2
            self.loss=loss_margin+loss_regul
            # print(loss_margin)
#             print(self.loss)

            pos_index=margins<1

            x_sub=x[pos_index,:]
            y_sub=y[pos_index]

            y_sub=y_sub.reshape(y_sub.shape[0],1)
            y_sub=np.tile(y_sub,(1,x_sub.shape[1]))
            deriv=np.mean(-x_sub*y_sub,0)

            deriv=deriv+2*self.lamb*self.w
#             print(deriv.shape)

            self.w=self.w-step_size*deriv
#             print(self.w)
    def predict(self,x):

        return self.w@x.T
