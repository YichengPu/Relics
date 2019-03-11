from decision_tree import decision_tree
import numpy as np
class random_forest:
    def __init__(self):
        self.model_list=[]
    def fit(self,train_x,train_y,n_estimator,sample_rate,max_depth,min_sample):
        sample_rate=0.7
        n_row=train_x.shape[1]
        for i in range(n_estimator):
            sampled_index=np.random.choice(np.arange(n_row),int(sample_rate*n_row))
            model=decision_tree()
            model.fit(train_x[sampled_index,:],train_y[sampled_index],max_depth=max_depth,min_sample=min_sample,random_feature=True)
            self.model_list.append(model)
    def predict(self, test_x):
        preds=[]
        for model in self.model_list:
            preds.append(model.predict(test_x))
        pred= np.array(preds).mean(axis=0)
        return pred
