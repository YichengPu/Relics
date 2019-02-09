import numpy as np

def g_log(x,epsilon=1e-7):
    return np.log(x+epsilon)

def entropy(arr,split,ep=1e-7):
    p=sum(arr[arr>=split])/len(arr)
    return -(p*g_log(p+ep)+(1-p)*g_log(1-p+ep))


def gini_index(arr,split):
    p=sum(arr[arr>=split])/len(arr)
    return (p*(1-p)+(1-p)*p)

def split(mat,col,v):
    left=[]
    right=[]

    for i in range(mat.shape[0]):
        if mat[i,col] <= v:
            left.append(i)
        else:
            right.append(i)


    return (left,right)

def branch(mat,target):

    nrow,ncol=mat.shape


    best_split=(-1,-1)
    best_gain=0

    for i in range(nrow):
        for j in range(ncol):
            left,right=split(mat,j,mat[i][j])

            leng_left=len(left)
            leng_right=len(right)
#             print(i,j,left,right)
            if leng_left>0 and leng_right>0:


                # info_gain=entropy(target,1)-len(left)/nrow*entropy(target[left],1)\
                #     -len(right)/nrow*entropy(target[right],1)

                info_gain=gini_index(target,1)-len(left)/nrow*gini_index(target[left],1)\
                    -len(right)/nrow*gini_index(target[right],1)

#                 print(entropy(target,1),len(left)/nrow*entropy(target[left],1)\
#                     ,len(right)/nrow*entropy(target[right],1))
#                 print(info_gain)
                if info_gain> best_gain:
                    best_split=(i,j)
                    best_gain=info_gain

    if best_split==(-1,-1):
            return best_split,[],[],[],[]
    left,right=split(mat,best_split[1],mat[best_split])
    return best_split,mat[left,:],mat[right,:],target[left],target[right]



def grow(node,n):

    if n>=10:
        return
    if len(node.data)<5:
        return

    if sum(node.target) == len(node.target):
        return

    best_split,mat_left,mat_right,target_left,target_right=branch(node.data,node.target)

    if (len(mat_left)==0) or (len(mat_right)==0):
        return

    node.record_split(best_split[1],node.data[best_split])

    node.data=None

    node.target=None

    left=dnode(mat_left,target_left)

    right=dnode(mat_right,target_right)

    node.left=left

    node.right=right

    n+=1
    grow(node.left,n)
    grow(node.right,n)
    return



def predict(x,root):
    node=root
    prev=node
    while node.col != -1:
        if x[node.col] <= node.value:
            prev=node
            node=node.left
        else:
            prev=node
            node=node.right

#     print(node.value,node.col,node.left,node.right)
    return sum(node.target)/len(node.target)>0.5


class dnode:
    def __init__(self,data,target):
        self.data=data
        self.target=target
        self.col=-1
        self.value=0
        self.left=None
        self.right=None
    def add_left(self,left_node):
        self.left=left_node
    def add_right(self,right_node):
        self.right=right_node
    def record_split(self,col,value):
        self.col=col
        self.value=value


class decision_tree:
    def __init__(self):
        self.root=None

        return

    def fit(self,data,target):
        '''
        data: array of size(n,p)
        target: binary arrary of size(n)
        '''

        self.root=dnode(data,target)
        grow(self.root,0)

        return

    def predict(self,test_data):
        preds=[]

        for i in range(len(test_data)):
            preds.append(predict(test_data[i],self.root))

        return np.array(preds)
