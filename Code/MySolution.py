import numpy as np
from sklearn.metrics import normalized_mutual_info_score, accuracy_score
import cvxpy as cp
### TODO: import any other packages you need for your solution

#--- Task 1 ---#
class MyClassifier:  
    def __init__(self, K):
        self.K = K  # number of classes

        ### TODO: Initialize other parameters needed in your algorithm
        # examples:
        self.w = None
        self.b = None
        self.lamb = 0.1  # regularization parameter
        print('MyClassifier initialized')
    
    def unique_labels(self, trueY):
        self.uniqueL = np.unique(trueY)

    def align_labels(self, trueY):
        return np.array([np.where(self.uniqueL == y)[0][0] for y in trueY])

    
    def train(self, trainX, trainY):
        ''' Task 1-2 
            TODO: train classifier using LP(s) and updated parameters needed in your algorithm 
        '''
        self.unique_labels(trainY)
        trainY = self.align_labels(trainY)
        n, d = trainX.shape
        self.w = [cp.Variable(d) for _ in range(self.K)]
        self.b = [cp.Variable() for _ in range(self.K)]
        eps = [cp.Variable(n) for _ in range(self.K)]

        constraints = []
        loss = 0
        for k in range(self.K):
            t = np.where(trainY == k, 1, -1)
            for i in range(n):
                constraints.append(t[i] * (self.w[k] @ trainX[i] + self.b[k]) >= 1 - eps[k][i])
                constraints.append(eps[k][i] >= 0)
                loss += eps[k][i]
                
        # Add L1 regularization term
        for k in range(self.K):
            loss += self.lamb * cp.norm(self.w[k], 2)

        problem = cp.Problem(cp.Minimize(loss), constraints)
        problem.solve(verbose=False)
        self.w = [w.value for w in self.w]
        self.b = [b.value for b in self.b]

        
    
    def predict(self, testX):
        ''' Task 1-2 
            TODO: predict the class labels of input data (testX) using the trained classifier
        '''
        n = testX.shape[0]
        scores = np.zeros((n, self.K))
        for k in range(self.K):
            scores[:,k] = testX @ self.w[k] + self.b[k]
        predY = np.argmax(scores, axis=1)

        # Return the predicted class labels of the input data (testX)
        return predY
    

    def evaluate(self, testX, testY):
        testY = self.align_labels(testY)
        predY = self.predict(testX)
        accuracy = accuracy_score(testY, predY)

        return accuracy
    

##########################################################################
#--- Task 2 ---#
class MyClustering:
    def __init__(self, K):
        self.K = K  # number of classes
        self.labels = None

        ### TODO: Initialize other parameters needed in your algorithm
        # examples: 
        # self.cluster_centers_ = None
        
    
    def train(self, trainX):
        ''' Task 2-2 
            TODO: cluster trainX using LP(s) and store the parameters that discribe the identified clusters
        '''
        


        # Update and teturn the cluster labels of the training data (trainX)
        return self.labels
    
    
    def infer_cluster(self, testX):
        ''' Task 2-2 
            TODO: assign new data points to the existing clusters
        '''

        # Return the cluster labels of the input data (testX)
        return pred_labels
    

    def evaluate_clustering(self, trainY):
        label_reference = self.get_class_cluster_reference(self.labels, trainY)
        aligned_labels = self.align_cluster_labels(self.labels, label_reference)
        nmi = normalized_mutual_info_score(trainY, aligned_labels)

        return nmi
    

    def evaluate_classification(self, trainY, testX, testY):
        pred_labels = self.infer_cluster(testX)
        label_reference = self.get_class_cluster_reference(self.labels, trainY)
        aligned_labels = self.align_cluster_labels(pred_labels, label_reference)
        accuracy = accuracy_score(testY, aligned_labels)

        return accuracy


    def get_class_cluster_reference(self, cluster_labels, true_labels):
        ''' assign a class label to each cluster using majority vote '''
        label_reference = {}
        for i in range(len(np.unique(cluster_labels))):
            index = np.where(cluster_labels == i,1,0)
            num = np.bincount(true_labels[index==1]).argmax()
            label_reference[i] = num

        return label_reference
    
    
    def align_cluster_labels(self, cluster_labels, reference):
        ''' update the cluster labels to match the class labels'''
        aligned_lables = np.zeros_like(cluster_labels)
        for i in range(len(cluster_labels)):
            aligned_lables[i] = reference[cluster_labels[i]]

        return aligned_lables



##########################################################################
#--- Task 3 (Option 1) ---#
class MyLabelSelection:
    def __init__(self, ratio):
        self.ratio = ratio  # percentage of data to label
        ### TODO: Initialize other parameters needed in your algorithm

    def select(self, trainX):
        ''' Task 3-2'''
        

        # Return an index list that specifies which data points to label
        return data_to_label
    




##########################################################################
#--- Task 3 (Option 2) ---#
class MyFeatureSelection:
    def __init__(self, num_features):
        self.num_features = num_features  # target number of features
        ### TODO: Initialize other parameters needed in your algorithm


    def construct_new_features(self, trainX, trainY=None):  # NOTE: trainY can only be used for construting features for classification task
        ''' Task 3-2'''
        


        # Return an index list that specifies which features to keep
        return feat_to_keep
    
    