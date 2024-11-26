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
        self.lamb = 0.01  # regularization parameter
    
    def unique_labels(self, trueY):
        self.uniqueL = np.sort(np.unique(trueY))

    def align_labels(self, trueY):
        res = np.array([np.where(self.uniqueL == y)[0][0] for y in trueY])
        return res
        
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
        delta = [cp.Variable() for _ in range(self.K)]

        constraints = []
        loss = 0
        for k in range(self.K):
            t = np.where(trainY == k, 1, -1)
            for i in range(n):
                constraints.append(t[i] * (self.w[k] @ trainX[i] + self.b[k]) >= 1 - eps[k][i])
                constraints.append(eps[k][i] >= 0)
                loss += eps[k][i]
            
            # L1 Regularization
            constraints.append(cp.abs(self.w[k]) <= delta[k])
            loss += self.lamb * delta[k]
            

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
        self.maxIter= 100
        self.centroids = None
        
    
    def train(self, trainX):
        ''' Task 2-2 
            TODO: cluster trainX using LP(s) and store the parameters that discribe the identified clusters
        '''
        n, d = trainX.shape
        centroids = trainX[np.random.choice(n, self.K, replace=False)] # (K, d)
        oldCentroids = np.zeros((self.K, d))
        self.labels = np.zeros(n)

        for iter in range(self.maxIter):
            x = cp.Variable((n, self.K), integer=True) # (n, K)
            constraints = []
            _trainX = trainX[:, None, :] # (n, 1, d)
            XMinusCentroids = _trainX - centroids # (n, K, d)
            distances = np.linalg.norm(XMinusCentroids, axis=2, ord=1) # (n, K)
            
            loss = cp.sum(cp.multiply(x, distances))

            constraints.append(cp.sum(x, axis=1) == 1) # each data point belongs to one cluster
            constraints.extend([x >= 0, x <= 1]) # ensure x is binary
            problem = cp.Problem(cp.Minimize(loss), constraints)
            problem.solve(solver=cp.GLPK_MI)

            x = x.value # (n, K)
            self.labels = np.argmax(x, axis=1) # (n,)
            for k in range(self.K):
                kPoints = trainX[self.labels == k] # (n_k, d)
                assert len(kPoints) > 0  # each cluster has at least one data point
                centroids[k] = np.mean(kPoints, axis=0) # (d,)
            
            if np.allclose(centroids, oldCentroids, atol=1e-4):
                print(f'Converged at iteration {iter}')
                self.centroids = centroids
                break 

            oldCentroids = centroids.copy()

        # Update and teturn the cluster labels of the training data (trainX)
        self.centroids = centroids
        return self.labels
    
    
    def infer_cluster(self, testX):
        ''' Task 2-2 
            TODO: assign new data points to the existing clusters
        '''
        testX_ = testX[:, None, :] # (n, 1, d)
        XMinusCentroids = testX_ - self.centroids # (n, K, d)
        distances = np.linalg.norm(XMinusCentroids, axis=2) # (n, K)
        pred_labels = np.argmin(distances, axis=1) # (n,)

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
        true_labels = true_labels.astype(int)
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
    
    


from utils import prepare_mnist_data, prepare_synthetic_data
def main():
    # data = prepare_mnist_data()
    # trainX, trainY, testX, testY = data['trainX'], data['trainY'], data['testX'], data['testY']
    # K = len(np.unique(trainY))
    # classifier = MyClassifier(K)
    # classifier.train(trainX, trainY)
    # testAcc = classifier.evaluate(testX, testY)
    # trainAcc = classifier.evaluate(trainX, trainY)
    # print(f'Training Accuracy: {trainAcc:.4f}')
    # print(f'Test Accuracy: {testAcc:.4f}')


    result2 = {'synthetic':{'K':[3, 5, 10], 'clustering_nmi':[0.6,0.6,0.6], 'classification_accuracy':[0.8,0.8,0.8]},
                'mnist':{'K':[3, 10, 32], 'clustering_nmi':[0.5,0.5,0.5], 'classification_accuracy':[0.7,0.7,0.7]}}
    syn_data = prepare_synthetic_data()
    trainX, trainY, testX, testY = syn_data['trainX'], syn_data['trainY'], syn_data['testX'], syn_data['testY']
    
    for i in range(3):
        k = result2['synthetic']['K'][i]
        synClustering = MyClustering(k)
        synClustering.train(trainX)
        result2['synthetic']['clustering_nmi'][i] = synClustering.evaluate_clustering(trainY)
        result2['synthetic']['classification_accuracy'][i] = synClustering.evaluate_classification(trainY, testX, testY)


    mnist_data = prepare_mnist_data()
    trainX, trainY, testX, testY = mnist_data['trainX'], mnist_data['trainY'], mnist_data['testX'], mnist_data['testY']
    for i in range(3):
        k = result2['mnist']['K'][i]
        mnistClustering = MyClustering(k)
        mnistClustering.train(trainX)
        result2['mnist']['clustering_nmi'][i] = mnistClustering.evaluate_clustering(trainY)
        result2['mnist']['classification_accuracy'][i] = mnistClustering.evaluate_classification(trainY, testX, testY)

    print(result2)
if __name__ == '__main__':
    main()