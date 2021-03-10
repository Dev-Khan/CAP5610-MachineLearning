import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import hashlib
import time


def manhattan_dist(P1, P2, space=2):
    dist = 0
    for dim in range(space):
        dist += np.abs(P1[dim] - P2[dim])
    return dist
        
def euclidean_dist(P1, P2, space=2):
    dist = 0
    for dim in range(space):
        dist += np.power( (P1[dim]-P2[dim]), 2)
    return np.sqrt(dist)

def cosine_dist(P1, P2, space=2):
    dist = 1 - (np.dot(P1, P2) / (P1.size * P2.size))
    return dist

def gen_jaccard_dist(P1, P2, space=2):
    numerator = 0
    denominator = 0
    for i in range(space):
        numerator += min(P1[i], P2[i])
        denominator += max(P1[i], P2[i])
    return 1 - (numerator/denominator)

def k_means(data, k, centroids=None, dist_func=manhattan_dist,  maxIter=100000, term_condition=1):
    if centroids is None:
        #print(data.shape)
        index = np.random.randint(0, data.shape[0], size=[k])
        centroids = np.array([data[p] for p in index])
    
    fail_safe = maxIter# max number of iterations before we manually end
    
    prev_centroids = centroids.copy()
    running = True
    
    cur_sse = 0
    prev_sse = 0
    clusters = dict()
    while running:
        
        # Create Cluster Holders
        clusters = dict()
        for center in centroids:
            #print(center)
            #print(type(center))
            #print(hashlib.sha512(center).hexdigest())
            clusters[hashlib.sha512(center).hexdigest()] = [center, []]
            
        # Calcualte the cluster for each data point
        for point in data:
            min_dist = [np.Inf, np.NAN] # pos inifinity
            for center in centroids:
                #print(center.shape[0])
                if dist_func(point, center, space=center.shape[0]) < min_dist[0]:
                    min_dist[0] = dist_func(point, center, space=center.shape[0])
                    min_dist[1] = hashlib.sha512(center).hexdigest()
            clusters[min_dist[1]][1].append(point)
            
                
        # Recalculate Centroids
        prev_centroids = centroids.copy()
        centroids = []
        for c, points in clusters.items():
            center = points[0]
            points = np.array(points[1])
            
            
            if points.shape[0] == 0:
                centroids.append(center)
            elif points.shape[0] == 1:
                centroids.append(points[0])
            else:
                latent_dim = points.shape[1]
                #print(latent_dim)
                center = []
                for i in range(latent_dim):
                    #print(points[i])
                    #print(points[i].sum())
                    #print(points[i].mean())
                    center.append(points[i].mean())
                #print(center)
                centroids.append(center)
        
        centroids = np.array(centroids)
        
        def repeat_centroids(centroids):
            for i in range(k):
                for j in range(i+1, k):
                    if np.array_equal(centroids[i], centroids[j]):
                        return i
            return None
            
        # Make sure that if two centroids that are the same are picked
        # a unique centroid is selected to replace one of the repeats
        while centroids.shape[0] < k:
            centroids = np.append(centroids, [data[np.random.randint(0, data.shape[0])]], axis=0)  
        while True:
            idx = repeat_centroids(centroids)
            if idx == None:
                break
            else:
                centroids[idx] = data[np.random.randint(0, data.shape[0])]
                
            
            
            
        
        #print(centroids)
        
        if fail_safe == maxIter:
            print("First Iteration Centers: ", centroids)
            
        fail_safe -= 1
        #print("Prev: ", prev_centroids)
        #print("Curr: ", centroids)
        # Break Conditions
        if (np.array_equal(prev_centroids, centroids) and term_condition==1):
            running = False
        if term_condition == 2:
             def calc_sse(clusters, dist_func):
                # SSE = sum([dist(point, cluster_center)**2 for point in cluster])
                #print(clusters)
                sse = 0
                for h, d in clusters.items():
                    c_sse = 0
                    center = d[0]
                    points = np.array(d[1])
                    for point in points:
                        c_sse += (dist_func(center, point))**2
                        sse += c_sse
                    print("SSE: {}".format(sse))
                    return sse
             prev_sse = cur_sse
             cur_sse = calc_sse(clusters, dist_func)
             
        if cur_sse > prev_sse and fail_safe != maxIter:
            running = False
        if term_condition == 100 and (maxIter - fail_safe) == 100:
            running = False
            
            
        if (fail_safe < 0):
            print("Fail Safe Engaged")
            running = False
    print(maxIter - fail_safe)
    return clusters


def task_one():
    t1_data = np.array([ [3,5], [3,4], [2,8], [2,3], [6,2], [6,4], [7,3], [7,4], [8,5], [7,6] ])
    colors = ["red", "blue"]
    
    # Q1
    q1_clusters = k_means(t1_data, 2, centroids=np.array([ [4,6], [5,4] ]))
    print(q1_clusters)
    i = 0
    for c, points in q1_clusters.items():
        points = np.array(points[1])
        center = points[0]
        x = points[:, 0]
        y = points[:, 1]
        plt.scatter(x, y, c=colors[i], label="Center: {}".format(center))
        plt.scatter(center[0], center[1], marker="s", c=colors[i])
        i += 1
    plt.title("Task 1 Q1")
    plt.xlabel("# Wins (2016 Season)")
    plt.ylabel("# Wins (2016 Season)")
    plt.show()
    
    # Q2
    q2_clusters = k_means(t1_data, 2, centroids=np.array([ [4,6], [5,4] ]), dist_func=euclidean_dist)
    print(q2_clusters)
    i = 0
    for c, points in q2_clusters.items():
        points = np.array(points[1])
        center = points[0]
        x = points[:, 0]
        y = points[:, 1]
        plt.scatter(x, y, c=colors[i], label="Center: {}".format(center))
        plt.scatter(center[0], center[1], marker="s", c=colors[i])
        i += 1
    plt.title("Task 1 Q2")
    plt.xlabel("# Wins (2016 Season)")
    plt.ylabel("# Wins (2016 Season)")
    plt.show()
    
    # Q3
    q3_clusters = k_means(t1_data, 2, centroids=np.array([ [3,3], [8,3] ]))
    print(q3_clusters)
    i = 0
    for c, points in q3_clusters.items():
        points = np.array(points[1])
        center = points[0]
        x = points[:, 0]
        y = points[:, 1]
        plt.scatter(x, y, c=colors[i], label="Center: {}".format(center))
        plt.scatter(center[0], center[1], marker="s", c=colors[i])
        i += 1
    plt.title("Task 1 Q3")
    plt.xlabel("# Wins (2016 Season)")
    plt.ylabel("# Wins (2016 Season)")
    plt.show()
    
    # Q4
    q4_clusters = k_means(t1_data, 2, centroids=np.array([ [3,2], [4,8] ]))
    print(q4_clusters)
    i = 0
    for c, points in q4_clusters.items():
        points = np.array(points[1])
        center = points[0]
        x = points[:, 0]
        y = points[:, 1]
        plt.scatter(x, y, c=colors[i], label="Center: {}".format(center))
        plt.scatter(center[0], center[1], marker="s", c=colors[i])
        i += 1
    plt.title("Task 1 Q4")
    plt.xlabel("# Wins (2016 Season)")
    plt.ylabel("# Wins (2016 Season)")
    plt.show()
    
def task_three():
    red = np.array([ [4.7,3.2], [4.9,3.1], [5.0,3.0], [4.6,2.9] ])
    blue = np.array([ [5.9,3.2], [6.7, 3.1], [6.0,3.0], [6.2,2.8] ])
    
    max_dist = [np.NINF, np.NAN, np.NAN]
    min_dist = [np.Inf, np.NAN, np.NAN]
    avg_dist = []
    
    for rP in red:
        for bP in blue:
            dist = euclidean_dist(rP, bP, space=2)
            
            if dist < min_dist[0]:
                min_dist = [dist, rP, bP]
            if dist > max_dist[0]:
                max_dist = [dist, rP, bP]
            avg_dist.append(dist)
    
    print("Max Distance: {} [RP: {}, BP:{}]".format(max_dist[0], max_dist[1], max_dist[2]))
    print("Min Distance: {} [RP: {}, BP:{}".format(min_dist[0], min_dist[1], min_dist[2]))
    print("Avg Distance: {}".format(np.array(avg_dist).mean()))
    
    
def task_two():
    """
    def calc_sse(clusters, np_data, df_data):
        # Find which label goes to the cluster
        def isEqual(subArray, target):
            return np.array_equal(subArray, target)
            
        print(np_data[0])
        print(df_data.iloc[0])
        print(isEqual(np_data[0], [4.9,3.0,1.4,0.2]))
        print(df_data.iloc[0]["Class"])
        for c, p in clusters.items():
            bins = {"Iris-setosa": 0, "Iris-versicolor": 0, "Iris-virginica": 0}
            center = p[0]
            points = p[1]
            for point in points:
                for i in range(np_data.shape[0]):
                    if isEqual(np_data[i], point):
                        bins[df_data.iloc[i]["Class"]] += 1
                        break
            print(bins)
        """
    def calc_sse(clusters, dist_func):
	# SSE = sum([dist(point, cluster_center)**2 for point in cluster])
        #print(clusters)
        sse = 0
        for h, d in clusters.items():
            c_sse = 0
            center = d[0]
            points = np.array(d[1])
            for point in points:
                c_sse += (dist_func(center, point))**2
            sse += c_sse
         
        print("SSE: {}".format(sse))
        return sse
            
    def calc_acc(clusters, np_data, df_data):
        # Find which label goes to the cluster            
        print(np_data[0])
        print(df_data.iloc[0])
        print(df_data.iloc[0]["Class"])
        total_correct = 0
        for c, p in clusters.items():
            bins = {"Iris-setosa": 0, "Iris-versicolor": 0, "Iris-virginica": 0}
            center = p[0]
            points = p[1]
            for point in points:
                for i in range(np_data.shape[0]):
                    if np.array_equal(np_data[i], point):
                        bins[df_data.iloc[i]["Class"]] += 1
                        break
            print(bins)
            maxKey = max(bins, key=bins.get)
            print(maxKey)
            total_correct+= bins[maxKey]
            
        print("Acc: {}".format(total_correct/150))
    
            
            
            
            
                
                
            
    iris_data = pd.read_csv("./iris.data")
    iris_data.columns=["SepLen", "SepWid", "PedLen", "PedWid", "Class"]
    
    non_labeled_data = iris_data[["SepLen", "SepWid", "PedLen", "PedWid"]]
    non_labeled_data = non_labeled_data.to_numpy()
    
    # Questions 1 and 2 (Uncomment to run since they are not gaurenteed to succeed
    #man_clusters = k_means(non_labeled_data, 3, centroids=None, dist_func=manhattan_dist)
    
    #euc_clusters = k_means(non_labeled_data, 3, centroids=None, dist_func=euclidean_dist)
    #calc_sse(euc_clusters, euclidean_dist)#, non_labeled_data, iris_data)
    #calc_acc(euc_clusters, non_labeled_data, iris_data)
    
    #cos_clusters = k_means(non_labeled_data, 3, centroids=None, dist_func=cosine_dist)
    #calc_sse(cos_clusters, cosine_dist)
    #calc_acc(cos_clusters, non_labeled_data, iris_data)
    
    #jac_clusters = k_means(non_labeled_data, 3, centroids=None, dist_func=gen_jaccard_dist)
    #calc_sse(jac_clusters, gen_jaccard_dist)
    #calc_acc(jac_clusters, non_labeled_data, iris_data)
    
    # Question 4
    
    #euc_clusters = k_means(non_labeled_data, 3, centroids=None, dist_func=euclidean_dist, term_condition=1)
    #euc_clusters = k_means(non_labeled_data, 3, centroids=None, dist_func=euclidean_dist, term_condition=2)
    #euc_clusters = k_means(non_labeled_data, 3, centroids=None, dist_func=euclidean_dist, term_condition=100)
    
    #cos_clusters = k_means(non_labeled_data, 3, centroids=None, dist_func=cosine_dist, term_condition=1)
    #cos_clusters = k_means(non_labeled_data, 3, centroids=None, dist_func=cosine_dist, term_condition=2)
    #cos_clusters = k_means(non_labeled_data, 3, centroids=None, dist_func=cosine_dist, term_condition=100)
    
    #jac_clusters = k_means(non_labeled_data, 3, centroids=None, dist_func=gen_jaccard_dist, term_condition=1)
    #jac_clusters = k_means(non_labeled_data, 3, centroids=None, dist_func=gen_jaccard_dist, term_condition=2)
    jac_clusters = k_means(non_labeled_data, 3, centroids=None, dist_func=gen_jaccard_dist, term_condition=100)

    
if __name__ == '__main__':
    #np.random.seed(2147483648)
    print(np.random.seed)
    #task_one()
    #task_three()
    task_two()
