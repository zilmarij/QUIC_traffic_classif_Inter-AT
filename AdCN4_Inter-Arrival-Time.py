"""
This program is the implementation for Classifier based on K-means and Packet Inter-Arrival Time
"""

import statistics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import distance
import math


pl= {'0':[],'1':[],'2':[],'3':[],'4':[],'5':[]}  #, '7':[],'8':[],'9':[]}   #packet lengths
pl_centroid={'0':[],'1':[],'2':[],'3':[],'4':[],'5':[]} #'6':[]} #'7':[],'8':[],'9':[]}  #for computing centroid of the lateset flow
pl_test=[]
IAT = []
Youtube={'1':[],'2':[],'3':[],'4':[],'5':[]} # '7':[],'8':[],'9':[]}
GSearch={'1':[],'2':[],'3':[],'4':[],'5':[]} #'7':[],'8':[],'9':[]}
GMusic={'1':[],'2':[],'3':[],'4':[],'5':[]} #'7':[],'8':[],'9':[]}
GDrive={'1':[],'2':[],'3':[],'4':[],'5':[]} #'7':[],'8':[],'9':[]}
GDocs={'1':[],'2':[],'3':[],'4':[],'5':[],} #'7':[],'8':[],'9':[]}

Y=[]
GS=[]; GM=[]; GDr=[]; GDo=[]
#cluster_compos = { '0':[], '1':[], '2':[], '3':[], '4':[]}
        #Youtube  ,   GSearch ,  GMusic,   #GDrive#,  GDoc
applications=['Y', 'GS', 'GM', 'GDo']
cluster_compos = { '0':[], '1':[], '2':[], '3':[], '4':[], '5':[], '6':[], '7':[], '8':[], '9':[],
                   '10':[], '11':[], '12':[], '13':[], '14':[]}


packet_lengths=dict()
global centroids; global tt; tt=[]; centroids=[]
global online; online=0
global centroidss

def label_centroids(label):
#fix this
    i=-30
    iter=-1
    for k in list(applications):
        i+=30
        iter+=1
        for j in range(i,i+30,1):
            cluster_compos["%d" %label[j]].append(k)

def compute_iat(iat, ds):
    #iat2=[]
    for i in range(len(iat)):
        if i>0:
          ds["%s" %i].append(iat[i]-iat[i-1])

def parse(path, ds):
    iat=[]

    f=open(path, "r")
    for i in range(6):
        t=f.readline()
        t=t.split()
        pl["%s" %i].append(t[2])
        pl_centroid["%s" %i].append(t[2])
        iat.append(float(t[1]))
    f.close()
    compute_iat(iat, ds)

def parse2(path):
    iat=[]
    iat2=[]
    f=open(path, "r")
    for i in range(6):
        t=f.readline()
        t=t.split()
        pl_test.append(int(t[2]))
        iat.append(float(t[1]))
    for i in range(len(iat)):
        if i>0:
          iat2.append(iat[i]-iat[i-1])
    return iat2

def generate_centroids():
    #Mean of the given packet lengths of an app
    for k in sorted(pl_centroid):
        t=0
        for i in range(len(pl_centroid[k])):
            t+= int(pl_centroid[k][int(i)])
        t=t//30
        tt.append(t)
    centroids.append(tt)

def Min_ED_APPS(iat, x):
    ed=1000
    app='x'
    if 'Y' in x and distance.euclidean(Y, iat) < ed:
        ed = distance.euclidean(Y, iat)
        app='Y'
    if 'GS' in x and distance.euclidean(GS, iat) < ed:
        ed= distance.euclidean(GS, iat)
        app='GS'
    if 'GM' in x and distance.euclidean(GM, iat) < ed:
        ed = distance.euclidean(GM, iat)
        app='GM'
    #ed.append(distance.euclidean(GDr, iat))
    if 'GDo' in x and distance.euclidean(GDo, iat) < ed:
        ed = distance.euclidean(GDo, iat)
        app = 'GDo'
    return ed, app

def cluster():
        df = pd.DataFrame(pl)

        centroids.append([316, 159, 475, 344, 135, 104]) #GSearch
        centroids.append([650, 74, 1294, 1294, 74, 1294])
        centroids.append([314, 74, 140, 104, 112, 74])
        centroids.append([86, 86, 74, 276, 74, 1294])
        centroids.append([317, 74, 140, 104, 112, 74])

        centroids.append([1285, 82, 304, 1412, 1412, 1412])   #GDoc
        centroids.append([1412, 1288, 1412, 103, 93, 710])
        centroids.append([356, 87, 325, 1412, 104, 1412])
        centroids.append([308, 87, 520, 1412, 995, 92])
        centroids.append([296, 87, 323, 1412, 1412, 1412])
        centroids.append([880, 67, 285, 1392, 1392, 1392])

        centr=np.array(centroids, np.float64)
        kmeans = KMeans(n_clusters= 15, init=centr, n_init=1, max_iter=1, algorithm='elkan')
        kmeans.fit(df)
        label= kmeans.labels_

        print(kmeans.labels_)
        #print(kmeans.cluster_centers_)
        centroidss=np.array(kmeans.cluster_centers_)

        label_centroids(label)   #tags clusters with app names
        #print(df.shape)
        #print(label)
        u_labels = np.unique(label)
        colors = ['#DF2020', '#81DF20', '#2095DF', '#DF2021', '#2097DE', '#82DFE2']
        #df['c'] = df.map({0: colors[0], 1: colors[1], 2: colors[2],
                                 # 3: colors[3], 4: colors[4], 5: colors[5]})
        #print(u_labels)
        df.columns=['a','b','c','d','e','f']
        #print(df.a)
        plt.scatter(x=df.a[:40], y=df.b[0:40])
        # plotting the results:
        #for i in u_labels:
         #   for j in range(len(df)):
          #      plt.scatter(df[j, label == i], label=i)
        #plt.legend()
        plt.show()

        #plt.scatter(df['x'].values, df['y'].values, c=kmeans.labels_.astype(float), s=50, alpha=0.5)
        #plt.scatter(centroidss[:, 0], centroidss[:, 1],marker='s', color ='red', s=50)
        #plt.show()
        return kmeans, centroidss

def classify(kmeans, centroidss, iat):
        #print(pl_test)
        l=kmeans.predict([pl_test,[1, 1, 1, 1, 1, 1]])
        ED=[]
        if(len(l)>1):
            #compute eucledian dist from the centroids
            for i in l:
                ED.append(math.sqrt(
                   (pow((pl_test[0] - centroidss[i][0]), 2)) + pow((pl_test[1] - centroidss[i][1]), 2) +
                    pow((pl_test[2] - centroidss[i][2]), 2) + pow((pl_test[3] - centroidss[i][3]), 2) +
                    pow((pl_test[4] - centroidss[i][4]), 2) + pow((pl_test[5] - centroidss[i][5]), 2)))
                    #pow((pl_test[6] - centroidss[i][6]), 2) + pow((pl_test[7] - centroidss[i][7]), 2) +
                    #pow((pl_test[8] - centroidss[i][8]), 2) + pow((pl_test[9] - centroidss[i][9]), 2))
        #print(l, ED)
        cl= ED.index(min(ED))
        cl=l[cl]   #predicted cluster
        x=cluster_compos["%s" %cl]
        ed, app= Min_ED_APPS(iat, x)
        #print(" eds are ", ed)
        print(" Application is ", app)


if __name__ == '__main__':
    #Read
    #Parse
    #Cluster
    #Classify
    s='link_to_dataset'  
    for i in range(30):   #15 files to take data from
        path= s + "Youtube-" + "%s" %i +'.txt'
        parse(path, Youtube)
    generate_centroids()   #call after each apps parsing
    pl_centroid = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': []} #'6': [], '7': [], '8': [], '9': []}
    for k in Youtube.keys():
        Y.append(statistics.mean(Youtube[k]))
    print("youtube", Y)

    tt=[]
    for i in range(30):
        path= s + "GoogleSearch-" + "%s" %i + '.txt'
        parse(path,GSearch)
    generate_centroids()
    pl_centroid = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': []} #'6': [], '7': [], '8': [], '9': []}
    for k in GSearch.keys():
        GS.append(statistics.mean(GSearch[k]))
    print("GSearch", GS)
    #print(statistics.mean(IAT), statistics.variance(IAT))

    tt=[]
    for i in range(30):
        path= s+ "GoogleMusic-" + "%s" %i + ".txt"    #shows fluctuating behav
        parse(path,GMusic)
    generate_centroids()
    pl_centroid = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': []} #'6': [], '7': [], '8': [], '9': []}
    for k in GMusic.keys():
        GM.append(statistics.mean(GMusic[k]))
    print("GMusic", GM)

    tt = []
    for i in range(30):
        path = s + "GoogleDrive-" + "%s" % i + '.txt'

    IAT = []
    tt = []
    for i in range(30):
        path = s + "GoogleDoc-" + "%s" % i + '.txt'
        parse(path, GDocs)
    generate_centroids()
    pl_centroid = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': []} #'6': [], '7': [], '8': [], '9': []}
    for k in GDocs.keys():
        GDo.append(statistics.mean(GDocs[k]))
    print("GDoc", GDo)

    kmeans, centroidss=cluster()
    #print(cluster_compos)

    path1 = "link_to_testing_data"  
    for i in range(11):
        pl_test=[]
        path = path1 + "Youtube-" + "%s" %i+ ".txt"
        iat = parse2(path)
        #classify(kmeans, centroidss, iat)

    for i in range(11):
        pl_test = []
        path = path1 + "GoogleSearch-" + "%s" % i + ".txt"
        iat = parse2(path)
        #classify(kmeans, centroidss, iat)

    for i in range(11):
        pl_test = []
        path = path1 + "GoogleMusic-" + "%s" % i + ".txt"
        iat = parse2(path)
        #classify(kmeans, centroidss, iat)

    for i in range(11):
        pl_test = []
        path =  path1 + "GoogleDoc-" + "%s" %i+ ".txt"
        iat = parse2(path)
        #classify(kmeans, centroidss, iat)
