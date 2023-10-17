import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA
import qnorm
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
import seaborn as sns
import scipy.stats as ss
from scipy.stats import poisson
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
import umap
#import umap.plot
import math
import matplotlib.pyplot as plt
import random
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
from time import time
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from time import time
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
import conorm
def type_to_list(sample_label):
    type_list=list()
    label_list=sample_label.iloc[:,1].unique()
    for i in range(0,len(sample_label.iloc[:,1].unique())):
        for j in range(i+1,len(sample_label.iloc[:,1].unique())):
            type_list.append(label_list[i]+label_list[j])
    return type_list




def ARI(full_data,sample_label,inlier_genes):
    full_data.index=full_data.iloc[:,0]
    full_data=full_data.loc[inlier_genes,:]
    full_data=full_data.iloc[:,1:]
    a=proc_qnorm(full_data)
    full_data=proc_scale(a)
    hierarchical_cluster = AgglomerativeClustering(n_clusters=len(sample_label['Subtype'].unique()))
    kmeans = KMeans(init="k-means++", n_clusters=len(sample_label['Subtype'].unique()), n_init=10, random_state=0,max_iter=500)
    labels=kmeans.fit_predict(reverse(full_data))
    a1=metrics.adjusted_rand_score(labels, sample_label.iloc[:,1])
    a2=silhouette_score(reverse(full_data), sample_label.iloc[:,1])
    
    labels=hierarchical_cluster.fit_predict(reverse(full_data))
    a3=metrics.adjusted_rand_score(labels, sample_label.iloc[:,1])
    a4=silhouette_score(reverse(full_data), sample_label.iloc[:,1])
    return a1,a2,a3,a4

def ARI_log_norm(full_data,sample_label,inlier_genes):
    full_data.index=full_data.iloc[:,0]
    full_data=full_data.loc[inlier_genes,:]
    full_data=full_data.iloc[:,1:]
    #a=proc_qnorm(full_data)
    #full_data=proc_scale(a)
    
    full_data=np.log(full_data+1)
    
    hierarchical_cluster = AgglomerativeClustering(n_clusters=len(sample_label['Subtype'].unique()))
    kmeans = KMeans(init="k-means++", n_clusters=len(sample_label['Subtype'].unique()), n_init=10, random_state=5)
    labels=kmeans.fit_predict(reverse(full_data))
    a1=metrics.adjusted_rand_score(labels, sample_label.iloc[:,1])
    a2=silhouette_score(reverse(full_data), sample_label.iloc[:,1])
    
    labels=hierarchical_cluster.fit_predict(reverse(full_data))
    a3=metrics.adjusted_rand_score(labels, sample_label.iloc[:,1])
    a4=silhouette_score(reverse(full_data), sample_label.iloc[:,1])
    return a1,a2,a3,a4

def ARI_SCR(full_data,sample_label):
    full_data.index=full_data.iloc[:,0]
    #full_data=full_data.loc[inlier_genes,:]
    full_data=full_data.iloc[:,1:]
    
    hierarchical_cluster = AgglomerativeClustering(n_clusters=len(sample_label['Subtype'].unique()))
    kmeans = KMeans(init="k-means++", n_clusters=len(sample_label['Subtype'].unique()), n_init=10, random_state=5,max_iter=1000)
    labels=kmeans.fit_predict(reverse(full_data))
    a1=metrics.adjusted_rand_score(labels, sample_label.iloc[:,1])
    a2=silhouette_score(reverse(full_data), sample_label.iloc[:,1])
    
    labels=hierarchical_cluster.fit_predict(reverse(full_data))
    a3=metrics.adjusted_rand_score(labels, sample_label.iloc[:,1])
    a4=silhouette_score(reverse(full_data), sample_label.iloc[:,1])
    return a1,a2,a3,a4



def bench_k_means(kmeans, name, data, labels):
    t0 = time()
    estimator = make_pipeline(StandardScaler(), kmeans).fit(data)
    fit_time = time() - t0
    results = [name, fit_time, estimator[-1].inertia_]

    # Define the metrics which require only the true labels and estimator
    # labels
    clustering_metrics = [
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.adjusted_rand_score,
        metrics.adjusted_mutual_info_score,
    ]
    results += [m(labels, estimator[-1].labels_) for m in clustering_metrics]

    # The silhouette score requires the full dataset
    results += [
        metrics.silhouette_score(
            data,
            estimator[-1].labels_,
            metric="euclidean",
            sample_size=300,
        )
    ]

    # Show the results
    formatter_result = (
        "{:9s}\t{:.3f}s\t{:.0f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}"
    )
    #print(formatter_result.format(*results))
    return results
def heatmap(tobo1):    
    data=tobo1.copy()
    score=data.copy()
    i=len(score)
    my_palette = dict(zip(score.iloc[i-1].unique(), ["orange","yellow","brown","red"]))
    row_colors = score.iloc[i-1,:].map(my_palette)
    #plt.figure(figsize=(20,15))
    #ax=subplot(111)
    sns.clustermap(score.iloc[:i-1,:], cmap="RdYlBu_r",standard_scale=0,col_colors=row_colors)
    #standard_scale=1
    #fig = sns_plot.get_figure()
    #fig.savefig("output.png")
def scale_anomaly_scores(s):
    """Changing (-0.5, 0.5) to (0, 1)"""
    a, b = (-0.5, 0.5), (1, 0)
    (a1, a2), (b1, b2) = a, b
    return (b1 + ((s - a1) * (b2 - b1) / (a2 - a1))) * 100
def reverse(a):
    ndata = a.to_numpy()
    ndata = ndata.T
    H = pd.DataFrame(data = ndata[:,:])
    
    return H
def proc_qnorm(d):
    d_ln=np.log2(d+1)
    d_q=qnorm.quantile_normalize(d_ln, axis=1, ncpus=8)
    return d_q

## min-max scaling
def proc_scale(d):
    scaler = MinMaxScaler()
    d_s = MinMaxScaler(feature_range=(0, 1)).fit(d).transform(d)
    df_s = pd.DataFrame(d_s, columns = list(d.columns))
    df_s=df_s.round(6)
    return df_s
def jaccard(list1, list2):
    intersection = len(set(list1)&set(list2))
    union = len(set(list1)|set(list2))
    return float(intersection) / union
def distribution_plot(matrix2,path,data_type,outlier,number,sub_sample):
    
    
    fig, axes = plt.subplots(3, 3, figsize=(10, 10), sharey=True, dpi=100)
    sns.distplot(matrix2.iloc[:,0] , color="dodgerblue", ax=axes[0][0], axlabel=matrix2.columns[0])
    sns.distplot(matrix2.iloc[:,2] , color="deeppink", ax=axes[0][1], axlabel=matrix2.columns[2])
    sns.distplot(matrix2.iloc[:,4] , color="dodgerblue", ax=axes[0][2], axlabel=matrix2.columns[4])
    sns.distplot(matrix2.iloc[:,6] , color="gold", ax=axes[1][0], axlabel=matrix2.columns[6])
    sns.distplot(matrix2.iloc[:,8] , color="red", ax=axes[1][1], axlabel=matrix2.columns[8])
    sns.distplot(matrix2.iloc[:,10] , color="blue", ax=axes[1][2], axlabel=matrix2.columns[10])
    sns.distplot(matrix2.iloc[:,12] , color="red", ax=axes[2][0], axlabel=matrix2.columns[12])
    sns.distplot(matrix2.iloc[:,14] , color="silver", ax=axes[2][1], axlabel=matrix2.columns[14])
    sns.distplot(matrix2.iloc[:,16] , color="yellow", ax=axes[2][2], axlabel=matrix2.columns[16])
    if not os.path.exists('/data1/projects/2021_MOPA/SMALL_NMF_V2/result/'+data_type+'/'+str(sub_sample)+'/gene_distribution/'):
        os.makedirs('/data1/projects/2021_MOPA/SMALL_NMF_V2/result/'+data_type+'/'+str(sub_sample)+'/gene_distribution/')
    if not os.path.exists('/data1/projects/2021_MOPA/SMALL_NMF_V2/result/'+data_type+'/'+str(sub_sample)+'/gene_distribution/'+path):
        os.makedirs('/data1/projects/2021_MOPA/SMALL_NMF_V2/result/'+data_type+'/'+str(sub_sample)+'/gene_distribution/'+path)
    if not os.path.exists('/data1/projects/2021_MOPA/SMALL_NMF_V2/result/'+data_type+'/'+str(sub_sample)+'/gene_distribution/'+path+'/'+outlier):
        os.makedirs('/data1/projects/2021_MOPA/SMALL_NMF_V2/result/'+data_type+'/'+str(sub_sample)+'/gene_distribution/'+path+'/'+outlier)
    fig.savefig("/data1/projects/2021_MOPA/SMALL_NMF_V2/result/"+data_type+"/"+str(sub_sample)+"/gene_distribution/"+path+"/"+outlier+'/'+str(number)+'.pdf')
def jaccard_corr(full_data): 
    
    full_data_sort=full_data.copy()
    full_data_score=pd.DataFrame(index=range(0,len(full_data.columns)),columns=range(0,len(full_data.columns)))
    for i in range(0,len(full_data.columns)):
        full_data_sort.iloc[:,i]=full_data_sort.iloc[:,i].sort_values(ascending=False).index
    corr_score=list()
    for i in range(0,len(full_data.columns)):
        for j in range(0,len(full_data.columns)):
            corr_score=list()
            for k in range(0,int(len(full_data_sort.index)/1000)):
                #print(k)
                if k== int(len(full_data_sort.index)/1000)-1:
                    t=jaccard(full_data_sort.iloc[k*1000:,i],full_data_sort.iloc[k*1000:,j])
                else:
                    t=jaccard(full_data_sort.iloc[k*1000:k*1000+1000,i],full_data_sort.iloc[k*1000:k*1000+1000,j])
                corr_score.append(t)
            full_data_score.iloc[i,j]=sum(corr_score)/len(corr_score)
    return full_data_score
def PCA_noise(full_data,sample_label,sample_index,sub,normalization,inlier_genes):
    
    full_data.index=full_data.iloc[:,0]
    
    if sub=='yes':
        sample_label=sample_label.iloc[sample_index.iloc[:,1],:]
        full_data=full_data.iloc[:,1:]
        full_data=full_data.iloc[:,list(sample_index.iloc[:,1])]
    else:
        full_data=full_data.iloc[:,1:]
    full=full_data.copy()
    if normalization=='norm':
        #full_data=np.log2(full_data+1)
        a=proc_qnorm(full_data)
        full_data1=proc_scale(a)
        full_data1.index=full_data.index
        full_data1.columns= full_data.columns
        full_data=full_data1
        #full_data_PCA =np.log2(full_data+1)
    else:
        full_data=np.log2(full_data+1)
    pca = PCA(svd_solver='full', n_components=5)
    pca.fit(full_data)
    pca_vals = pca.transform(full_data)
    topk=len(inlier_genes)
    pc0=(np.argsort(pca_vals[:,0])[::-1])[:topk]
    df=pd.DataFrame(data=pc0)
    
    return full.iloc[pc0,:].index
def PCA_for_NMF(full_data,sample_label,sample_index,sub,normalization,lenofPCA):
    
    full_data.index=full_data.iloc[:,0]
    
    if sub=='yes':
        sample_label=sample_label.iloc[sample_index.iloc[:,1],:]
        full_data=full_data.iloc[:,1:]
        full_data=full_data.iloc[:,list(sample_index.iloc[:,1])]
    else:
        full_data=full_data.iloc[:,1:]
    full=full_data.copy()
    if normalization=='norm':
        #full_data=np.log2(full_data+1)
        a=proc_qnorm(full_data)
        full_data1=proc_scale(a)
        full_data1.index=full_data.index
        full_data1.columns= full_data.columns
        full_data=full_data1
        #full_data_PCA =np.log2(full_data+1)
    else:
        full_data=np.log2(full_data+1)
    pca = PCA(svd_solver='full', n_components=5)
    pca.fit(full_data)
    pca_vals = pca.transform(full_data)
    topk=lenofPCA
    pc0=(np.argsort(pca_vals[:,0])[::-1])[:topk]
    df=pd.DataFrame(data=pc0)
    
    return full.iloc[pc0,:].index


def weibull(x,m,n):
    return (m/n)*(x/n)**(m-1)*np.exp(-(x/n)**m)
def weibull_cdf(x,m,n):
    return 1-np.exp(-(x/n)**m)


def NMF_noise_v2(standard,full_data,iteration,normalization,rank,version,threshold,outdir,pvalue,pvalue1):
    n_MC = 10**6
    m=1.5
    n=2
    full_data.index=full_data.iloc[:,0]
    full_data=full_data.iloc[:,1:]
    full_data_NMF=full_data.loc[(full_data!=0).any(axis=1)]
    for i in range(0,iteration):
        matrix=reverse(full_data_NMF).copy()
        matrix.index=full_data_NMF.columns
        matrix.columns=full_data_NMF.index
        inlier_genes = set(matrix.columns)
        ALL_genes = set(matrix.columns)
        outg=list()
        outlier_genes=set()
        matrix=matrix.loc[:,inlier_genes]
        #matrix=matrix.loc[:,inlier_genes]
        if normalization =='norm':    
            a=proc_qnorm(full_data_NMF.loc[inlier_genes,:])
            full_data1=proc_scale(a)
            matrix=reverse(full_data1).copy()
            full_data1.index=full_data_NMF.loc[inlier_genes,:].index
            full_data1.columns= full_data_NMF.loc[inlier_genes,:].columns
            matrix.columns=full_data_NMF.loc[inlier_genes,:].index
            matrix.index= full_data_NMF.loc[inlier_genes,:].columns
        if normalization =='log_norm':
            matrix2=np.log(full_data_NMF.loc[inlier_genes,:]+1)
            matrix=reverse(matrix2)
            matrix.columns = matrix2.index
            matrix.index = matrix2.columns
        if normalization =='TMM':
            matrix2= conorm.tmm(full_data_NMF.loc[inlier_genes,:])
            #matrix2=np.log(full_data_NMF.loc[inlier_genes,:]+1)
            matrix=reverse(matrix2)
            matrix.columns = matrix2.index
            matrix.index = matrix2.columns
        model = NMF(n_components=rank,init='random',solver='mu',max_iter=1000,beta_loss='kullback-leibler',random_state=0)
        model.fit(matrix)
        sample_NMF = model.transform(matrix)
        gene_NMF=model.components_
        sample_NMF=pd.DataFrame(data=sample_NMF) #### sample_num X rank_freatures
        gene_NMF=pd.DataFrame(data=gene_NMF) ### rank_featrues X gene_num (ex, 40, 20,000)
        outlier=list()
        outlier_score=list()

        ##### finding outlier features , when one feature is very different in one sample it is outlier
        if standard !='yes':    
            for i in range(0,len(sample_NMF.columns)):
                model=IsolationForest(n_estimators=100,max_samples='auto',random_state=0)
                model.fit(sample_NMF.iloc[:,i:i+1])
                anomaly_scores = model.decision_function(sample_NMF.iloc[:,i:i+1])
                ###scale anomaly scores
                scaled_scores = pd.Series(anomaly_scores).map(scale_anomaly_scores).to_numpy()
                ###selecting outlier rank features
                if len(scaled_scores[scaled_scores>threshold])==1:
                    outlier.append(i)
                    anomaly_scores=[-1*s + 0.5 for s in anomaly_scores]
                    scaled_scores.sort()
                    outlier_score.append(scaled_scores[-1]-scaled_scores[-2])
                    gene_NMF.columns=list(matrix.columns)
            ##selecting biggest features in each genes
            gene_features=pd.DataFrame(data=gene_NMF.idxmax(axis=0))
            gene_features.index=list(matrix.columns)
            gene_NMF=reverse(gene_NMF)
            gene_NMF.index=list(matrix.columns)

        #choosing genes in outlier features
        if version=='V2':
            for outlier_gene in gene_features[(gene_features[0].isin(outlier))].index:
                model=IsolationForest(n_estimators=100,max_samples='auto',random_state=0)
                outlier_gene_score=pd.DataFrame(data=gene_NMF.loc[outlier_gene,:])
                model.fit(outlier_gene_score)
                anomaly_scores = model.decision_function(outlier_gene_score)
                p=pd.Series(anomaly_scores).map(scale_anomaly_scores).to_numpy()
                for feature in outlier:
                    if p[feature]>85:
                        outlier_genes.add(outlier_gene)
        if standard =='yes':
            isolation_fores_df=sample_NMF.copy()
            isolation_fores_scaled_df=sample_NMF.copy()
            for i in range(0,len(sample_NMF.columns)):
                #model=IsolationForest(n_estimators=100,max_samples='auto',random_state=0)
                anomaly_scores=weibull_cdf(sample_NMF.iloc[:,i:i+1],m,n)
                isolation_fores_df.loc[:,i] = anomaly_scores
                isolation_fores_scaled_df.loc[:,i] = anomaly_scores
            outlier=list()
            
            average = np.mean(isolation_fores_df)

            average = np.mean(average)
            #pvalue=1/len(sample_NMF.index)
            stat_outlier=list()
            for i in range(0,len(sample_NMF.columns)):
                data = isolation_fores_df.iloc[:,i:i+1]
                stat=stats.ttest_1samp(a=data, popmean=average)
                stat_outlier.append(stat[1][0])
                if stat[1][0] < pvalue:
                    outlier.append(i)
                if stat[1][0] > 0.95:
                    outlier.append(i)
            gene_NMF.columns=list(matrix.columns)
            ##selecting biggest features in each genes
            gene_features=pd.DataFrame(data=gene_NMF.idxmax(axis=0))
            gene_features.index=list(matrix.columns)
            gene_NMF=reverse(gene_NMF)
            gene_NMF.index=list(matrix.columns)
            
            outlier_genes_v2=set()
        
        outlier_genes=outlier_genes|set(gene_features[(gene_features[0].isin(outlier))].index)|outlier_genes_v2
        inlier_genes=ALL_genes-outlier_genes
        a=pd.DataFrame(data=inlier_genes)
        
        #outlier=list()
        outlier_score=list()
        outlier_sample_list=list()
        outlier_rank_list=list()
        stat_outlier_df = isolation_fores_df.copy()
        for t1 in isolation_fores_df.columns: ## feature
            for t2 in isolation_fores_df.index: ## samples
                t_value, p_value = stats.ttest_1samp(np.array(isolation_fores_df).flatten(), isolation_fores_df.loc[t2,t1])
                stat_outlier_df.loc[t2,t1]=p_value
                if p_value <pvalue1:
                    outlier_sample_list.append(t2)
                    outlier_rank_list.append(t1)
        outlier_genes=set()
        outlier_genes_v2=set()
        outlier_genes=outlier_genes|set(gene_features[(gene_features[0].isin(outlier_rank_list))].index)|outlier_genes_v2
        inlier_genes=ALL_genes-outlier_genes
        b=pd.DataFrame(data=inlier_genes)
        
        
        
        a.to_csv(outdir+'/0.csv',index=False)
        b.to_csv(outdir+'/2.csv',index=False)
        pd.DataFrame(data=outlier_sample_list).to_csv(outdir+'/2_outlier_sample.csv',index=False) 
        pd.DataFrame(data=stat_outlier).to_csv(outdir+'/outlier_stat.csv',index=False)
        pd.DataFrame(data=stat_outlier_df).to_csv(outdir+'/2_outlier_stat.csv',index=False)
        pd.DataFrame(data=outlier_rank_list).to_csv(outdir+'/2_outlier_rank.csv',index=False) 
        pd.DataFrame(data=outlier).to_csv(outdir+'/outlier.csv',index=False) 
        pd.DataFrame(data=isolation_fores_df).to_csv(outdir+'/weibull_cdf.csv',index=False) 
    return inlier_genes

def NMF_noise(standard,full_data,sample_label,iteration,normalization,rank,version,threshold,outdir,pvalue):
    
    full_data.index=full_data.iloc[:,0]
    full_data=full_data.iloc[:,1:]
    full_data_NMF=full_data.loc[(full_data!=0).any(axis=1)]
    for i in range(0,iteration):
        matrix=reverse(full_data_NMF).copy()
        matrix.index=full_data_NMF.columns
        matrix.columns=full_data_NMF.index
        inlier_genes = set(matrix.columns)
        ALL_genes = set(matrix.columns)
        outg=list()
        outlier_genes=set()
        matrix=matrix.loc[:,inlier_genes]
        #matrix=matrix.loc[:,inlier_genes]
        if normalization =='norm':    
            a=proc_qnorm(full_data_NMF.loc[inlier_genes,:])
            full_data1=proc_scale(a)
            matrix=reverse(full_data1).copy()
            full_data1.index=full_data_NMF.loc[inlier_genes,:].index
            full_data1.columns= full_data_NMF.loc[inlier_genes,:].columns
            matrix.columns=full_data_NMF.loc[inlier_genes,:].index
            matrix.index= full_data_NMF.loc[inlier_genes,:].columns
        if normalization =='log_norm':
            matrix2=np.log(full_data_NMF.loc[inlier_genes,:]+1)
            matrix=reverse(matrix2)
            matrix.columns = matrix2.index
            matrix.index = matrix2.columns
        if normalization =='TMM':
            matrix2= conorm.tmm(full_data_NMF.loc[inlier_genes,:])
            #matrix2=np.log(full_data_NMF.loc[inlier_genes,:]+1)
            matrix=reverse(matrix2)
            matrix.columns = matrix2.index
            matrix.index = matrix2.columns
        model = NMF(n_components=rank,init='random',solver='mu',max_iter=1000,beta_loss='kullback-leibler',random_state=0)
        model.fit(matrix)
        sample_NMF = model.transform(matrix)
        gene_NMF=model.components_
        sample_NMF=pd.DataFrame(data=sample_NMF) #### sample_num X rank_freatures
        gene_NMF=pd.DataFrame(data=gene_NMF) ### rank_featrues X gene_num (ex, 40, 20,000)
        outlier=list()
        outlier_score=list()

        ##### finding outlier features , when one feature is very different in one sample it is outlier
        if standard !='yes':    
            for i in range(0,len(sample_NMF.columns)):
                model=IsolationForest(n_estimators=100,max_samples='auto',random_state=0)
                model.fit(sample_NMF.iloc[:,i:i+1])
                anomaly_scores = model.decision_function(sample_NMF.iloc[:,i:i+1])
                ###scale anomaly scores
                scaled_scores = pd.Series(anomaly_scores).map(scale_anomaly_scores).to_numpy()
                ###selecting outlier rank features
                if len(scaled_scores[scaled_scores>threshold])==1:
                    outlier.append(i)
                    anomaly_scores=[-1*s + 0.5 for s in anomaly_scores]
                    scaled_scores.sort()
                    outlier_score.append(scaled_scores[-1]-scaled_scores[-2])
                    gene_NMF.columns=list(matrix.columns)
            ##selecting biggest features in each genes
            gene_features=pd.DataFrame(data=gene_NMF.idxmax(axis=0))
            gene_features.index=list(matrix.columns)
            gene_NMF=reverse(gene_NMF)
            gene_NMF.index=list(matrix.columns)

        #choosing genes in outlier features
        if version=='V2':
            for outlier_gene in gene_features[(gene_features[0].isin(outlier))].index:
                model=IsolationForest(n_estimators=100,max_samples='auto',random_state=0)
                outlier_gene_score=pd.DataFrame(data=gene_NMF.loc[outlier_gene,:])
                model.fit(outlier_gene_score)
                anomaly_scores = model.decision_function(outlier_gene_score)
                p=pd.Series(anomaly_scores).map(scale_anomaly_scores).to_numpy()
                for feature in outlier:
                    if p[feature]>85:
                        outlier_genes.add(outlier_gene)
        if standard =='yes':
            isolation_fores_df=sample_NMF.copy()
            isolation_fores_scaled_df=sample_NMF.copy()
            for i in range(0,len(sample_NMF.columns)):
                model=IsolationForest(n_estimators=100,max_samples='auto',random_state=0)
                model.fit(sample_NMF.iloc[:,i:i+1])
                anomaly_scores = model.decision_function(sample_NMF.iloc[:,i:i+1])
                ###scale anomaly scores
                scaled_scores = pd.Series(anomaly_scores).map(scale_anomaly_scores).to_numpy()
                ###selecting outlier rank features
                isolation_fores_df.loc[:,i] = anomaly_scores
                isolation_fores_scaled_df.loc[:,i] = scaled_scores
            outlier=list()
            
            average = np.mean(isolation_fores_df)

            average = np.mean(average)
            #pvalue=1/len(sample_NMF.index)
            for i in range(0,len(sample_NMF.columns)):
                data = isolation_fores_df.iloc[:,i:i+1]
                stat=stats.ttest_1samp(a=data, popmean=average)
                if pvalue > 0.5:
                    if stat[1][0] >pvalue:
                        outlier.append(i)
                else:    
                    if stat[1][0] <pvalue:
                        outlier.append(i)
            gene_NMF.columns=list(matrix.columns)
            ##selecting biggest features in each genes
            gene_features=pd.DataFrame(data=gene_NMF.idxmax(axis=0))
            gene_features.index=list(matrix.columns)
            gene_NMF=reverse(gene_NMF)
            gene_NMF.index=list(matrix.columns)
            
            #outlier_genes_v2=set()
            
            #for outlier_gene in gene_NMF.index:
            #    model=IsolationForest(n_estimators=1,max_samples='auto',random_state=0)
            #    outlier_gene_score=pd.DataFrame(data=gene_NMF.loc[outlier_gene,:])
            #    model.fit(outlier_gene_score)
            #    anomaly_scores = model.decision_function(outlier_gene_score)
            #    p=pd.Series(anomaly_scores).map(scale_anomaly_scores).to_numpy()
            #    #for feature in outlier:
            #    if max(p)<80:
            #        outlier_genes_v2.add(outlier_gene)
        
        #outlier_genes=outlier_genes|set(gene_features[(gene_features[0].isin(outlier))].index)|outlier_genes_v2
        outlier_genes=outlier_genes|set(gene_features[(gene_features[0].isin(outlier))].index)
        inlier_genes=ALL_genes-outlier_genes
        a=pd.DataFrame(data=inlier_genes)
        a.to_csv(outdir+'/0.csv',index=False)                    
    return inlier_genes
def NMF_noise_v3(full_data,normalization,rank,threshold,outdir):
    
    full_data.index=full_data.iloc[:,0]
    full_data=full_data.iloc[:,1:]
    full_data_NMF=full_data.loc[(full_data!=0).any(axis=1)]
    matrix=reverse(full_data_NMF).copy()
    matrix.index=full_data_NMF.columns
    matrix.columns=full_data_NMF.index
    inlier_genes = set(matrix.columns)
    ALL_genes = set(matrix.columns)
    outlier_genes=set()
    matrix=matrix.loc[:,inlier_genes]
    if normalization =='norm':    
        a=proc_qnorm(full_data_NMF.loc[inlier_genes,:])
        full_data1=proc_scale(a)
        matrix=reverse(full_data1).copy()
        full_data1.index=full_data_NMF.loc[inlier_genes,:].index
        full_data1.columns= full_data_NMF.loc[inlier_genes,:].columns
        matrix.columns=full_data_NMF.loc[inlier_genes,:].index
        matrix.index= full_data_NMF.loc[inlier_genes,:].columns
    if normalization =='log_norm':
        matrix2=np.log(full_data_NMF.loc[inlier_genes,:]+1)
        matrix=reverse(matrix2)
        matrix.columns = matrix2.index
        matrix.index = matrix2.columns
    if normalization =='TMM':
        matrix2= conorm.tmm(full_data_NMF.loc[inlier_genes,:])
        #matrix2=np.log(full_data_NMF.loc[inlier_genes,:]+1)
        matrix=reverse(matrix2)
        matrix.columns = matrix2.index
        matrix.index = matrix2.columns
    model = NMF(n_components=rank,init='random',solver='mu',max_iter=1000,beta_loss='kullback-leibler',random_state=0)
    model.fit(matrix)
    sample_NMF = model.transform(matrix)
    gene_NMF=model.components_
    sample_NMF=pd.DataFrame(data=sample_NMF) #### sample_num X rank_freatures
    gene_NMF=pd.DataFrame(data=gene_NMF) ### rank_featrues X gene_num (ex, 40, 20,000)
    gene_NMF.to_csv(outdir+'/gene_NMF.csv',index=False)
    sample_NMF.to_csv(outdir+'/sample_NMF.csv',index=False)
    outlier=list()
    outlier_score=list()
    ##### process start
    outlier_list=list()
    isolation_fores_df=sample_NMF.copy()
    isolation_fores_scaled_df=sample_NMF.copy()
    for i in range(0,len(sample_NMF.columns)):
        model=IsolationForest(n_estimators=5,max_samples='auto',random_state=0)
        model.fit(sample_NMF.iloc[:,i:i+1])
        anomaly_scores = model.decision_function(sample_NMF.iloc[:,i:i+1])
        scaled_scores = pd.Series(anomaly_scores).map(scale_anomaly_scores).to_numpy()
        if scaled_scores.max() <threshold:
            outlier_list.append(i)
        isolation_fores_df.loc[:,i] = anomaly_scores
        isolation_fores_scaled_df.loc[:,i] = scaled_scores
   # gene_features=pd.DataFrame(data=gene_NMF.idxmax(axis=0))
            
    gene_NMF=reverse(gene_NMF)
    gene_NMF.index=list(matrix.columns) ### columns genes
    gene_features1=pd.DataFrame(data=gene_NMF.idxmax(axis=1))
    gene_features1.index=list(matrix.columns)
    gene_features2= gene_NMF.apply(lambda x: gene_NMF.columns[np.argpartition(-x.values, 1)[1]], axis=1)
    gene_features2=pd.DataFrame(gene_features2)
    
    first_inlier_genes=gene_features1[gene_features1.iloc[:,0].isin(outlier_list)].index
    second_inlier_genes=gene_features2[gene_features2.iloc[:,0].isin(outlier_list)].index
    
    #gene_NMF=reverse(gene_NMF)
    #gene_NMF.index=list(matrix.columns)
    #outlier_genes=outlier_genes|set(gene_features[(gene_features[0].isin(outlier))].index)
    #inlier_genes=ALL_genes-outlier_genes
    inlier_genes = set(first_inlier_genes)
    a=pd.DataFrame(data=inlier_genes)
    b=pd.DataFrame(data=outlier_list)
    isolation_fores_df.to_csv(outdir+'/isolation_fores_0.csv',index=False)
    isolation_fores_scaled_df.to_csv(outdir+'/isolation_fores_scaled_0.csv',index=False)
    b.to_csv(outdir+'/outlier_rank.csv',index=False)
    a.to_csv(outdir+'/0.csv',index=False)
    
    
    return inlier_genes
def NMF_noise_v4(full_data,normalization,rank,threshold,outdir):
    
    full_data.index=full_data.iloc[:,0]
    full_data=full_data.iloc[:,1:]
    full_data_NMF=full_data.loc[(full_data!=0).any(axis=1)]
    matrix=reverse(full_data_NMF).copy()
    matrix.index=full_data_NMF.columns
    matrix.columns=full_data_NMF.index
    inlier_genes = set(matrix.columns)
    ALL_genes = set(matrix.columns)
    outlier_genes=set()
    matrix=matrix.loc[:,inlier_genes]
    if normalization =='norm':    
        a=proc_qnorm(full_data_NMF.loc[inlier_genes,:])
        full_data1=proc_scale(a)
        matrix=reverse(full_data1).copy()
        full_data1.index=full_data_NMF.loc[inlier_genes,:].index
        full_data1.columns= full_data_NMF.loc[inlier_genes,:].columns
        matrix.columns=full_data_NMF.loc[inlier_genes,:].index
        matrix.index= full_data_NMF.loc[inlier_genes,:].columns
    if normalization =='log_norm':
        matrix2=np.log(full_data_NMF.loc[inlier_genes,:]+1)
        matrix=reverse(matrix2)
        matrix.columns = matrix2.index
        matrix.index = matrix2.columns
    model = NMF(n_components=rank,init='random',solver='mu',max_iter=1000,beta_loss='kullback-leibler',random_state=0)
    model.fit(matrix)
    sample_NMF = model.transform(matrix)
    gene_NMF=model.components_
    sample_NMF=pd.DataFrame(data=sample_NMF) #### sample_num X rank_freatures
    gene_NMF=pd.DataFrame(data=gene_NMF) ### rank_featrues X gene_num (ex, 40, 20,000)
    gene_NMF.to_csv(outdir+'/gene_NMF.csv',index=False)
    sample_NMF.to_csv(outdir+'/sample_NMF.csv',index=False)
    outlier=list()
    outlier_score=list()
    ##### process start
    outlier_list=list()
    isolation_fores_df=sample_NMF.copy()
    isolation_fores_scaled_df=sample_NMF.copy()
    for i in range(0,len(sample_NMF.columns)):
        model=IsolationForest(n_estimators=5,max_samples='auto',random_state=0)
        model.fit(sample_NMF.iloc[:,i:i+1])
        anomaly_scores = model.decision_function(sample_NMF.iloc[:,i:i+1])
        scaled_scores = pd.Series(anomaly_scores).map(scale_anomaly_scores).to_numpy()
        if scaled_scores.max() <threshold:
            outlier_list.append(i)
        isolation_fores_df.loc[:,i] = anomaly_scores
        isolation_fores_scaled_df.loc[:,i] = scaled_scores
   # gene_features=pd.DataFrame(data=gene_NMF.idxmax(axis=0))
            
    gene_NMF=reverse(gene_NMF)
    gene_NMF.index=list(matrix.columns) ### columns genes
    gene_features1=pd.DataFrame(data=gene_NMF.idxmax(axis=1))
    gene_features1.index=list(matrix.columns)
    gene_features2= gene_NMF.apply(lambda x: gene_NMF.columns[np.argpartition(-x.values, 1)[1]], axis=1)
    gene_features2=pd.DataFrame(gene_features2)
    
    first_inlier_genes=gene_features1[gene_features1.iloc[:,0].isin(outlier_list)].index
    second_inlier_genes=gene_features2[gene_features2.iloc[:,0].isin(outlier_list)].index
    
    #gene_NMF=reverse(gene_NMF)
    #gene_NMF.index=list(matrix.columns)
    #outlier_genes=outlier_genes|set(gene_features[(gene_features[0].isin(outlier))].index)
    #inlier_genes=ALL_genes-outlier_genes
    inlier_genes = set(first_inlier_genes) - set(second_inlier_genes)
    a=pd.DataFrame(data=inlier_genes)
    b=pd.DataFrame(data=outlier_list)
    isolation_fores_df.to_csv(outdir+'/isolation_fores_0.csv',index=False)
    isolation_fores_scaled_df.to_csv(outdir+'/isolation_fores_scaled_0.csv',index=False)
    b.to_csv(outdir+'/outlier_rank.csv',index=False)
    a.to_csv(outdir+'/0.csv',index=False)
    
    return inlier_genes

def subsampling(sample_num,data_typ,data_type):
    sample_info=pd.read_csv('/data/projects/2021_MOPA/SMALL_NMF_V2/data/'+data_typ+'/full_data/subtype.csv') ### label data
    types=sample_info['Subtype'].unique()
    num=0
    num1=0
    
    f1=sample_num
    for f in range(0,20):
        selected_samples=set()
        for i in types:
            a=list(sample_info[sample_info['Subtype']==i].index)
            a1= random.sample(a,f1)
            
            selected_samples=selected_samples|set(a1)
            select=list(selected_samples)
            select.sort()
            samples=pd.DataFrame(data=list(select))
        num=f1*len(types)
        if not os.path.exists('/data1/projects/2021_MOPA/SMALL_NMF_V2/data/'+data_typ+'/subsample/'+str(num)):
            os.makedirs('/data1/projects/2021_MOPA/SMALL_NMF_V2/data/'+data_typ+'/subsample/'+str(num))
        samples.to_csv('/data1/projects/2021_MOPA/SMALL_NMF_V2/data/'+data_typ+'/subsample/'+str(num)+'/'+'samples_'+str(f)+'.csv')
        #matrix=gene.loc[select.iloc[:,1],:]
#def draw_umap(path,num,full_sample):
def draw_umap(full_data,sample_label,path,data_type,sub_sample,method,num):
    
    full_data=full_data.iloc[:,1:]
    #embedding_gene_corr_full = umap.UMAP(n_neighbors=10,
    #                  min_dist=0.1).fit_transform(reverse(full_data))
    a=proc_qnorm(full_data)
    full_data1=proc_scale(a)
    matrix=reverse(full_data1).copy()
    
    
    
    
    embedding_gene_corr_full = umap.UMAP().fit_transform(matrix)
    plot=pd.DataFrame(index=range(len(embedding_gene_corr_full)),columns =['x','y','type'])
    plot['x']=embedding_gene_corr_full[:,0]
    plot['y']=embedding_gene_corr_full[:,1]
    plot['type']=sample_label['Subtype'].to_list()
    if len(sample_label['Subtype'].unique())==2:
        umall=sns.scatterplot(x='x',y='y',hue='type',data=plot,s=90,palette=["#2c7fb8", "#f03b20"],linewidth=0)
    if len(sample_label['Subtype'].unique())==3:
        umall=sns.scatterplot(x='x',y='y',hue='type',data=plot,s=90,palette=["#2c7fb8", "#7fcdbb", "#edf8b1"],linewidth=0)
    if len(sample_label['Subtype'].unique())==4:
        umall=sns.scatterplot(x='x',y='y',hue='type',data=plot,s=90,palette=["#2c7fb8", "#7fcdbb", "#edf8b1", "#f03b20"],linewidth=0)
    if len(sample_label['Subtype'].unique())==8:
        umall=sns.scatterplot(x='x',y='y',hue='type',data=plot,s=90,palette=['#F8766D', '#F17CBE', '#7CE88D', '#00BFC4','red'],linewidth=0)
        #umall=sns.scatterplot(x='x',y='y',hue='type',data=plot,s=90,palette=['red','#ff7f0e','#2ca02c','#F25F5C','#247BA0','#70C1B3','#50514F','#628B48'],linewidth=0)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    umall = umall.get_figure()
    fig_size = (12, 12)
    umall.set_size_inches(fig_size)
    
    if not os.path.exists(path+data_type+'/'+sub_sample+'/'+'UMAP'):
        os.makedirs(path+data_type+'/'+sub_sample+'/'+'UMAP')
    if not os.path.exists(path+data_type+'/'+sub_sample+'/'+'UMAP'+'/'+method):
        os.makedirs(path+data_type+'/'+sub_sample+'/'+'UMAP'+'/'+method)
    umall.savefig(path+data_type+'/'+sub_sample+'/'+'UMAP'+'/'+method+'/'+str(num)+'.pdf')
    plt.clf()
def draw_umap_for_paper(full_data,sample_label,outdir,file_name):
    
    full_data=full_data.iloc[:,1:]
    a=proc_qnorm(full_data)
    full_data1=proc_scale(a)
    matrix=reverse(full_data1).copy()
    embedding_gene_corr_full = umap.UMAP().fit_transform(matrix)
    plot=pd.DataFrame(index=range(len(embedding_gene_corr_full)),columns =['x','y','type'])
    plot['x']=embedding_gene_corr_full[:,0]
    plot['y']=embedding_gene_corr_full[:,1]
    plot['type']=sample_label['Subtype'].to_list()
    if len(sample_label['Subtype'].unique())==2:
        umall=sns.scatterplot(x='x',y='y',hue='type',data=plot,s=90,palette=["#2c7fb8", "#f03b20"],linewidth=0)
    if len(sample_label['Subtype'].unique())==3:
        umall=sns.scatterplot(x='x',y='y',hue='type',data=plot,s=90,palette=["#2c7fb8", "#7fcdbb", "#edf8b1"],linewidth=0)
    if len(sample_label['Subtype'].unique())==4:
        umall=sns.scatterplot(x='x',y='y',hue='type',data=plot,s=90,palette=["#2c7fb8", "#7fcdbb", "#edf8b1", "#f03b20"],linewidth=0)
    if len(sample_label['Subtype'].unique())==5:
        umall=sns.scatterplot(x='x',y='y',hue='type',data=plot,s=90,palette=['#F8766D', '#F17CBE', '#7CE88D', '#00BFC4','red'],linewidth=0)
    if len(sample_label['Subtype'].unique())>5:
        umall=sns.scatterplot(x='x',y='y',hue='type',data=plot,s=90,linewidth=0)
    #plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    umall = umall.get_figure()
    fig_size = (8,8)
    umall.set_size_inches(fig_size)
    plt.show()
    umall.savefig(outdir+'/'+file_name+'.pdf')
    plt.clf()
def draw_umap1(full_data,sample_label,path,data_type,sub_sample,method,num,x):
    
    full_data=full_data.iloc[:,1:]
    #embedding_gene_corr_full = umap.UMAP(n_neighbors=10,
    #                  min_dist=0.1).fit_transform(reverse(full_data))
    a=proc_qnorm(full_data)
    full_data1=proc_scale(a)
    matrix=reverse(full_data1).copy()
    
    
    
    
    embedding_gene_corr_full = umap.UMAP().fit_transform(matrix)
    plot=pd.DataFrame(index=range(len(embedding_gene_corr_full)),columns =['x','y','type'])
    plot['x']=embedding_gene_corr_full[:,0]
    plot['y']=embedding_gene_corr_full[:,1]
    plot['type']=sample_label['Subtype'].to_list()
    kmeans = KMeans(n_clusters=x, random_state=0).fit(plot.iloc[:,0:2])
    labels = kmeans.labels_
    plot['cluster'] = labels
    return plot
    




# sample filter + iteration
def NMF_noise_Bulk(full_data,normalization,rank,outdir):
    n_MC = 10**6
    m=1.5
    n=2
    full_data.index=full_data.iloc[:,0]
    full_data=full_data.iloc[:,1:]
    full_data_NMF=full_data.loc[(full_data!=0).any(axis=1)]
    matrix=reverse(full_data_NMF).copy()
    matrix.index=full_data_NMF.columns
    matrix.columns=full_data_NMF.index
    inlier_genes = set(matrix.columns)
    ALL_genes = set(matrix.columns)
    outlier_genes=set()
    matrix=matrix.loc[:,inlier_genes]
    if normalization =='norm':    
        a=proc_qnorm(full_data_NMF.loc[inlier_genes,:])
        full_data1=proc_scale(a)
        matrix=reverse(full_data1).copy()
        full_data1.index=full_data_NMF.loc[inlier_genes,:].index
        full_data1.columns= full_data_NMF.loc[inlier_genes,:].columns
        matrix.columns=full_data_NMF.loc[inlier_genes,:].index
        matrix.index= full_data_NMF.loc[inlier_genes,:].columns
    if normalization =='log_norm':
        matrix2=np.log(full_data_NMF.loc[inlier_genes,:]+1)
        matrix=reverse(matrix2)
        matrix.columns = matrix2.index
        matrix.index = matrix2.columns
    if normalization =='TMM':
        matrix2= conorm.tmm(full_data_NMF.loc[inlier_genes,:])
        matrix=reverse(matrix2)
        matrix.columns = matrix2.index
        matrix.index = matrix2.columns
    model = NMF(n_components=rank,init='random',solver='mu',max_iter=1000,beta_loss='kullback-leibler',random_state=0)
    model.fit(matrix)
    sample_NMF = model.transform(matrix)
    gene_NMF=model.components_
    sample_NMF=pd.DataFrame(data=sample_NMF) #### sample_num X rank_freatures
    gene_NMF=pd.DataFrame(data=gene_NMF) ### rank_featrues X gene_num (ex, 40, 20,000)
    t1=sample_NMF.copy()
    t2=gene_NMF.copy()
    
    
    for pvalue in [0.1,0.05,0.01,0.001,0.0001]:
        sample_NMF=t1.copy()
        gene_NMF=t2.copy()
        
        outdir2=outdir+'_'+str(pvalue)
        if not os.path.exists(outdir2):
            os.makedirs(outdir2)
               
        isolation_fores_df=sample_NMF.copy()
        isolation_fores_scaled_df=sample_NMF.copy()
        outlier_max_sample_list=list()
        outlier_score_list=list()
        for i in range(0,len(sample_NMF.columns)):
            model=IsolationForest(n_estimators=100,max_samples='auto',random_state=0)
            model.fit(sample_NMF.iloc[:,i:i+1])
            anomaly_scores = model.decision_function(sample_NMF.iloc[:,i:i+1])
            ###scale anomaly scores
            scaled_scores = pd.Series(anomaly_scores).map(scale_anomaly_scores).to_numpy()
            ###selecting outlier rank features
            isolation_fores_df.loc[:,i] = anomaly_scores
            isolation_fores_scaled_df.loc[:,i] = scaled_scores
            outlier_score_list.append(scaled_scores.max())
        outlier=list()
        
        average = np.mean(isolation_fores_df)
        average = np.mean(average)
        outlier_stat=list()
        for i in range(0,len(sample_NMF.columns)):
            data = isolation_fores_df.iloc[:,i:i+1]
            stat=stats.ttest_1samp(a=data, popmean=average)
            if pvalue > 0.5:
                if stat[1][0] >pvalue:
                    outlier.append(i)
            else:    
                if stat[1][0] <pvalue:
                    outlier.append(i)
            outlier_max_sample_list.append(isolation_fores_df.index[np.argmax(isolation_fores_df.iloc[:,i])])  
            outlier_stat.append(stat[1][0])
        gene_NMF.columns=list(matrix.columns)
        gene_features=pd.DataFrame(data=gene_NMF.idxmax(axis=0))
        gene_features.index=list(matrix.columns)
        gene_NMF=reverse(gene_NMF)
        gene_NMF.index=list(matrix.columns)

        outlier_genes=outlier_genes|set(gene_features[(gene_features[0].isin(outlier))].index)
        if pvalue == 0.001:
            new111=outlier_genes.copy()
        inlier_genes=ALL_genes-outlier_genes
        a=pd.DataFrame(data=inlier_genes)
        
        
        a.to_csv(outdir2+'/0.csv',index=False)
        
        a=pd.DataFrame(data=outlier_score_list)
        a['sample']=outlier_max_sample_list
        a.to_csv(outdir2+'/outlier_score_list.csv',index=False)
        pd.DataFrame(data=outlier_stat).to_csv(outdir2+'/wb_outlier.csv',index=False) 
        pd.DataFrame(data=isolation_fores_scaled_df).to_csv(outdir2+'/isolation_matrix.csv',index=False) 
    return inlier_genes
def NMF_Bulk_gene_remove(full_data,normalization,rank,outdir,outdir_name,threadss=0.05):
    num=outdir_name
    #outdir_name=int(outdir_name)
    n_MC = 10**6
    m=1.5
    n=2
    full_data.index=full_data.iloc[:,0]
    full_data=full_data.iloc[:,1:]
    full_data_NMF=full_data.loc[(full_data!=0).any(axis=1)]
    matrix=reverse(full_data_NMF).copy()
    matrix.index=full_data_NMF.columns
    matrix.columns=full_data_NMF.index
    inlier_genes = list(set(matrix.columns))
    ALL_genes = set(matrix.columns)
    outlier_genes=set()
    
    matrix=matrix.loc[:,inlier_genes]
    
    #### Normalization
    if normalization =='norm':    
        a=proc_qnorm(full_data_NMF.loc[inlier_genes,:])
        full_data1=proc_scale(a)
        matrix=reverse(full_data1).copy()
        full_data1.index=full_data_NMF.loc[inlier_genes,:].index
        full_data1.columns= full_data_NMF.loc[inlier_genes,:].columns
        matrix.columns=full_data_NMF.loc[inlier_genes,:].index
        matrix.index= full_data_NMF.loc[inlier_genes,:].columns
    if normalization =='log_norm':
        matrix2=np.log(full_data_NMF.loc[inlier_genes,:]+1)
        matrix=reverse(matrix2)
        matrix.columns = matrix2.index
        matrix.index = matrix2.columns
    if normalization =='TMM':
        matrix2= conorm.tmm(full_data_NMF.loc[inlier_genes,:])
        matrix=reverse(matrix2)
        matrix.columns = matrix2.index
        matrix.index = matrix2.columns
        
        
    #### NMF    
    model = NMF(n_components=rank,init='random',solver='mu',max_iter=1000,beta_loss='kullback-leibler',random_state=0)
    model.fit(matrix)
    sample_NMF = model.transform(matrix)
    gene_NMF=model.components_
    sample_NMF=pd.DataFrame(data=sample_NMF) #### sample_num X rank_freatures
    gene_NMF=pd.DataFrame(data=gene_NMF) ### rank_featrues X gene_num (ex, 40, 20,000)
    t1=sample_NMF.copy()
    t2=gene_NMF.copy()
    #if int(outdir_name)!=0:
    #    P_list=[float(threadss)]
    #else:
    #    P_list=[0.05,0.01,0.001,1e-5,1e-10,1e-20,1e-30,0.8]
    P_list=[float(threadss)]
    for pvalue in P_list:
        inlier_genes = set(matrix.columns)
        ALL_genes = set(matrix.columns)
        outlier_genes=set()
        
        
        sample_NMF=t1.copy()
        gene_NMF=t2.copy()
        if len(P_list)==1:
            outdir2=outdir
        else:
            outdir2=outdir+'_'+str(pvalue)
        if not os.path.exists(outdir2):
            os.makedirs(outdir2)
               
        isolation_fores_df=sample_NMF.copy()
        isolation_fores_scaled_df=sample_NMF.copy()
        outlier_max_sample_list=list()
        outlier_score_list=list()
        for i in range(0,len(sample_NMF.columns)):
            model=IsolationForest(n_estimators=100,max_samples='auto',random_state=0)
            model.fit(sample_NMF.iloc[:,i:i+1])
            anomaly_scores = model.decision_function(sample_NMF.iloc[:,i:i+1])
            ###scale anomaly scores
            scaled_scores = pd.Series(anomaly_scores).map(scale_anomaly_scores).to_numpy()
            ###selecting outlier rank features
            isolation_fores_df.loc[:,i] = anomaly_scores
            isolation_fores_scaled_df.loc[:,i] = scaled_scores
            outlier_score_list.append(scaled_scores.max())
        outlier=list()
        
        average = np.mean(isolation_fores_df)
        average = np.mean(average)
        outlier_stat=list()
        for i in range(0,len(sample_NMF.columns)):
            data = isolation_fores_df.iloc[:,i:i+1]
            stat=stats.ttest_1samp(a=data, popmean=average)
            if pvalue > 0.5:
                if stat[1][0] >pvalue:
                    outlier.append(i)
                if stat[1][0] <0.001:
                    outlier.append(i)
            else:    
                if stat[1][0] <pvalue:
                    outlier.append(i)
            outlier_max_sample_list.append(isolation_fores_df.index[np.argmax(isolation_fores_df.iloc[:,i])])  
            outlier_stat.append(stat[1][0])
        gene_NMF.columns=list(matrix.columns)
        gene_features=pd.DataFrame(data=gene_NMF.idxmax(axis=0))
        gene_features.index=list(matrix.columns)
        gene_NMF=reverse(gene_NMF)
        gene_NMF.index=list(matrix.columns)

        outlier_genes=outlier_genes|set(gene_features[(gene_features[0].isin(outlier))].index)
        if pvalue == 0.001:
            new111=outlier_genes.copy()
        inlier_genes=ALL_genes-outlier_genes
        a=pd.DataFrame(data=inlier_genes)
        
        a.to_csv(outdir2+'/'+str(num)+'.csv',index=False)
        a=pd.DataFrame(data=outlier_score_list)
        a['sample']=outlier_max_sample_list
        a.to_csv(outdir2+'/outlier_score_list_'+str(num)+'.csv',index=False)
        pd.DataFrame(data=outlier_stat).to_csv(outdir2+'/outlier_p_value_'+str(num)+'.csv',index=False) 
        pd.DataFrame(data=outlier).to_csv(outdir2+'/outlier_'+str(num)+'.csv',index=False) 
        pd.DataFrame(data=isolation_fores_scaled_df).to_csv(outdir2+'/isolation_matrix_'+str(num)+'.csv',index=False)
    return inlier_genes
def NMF_Bulk_gene_remove_iteration_sample(full_data,normalization,rank,outdir):
    n_MC = 10**6
    m=1.5
    n=2
    full_data.index=full_data.iloc[:,0]
    full_data=full_data.iloc[:,1:]
    full_data_NMF=full_data.loc[(full_data!=0).any(axis=1)]
    matrix=reverse(full_data_NMF).copy()
    matrix.index=full_data_NMF.columns
    matrix.columns=full_data_NMF.index
    inlier_genes = set(matrix.columns)
    ALL_genes = set(matrix.columns)
    outlier_genes=set()
    
    matrix=matrix.loc[:,inlier_genes]
    
    #### Normalization
    if normalization =='norm':    
        a=proc_qnorm(full_data_NMF.loc[inlier_genes,:])
        full_data1=proc_scale(a)
        matrix=reverse(full_data1).copy()
        full_data1.index=full_data_NMF.loc[inlier_genes,:].index
        full_data1.columns= full_data_NMF.loc[inlier_genes,:].columns
        matrix.columns=full_data_NMF.loc[inlier_genes,:].index
        matrix.index= full_data_NMF.loc[inlier_genes,:].columns
    if normalization =='log_norm':
        matrix2=np.log(full_data_NMF.loc[inlier_genes,:]+1)
        matrix=reverse(matrix2)
        matrix.columns = matrix2.index
        matrix.index = matrix2.columns
    if normalization =='TMM':
        matrix2= conorm.tmm(full_data_NMF.loc[inlier_genes,:])
        matrix=reverse(matrix2)
        matrix.columns = matrix2.index
        matrix.index = matrix2.columns
        
        
    #### NMF    
    model = NMF(n_components=rank,init='random',solver='mu',max_iter=1000,beta_loss='kullback-leibler',random_state=0)
    model.fit(matrix)
    sample_NMF = model.transform(matrix)
    gene_NMF=model.components_
    sample_NMF=pd.DataFrame(data=sample_NMF) #### sample_num X rank_freatures
    gene_NMF=pd.DataFrame(data=gene_NMF) ### rank_featrues X gene_num (ex, 40, 20,000)
    t1=sample_NMF.copy()
    t2=gene_NMF.copy()

    P_list=[0.05,0.01,0.001,1e-5,1e-10,1e-20,1e-30,0.8]
    for pvalue in P_list:
        inlier_genes = set(matrix.columns)
        ALL_genes = set(matrix.columns)
        outlier_genes=set()
        
        
        sample_NMF=t1.copy()
        gene_NMF=t2.copy()
        
        outdir2=outdir+'_'+str(pvalue)
        if not os.path.exists(outdir2):
            os.makedirs(outdir2)
               
        isolation_fores_df=sample_NMF.copy()
        isolation_fores_scaled_df=sample_NMF.copy()
        outlier_max_sample_list=list()
        outlier_score_list=list()
        for i in range(0,len(sample_NMF.columns)):
            model=IsolationForest(n_estimators=100,max_samples='auto',random_state=0)
            model.fit(sample_NMF.iloc[:,i:i+1])
            anomaly_scores = model.decision_function(sample_NMF.iloc[:,i:i+1])
            ###scale anomaly scores
            scaled_scores = pd.Series(anomaly_scores).map(scale_anomaly_scores).to_numpy()
            ###selecting outlier rank features
            isolation_fores_df.loc[:,i] = anomaly_scores
            isolation_fores_scaled_df.loc[:,i] = scaled_scores
            outlier_score_list.append(scaled_scores.max())
        outlier=list()
        
        average = np.mean(isolation_fores_df)
        average = np.mean(average)
        outlier_stat=list()
        for i in range(0,len(sample_NMF.columns)):
            data = isolation_fores_df.iloc[:,i:i+1]
            stat=stats.ttest_1samp(a=data, popmean=average)
            if pvalue > 0.5:
                if stat[1][0] >pvalue:
                    outlier.append(i)
                if stat[1][0] <0.001:
                    outlier.append(i)
            else:    
                if stat[1][0] <pvalue:
                    outlier.append(i)
            outlier_max_sample_list.append(isolation_fores_df.index[np.argmax(isolation_fores_df.iloc[:,i])])  
            outlier_stat.append(stat[1][0])
        gene_NMF.columns=list(matrix.columns)
        gene_features=pd.DataFrame(data=gene_NMF.idxmax(axis=0))
        gene_features.index=list(matrix.columns)
        gene_NMF=reverse(gene_NMF)
        gene_NMF.index=list(matrix.columns)

        outlier_genes=outlier_genes|set(gene_features[(gene_features[0].isin(outlier))].index)
        if pvalue == 0.001:
            new111=outlier_genes.copy()
        inlier_genes=ALL_genes-outlier_genes
        a=pd.DataFrame(data=inlier_genes)
        
        #a.to_csv(outdir2+'/0.csv',index=False)
        a=pd.DataFrame(data=outlier_score_list)
        a['sample']=outlier_max_sample_list
        #a.to_csv(outdir2+'/outlier_score_list.csv',index=False)
        #pd.DataFrame(data=outlier_stat).to_csv(outdir2+'/outlier_p_value.csv',index=False) 
        #pd.DataFrame(data=outlier).to_csv(outdir2+'/outlier.csv',index=False) 
        #pd.DataFrame(data=isolation_fores_scaled_df).to_csv(outdir2+'/isolation_matrix.csv',index=False)
    return a,pd.DataFrame(data=outlier_stat)
def NMF_Bulk_gene_remove_iteration_gene(full_data,normalization,rank,outdir):
    n_MC = 10**6
    m=1.5
    n=2
    full_data.index=full_data.iloc[:,0]
    full_data=full_data.iloc[:,1:]
    full_data_NMF=full_data.loc[(full_data!=0).any(axis=1)]
    matrix=reverse(full_data_NMF).copy()
    matrix.index=full_data_NMF.columns
    matrix.columns=full_data_NMF.index
    inlier_genes = set(matrix.columns)
    ALL_genes = set(matrix.columns)
    outlier_genes=set()
    
    matrix=matrix.loc[:,inlier_genes]
    
    #### Normalization
    if normalization =='norm':    
        a=proc_qnorm(full_data_NMF.loc[inlier_genes,:])
        full_data1=proc_scale(a)
        matrix=reverse(full_data1).copy()
        full_data1.index=full_data_NMF.loc[inlier_genes,:].index
        full_data1.columns= full_data_NMF.loc[inlier_genes,:].columns
        matrix.columns=full_data_NMF.loc[inlier_genes,:].index
        matrix.index= full_data_NMF.loc[inlier_genes,:].columns
    if normalization =='log_norm':
        matrix2=np.log(full_data_NMF.loc[inlier_genes,:]+1)
        matrix=reverse(matrix2)
        matrix.columns = matrix2.index
        matrix.index = matrix2.columns
    if normalization =='TMM':
        matrix2= conorm.tmm(full_data_NMF.loc[inlier_genes,:])
        matrix=reverse(matrix2)
        matrix.columns = matrix2.index
        matrix.index = matrix2.columns
        
        
    #### NMF    
    model = NMF(n_components=rank,init='random',solver='mu',max_iter=1000,beta_loss='kullback-leibler',random_state=0)
    model.fit(matrix)
    sample_NMF = model.transform(matrix)
    gene_NMF=model.components_
    sample_NMF=pd.DataFrame(data=sample_NMF) #### sample_num X rank_freatures
    gene_NMF=pd.DataFrame(data=gene_NMF) ### rank_featrues X gene_num (ex, 40, 20,000)
    t1=sample_NMF.copy()
    t2=gene_NMF.copy()

    P_list=[0.05,0.01,0.001,1e-5,1e-10,1e-20,1e-30,0.8]
    for pvalue in P_list:
        inlier_genes = set(matrix.columns)
        ALL_genes = set(matrix.columns)
        outlier_genes=set()
        
        
        sample_NMF=t1.copy()
        gene_NMF=t2.copy()
        
        outdir2=outdir+'_'+str(pvalue)
        if not os.path.exists(outdir2):
            os.makedirs(outdir2)
               
        isolation_fores_df=sample_NMF.copy()
        isolation_fores_scaled_df=sample_NMF.copy()
        outlier_max_sample_list=list()
        outlier_score_list=list()
        for i in range(0,len(sample_NMF.columns)):
            model=IsolationForest(n_estimators=100,max_samples='auto',random_state=0)
            model.fit(sample_NMF.iloc[:,i:i+1])
            anomaly_scores = model.decision_function(sample_NMF.iloc[:,i:i+1])
            ###scale anomaly scores
            scaled_scores = pd.Series(anomaly_scores).map(scale_anomaly_scores).to_numpy()
            ###selecting outlier rank features
            isolation_fores_df.loc[:,i] = anomaly_scores
            isolation_fores_scaled_df.loc[:,i] = scaled_scores
            outlier_score_list.append(scaled_scores.max())
        outlier=list()
        
        average = np.mean(isolation_fores_df)
        average = np.mean(average)
        outlier_stat=list()
        for i in range(0,len(sample_NMF.columns)):
            data = isolation_fores_df.iloc[:,i:i+1]
            stat=stats.ttest_1samp(a=data, popmean=average)
            if pvalue > 0.5:
                if stat[1][0] >pvalue:
                    outlier.append(i)
                if stat[1][0] <0.001:
                    outlier.append(i)
            else:    
                if stat[1][0] <pvalue:
                    outlier.append(i)
            outlier_max_sample_list.append(isolation_fores_df.index[np.argmax(isolation_fores_df.iloc[:,i])])  
            outlier_stat.append(stat[1][0])
        gene_NMF.columns=list(matrix.columns)
        gene_features=pd.DataFrame(data=gene_NMF.idxmax(axis=0))
        gene_features.index=list(matrix.columns)
        gene_NMF=reverse(gene_NMF)
        gene_NMF.index=list(matrix.columns)

        outlier_genes=outlier_genes|set(gene_features[(gene_features[0].isin(outlier))].index)
        if pvalue == 0.001:
            new111=outlier_genes.copy()
        inlier_genes=ALL_genes-outlier_genes
        a=pd.DataFrame(data=inlier_genes)
        
        #a.to_csv(outdir2+'/0.csv',index=False)
        a=pd.DataFrame(data=outlier_score_list)
        a['sample']=outlier_max_sample_list
        #a.to_csv(outdir2+'/outlier_score_list.csv',index=False)
        #pd.DataFrame(data=outlier_stat).to_csv(outdir2+'/outlier_p_value.csv',index=False) 
        #pd.DataFrame(data=outlier).to_csv(outdir2+'/outlier.csv',index=False) 
        #pd.DataFrame(data=isolation_fores_scaled_df).to_csv(outdir2+'/isolation_matrix.csv',index=False)
    return inlier_genes