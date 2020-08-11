---
layout: post
title: Hierarchical Clustering Visualizations
subtitle: A Brief Look at Dendrograms
date: 2018-02-06
background: '/img/posts/h-clusts/genome-2d-dend.png'
---

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

### The Algorithm

Agglomerative hierarchical clustering is a clustering algorithm that follows these simple rules:
* Every point in your dataset starts out as a cluster of size one
* Find the two closest (spatially) clusters and merge them
* Repeat this until all points are part of one final cluster
By following these steps you can draw out a *dendrogram*. A dendrogram is a tree-like structure that shows the way in which the clusters were merged. Here is a simple example:

| ![dend-no-cut.png]({{ site.baseurl }}/img/posts/h-clusts/dend-no-cut.png)
|:--:|
| *A simple dendrogram showing the merging of ten points* |

Each tick on the x-axis is a point in the dataset, the y-axis represents the distance between two clusters at the point of merging. So in this example, at the bottom left of the graph we can see that two clusters, one containing points 5 and 6 and one containing points 9, 7, and 8, were merged at a distance of about 400. That means that, by our distance metric, these two sub-clusters were a distance of 400 apart when they were the two closest clusters in the set.

It is important to note that the definition of "distance between clusters" can vary. There are many ways to define this distance. One could use the distance between the two closest points in the two clusters being compared (this is called single-linkage clustering). We could also use the maximum distance between points in the clusters (complete-linkage clustering). Other methods include defining the center of a cluster, or using the mean of all points in the clusters. For this demonstration I will be using single-linkage clustering due to the nature of my dataset. This method gives me the added knowledge that for any cluster in my dendrogram, the greatest distance between any two given elements in the cluster must equal the y-value of the top horizontal line in the graph (of that cluster). For the cluster containing {5, 6, 7, 8, 9} from the above image, we know all elements in the cluster set must be $$\leq$$ approximately 400 apart.

Once we have a dendrogram we can *cut* at various distances, essentially adding the constraint that no merges greater than the distance specified can occur. If this distance is less than the final merge, we will have at least two clusters resulting from this constraint. Visually, this cut looks like:

| ![dend-cut.png]({{ site.baseurl }}/img/posts/h-clusts/dend-cut.png)
|:--:|
| *Dendrogram cut at dist = 2000* |

By varying this cut threshold we can see how the resulting clusters change. From here, we can analyze the clusters at various cut levels using techniques that I will not go into in this writeup. If you are interested, [here is a page](https://www.stat.berkeley.edu/~spector/s133/Clus.html) that demonstrates some of these methods.

### My Visualization: Genomics Data

The following is a particular implementation of hierarchical clustering that I found interesting and visually helpful. The dataset that I will be using is from the paper  [*Genome-wide assessment of sequence-intrinsic enhancer responsiveness at single-base-pair resolution*](https://www.nature.com/articles/nbt.3739). The data generated gives the results of an assay on the *Drosophila Melanogaster* genome that examines the responsiveness of genomic sequences to enhancers[^fn1]. We end up with a dataset that gives the index of base pairs at which genomic transcription was observed to start (Transcription Start Sites) and the number of reporter transcripts observed at that TSS.

[^fn1]: Nature Biotechnology volume 35, pages 136–144 (2017)  
        doi:10.1038/nbt.3739  

So let's say I want to cluster these TSSs. I want to cluster them because I eventually want to have a dataset of "promoters", which are approximately length 100 sequences that are the site of the beginning of gene transcription. I don't want to use every TSS as an individual promoter because there will be a lot of inaccuracy and redundancy. Instead, I want to find clusters of genomic activity, and place each promoter location based on those clusters. I have spatial information since the distance between TSSs can be defined by number of base pairs. This is a one dimensional hierarchical clustering, so the computational cost is low. Here is a dendrogram from this clustering, for the first 1000 TSSs and a cut of 5000 (I used the scipy hierarchical clustering functions to cluster and plot):

| ![genome-2d-dend.png]({{ site.baseurl }}/img/posts/h-clusts/genome-2d-dend.png)
|:--:|
| *Dendrogram of first 1000 base pairs w/ cut at dist = 5000* |

Obviously there is overplotting in this graph, but I want to get a general idea of what my clustered data looks like. There are three things that I think would be helpful in this case. First, I would rather have all of the data on the x-axis be sequentially ordered. I can get away with this because my data is one dimensional and I am using single-linkage clustering. Second, though the dendrogram already shows the distance between clusters (the y-axis), it would be interesting to also include that spatial component on the x-axis. I know this is redundant, but given the nature of the data it would be a nice visual aid as well. Finally, I can add a third dimension to the data: reporter transcript count. Each TSS has an associated integer reporter transcript count, so I can plot those on the plane $$y = 0$$.  Here is the result for the first 100 base pairs using these new attributes:

| ![genome-3d-dend.png]({{ site.baseurl }}/img/posts/h-clusts/genome-3d-dend.png)
|:--:|
| *3D dendrogram with ordered spatial x-axis and reporter transcript dimension* |

I like this view because we are now visualizing TSSs on the genome.

Realistically, our clusters are going to be much smaller than a distance threshold of 5000 (if promoters are only approx. 100 bp then we don't want that big of a threshold). Let's now use a threshold of 50 and compare the various clusters in our dataset side by side to gain insight about the overall structure of our clusters. To do this, I grabbed 16 random individual clusters from the dataset that were clustered at a distance threshold of 50 and plotted each of their 3D dendrograms side by side:

| ![dend-compare.png]({{ site.baseurl }}/img/posts/h-clusts/dend-compare.png)
|:--:|
| *Comparison of clusters with distance threshold = 50* |

Here we can see some interesting features about the clusters. Some of the clusters look very weak; they have only a few scattered TSSs and each TSS has only a couple reporter transcripts. We also see clusters with one very large peak of reporter transcripts, and "bi-modal" clusters with two very large peaks somewhat close to each other. These insights can be used in the construction of a working definition of a promoter, based on clusters of TSSs.

Further analysis could shed light on the relative quantities of these "peaked" vs. "bi-modal" vs. "noisy" promoters. Varying the cut threshold and examining these statistics could help lead to a conclusion about what the "best" cut value for our data is.
