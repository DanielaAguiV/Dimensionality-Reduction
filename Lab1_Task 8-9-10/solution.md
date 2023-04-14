# **More robuste PCA**

Robust PCA factors a matrix into the sum of two matrices, $M=L+S$, where $M$ is the original matrix, $L$ is low-rank, and $S$ is sparse. This is what we'll be using for the background removal problem! Low-rank means that the matrix has a lot of redundant information. Sparse means that the matrix has mostly zero entries 

Original paper is here https://arxiv.org/pdf/0912.3599.pdf







# **About UMAP and LDA**
  
## **UMAP**
Uniform Manifold Approximation and Projection (UMAP) is a dimension reduction technique that can be used for visualisation similarly to t-SNE, but also for general non-linear dimension reduction. The algorithm is founded on three assumptions about the data

* *The data is uniformly distributed on Riemannian manifold*: This is any ball of fixed volumen should contain approximatly the same numbers osf $X$, regardles of where the manifols it is centered

* *The Riemannian metric is locally constant (or can be approximated as such)*

    Definition: Lef $f :X \to  S $ be a function from a topology space $X$ into a set $S$. 
    If $x \in X$ the $f$ is locally constant at $x$ if  there is a neighborhodd $U \subseteq X$ of $x$ such that $f$ is constant in $U$, ie, $f(u) = f(v)$  $\forall u,v \in U$. The function $f$ is locally constant if it is locally constant at every point of $ x \in X $ in the domain  


* *The manifold is locally connected.* $X$ in locally connected if every point admits a negouborhood basis consisting enterly of open connected sets
    
        conneted: can be represented for the union of two disjoint not-empty sets

From these assumptions it is possible to model the manifold with a fuzzy topological structure. The embedding is found by searching for a low dimensional projection of the data that has the closest possible equivalent fuzzy topological structure.

For more information we can see https://arxiv.org/pdf/1802.03426.pdf in section 2.

**Note** A manifold looks like $\R^n$ locally


## **LDA**
We describe latent Dirichlet allocation (LDA), a generative probabilistic model for collections of
discrete data such as text corpora. LDA is a three-level hierarchical Bayesian model, in which each
item of a collection is modeled as a finite mixture over an underlying set of topics. Each topic is, in
turn, modeled as an infinite mixture over an underlying set of topic probabilities. In the context of
text modeling, the topic probabilities provide an explicit representation of a document. We present
efficient approximate inference techniques based on variational methods and an EM algorithm for
empirical Bayes parameter estimation. We report results in document modeling, text classification,
and collaborative filtering, comparing to a mixture of unigrams model and the probabilistic LSI
model.

Latent Dirichlet allocation (LDA) is a generative probabilistic model of a corpus. The basic idea is
that documents are represented as random mixtures over latent topics, where each topic is characterized by a distribution over words.1
LDA assumes the following generative process for each document w in a corpus D:

* $N \sim Poisson(\xi)$ , $N$ document shape
* Choose $\theta \sim Dir(\alpha)$
* For each of the $N$ words $w_n$:
     
     a. Choose topic $z_n \sim Multinomial(\theta)$ 

     b. Choose a word $w_n$ from $p(w_n|z_n,\beta)$, a multinomial probability conditioned on the topic $z_n$

For more informarion you can see https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf