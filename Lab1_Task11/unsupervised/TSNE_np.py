import numpy as np

class TSNE:

    def __init__(self, n_components=2, perplexity=20.0, max_iter=500, learning_rate=10,random_state=1):
        """A t-Distributed Stochastic Neighbor Embedding implementation.
        Parameters
        ----------
        max_iter : int, default 200
        perplexity : float, default 30.0
        n_components : int, default 2
        """
        self.max_iter = max_iter
        self.perplexity = perplexity
        self.n_components = n_components
        self.momentum = 0.9
        self.lr = learning_rate
        self.seed=random_state

    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)

    def fit(self,X):
        self.Y = np.random.RandomState(self.seed).normal(0., 0.0001, [X.shape[0], self.n_components])
        self.Q, self.distances = self.q_tsne()
        self.P=self.p_joint(X)
        
    def transform(self,X):
        if self.momentum:
            Y_m2 = self.Y.copy()
            Y_m1 = self.Y.copy()
            
        for i in range(self.max_iter):
            # Get Q and distances (distances only used for t-SNE)
            self.Q, self.distances = self.q_tsne()
            # Estimate gradients with respect to Y
            grads = self.tsne_grad()
            # Update Y
            self.Y = self.Y - self.lr * grads

            if self.momentum:  # Add momentum
                self.Y += self.momentum * (Y_m1 - Y_m2)
                # Update previous Y's for momentum
                Y_m2 = Y_m1.copy()
                Y_m1 = self.Y.copy()
        return self.Y

    def p_joint(self,X):
        """Given a data matrix X, gives joint probabilities matrix.
    # Arguments
        X: Input data matrix.
    # Returns:
        P: Matrix with entries p_ij = joint probabilities.
    """
        def p_conditional_to_joint(P):
            """Given conditional probabilities matrix P, return
            approximation of joint distribution probabilities."""
            return (P + P.T) / (2. * P.shape[0])
        
        def calc_prob_matrix(distances, sigmas=None, zero_index=None):
            """Convert a distances matrix to a matrix of probabilities."""
            if sigmas is not None:
                two_sig_sq = 2. * np.square(sigmas.reshape((-1, 1)))
                return self.softmax(distances / two_sig_sq, zero_index=zero_index)
            else:
                return self.softmax(distances, zero_index=zero_index)
            
        # Get the negative euclidian distances matrix for our data
        distances = self.neg_squared_euc_dists(X)
        # Find optimal sigma for each row of this distances matrix
        sigmas = self.find_optimal_sigmas()
        # Calculate the probabilities based on these optimal sigmas
        p_conditional = calc_prob_matrix(distances, sigmas)
        # Go from conditional to joint probabilities matrix
        self.P = p_conditional_to_joint(p_conditional)
        return self.P
  
    def find_optimal_sigmas(self):
        """For each row of distances matrix, find sigma that results
        in target perplexity for that role."""
        def binary_search(eval_fn, target, tol=1e-10, max_iter=10000,
                  lower=1e-20, upper=1000.):
            """Perform a binary search over input values to eval_fn.
            # Arguments
            eval_fn: Function that we are optimising over.
            target: Target value we want the function to output.
            tol: Float, once our guess is this close to target, stop.
            max_iter: Integer, maximum num. iterations to search for.
            lower: Float, lower bound of search range.
            upper: Float, upper bound of search range.
            # Returns:
            Float, best input value to function found during search."""
            for i in range(max_iter):
                guess = (lower + upper) / 2.
                val = eval_fn(guess)
                if val > target:
                    upper = guess
                else:
                    lower = guess
                if np.abs(val - target) <= tol:
                    break
            return guess
        
        def calc_perplexity(prob_matrix):
            """Calculate the perplexity of each row
            of a matrix of probabilities."""
            entropy = -np.sum(prob_matrix * np.log2(prob_matrix), 1)
            perplexity = 2 ** entropy
            return perplexity

        def perplexity(distances, sigmas, zero_index):
            """Wrapper function for quick calculation of
            perplexity over a distance matrix."""
            def calc_prob_matrix(distances, sigmas=None, zero_index=None):
                """Convert a distances matrix to a matrix of probabilities."""
                if sigmas is not None:
                    two_sig_sq = 2. * np.square(sigmas.reshape((-1, 1)))
                    return self.softmax(distances / two_sig_sq, zero_index=zero_index)
                else:
                    return self.softmax(distances, zero_index=zero_index)
                return calc_perplexity(calc_prob_matrix(distances, sigmas, zero_index))
    
            sigmas = []
            # For each row of the matrix (each point in our dataset)
            for i in range(self.distances.shape[0]):
                # Make fn that returns perplexity of this row given sigma
                    eval_fn = lambda sigma: perplexity(self.distances[i:i+1, :], np.array(sigma), i)
                    # Binary search over sigmas to achieve target perplexity
                    correct_sigma = binary_search(eval_fn, self.perplexity)
                    # Append the resulting sigma to our output array
                    sigmas.append(correct_sigma)
            return np.array(sigmas)