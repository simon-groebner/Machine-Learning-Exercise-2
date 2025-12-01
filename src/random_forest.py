import numpy as np
from .regression_tree import RegressionTree

class RandomForest:
    def __init__(self, n_trees=10, min_samples_split=2, max_depth=float('inf'), seed=None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.trees = []
        self.rng = np.random.default_rng(seed)

    def fit(self, X, y):
        """
        Trains n_trees on bootstrap samples(random subset) of the data.
        """
        self.trees = []
        
        for _ in range(self.n_trees):
            #Create a new tree
            tree = RegressionTree(
                min_samples_split=self.min_samples_split, 
                max_depth=self.max_depth
            )
            
            #Create a "Bootstrap Sample" (Random subset of data)
            X_sample, y_sample = self._bootstrap_sample(X, y)
            
            # Train the tree on this random subset
            tree.fit(X_sample, y_sample)
            
            # 4. Store the trained tree
            self.trees.append(tree)

    def _bootstrap_sample(self, X, y):
        """
        Creates a random sample of the data with replacement.
        """
        n_samples = X.shape[0]
        # Pick n_samples random(with a seed) indices (some will be repeated, some skipped)
        idxs = self.rng.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def predict(self, X):
        """
        Predicts the class by averaging predictions from all trees.
        """
        # Collect predictions from every tree
        # Shape: (n_trees, n_samples)
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        
        # Return the mean prediction across all trees (axis=0)
        return np.mean(tree_preds, axis=0)