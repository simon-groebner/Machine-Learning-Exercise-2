import numpy as np

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        """
        Stores the information of a node.
        """
        # DECISION NODE
        self.feature_index = feature_index  
        self.threshold = threshold          
        self.left = left                    
        self.right = right                  
        
        # LEAF NODE
        self.value = value                  

class RegressionTree:
    def __init__(self, min_samples_split=2, max_depth=float('inf')):
        # Hyperparameters to stop the tree from growing too large (overfitting)
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        """
        Builds the tree based on X and y.
        """
        # Start building the tree recursively from the root
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, current_depth=0):
        """
        Recursive function to build the tree.
        """
        num_samples, num_features = X.shape
        
        # Max depth or too few samples --> split
        if num_samples < self.min_samples_split or current_depth >= self.max_depth:
            leaf_value = self._calculate_leaf_value(y)
            return Node(value=leaf_value)
        
        # Identify best split
        best_split = self._get_best_split(X, y, num_features)
        
        # Builds the tree with the split
        if best_split["gain"] > 0:
            left_subtree = self._build_tree(best_split["X_left"], best_split["y_left"], current_depth + 1)
            right_subtree = self._build_tree(best_split["X_right"], best_split["y_right"], current_depth + 1)
            
            return Node(
                feature_index=best_split["feature_index"],
                threshold=best_split["threshold"],
                left=left_subtree,
                right=right_subtree
            )
        
        return Node(value=self._calculate_leaf_value(y))

    def _calculate_leaf_value(self, y):
        """
        Calculates the value of a leaf.(mean of target)
        """
        return np.mean(y)

    def _get_best_split(self, X, y, num_features):
        """
        Find the best split for a node.
        """
        best_split = {
            "feature_index": None,
            "threshold": None,
            "gain": -float("inf"), # Start with very low gain
            "X_left": None,
            "X_right": None,
            "y_left": None,
            "y_right": None
        }
        
        # Calculate the variance of the parent node
        parent_variance = self._calculate_variance(y)
        
        # Loop over every feature (column)
        for feature_index in range(num_features):
            
            # Get all unique values in this feature to test as thresholds
            thresholds = np.unique(X[:, feature_index])
            
            for threshold in thresholds:
                # Split the data
                X_left, X_right, y_left, y_right = self._split_data(X, y, feature_index, threshold)
                
                if len(X_left) == 0 or len(X_right) == 0:
                    continue
                
                # Calculate the weighted variance of children
                n = len(y)
                n_l, n_r = len(y_left), len(y_right)
                var_l = self._calculate_variance(y_left)
                var_r = self._calculate_variance(y_right)
                
                child_variance = (n_l / n) * var_l + (n_r / n) * var_r
                
                # Calculate gain
                gain = parent_variance - child_variance
                
                if gain > best_split["gain"]:
                    best_split["feature_index"] = feature_index
                    best_split["threshold"] = threshold
                    best_split["gain"] = gain
                    best_split["X_left"] = X_left
                    best_split["X_right"] = X_right
                    best_split["y_left"] = y_left
                    best_split["y_right"] = y_right
                    
        return best_split

    def _calculate_variance(self, y):
        """
        Calculates the variance of a target,
        """
        m = len(y)
        if m == 0:
            return 0
        mean = np.mean(y)
        return np.mean((y - mean) ** 2)    

    def _split_data(self, X, y, feature_index, threshold):
        """
        Splits data based on a threshold.
        """
        # Finds rows where the feature value is less than or equal to the threshold
        left_mask = X[:, feature_index] <= threshold
        
        right_mask = ~left_mask
        
        return X[left_mask], X[right_mask], y[left_mask], y[right_mask]


    

    def predict(self, X):
        """
        Predicts values for an array of samples X.
        """
        predictions = []
        
        # Loop over each row (sample) in the input data
        for x in X:
            prediction = self._make_prediction(x, self.root)
            predictions.append(prediction)
            
        return np.array(predictions)

    def _make_prediction(self, x, node):
        """
        Recursive helper to traverse the tree for a single sample x.
        """
        if node.value is not None:
            return node.value
        
        feature_val = x[node.feature_index]
        
        if feature_val <= node.threshold:
            return self._make_prediction(x, node.left)
        else:
            return self._make_prediction(x, node.right)