class QDA:
    def fit(self, x, y):
        self.n_features = x.shape[1]
        self.n_classes = len(np.unique(y))
        self.phi = self.cal_phi(y)
        self.mean = self.cal_mean(x, y)
        self.sigma = self.cal_sigma(x, y)
    
    def cal_phi(self, y):
        phi = [] # each phi correspon to each class
        for i in range(self.n_classes):
            phi.append((y==i).mean())
        return np.array(phi)
    
    def cal_mean(self, x, y):
        mean = [] # first dimension correspond to class
        for i in range(self.n_classes):
            tmp_mean = x[(y==i).squeeze()].mean(0)
            mean.append(tmp_mean)
        return np.array(mean)
    
    def cal_sigma(self, x, y):
        y = y.squeeze()
        n = x.shape[1]
        sigma = np.zeros((self.n_classes, n, n))
        for i in range(self.n_classes):
            sigma[i] = ((x[y==i] - self.mean[i]).T.dot(x[y==i] - self.mean[i])) / (len(x[y==i]) - 1)
        return sigma
    
    def decision_function(self, x):
        sigma_inv = np.linalg.pinv(self.sigma)
        decisions = []

        for comb in combinations(range(self.n_classes), 2):
            i, j = comb
            decision = np.log(self.phi[i] / self.phi[j]) + \
                    0.5 * np.log(np.linalg.det(self.sigma[j]) / np.linalg.det(self.sigma[i])) + \
                    -0.5 * ((x - self.mean[i]) @ np.linalg.pinv(self.sigma[i]) * (x - self.mean[i])).sum(1) + \
                    0.5 * ((x - self.mean[j]) @ np.linalg.pinv(self.sigma[j]) * (x - self.mean[j])).sum(1)            
            decisions.append(decision)
            
        return np.array(decisions).T
        
    def generate(self, class_):
        return np.random.multivariate_normal(self.mean[class_], self.sigma[class_])
    
    def predict(self, x):
        # x -> shape=[batch_size, n_features]
        probs = []
        n = x.shape[1]
        sigma_inv = np.linalg.pinv(self.sigma)
        for i in range(self.n_classes):
            if np.linalg.det(self.sigma[i]) != 0:
                tmp = - 0.5 * np.log(np.linalg.det(self.sigma[i]))
            else:
                tmp = 0
            prob =  tmp - 0.5 * ((x - self.mean[i]).dot(sigma_inv[i]) * (x - self.mean[i])).sum(1) + np.log(self.phi[i])
            probs.append(prob)
        return np.array(probs).T
    
    def empirical_rule(self, x):
        # x -> shape=[batch_size, n_features]
        result = []
        for i in range(self.n_classes):
            v, _ = np.linalg.eig(self.sigma[i])
            upper_limit = self.mean[i] + (v * 3)
            lower_limit = self.mean[i] - (v * 3)
            result.append(((lower_limit > x) | (upper_limit < x)).any())
            
        return np.array(result).T