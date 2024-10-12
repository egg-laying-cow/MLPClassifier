# Thm th vin
import numpy as np

# Hm sigmoid
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)
# Lp neural network
class MLPClassifier:
    def __init__(self, layers, alpha=0.1):
# M hnh layer v d [2,2,1]
        self.layers = layers
        # H sè learning rate
        self.alpha = alpha
        # Tham sè W, b
        self.W = []
        self.b = []

 # Khi to cc tham sè méi layer
        for i in range(0, len(layers)-1):
            w_ = np.random.randn(layers[i], layers[i+1])
            b_ = np.zeros((layers[i+1], 1))
            self.W.append(w_/layers[i])
            self.b.append(b_)

 # Train m hnh vi dú liu
    def fit_partial(self, x, y):
        A = [x]
        # qu trnh feedforward
        out = A[-1]
        for i in range(0, len(self.layers)- 1):
            out = sigmoid(np.dot(out, self.W[i]) + (self.b[i].T))
            A.append(out)
        # qu trnh backpropagation
        y = y.reshape(-1, 1)
        dA = [-(y/A[-1]- (1-y)/(1-A[-1]))]
        dW = []
        db = []
        for i in reversed(range(0, len(self.layers)-1)):
            dw_ = np.dot((A[i]).T, dA[-1] * sigmoid_derivative(A[i+1]))
            db_ = (np.sum(dA[-1] * sigmoid_derivative(A[i+1]), 0)).reshape(-1,1)
            dA_ = np.dot(dA[-1] * sigmoid_derivative(A[i+1]), self.W[i].T)
            dW.append(dw_)
            db.append(db_)
            dA.append(dA_)
 # o
        dW = dW[::-1]
        db = db[::-1]
 # Gradient descent
        for i in range(0, len(self.layers)-1):
            self.W[i] = self.W[i]- self.alpha * dW[i]
            self.b[i] = self.b[i]- self.alpha * db[i]

    def fit(self, X, y, epochs=20, verbose=10):
        for epoch in range(0, epochs):
            self.fit_partial(X, y)
            if epoch % verbose == 0:
                loss = self.calculate_loss(X, y)
                print("Epoch {}, loss {}".format(epoch, loss))
 # D on
    def predict(self, X):
        for i in range(0, len(self.layers)- 1):
            X = sigmoid(np.dot(X, self.W[i]) + (self.b[i].T))
        return X
 # Tnh loss function
    def calculate_loss(self, X, y):
        y_predict = self.predict(X)
    #return np.sum((y_predict-y)**2)/2
        return-(np.sum(y*np.log(y_predict) + (1-y)*np.log(1-y_predict)))
    
def main():
    # Du lieu dau vao
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])

    # Khoi tao mo hinh
    model = MLPClassifier([2, 2, 1], 0.1)

    # Huan luyen mo hinh
    model.fit(X, y, 100, 10)

    # Du doan
    y_pred = model.predict(X)
    print(y_pred)

if __name__ == "__main__":
    main()