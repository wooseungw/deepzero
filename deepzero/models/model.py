class SimpleCNN:
    def __init__(self, input_dim, hidden_dim, num_classes, dropout):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout = dropout
        self.model = self.build_model()

    def build_model(self):
        # Define the architecture of the CNN model
        model = {
            'conv1': {'filters': 32, 'kernel_size': 3, 'activation': 'relu'},
            'pool1': {'pool_size': 2},
            'conv2': {'filters': 64, 'kernel_size': 3, 'activation': 'relu'},
            'pool2': {'pool_size': 2},
            'fc1': {'units': self.hidden_dim, 'activation': 'relu'},
            'dropout': {'rate': self.dropout},
            'output': {'units': self.num_classes, 'activation': 'softmax'}
        }
        return model

    def forward(self, x):
        # Implement forward propagation logic
        pass

class SimpleRNN:
    def __init__(self, input_dim, hidden_dim, num_classes, dropout):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout = dropout
        self.model = self.build_model()

    def build_model(self):
        # Define the architecture of the RNN model
        model = {
            'rnn': {'units': self.hidden_dim, 'activation': 'tanh'},
            'dropout': {'rate': self.dropout},
            'output': {'units': self.num_classes, 'activation': 'softmax'}
        }
        return model

    def forward(self, x):
        # Implement forward propagation logic
        pass