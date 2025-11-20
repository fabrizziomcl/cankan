import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth=2):
        super(SimpleMLP, self).__init__()
        self.to(torch.float64) # Importante para precisión científica
        
        layers = []
        # Capa entrada
        layers.append(nn.Linear(input_dim, hidden_dim, dtype=torch.float64))
        layers.append(nn.Tanh()) # Tanh es mejor para aproximar funciones suaves que ReLU
        
        # Capas ocultas
        for _ in range(depth - 2): # depth incluye entrada y salida
            layers.append(nn.Linear(hidden_dim, hidden_dim, dtype=torch.float64))
            layers.append(nn.Tanh())
            
        # Capa salida
        layers.append(nn.Linear(hidden_dim, output_dim, dtype=torch.float64))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def fit(self, dataset, opt="Adam", steps=100, lr=0.01, lamb=0.0):
        optimizer = None
        if opt == "Adam":
            optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        elif opt == "SGD":
            optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        elif opt == "LBFGS":
            optimizer = optim.LBFGS(self.parameters(), lr=1.0, history_size=10, line_search_fn="strong_wolfe")
        
        criterion = nn.MSELoss()
        results = {'train_loss': [], 'test_loss': []}
        
        # Loop de entrenamiento
        for _ in range(steps):
            def closure():
                optimizer.zero_grad()
                pred = self(dataset['train_input'])
                loss = criterion(pred, dataset['train_label'])
                loss.backward()
                return loss

            if opt == "LBFGS":
                optimizer.step(closure)
                with torch.no_grad():
                    train_loss = criterion(self(dataset['train_input']), dataset['train_label'])
            else:
                train_loss = closure()
                optimizer.step()

            # Evaluar Test
            with torch.no_grad():
                test_loss = criterion(self(dataset['test_input']), dataset['test_label'])

            results['train_loss'].append(train_loss.item())
            results['test_loss'].append(test_loss.item())

        return results