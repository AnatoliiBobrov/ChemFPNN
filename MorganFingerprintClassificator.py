import csv
import json
import time
import itertools
from rdkit import Chem #rdkit==2023.9.5
from rdkit.Chem import AllChem
import numpy as np #numpy==1.26.4
import torch #torch==2.4.1
import torch.nn as nn
from sklearn.model_selection import train_test_split #scikit-learn==1.4.2

DATASETS = [["freesolv.csv", "smiles", "expt", "calc"], ["esol.csv", "smiles", "measured log solubility in mols per litre", "ESOL predicted log solubility in mols per litre"],]
DATASET_NUM = 1 #0 freesolv 1 esol
VECTOR_SIZE = [2 ** i for i in range(4, 12)]
FP_RADIUS = [1, ]#FP_RADIUS = list(range(1,7))
HIDDEN_SIZE = [10 * i for i in range(1, 11)]
LEARNING_RATE = [0.0025, 0.005, 0.01, 0.025, 0.05]
EPOCHS = 100

start = 0
best_index = 0
best_loss = None
params = None
try:
    with open('saved_success.json', 'r') as json_file:
        data = json.load(json_file)
        if data["EPOCHS"] == EPOCHS:
            start = data["start"]
            best_index = data["best_index"]
            best_loss = data["best_loss"]
            params = data["params"]
            print (f"Loaded last success {str(data)}")
        else:
            print ("Loaded data is not for this EPOCHS. Start from the beginning")
except FileNotFoundError:
    pass
    
def set_zero_seed():
    np.random.seed(0)
    torch.manual_seed(0)

def smiles_to_fp(smiles, radius=4, n_bits=64):
    mol = Chem.MolFromSmiles(smiles)
    #mol = Chem.AddHs(mol)
    finger_print = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(finger_print)

def read_dataset(filename, smiles_pos, exp_pos, comp_pos):
    smiles_ = []
    experimental_values_ = []
    computed_values_ = []
    with open(filename) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        for row in csv_reader:
            smiles_.append(row[smiles_pos])
            experimental_values_.append(float(row[exp_pos]))
            computed_values_.append(float(row[comp_pos]))
    return smiles_, experimental_values_, computed_values_

set_zero_seed()
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST, _, computed_energy = train_test_split(*read_dataset(*DATASETS[DATASET_NUM]), test_size=0.2)
Y_TRAIN = [torch.FloatTensor([i]) for i in Y_TRAIN]
TRAIN_SET_LEN = len(Y_TRAIN)
y_test_ = torch.FloatTensor(Y_TEST)
computed_energy = torch.FloatTensor(computed_energy)
Y_TEST = [torch.FloatTensor([i]) for i in Y_TEST]
TEST_LEN_SET = len(Y_TEST)

crit = nn.MSELoss()
loss = crit(y_test_, computed_energy)
print(f"Computation Loss: {loss:.4f}")

class ChemNet(nn.Module):
    def __init__(self, input_size=64, hidden_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        
    def forward(self, input):
        output = torch.relu(self.fc1(input))#relu
        output = self.fc2(output)
        return output

def train_and_test(fp_radius, vector_size, hidden_size, lr):
    x_train = [torch.tensor(smiles_to_fp(smile, radius=fp_radius, n_bits=vector_size), dtype=torch.float) for smile in X_TRAIN]
    x_test = [torch.tensor(smiles_to_fp(smile, radius=fp_radius, n_bits=vector_size), dtype=torch.float) for smile in X_TEST]
    set_zero_seed()
    model = ChemNet(vector_size, hidden_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(EPOCHS):
        for i in range(TRAIN_SET_LEN):
            optimizer.zero_grad()
            outputs = model(x_train[i])
            loss = criterion(outputs, Y_TRAIN[i])
            loss.backward()
            optimizer.step()
    
    test_loss = 0
    with torch.no_grad():
        for i in range(TEST_LEN_SET):
            preds = model(x_test[i])
            test_loss += criterion(preds, Y_TEST[i]).item()
    return test_loss/TEST_LEN_SET

query = list(itertools.product(FP_RADIUS, VECTOR_SIZE, HIDDEN_SIZE, LEARNING_RATE))
length = len(query)
print(f"Amoung of hyperparameters' combinations {length}")

def save_success(start, best_index, best_loss, params):
     json_data = {'start': start, 'best_index': best_index, 'best_loss': best_loss, 'params': params, 'EPOCHS': EPOCHS}
     with open('saved_success.json', 'w') as json_file:
        json.dump(json_data, json_file)

#loss = train_and_test(1, 512, 100, 0.0025)
#print(f"Loss: {loss:.4f}")
if best_loss == None:
    best_loss = train_and_test(*query[0])
    params = str(query[0])

while start < length:
    begin = time.time()
    loss = train_and_test(*query[start])
    if loss < best_loss:
        best_loss = loss
        best_index = start
        params = str(query[start])
        print (f"New best loss {best_loss:.4f} with params {params}, index {best_index}")
    if start % 10 == 0:
        end = time.time()
        print(f"Excecution time is {end-begin:.2f} seconds, iteration {start}")
        save_success(start+1, best_index, best_loss, params)
    start += 1
save_success(start, best_index, best_loss, params)
