import torch
import torch.nn as nn
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report, mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm import tqdm


class Ensemble(nn.Module):
    ''' Use NN to ensemble the predictions from different models for a regression problem '''
    def __init__(self, input_size, hidden_size, output_size):
        super(Ensemble, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size,output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
    

def train_nn_ensemble(train_data, train_y, test_data, test_y, input_size, hidden_size, output_size, num_epochs, learning_rate, batch_size=100):
    ''' Train the NN ensemble '''
    # convert to tensors
    train_data = np.array(train_data)
    train_y = np.array(train_y)
    test_data = np.array(test_data)
    test_y = np.array(test_y)
    train_data = torch.from_numpy(train_data).float()
    train_y = torch.from_numpy(train_y).float()
    test_data = torch.from_numpy(test_data).float()
    test_y = torch.from_numpy(test_y).float()
    # define the model
    model = Ensemble(input_size, hidden_size, output_size)
    # define the loss function
    criterion = nn.MSELoss()
    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # train the model with mini-batches
    for epoch in tqdm(range(num_epochs)):
        permutation = torch.randperm(train_data.size()[0])
        for i in range(0, train_data.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = train_data[indices], train_y[indices]
            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
    # test the model
    with torch.no_grad():
        outputs = model(test_data)
        loss = criterion(outputs, test_y)
        print('Test loss: {:.4f}'.format(loss.item()))
    return model


train = pd.read_csv('data/train_regression.csv')
test = pd.read_csv('data/test_regression.csv')

pd.read_csv('data/train.csv')
scaler = StandardScaler()
scaler.fit(train.iloc[:, 1:])
scaled_train = scaler.transform(train.iloc[:, 1:])
y_test = test['popularity']
y_train = train['popularity']
test.drop("popularity", axis=1, inplace=True)
train.drop("popularity", axis=1, inplace=True)
scaled_test_data = scaler.transform(test)

model_xgb = pickle.load(open('models_path_regression/xgboost_regressor.sav', 'rb'))
model_rf = pickle.load(open('models_path_regression/rf_regressor.sav', 'rb'))
model_ada = pickle.load(open('models_path_regression/adaboost_regressor.sav', 'rb'))
model_linear = pickle.load(open('models_path_regression/linear.sav', 'rb'))
model_ridge = pickle.load(open('models_path_regression/ridgeGLM.sav', 'rb'))
model_glm = pickle.load(open('models_path_regression/glm.sav', 'rb'))
model_knn = pickle.load(open('models_path_regression/knnreg_bestk23.sav', 'rb'))

pred_xgb = model_xgb.predict(train)
pred_rf = model_rf.predict(train)
pred_ada = model_ada.predict(train)
pred_linear = model_linear.predict(scaled_train)
pred_ridge = model_ridge.predict(scaled_train)
pred_glm = model_glm.predict(scaled_train)
pred_knn = model_knn.predict(scaled_train)

pred_xgb_test = model_xgb.predict(test)
pred_rf_test = model_rf.predict(test)
pred_ada_test = model_ada.predict(test)
pred_linear_test = model_linear.predict(scaled_test_data)
pred_ridge_test = model_ridge.predict(scaled_test_data)
pred_glm_test = model_glm.predict(scaled_test_data)
pred_knn_test = model_knn.predict(scaled_test_data)

train_ensemble = np.column_stack((pred_xgb, pred_rf, pred_ada, pred_linear, pred_ridge, pred_glm, pred_knn))
test_ensemble = np.column_stack((pred_xgb_test, pred_rf_test, pred_ada_test, pred_linear_test, pred_ridge_test, pred_glm_test, pred_knn_test))

# scaled training and test set with a new scaler
scaler_ensemble = StandardScaler()
scaler_ensemble.fit(train_ensemble)
train_ensemble = scaler_ensemble.transform(train_ensemble)
test_ensemble = scaler_ensemble.transform(test_ensemble)

# scale the y values
scaler_y = StandardScaler()
scaler_y.fit(y_train.values.reshape(-1, 1))
y_train = scaler_y.transform(y_train.values.reshape(-1, 1))
y_test = scaler_y.transform(y_test.values.reshape(-1, 1))

# save both scalers
pickle.dump(scaler_ensemble, open('models_path_regression/scaler_ensemble.sav', 'wb'))
pickle.dump(scaler_y, open('models_path_regression/scaler_ensemble_y.sav', 'wb'))

if __name__ == '__main__':
    # # train the NN ensemble
    model = train_nn_ensemble(train_ensemble, y_train, test_ensemble, y_test, 7, 5, 1, 50, 0.01, 100)
    # # save the model
    torch.save(model.state_dict(), 'models_path_regression/nn_ensemble.ckpt')
    # # load the model
    model = Ensemble(7, 5, 1)
    model.load_state_dict(torch.load('models_path_regression/nn_ensemble.ckpt'))
    model.eval()
    # predict the popularity
    with torch.no_grad():
        # try predict random sample
        outputs = model(torch.from_numpy(test_ensemble).float())
        print(outputs)
        outputs_train = model(torch.from_numpy(train_ensemble).float())
        criterion = nn.MSELoss()
        loss = criterion(outputs, torch.from_numpy(np.array(y_test)).float().view(-1, 1))
        # transform the dataset back to the original scale
        outputs = scaler_y.inverse_transform(outputs)
        outputs_train = scaler_y.inverse_transform(outputs_train)
        y_test = scaler_y.inverse_transform(y_test)
        y_train = scaler_y.inverse_transform(y_train)
        print('Test loss: {:.4f}'.format(loss.item()))
        print('R2 on test: ', r2_score(y_test, outputs))
        print('MAE test', mean_absolute_error(y_test, outputs))
        print('RMSE test', np.sqrt(mean_squared_error(y_test, outputs)))
        print('MAE train', mean_absolute_error(y_train, outputs_train))
        print('R2 on train: ', r2_score(y_train, outputs_train))


    # train linear regression model with train_ensemble and y_train and test on test_ensemble and y_test
    # train the model
    # from sklearn.linear_model import LinearRegression
    # model = LinearRegression()
    # model.fit(train_ensemble, y_train)
    # # predict the popularity
    # y_pred = model.predict(test_ensemble)
    # print('R2 on test: ', r2_score(y_test, y_pred))
    # print('MAE test', mean_absolute_error(y_test, y_pred))
    # print('RMSE test', np.sqrt(mean_squared_error(y_test, y_pred)))
    # print('MAE train', mean_absolute_error(y_train, model.predict(train_ensemble)))
    # print('R2 on train: ', r2_score(y_train, model.predict(train_ensemble)))
    # print('RMSE train', np.sqrt(mean_squared_error(y_train, model.predict(train_ensemble))))
