# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:ARAVIND SAMY.P
### Register Number:212222230011
```python
class NeuralNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(1, 4)
    self.fc2 = nn.Linear(4, 6)
    self.fc3 = nn.Linear(6, 1)
    self.relu = nn.ReLU()
    self.history = {'loss':[]}
  def forward(self,x):
    x=self.relu(self.fc1(x))
    x=self.relu(self.fc2(x))
    x=self.fc3(x)
    return x




ai_brain=NeuralNet()
criterion=nn.MSELoss()
optimizer=optim.RMSprop(ai_brain.parameters(),lr=0.001)



def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(ai_brain(X_train), y_train)
        loss.backward()
        optimizer.step()

        ai_brain.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

```
## Dataset Information

![image](https://github.com/user-attachments/assets/c438d7c0-5009-4529-9f34-c1b960672654)

## OUTPUT


### Training Loss Vs Iteration Plot

![image](https://github.com/user-attachments/assets/7a2b63ea-f486-4df8-a9bf-3cf3a44df3dd)

### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/09aeb8dc-7e55-4996-98fe-0f6d6c289e7e)
![image](https://github.com/user-attachments/assets/7f28d5ee-bcd7-4ef8-a165-374d07cf050f)


## RESULT

The neural network regression model was successfully trained and evaluated. The model demonstrated strong predictive performance on unseen data, with a low error rate.

