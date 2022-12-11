# Network Definition
import torch
import torch.nn as nn

class PrimeNumberPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PrimeNumberPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        out = self.sigmoid(out)
        return out

input_size = 5  # the number of properties (sum of digits, number itself, parity, number of digits, sum of all digits)
hidden_size = 128  # size of the hidden layer
output_size = 1  # output is a single value (probability of being prime)

model = PrimeNumberPredictor(input_size, hidden_size, output_size)


# define a function to check if a number is prime
def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n ** 0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

# initialize empty lists to hold the data
numbers = []
sum_of_digits = []
parity = []
num_digits = []
sum_all_digits = []
labels = []  # 0 for not prime, 1 for prime

# iterate over the first 100,000 numbers
for i in range(1, 1000000):
    # determine if the number is prime or not
    prime = is_prime(i)

    # calculate the sum of the digits
    digits = [int(d) for d in str(i)]
    digit_sum = sum(digits)

    # calculate the parity
    parity_val = 0 if i % 2 == 0 else 1

    # calculate the number of digits
    num_digits_val = len(digits)

    # calculate the sum of all digits
    sum_all_digits_val = sum([d * (10 ** i) for i, d in enumerate(digits[::-1])])

    # append the data to the lists
    numbers.append(i)
    sum_of_digits.append(digit_sum)
    parity.append(parity_val)
    num_digits.append(num_digits_val)
    sum_all_digits.append(sum_all_digits_val)
    labels.append(int(prime))

# combine the data into a single dataset
data = list(zip(numbers, sum_of_digits, parity, num_digits, sum_all_digits, labels))

print("Size of training data:", len(data))
print("Sample data:")
for i in range(2):
  print(data[i])


from sklearn.model_selection import train_test_split
# split the data into training and validation sets
train_data, val_data = train_test_split(data, test_size=0.2)
# create tensors from the data
train_x = torch.tensor([x[:-1] for x in train_data], dtype=torch.float)
train_y = torch.tensor([x[-1] for x in train_data], dtype=torch.float)
val_x = torch.tensor([x[:-1] for x in val_data], dtype=torch.float)
val_y = torch.tensor([x[-1] for x in val_data], dtype=torch.float)

# define the optimizer and loss
optimizer = torch.optim.Adam(model.parameters())
loss_fn = nn.BCELoss()

# train the model for a certain number of epochs
num_epochs = 1000
for epoch in range(num_epochs):
    # feed the training data to the model
    train_output = model(train_x)

    # compute the loss on the training data
    train_loss = loss_fn(train_output, train_y.unsqueeze(1))

    # zero the gradients before backpropagation
    optimizer.zero_grad()

    # backpropagate the loss
    train_loss.backward()

    # update the model's parameters
    optimizer.step()

    # feed the validation data to the model
    val_output = model(val_x)

    # compute the loss on the validation data
    val_loss = loss_fn(val_output, val_y.unsqueeze(1))

    # print the loss on the validation data
    print(f'epoch: {epoch+1}, val_loss: {val_loss.item()}')

# initialize empty lists to hold the data
numbers = []
sum_of_digits = []
parity = []
num_digits = []
sum_all_digits = []

i = 7

# calculate the sum of the digits
digits = [int(d) for d in str(i)]
digit_sum = sum(digits)

# calculate the parity
parity_val = 0 if i % 2 == 0 else 1

# calculate the number of digits
num_digits_val = len(digits)

# calculate the sum of all digits
sum_all_digits_val = sum([d * (10 ** i) for i, d in enumerate(digits[::-1])])

# append the data to the lists
numbers.append(i)
sum_of_digits.append(digit_sum)
parity.append(parity_val)
num_digits.append(num_digits_val)
sum_all_digits.append(sum_all_digits_val)

# combine the data into a single dataset
eval = list(zip(numbers, sum_of_digits, parity, num_digits, sum_all_digits, labels))



print (eval)


evalData = (i, digit_sum, parity_val, num_digits_val, sum_all_digits_val)
evalData = torch.Tensor(evalData)

output = model(evalData)
print(f'Output: {output.item()}')
