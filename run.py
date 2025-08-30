import argparse
import torch
import torch.nn as nn

from data_preprocessing import train_data, train_labels, preprocess_sentence
from data_loader import get_dataloader, word_to_ix, custom_collate_fn, test_collate_fn
from model import WordWindowClassifier

parser = argparse.ArgumentParser(description="Word Window Classifier")
parser.add_argument("--mode", 
                    type=str, required=True, 
                    choices=["test"],
                    help="Please type \"test\"")
args = parser.parse_args()

# Define a loss function, which computes to binary cross entropy loss
def loss_function(batch_outputs, batch_labels, batch_lengths):
    # Calculate the loss for the whole batch
    bceloss = nn.BCELoss()
    loss = bceloss(batch_outputs, batch_labels.float())

    # Rescale the loss. Remember that we have used lengths to store the
    # number of words in each training example
    loss = loss / batch_lengths.sum().float()

    return loss

# Function that will be called in every epoch
def train_epoch(loss_function, optimizer, model, loader):
  # Keep track of the total loss for the batch
  total_loss = 0
  for batch_inputs, batch_labels, batch_lengths in loader:
    # Clear the gradients
    optimizer.zero_grad()
    # Run a forward pass
    outputs = model.forward(batch_inputs)
    # Compute the batch loss
    loss = loss_function(outputs, batch_labels, batch_lengths)
    # Calculate the gradients
    loss.backward()
    # Update the parameteres
    optimizer.step()
    total_loss += loss.item()

  return total_loss

# Function containing our main training loop
def train(loss_function, optimizer, model, loader, num_epochs=10000):
  # Iterate through each epoch and call our train_epoch function
  i = 0
  for epoch in range(num_epochs):
    epoch_loss = train_epoch(loss_function, optimizer, model, loader)
    if epoch % 100 == 0:
       print(f'epoch {i+1}: {epoch_loss}')
       i += 1
  print("\n")

if __name__ == "__main__":
    # Instantiate a DataLoader
    loader = get_dataloader(data=list(zip(train_data(), train_labels())),
                            batch_size=2,
                            shuffle=True,
                            window_size=2,
                            collate_fn=custom_collate_fn)
    # Initialize a model
    # It is useful to put all the model hyperparameters in a dictionary
    model_hyperparameters = {
        "batch_size": 4,
        "window_size": 2,
        "embed_dim": 25,
        "hidden_dim": 25,
        "freeze_embeddings": False,
    }

    vocab_size = len(word_to_ix())
    model = WordWindowClassifier(model_hyperparameters, vocab_size)

    # Define an optimizer
    learning_rate = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Train the model
    num_epochs = 1000
    train(loss_function, optimizer, model, loader, num_epochs=num_epochs)

    # Test the model
    if args.mode == "test":
        while True:
            print("영어 문장을 입력하세요. 종료하려면 \"q\"를 입력하세요.")
            test_corpus = input()
            if test_corpus == "q":
                print("종료합니다. . .")
                break
            test_sentence = [preprocess_sentence(test_corpus)]

            test_loader = get_dataloader(test_sentence,
                                         batch_size=1,
                                         shuffle=False,
                                         window_size=2,
                                         collate_fn = test_collate_fn)
            for test_instance in test_loader:
              outputs = model.forward(test_instance)
              sentence = list(test_sentence)[0]
              # 확률 분포
              for i in range(len(sentence)):
                 print(f'{sentence[i]}: {outputs[0][i].item()}')
              print("\n", f'위치를 의미하는 단어는 {sentence[int(outputs.argmax().item())]}입니다.')