import json

def plot_experiment_stats():
    fopen = open('experiment_stats.json', 'r')
    data = json.load(fopen)
    training_loss = data["training_loss"] 
    dev_loss = data["dev_loss"]

    training_rouge = data["training_rouge"]
    dev_rouge = data["dev_rouge"]

    print('-----')
    print(training_loss)
    print(len(training_loss))

    print('-----')
    print(dev_loss)
    print(len(dev_loss))

    print('-----')
    print(training_rouge)
    print(len(training_rouge))

    print('-----')
    print(dev_rouge)
    print(len(dev_rouge))

def plot_best_dev_output():
    fopen = open('experiment_stats.json', 'r')
    data = json.load(fopen)
    predictions, ans1, ans2 = [], [], []
    
plot_experiment_stats()
