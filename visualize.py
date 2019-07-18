import matplotlib.pyplot as plt

def plot(metrics, title='Model Accuracy', xlabel='epoch', ylabel='accuracy'):
    legend = []
    for metric in metrics:
        legend.append(metric)
        plt.plot(metrics[metric])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(legend, loc='lower right')
    plt.show()

def visualize(results):
    for experiment in results:
        if experiment == 'train':
            continue

        metrics = {}
        for model in results[experiment]:
            for eval_type in results[experiment][model]:
                metrics[ model + '_' + eval_type ] = results[experiment][model][eval_type]['acc']
        plot(metrics, title=experiment, xlabel='epoch', ylabel='accuracy')
    