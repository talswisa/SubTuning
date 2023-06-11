from src.train import train
from src.models import MODEL_LAYERS


def set_max_epochs(config, n_epochs):
    config.trainer.max_epochs = n_epochs
    config.model.max_epochs = n_epochs


class GreedySubsetSearch:

    def __init__(self, all_layers, epsilon, train_config, start=None, cross_val=False):
        self.all_layers = all_layers
        self.epsilon = epsilon
        self.train_config = train_config
        self.cross_val = cross_val
        self.best_layers, self.best_acc, self.start = self.determine_start(start)

    def train_and_get_acc(self, train_config):
        if self.cross_val:
            accs = []
            for i in range(5):
                train_config.datamodule.cross_val_index = i
                acc = train(train_config)
                accs.append(acc)
            mean_acc = sum(accs) / len(accs)
            return mean_acc
        else:
            acc = train(train_config)
            return acc

    def determine_start(self, start):
        if start == "all":
            self.train_config.model.layers_to_finetune = self.all_layers
            all_acc = self.train_and_get_acc(self.train_config)
            return self.all_layers, all_acc, "all"
        elif start == "non":
            self.train_config.model.layers_to_finetune = []
            non_acc = self.train_and_get_acc(self.train_config)
            return [], non_acc, "non"
        else:
            # running both with all and non layers to determine which is better
            self.train_config.model.layers_to_finetune = self.all_layers
            all_acc = self.train_and_get_acc(self.train_config)
            self.train_config.model.layers_to_finetune = []
            non_acc = self.train_and_get_acc(self.train_config)
            if all_acc > non_acc:
                return self.all_layers, all_acc, "all"
            else:
                return [], non_acc, "non"

    def handle_layer(self, layers, layer):
        if self.start == "non":  # then we add the layer
            if layer not in layers:
                return layers + [layer]
            else:
                # in case we already have the layer, we don't add it
                return None
        else:  # then we remove the layer
            if layer in layers:
                return [l for l in layers if l != layer]
            else:
                # in case we don't have the layer, we don't remove it
                return None

    def search(self):
        layers = [] if self.start == 'non' else self.all_layers
        for _ in range(len(self.all_layers)):
            acc, layers = self.search_iter(layers)
            if acc > self.best_acc:
                self.best_acc, self.best_layers = acc, layers
            elif acc < self.best_acc - self.epsilon:
                break
        return self.best_layers

    def search_iter(self, layers):
        iter_best_acc = 0
        best_layer_to_handle = None
        for layer in self.all_layers:
            layers_to_train = self.handle_layer(layers, layer)
            if layers_to_train is None:
                continue  #in case we already added / removed the layer
            self.train_config.model.layers_to_finetune = layers_to_train
            acc = self.train_and_get_acc(self.train_config)
            if acc > iter_best_acc:
                iter_best_acc, best_layer_to_handle = acc, layer
        layers = self.handle_layer(layers, best_layer_to_handle)
        return iter_best_acc, layers


def find_best_layers(config):
    model = config.model.arch
    all_layers = MODEL_LAYERS[model]
    best_layers = GreedySubsetSearch(all_layers,
                                     config.greedy_eposilon,
                                     config,
                                     start=config.greedy_start,
                                     cross_val=config.greedy_cross_val).search()
    return best_layers
