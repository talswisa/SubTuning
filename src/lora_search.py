from src.train import train


class LoraSearch:

    def __init__(self, train_config):
        self.train_config = train_config

    def train_and_get_acc(self, train_config):
        accs = []
        for i in range(5):
            train_config.datamodule.cross_val_index = i
            acc = train(train_config)
            accs.append(acc)
        mean_acc = sum(accs) / len(accs)
        return mean_acc

    def search(self):
        best_rank = 0
        best_acc = 0
        for rank in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
            self.train_config.model.layers_ranks = rank
            acc = self.train_and_get_acc(self.train_config)
            if acc > best_acc:
                best_acc = acc
                best_rank = rank

        return best_rank


def find_best_lora(train_config):
    lora_search = LoraSearch(train_config)
    best_rank = lora_search.search()
    return best_rank
