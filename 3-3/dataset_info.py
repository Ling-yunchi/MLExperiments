from torchvision import datasets
from prettytable import PrettyTable


train_dataset = datasets.CIFAR10("data", train=True)
test_dataset = datasets.CIFAR10("data", train=False)

print("Train Dataset size:", len(train_dataset))
print("Test dataset size:", len(test_dataset))


def calc_label_count(dataset):
    class_counts = {}
    for i in range(len(dataset)):
        _, label = dataset[i]
        class_name = dataset.classes[label]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    class_counts = dict(sorted(class_counts.items()))
    return class_counts


# table = PrettyTable(["Name", "Train", "Test"])
# train = calc_label_count(train_dataset)
# test = calc_label_count(test_dataset)
# for k in train_dataset.classes:
#     table.add_row([k, train.get(k, 0), test.get(k, 0)])
#
# print(table)

print(str.join(",", train_dataset.classes))
