import torch

def get_cifar100_superclass_mapping():
    super_classes = [
        [4, 30, 55, 72, 95],        # aquatic mammals
        [1, 32, 67, 73, 91],        # fish
        [54, 62, 70, 82, 92],       # flowers
        [9, 10, 16, 28, 61],        # food containers
        [0, 51, 53, 57, 83],        # fruit and vegetables
        [22, 39, 40, 86, 87],       # household electrical devices
        [5, 20, 25, 84, 94],        # household furniture
        [6, 7, 14, 18, 24],         # insects
        [3, 42, 43, 88, 97],        # large carnivores
        [12, 17, 37, 68, 76],       # large man-made outdoor things
        [23, 33, 49, 60, 71],       # large natural outdoor scenes
        [15, 19, 21, 31, 38],       # large omnivores and herbivores
        [34, 63, 64, 66, 75],       # medium-sized mammals
        [26, 45, 77, 79, 99],       # non-insect invertebrates
        [2, 11, 35, 46, 98],        # people
        [27, 29, 44, 78, 93],       # reptiles
        [36, 50, 65, 74, 80],       # small mammals
        [47, 52, 56, 59, 96],       # trees
        [8, 13, 48, 58, 90],        # vehicles 1
        [41, 69, 81, 85, 89],       # vehicles 2
    ]

    mapping = {}
    for super_idx, fine_list in enumerate(super_classes):
        for fine in fine_list:
            mapping[fine] = super_idx

    return mapping

def acc_top1(output, target):
    _, pred = output.topk(1, dim=1, largest=True, sorted=True)
    pred = pred.squeeze(1)
    correct = pred.eq(target).float().sum()
    return correct

def super_class_accuracy(output, target, class_to_super, k=5):
    _, pred = output.topk(k, dim=1, largest=True, sorted=True)

    target_super = torch.tensor(
        [class_to_super[t.item()] for t in target],
        device=target.device
    )

    pred_super = torch.tensor(
        [[class_to_super[p.item()] for p in row] for row in pred],
        device=target.device
    )

    match = pred_super.eq(target_super.unsqueeze(1))
    score = match.float().sum(dim=1) / k

    return score.sum()

def evaluate(model, loader, criterion, device):
    model.eval()

    class_to_super = get_cifar100_superclass_mapping()

    running_loss = 0.0
    total = 0

    top1 = 0.0
    super_class = 0.0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            total += labels.size(0)

            top1 += acc_top1(outputs, labels).item()

            super_class += super_class_accuracy(
                outputs, labels, class_to_super
            ).item()

    epoch_loss = running_loss / len(loader)
    top1_acc = top1 / total
    super_class_acc = super_class / total

    return epoch_loss, top1_acc, super_class_acc