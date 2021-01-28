from typing import List, Tuple, cast
import numpy as np

XY = Tuple[np.ndarray, np.ndarray]
XYList = List[XY]
PartitionedDataset = List[Tuple[XY, XY]]


def shuffle(x: np.ndarray, y: np.ndarray) -> XY:
    """Shuffle x and y."""
    idx = np.random.permutation(len(x))
    return x[idx], y[idx]


def partition(x: np.ndarray, y: np.ndarray, num_partitions: int) -> XYList:
    """Split x and y into a number of partitions."""
    return list(zip(np.array_split(x, num_partitions), np.array_split(y, num_partitions)))


def create_partitions(
        source_dataset: XY,
        num_partitions: int,
) -> XYList:
    """Create partitioned version of a source dataset."""
    x, y = source_dataset
    # x, y = shuffle(x, y)
    xy_partitions = partition(x, y, num_partitions)

    return xy_partitions


def load_data(num_partitions: int, pct_test: float = 0.2) -> PartitionedDataset:
    inputs = np.load('../data/inputs.npy')
    labels = np.load('../data/labels.npy')
    train_inputs = inputs[:-int(len(labels) * pct_test)]
    train_labels = labels[:-int(len(labels) * pct_test)]
    test_inputs = inputs[-int(len(labels) * pct_test):]
    test_labels = labels[-int(len(labels) * pct_test):]

    # train_set = [(x, y) for x, y in zip(train_inputs, train_labels)]
    # test_set = [(x, y) for x, y in zip(test_inputs, test_labels)]

    xy_train_partitions = create_partitions((train_inputs, train_labels), num_partitions)
    xy_test_partitions = create_partitions((test_inputs, test_labels), num_partitions)
    return list(zip(xy_train_partitions, xy_test_partitions))
