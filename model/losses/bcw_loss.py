import numpy as np


class BalancedClassWeight:
    """
        Balanced Class Weight
        If ratio of classes of image is [n1:n2:n3...:nm], m is class number, and the min ratio is 1
        So the weight is [r_sum - n1, r_sum - n2, r_sum - n3, ..., r_sum - nm]
        where r_sum = n1 + n2 + ... +nm
    """

    def __init__(self, class_num: int = 2) -> None:
        self._class_num = class_num
    
    def get_weight(self, label: np.ndarray) -> np.ndarray:
        label[label > 0] = 1
        weight = np.zeros((label.shape[0], label.shape[1], self._class_num))
        class_weight = np.zeros((self._class_num, 1))
        for idx in range(self._class_num):
            idx_num = np.count_nonzero(label == idx)
            class_weight[idx, 0] = idx_num
        t_matrix = class_weight[class_weight != 0]
        min_num = np.amin(t_matrix)
        # min_num = np.amin(class_weight)
        class_weight = class_weight * 1.0 / min_num
        class_weight = np.sum(class_weight) - class_weight
        for idx in range(self._class_num):
            weight[:, :, idx][label == idx] = class_weight[idx, 0]
        return weight
