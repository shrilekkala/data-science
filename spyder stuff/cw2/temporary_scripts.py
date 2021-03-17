import pandas as pd
import numpy as np

MLP_Metrics = {}

l1 = [(1, 2), (3, 4), (5, 6)]
a1 = [(3, 1), (2, 4), (6, 5)]
tt1 = "10"

MLP_Metrics["80, 0.01"] = l1, a1, tt1

l2 = [(5, 2), (3, 4), (10, 6)]
a2 = [(5, 1), (2, 4), (20, 5)]
tt2 = "20"

MLP_Metrics["80, 0.0001"] = l2, a2, tt2

def get_last_metrics(metric_list):
    final_loss = list(metric_list[0][-1])
    final_accuracy = list(metric_list[1][-1])
    train_time = str(tt2)
    return final_loss + final_accuracy + [train_time]

g1 = get_last_metrics(MLP_Metrics["80, 0.01"])
g2 = get_last_metrics(MLP_Metrics["80, 0.0001"])

MLP_comparison = pd.DataFrame(columns=['Training Loss', 'Validation Loss',
                                       'Training Accuracy', 'Validation Accuracy',
                                       'Training Time'],
                              index=['40 Epochs, $\eta$ = 0.0001', '40 Epochs, $\eta$ = 0.01'],
                              data = np.array((get_last_metrics(MLP_Metrics["80, 0.0001"]),
                                               get_last_metrics(MLP_Metrics["80, 0.01"]))))