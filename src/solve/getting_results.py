import os
import pandas as pd
from fastsdp_tools import add_row_from_dict, get_project_path

def get_results_trivially_solved(self):
    """
    Recuperation of optimization results for trivially solved problems
    """
    dic_benchmark = {
        "network": self.network_name,
        "model": self.name,
        "dataset": self.dataset_name,
        "data_index": self.data_index,
        "label": self.ytrue,
        "label_predicted": self.label_predicted,
        "target": self.ytarget if "Lan" in self.__class__.__name__ else None,
        "epsilon": self.epsilon,
        "status": "trivially_solved",
        "iterations": 0,
        "time": 0.0,
        "pretreatment_time": 0.0,
        "bound_time": self.compute_bounds_time,
        "LAST_LAYER": self.LAST_LAYER,
        "USE_STABLE_ACTIVES": self.use_active_neurons,
        "USE_STABLE_INACTIVES": self.use_inactive_neurons,
        "Nb_stable_inactives": len(self.stable_inactives_neurons),
        "Nb_stable_actives": len(self.stable_actives_neurons),
    }
    if self.__class__.__name__=="TargetedSDP" or self.__class__.__name__=="UntargetedSDP":
        dic_benchmark["CHORDAL_DECOMPOSITION"] = self.CHORDAL_DECOMPOSITION
    print("dic benchmark keys : ", dic_benchmark)
    if self.benchmark_dataframe is None:
        self.benchmark_dataframe = pd.DataFrame(dic_benchmark, index=[0])
    else:
        self.benchmark_dataframe = add_row_from_dict(
            self.benchmark_dataframe, dic_benchmark
        )
    print("\n \n self.benchmark_dataframe   : ", self.benchmark_dataframe)
    path = get_project_path(f"{self.folder_name}/results.csv")
    row_df = pd.DataFrame(dic_benchmark, index=[0])
    if os.path.exists(path):
        existing = pd.read_csv(path)
        existing = existing[existing.get("status", pd.Series(dtype=str)) != "pre-solve"]
        merged = pd.concat([existing, row_df], ignore_index=True)
    else:
        merged = row_df
    merged.to_csv(path, index=False)
