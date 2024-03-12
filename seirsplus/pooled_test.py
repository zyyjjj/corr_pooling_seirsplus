from typing import Dict, List, Optional, Union

import numpy as np

PCR_PARAMS = {
    "V_sample": 1,
    # V_subsample/V_sample, fraction of original sample volume used for PCR test
    "c_1": 1 / 10,
    # probability that each RNA copy attaches to the glass fiber plate (indep)
    "xi": 1 / 2,
    "c_2": 1,
    # limit of detection
    "LoD": 174,
}


class OneStageGroupTesting:
    r"""One-stage hierarchical group testing."""

    def __init__(
        self,
        ids: List[List[int]],
        viral_loads: List[List[int]],
        pcr_params: Optional[Dict[str, Union[float, int]]] = PCR_PARAMS,
    ) -> None:
        """Initialize a OneStageGroupTesting object.

        Args:
            ids: List of list of individual IDs in each testing group.
            viral_loads: List of list of viral loads (copies/mL) in each testing group.
            pcr_params: dictionary of PCR test parameters for subsampling and dilution.

        Returns:
            None.
        """
        self.ids = ids
        self.viral_loads = viral_loads
        self.pcr_params = pcr_params
        self.pool_size = len(ids[0])

    def run_one_stage_group_testing(self, seed: Optional[int] = None):
        """Run one-stage hierarchical group testing.

        Args:
            seed: random seed used for one-stage group testing.

        Returns:
            A two-element tuple containing:
            - A list of list of booleans indicating the group testing results.
            - A dictionary storing the sensitivity and test consumption.
        """
        res = []
        if seed is not None:
            np.random.seed(seed)
        num_tests = len(self.viral_loads)

        # sensitivity, test consumption
        for vl_ in self.viral_loads:
            # append zero(s) if the pool size is larger than the number of samples
            vl = vl_ + [0] * (self.pool_size - len(vl_))
            if run_one_PCR_test(mu=vl, individual=False, params=self.pcr_params):
                num_tests += len(vl_)
                res.append(
                    run_one_PCR_test(mu=vl_, individual=True, params=self.pcr_params)
                )
            else:
                res.append([False] * len(vl_))

        num_positives_per_pool = list(
            filter(
                lambda x: x > 0,
                [sum([v > 0 for v in vl]) for vl in self.viral_loads],
            )
        )
        # number of positives expected to be identifiable by individual test
        num_identifiable_positives_per_pool = list(
            filter(
                lambda x: x > 0,
                [sum([v*self.pcr_params["c_1"]*self.pcr_params["xi"]*self.pcr_params["c_2"]*self.pcr_params["V_sample"]>self.pcr_params["LoD"] for v in vl]) for vl in self.viral_loads],
            )
        )
        num_positives = sum(num_positives_per_pool)
        num_identified = sum([sum(pool_res) for pool_res in res])
        if num_positives:
            sensitivity = num_identified / num_positives
        else:
            sensitivity = float("nan")
        return res, {
            "sensitivity": sensitivity,
            "num_tests": num_tests,
            "num_positives_per_positive_pool": num_positives_per_pool,
            "num_identifiable_positives_per_positive_pool": num_identifiable_positives_per_pool,
            "num_positives": num_positives,
            "num_identified": num_identified,
        }


def run_one_PCR_test(
    mu: List[int],
    individual: bool,
    params: Dict[str, Union[float, int]],
) -> List[int]:
    """Perform a single PCR test.

    Args:
        mu: viral loads (copies/mL) in the samples.
        individual: boolean, whether the PCR test is a pooled test or an individual test.
        params: dictionary of PCR test parameters for subsampling and dilution.
        LoD: limit of detection (LoD) for the PCR test, the minimum number of
            copies of viral RNA in the sample needed for the sample to be
            considered positive.

    Returns:
        Boolean indicating whether the PCR test returns positive or negative.
    """
    V_sample = params["V_sample"]
    c_1 = params["c_1"]
    xi = params["xi"]
    c_2 = params["c_2"]
    LoD = params["LoD"]
    dilute = params.get("dilute", "average") # one of {"average", "sum", "max", "max_average"}

    if individual:
        if not dilute.startswith("constant"):
            N_templates = np.random.binomial(V_sample * np.array(mu), c_1 * xi * c_2)
            return list(N_templates >= LoD)
        else:
            constant_sensitivity = float(dilute.split("_")[1])
            return [np.random.binomial(1, constant_sensitivity * (v > 0)) == 1 for v in mu]

    # number of copies of RNA in a subsample is c_1 of the original volume
    # then apply different dilution schemes
    if dilute == "average":
        N_pre_extraction = np.sum(np.random.binomial(V_sample * np.array(mu), c_1 / len(mu)))
    elif dilute == "sum":
        N_pre_extraction = np.sum(np.random.binomial(V_sample * np.array(mu), c_1))
    elif dilute == "max":
        N_pre_extraction = np.random.binomial(V_sample * np.max(mu), c_1)
    elif dilute == "max_average":
        N_pre_extraction = np.random.binomial(V_sample * np.max(mu), c_1 / len(mu))
    elif dilute.startswith("constant"):
        constant_sensitivity = float(dilute.split("_")[1])
        return np.random.binomial(1, constant_sensitivity * (np.sum(mu) > 0)) == 1
    else:
        raise RuntimeError(f"Unrecognized dilution scheme {dilute}!")

    # copies of extracted RNA in the PCR template
    N_templates = np.random.binomial(N_pre_extraction, xi * c_2)
    
    return N_templates >= LoD