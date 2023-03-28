from typing import Dict, List, Optional, Union
import numpy as np


PCR_PARAMS = {
    'V_sample': 1, 
    'c_1': 1/10, 
    'xi': 1/2, 
    'c_2': 1, 
    'LoD': 100,
}  # c_1 = V_subsample / V_sample


class OneStageGroupTesting:
    r"""One-stage hierarchical group testing."""
    def __init__(
        self,
        ids: List[List[int]],
        viral_loads: List[List[Union[int, float]]],
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
    
    def run_one_stage_group_testing(self, seed: int = 0):
        """Run one-stage hierarchical group testing."""
        res = []
        np.random.seed(seed)
        
        for vl_ in self.viral_loads:
            # append zero(s) if the pool size is larger than the number of samples
            vl = vl_ + [0] * (self.pool_size - len(vl_))
            if run_one_PCR_test(mu=vl, individual=False, params=self.pcr_params):
                res.append(
                    run_one_PCR_test(
                        mu=vl_, 
                        individual=True,
                        params=self.pcr_params
                    )
                )
            else:
                res.append([False] * len(vl_))             
        return res


def run_one_PCR_test(
    mu: List[int],
    individual: bool, 
    params: Dict[str, Union[float, int]], 
) -> List[int]:
    """Perform a single PCR test

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
    V_sample = params['V_sample']
    c_1 = params['c_1']
    xi = params['xi']
    c_2 = params['c_2']
    LoD = params['LoD']
    
    if individual:
        N_templates = np.random.binomial(V_sample * np.array(mu, dtype=int), c_1 * xi * c_2)    
        return list(N_templates >= LoD)

    pool_size = len(mu)
    # copies of RNA in a subsample that is c_1 of the original volume
    N_subsamples = np.random.binomial(V_sample * np.array(mu, dtype=int), c_1 / pool_size)
    N_pre_extraction = np.sum(N_subsamples)
    # copies of extracted RNA in the PCR template
    N_templates = np.random.binomial(N_pre_extraction, xi * c_2)
    return N_templates >= LoD