# The Hadamard utilities live in the signrot package; re-exported here for the fake_quant modules.
from signrot.functional.hadamard import (
    KNOWN_HADAMARD_MATRICES,
    get_hadK,
    matmul_hadU,
    matmul_hadUt,
    random_hadamard_matrix,
    matmul_hadU_cuda,
    matmul_hadUt_cuda,
    apply_exact_had_to_linear,
    is_pow2,
)
