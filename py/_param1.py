EXP_NAME = "simple_test_cov"
RUN_NAME = "simple_1"
# RUN_NAME_cmplx3 = "cmplx_3_400"

###########################
# ITERATIONS
ITERS = 100
SEED_ADDL=99
N = [400]
PLOT_ALL=True
###########################

# comp_1
SPLIT_RULES1 =  [
    "pmb.ContinuousSplitRule()", # time
	"pmb.ContinuousSplitRule()"
]
MODEL_DICT1 = {"trees": 50,
	"split_rules": SPLIT_RULES1
}
SAMPLER_DICT1 = {
	"draws": 500,
	"tune": 500,
	"cores": 4,
	"chains": 4,
	"compute_convergence_checks": False,
    "k":1.5
}


###########################
# Comp2
SPLIT_RULES2 =  [
    "pmb.ContinuousSplitRule()", # time
    "pmb.ContinuousSplitRule()" # time
]
MODEL_DICT2 = {"trees": 50,
	"split_rules": SPLIT_RULES2
}
SAMPLER_DICT2 = {
	"draws": 1000,
	"tune": 1000,
	"cores": 4,
	"chains": 4,
	"compute_convergence_checks": False,
    "k":2,
    "update_fraction":.05,
    "step_size":.3,
    "thin":10
}

