EXP_NAME = "cmplx_updated_bart_1000"
RUN_NAME = "cmplx_1_test"


###########################
# ITERATIONS
ITERS = 1000
SEED_ADDL=99
N = [400]
PLOT_ALL=True
###########################

# comp_1
SPLIT_RULES1 =  [
    "pmb.ContinuousSplitRule()", # time
    "pmb.ContinuousSplitRule()", # 
	"pmb.ContinuousSplitRule()", # 
	"pmb.ContinuousSplitRule()", # 
	"pmb.ContinuousSplitRule()", # 
    "pmb.ContinuousSplitRule()", # 
    "pmb.ContinuousSplitRule()", # 
    "pmb.ContinuousSplitRule()", # 
    "pmb.ContinuousSplitRule()", # 
    "pmb.ContinuousSplitRule()", # 
    "pmb.ContinuousSplitRule()", # 
	
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
	# "pgbart": {"num_particles":10}
}


###########################
# Comp2
SPLIT_RULES2 =  [
    "pmb.ContinuousSplitRule()", # time
	"pmb.ContinuousSplitRule()", # 
	"pmb.ContinuousSplitRule()", # 
	"pmb.ContinuousSplitRule()", # 
	"pmb.ContinuousSplitRule()", # 
    "pmb.ContinuousSplitRule()", # 
    "pmb.ContinuousSplitRule()", # 
    "pmb.ContinuousSplitRule()", # 
    "pmb.ContinuousSplitRule()", # 
    "pmb.ContinuousSplitRule()", # 
    "pmb.ContinuousSplitRule()", # 
]
MODEL_DICT2 = {"trees": 50,
	"split_rules": SPLIT_RULES2
}
SAMPLER_DICT2 = {
	"draws": 500,
	"tune": 500,
	"cores": 4,
	"chains": 4,
	"compute_convergence_checks": False
}
