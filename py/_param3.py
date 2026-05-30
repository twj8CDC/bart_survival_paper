EXP_NAME = "cmplx_3_updated_bart_1000"
RUN_NAME = "cmplx_3_test"


###########################
# ITERATIONS
ITERS = 1000
SEED_ADDL=99
N_cmplx3 = [400, 1000, 4000]
ITERS_cmplx3 = 3
PLOT_ALL=True
###########################

# comp 3
SPLIT_RULES3 =  [
    "pmb.ContinuousSplitRule()", # time
    "pmb.ContinuousSplitRule()", 
 	"pmb.ContinuousSplitRule()",
  	"pmb.ContinuousSplitRule()",
  	"pmb.ContinuousSplitRule()",
	"pmb.ContinuousSplitRule()",
	"pmb.ContinuousSplitRule()",
	"pmb.ContinuousSplitRule()",
	"pmb.ContinuousSplitRule()",
 	"pmb.ContinuousSplitRule()",
  	"pmb.ContinuousSplitRule()"
]
MODEL_DICT3 = {"trees": 50,
	"split_rules": SPLIT_RULES3
}

SAMPLER_DICT3 = { # new
	"draws": 500,
	"tune": 500,
	"cores": 4,
	"chains": 4,
	"compute_convergence_checks": False
}


