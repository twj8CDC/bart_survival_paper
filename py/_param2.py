EXP_NAME = "cmplx_new"
RUN_NAME = "cmplx_1_400_200_2_test"
RUN_NAME_cmplx3 = "cmplx_3_400"

###########################
# ITERATIONS
ITERS = 200
SEED_ADDL=99
N = [400]
N_cmplx3 = [400, 1000, 4000]
ITERS_cmplx3 = 3
PLOT_ALL=True
###########################

# comp_1
SPLIT_RULES1 =  [
    "pmb.ContinuousSplitRule()", # time
	"pmb.OneHotSplitRule()",
	"pmb.OneHotSplitRule()",
	"pmb.OneHotSplitRule()",
	"pmb.OneHotSplitRule()",
	"pmb.OneHotSplitRule()",
	"pmb.OneHotSplitRule()",
	"pmb.OneHotSplitRule()",
	"pmb.OneHotSplitRule()",
	"pmb.OneHotSplitRule()",
	"pmb.OneHotSplitRule()",
]
MODEL_DICT1 = {"trees": 100,
	"split_rules": SPLIT_RULES1
}
SAMPLER_DICT1 = {
	"draws": 500,
	"tune": 100,
	"cores": 4,
	"chains": 4,
	"compute_convergence_checks": False,
	# "pgbart": {"num_particles":10}
}


###########################
# Comp2
SPLIT_RULES2 =  [
    "pmb.ContinuousSplitRule()", # time
	"pmb.OneHotSplitRule()",
	"pmb.OneHotSplitRule()",
	"pmb.OneHotSplitRule()",
	"pmb.OneHotSplitRule()",
	"pmb.OneHotSplitRule()",
	"pmb.OneHotSplitRule()",
	"pmb.OneHotSplitRule()",
	"pmb.OneHotSplitRule()",
	"pmb.OneHotSplitRule()",
	"pmb.OneHotSplitRule()",
]
MODEL_DICT2 = {"trees": 100,
	"split_rules": SPLIT_RULES2
}
SAMPLER_DICT2 = {
	"draws": 500,
	"tune": 100,
	"cores": 4,
	"chains": 4,
	"compute_convergence_checks": False
}

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
MODEL_DICT3 = {"trees": 100,
	"split_rules": SPLIT_RULES3
}
# SAMPLER_DICT3 = { # old
# 	"draws": 500,
# 	"tune": 100,
# 	"cores": 4,
# 	"chains": 4,
# 	"compute_convergence_checks": False
# }

SAMPLER_DICT3 = { # new
	"draws": 300,
	"tune": 50,
	"cores": 12,
	"chains": 12,
	"compute_convergence_checks": False,
	"pgbart":{
                "num_particles":5,
                "batch":(.1,.1)
            }
}


