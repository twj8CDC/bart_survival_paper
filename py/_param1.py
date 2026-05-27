EXP_NAME = "simple1"
# RUN_NAME = "simple_12_200_iters_400_n"
# RUN_NAME = "small_particles_100iter"
RUN_NAME = "small_particles_1000iter"

###########################
# ITERATIONS
ITERS = 2
SEED_ADDL=13
# N  = [200, 400]
N  = [400]

###########################
# Cond 1
SPLIT_RULES1 =  [
	"pmb.ContinuousSplitRule()", 
]
MODEL_DICT1 = {"trees": 50,
	"split_rules": SPLIT_RULES1
}
SAMPLER_DICT1 = {
	"draws": 300,
	"tune": 50,
	"cores": 4,
	"chains": 4,
	"compute_convergence_checks": False,
	# "pgbart":{
    #             "num_particles":5,
    #             "batch":(.1,.1)
    #         }
}


###########################
# Cond 2
SPLIT_RULES2 =  [
	"pmb.ContinuousSplitRule()", 
	# "pmb.OneHotSplitRule()"	
	"pmb.ContinuousSplitRule()"
]
MODEL_DICT2 = {"trees": 100,
	"split_rules": SPLIT_RULES2,
	# "split_prior":[10,10]
}
SAMPLER_DICT2 = {
	"draws": 300,
	"tune": 50,
	"cores": 4,
	"chains": 4,
	"compute_convergence_checks": False,
	"pgbart":{
                "num_particles":5,
                "batch":(.1,.1)
            }
}
PLOT_ALL=True