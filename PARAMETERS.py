import CVRPInstance

# GLOBAL VARIABLES USED BY ALL MODELS AND GENERATION
CVRP_INSTANCE = CVRPInstance.CVRPInstance()

# ROUTING VARIABLES
LOCAL_NEIGHBOURHOOD = 2

# ROUTING VARIABLES
LOCAL_CLUSTER_NEIGHBOURHOOD = 3

WEIGHING_SCHEME = 'time_exp'  # uniform, time_linear, time_squared, time_exp
ALPHA = 0.7  # Exponential smoothing parameter (between 0 and 1)
LAMBDA = 0.0001  # Laplace smoothing parameter (>= 0)