import CVRPInstance

# GLOBAL VARIABLES USED BY ALL MODELS AND GENERATION
CVRP_INSTANCE = CVRPInstance.CVRPInstance()

# INCREMENTAL LEARNING PARAMETERS
TAU = 130
RHO = 165
OMEGA = 200


# ROUTING VARIABLES
LOCAL_NEIGHBOURHOOD = 2 # Number of stops before and after stop in local path
PEAK_ROUTING = "exp"    #'exp', 'exp2', 'lin', 'ell'

# CLUSTERING VARIABLES
LOCAL_CLUSTER_NEIGHBOURHOOD = 3 # Number of nearest neighbours in stop in local path
PEAK_CLUSTERING = "exp"    #'exp', 'exp2', 'lin', 'ell'

# VARIABLES FOR TRANSITION PROBABILITY MATRIX ()
WEIGHING_SCHEME = 'time_exp'  # uniform, time_linear, time_squared, time_exp
ALPHA = 0.7  # Exponential smoothing parameter (between 0 and 1)
LAMBDA = 0.0001  # Laplace smoothing parameter (>= 0)