import math

from Problem import Problem
from Solution import Solution
from ClusteringModel import ClusteringModel, save_clustering_model, load_clustering_model
from RoutingModel import RoutingModel, save_routing_model, load_routing_model
gen_route = False
if gen_route:
    crm = RoutingModel()
    for i in range(18):
        crm.train(Solution(i))
    save_routing_model(crm)
    crm = load_routing_model()

plot_route = False
if plot_route:
    crm = load_routing_model()
    for i in ["p", "gdb", "gda", "lcbx", "lcby", "lcax", "lcay"]:
        crm.node_models[51].attribute_probability_models[i].plot_attribute_probability()

solve_route = False
if solve_route:
    sol = Solution(18)

    for path in sol.paths:
        print(path)
        print(crm.predict(path))
        """
        b, c = None, math.inf
        for p in itertools.permutations(sorted(path[1:-1])):
            sc = crm.predict([0] + list(p) + [0])
            if sc < c:
                c = sc
                b = [0] + list(p) + [0]
                print("best:", b, c)
        """


        gsol = RoutingSolver(sorted(path[1:-1]), crm)
        solution = gsol.solve()
        print("Orig path", path)
        print("Pred path", solution)

solve_route_exh = False
if solve_route_exh:
    crm = load_routing_model()
    sol = Solution(6)
    for path in sol.paths:
        print('----')
        print(path)
        print(crm.predict(path))
        gsol = RoutingSolverExhaustive(sorted(path[1:-1]), crm)
        solution = gsol.solve()
        print("Orig path", path)
        print("Pred path", solution)


plot_cluster = False
if plot_cluster:
    ccm = ClusteringModel()
    for i in range(50):
        ccm.train(Solution(i))
    save_clustering_model(ccm)
    print('Done')
    ccm.node_models[51].attribute_probability_models['ccx'].predictors[0].plot_peak_linear()
    ccm.node_models[51].attribute_probability_models['ccx'].predictors[0].plot_peak_elliptoid()
    ccm.node_models[51].attribute_probability_models['ccx'].predictors[0].plot_peak_exponential()

plot_cluster2 = True
if plot_cluster2:
    ccm = ClusteringModel()
    for i in range(100):
        ccm.train(Solution(i))
    ccm.node_models[51].attribute_probability_models['gccx'].plot()
    ccm.node_models[51].attribute_probability_models['gccx'].plot_smooth()
