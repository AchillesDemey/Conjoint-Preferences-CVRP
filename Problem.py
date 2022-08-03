from PARAMETERS import CVRP_INSTANCE

class Problem:

    def __init__(self, problem_id):
        self.problem_id = problem_id
        self.stops = CVRP_INSTANCE.stops[problem_id]
        self.demands = CVRP_INSTANCE.demands[problem_id]
        self.vehicles = CVRP_INSTANCE.n_vehicles[problem_id]
        self.weekday = CVRP_INSTANCE.weekday[problem_id]
        self.capacity = CVRP_INSTANCE.capacities[problem_id]

    def get_problem_data(self):
        record = {
            'problem_id': self.problem_id,
            'weekday': self.weekday,
            'nb_vehicles': self.vehicles,
            'capacity': self.capacity,
            'demands': self.demands,
            'stops': [1 if stop in self.stops else 0 for stop in range(CVRP_INSTANCE.nb_nodes)],
        }
        return record
