from attention.problems.nc.state_nc import StateNC

class NC(object):

    NAME = 'nc'

    @staticmethod
    def get_costs(dataset, pi):
        
        return 0, None


    @staticmethod
    def make_state(*args, **kwargs):
        return StateNC.initialize(*args, **kwargs)
