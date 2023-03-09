
class BaseSAApproachResult:
    """ Base class for all result classes.
    Specifies the main logic of setting and getting calculated metrics. """

    def get_worst_result(self):
        """ Returns the worst result among all metrics. """
        raise NotImplementedError()

    def get_all_results(self):
        """ Returns all calculated metrics. """
        raise NotImplementedError()
