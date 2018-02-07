

class VariationalMLFM:

    def __init__(self,
                 xkp, gkp,
                 sigmas, gammas,
                 As=None,
                 data_time=None,
                 data_Y=None):

        self._As = np.array(As)
