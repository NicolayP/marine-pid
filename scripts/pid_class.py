import numpy as np

class PIDClass(object):
    def __init__(self, kp, ki, kd):
        self._Kp = kp
        self._Ki = ki
        self._Kd = kd
        self._windup = 1.

        self._error_prev = np.zeros(6)
        self._int = np.zeros(6)
    
    def update_pid(self, error, dt):
        self._int += 0.5 * (self._error_prev + error["prop"]) * dt
        self._windup_protection()
        self._error_prev = error["prop"]
        return np.dot(self._Kp, error["prop"]) \
             + np.dot(self._Ki, self._int) \
             + np.dot(self._Kd, error["deriv"]), self._int
    
    def _windup_protection(self):
        """ If windup protection is used we limit the maximum value of the integral term
        https://en.wikipedia.org/wiki/Integral_windup """
        # compute windup
        self._int = np.clip(self._int, -self._windup, self._windup)