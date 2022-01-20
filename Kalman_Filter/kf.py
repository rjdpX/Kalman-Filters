import numpy as np
import matplotlib as plt

# offsets of each variable in the state vector
iX = 0
iV = 1
NUMVARS = iV + 1 # Number of variables in the Kalman filter state vector

class KF:
    
    def __init__(self,  inx : float, 
                        inv : float,
                        accl_variance : float) -> None:
        
        # Mean of state GRV
        self._x = np.zeros(NUMVARS)
        self._x[iX] = inx
        self._x[iV] = inv
        # self._x = np.array([inx, inv])
        self._accl_variance = accl_variance
        
        # Covariance of state GRV
        # .eye creates a 2X2 indentity matrix
        self._P = np.eye(NUMVARS)
        # self._P = np.eye(2)

    def predict(self, dt: float) -> None:
        # x = F  x
        # P = F  P  Ft + G  a  Gt
        F = np.eye(NUMVARS)
        F[iX, iV] = dt # X is dt times iV // distance = speed*time
        # F = np.array([[1, dt], [0, 1]])
        # new_x = F * self._x # Element-by-element wise multiplication (broadcasting)
        new_x = F.dot(self._x) # For matrix-matrix product

        G = np.zeros((2, 1))

        # G = np.array([0.5*(dt**2), dt]).reshape(2, 1) // Below we have a better method
        G[iX] = 0.5 * dt ** 2 # Assigning the value to the [1st row] of the G vector
        G[iV] = dt # Assigning the value to the [2nd row] of the G vector
        new_P = F.dot(self._P).dot(F.T) + G.dot(G.T) * self._accl_variance

        self._x = new_x
        self._P = new_P
    
    def update(self, meas_value: float, meas_variance: float):
        
        # y = z - H x       // fetching Innovation
        # S = H P Ht + R    // fetching Innovation covariance 
        # K = P Ht S^-1     // fetching the Kalman gain
        # x = x + K y       // fetching new x
        # P = (I - K H) * P // fetching new covariance value

        z = np.array([meas_value])
        R = np.array([meas_variance])

        H = np.zeros((1, NUMVARS))
        H[0, iX] = 1
        # H = np.array([1, 0]).reshape(1, 2)

        y = z - H.dot(self._x)
        S = H.dot(self._P).dot(H.T) + R

        K = self._P.dot(H.T).dot(np.linalg.inv(S))

        new_x = self._x + K.dot(y)
        new_P = (np.eye(2) - K.dot(H)).dot(self._P)

        self._x = new_x
        self._P = new_P
    
    @property
    def cov(self) -> np.array:
        return self._P
    
    @property
    def mean(self) -> np.array:
        return self._x

    @property
    def pos(self) -> float:
        return self._x[iX]

    @property
    def vel(self) -> float:
        return self._x[iV]

