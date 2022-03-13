import numpy as np


def length(v):
    """
    given v, return the length
    """

    s = 0
    for num in v:
        s += num[0] ** 2

    return np.sqrt(s)


def normalize(v):
    """
    for 3by3 only
    eg: w01 = np.array([[-1.42], [0.43], [-0.42]])
    """
    mag = length(v)
    if mag == 0:
        raise ZeroDivisionError("zero vector cannot be normalized")
    return v/mag, mag


def Rx(theta):
    """
    rotation matrix that rotate around x axis
    """

    R = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)],
                  [0, np.sin(theta), np.cos(theta)]])

    return R


def Ry(theta):
    """
    rotation matrix that rotate around y axis
    """

    R = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0],
                  [-np.sin(theta), 0, np.cos(theta)]])

    return R


def Rz(theta):
    """
    rotation matrix that rotate around z axis
    """

    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

    return R


def wToSkew(w):
    """
    for 3by3 only
    eg: w01 = np.array([[-1.42], [0.43], [-0.42]])
    """
    ret = np.zeros((3, 3))
    ret[0, 1] = -w[2, 0]
    ret[0, 2] = w[1, 0]
    ret[1, 0] = w[2, 0]
    ret[1, 2] = -w[0, 0]
    ret[2, 0] = -w[1, 0]
    ret[2, 1] = w[0, 0]
    return ret


def skewToW(skew):
    ret = np.zeros((3, 1))
    ret[0, 0] = skew[2, 1]
    ret[1, 0] = skew[0, 2]
    ret[2, 0] = skew[1, 0]
    return ret


def rodrigues(omega_hat, theta):
    skew_omega = wToSkew(omega_hat)
    return np.eye(3) + np.sin(theta)*skew_omega + (1-np.cos(theta))*skew_omega@skew_omega


def wToR(omega):
    """
    turn omega into R
    """
    if (omega == np.zeros((3, 1))).all():
        omega_hat, theta = omega, 0
    else:
        omega_hat, theta = normalize(omega)
    return rodrigues(omega_hat, theta)


def rToW(R):
    """
    turn R into omega
    R should by 3by3
    """
    omega_hat = np.zeros((3, 1))
    theta = 0
    if (R == np.eye(3)).all():
        pass  # same as default
    elif np.trace(R) == -1:
        theta = np.pi
        # TODO: not sure if it's corret?
        # only choose one to return
        if np.sqrt(2*(1+R[2, 2])) != 0:
            omega_hat = 1 / \
                np.sqrt(2*(1+R[2, 2])) * \
                np.array([[R[0, 2], [R[1, 2]], [1+R[2, 2]]]])
        elif np.sqrt(2*(1+R[1, 1])) != 0:
            omega_hat = 1 / \
                np.sqrt(2*(1+R[1, 1])) * \
                np.array([[R[0, 1], [1+R[1, 1]], [1+R[2, 1]]]])
        else:  # np.sqrt(2*(1+R[0, 0])) != 0
            omega_hat = 1 / \
                np.sqrt(2*(1+R[0, 0])) * \
                np.array([[1+R[0, 0], [R[1, 0]], [1+R[2, 0]]]])
    else:
        theta = np.arccos(1/2*(np.trace(R)-1))
        omega_hat_skew = 1/(2*np.sin(theta)) * (R - R.T)
        omega_hat = skewToW(omega_hat_skew)

    return theta, omega_hat


def xyzToP(x, y, z):
    """
    turn x, y, z to p
    """

    return np.array([[x], [y], [z]])


def RpToT(R, p):
    """
    turn R, p to T

    ex:
        R = np.eye(3)
        p = np.array([[2], [0], [0]])
    """
    T = np.zeros((4, 4))
    T[3, 3] = 1
    T[0:3, 0:3] = R
    T[0:3, [3]] = p

    return T


def twistToWV(twist):
    """
    turn twist V to omega and v

    V_sbinb = np.array([[ 6], [-5], [ 3], [-3], [ 7], [-5]])
    """

    omega = twist[0:3]
    v = twist[3:]
    return omega, v


def wVToTwist(omega, v):
    """
    turn omega and v to twist V
    """
    V = np.zeros((6, 1))
    V[0:3] = omega
    V[3:6] = v

    return V


def twistToTmatrix(twist):
    """
    turn V to [V]

    V = np.array([[6], [2], [5], [6], [4], [1]])
    """

    omega, v = twistToWV(twist)

    V_matrix = np.zeros([4, 4])
    V_matrix[0:3, 0:3] = wToSkew(omega)
    V_matrix[0:3, [3]] = v

    return V_matrix


def tmatrixToTwist(tmatrix):
    """
    turn [v] to v
    """
    skew_omega = tmatrix[0:3, 0:3]
    omega = skewToW(skew_omega)
    v = tmatrix[0:3, [3]]

    return wVToTwist(omega, v)


def twistToScrew(twist):
    """
    turn twist into screw
    """

    omega, v = twistToWV(twist)
    # |w| is not zero
    if np.any(omega):
        omega_hat, theta = normalize(omega)
        screw = twist/theta
    # |w| is zero
    else:
        # |v| is not zero
        if np.any(v):
            v_hat, theta = normalize(v)
            screw = twist/theta
        else:
            theta = 0
            screw = np.zeros_like(twist)

    return theta, screw


def twistToT(twist):
    """
    turn twist V to homogenous matrix T
    """
    T = np.zeros((4, 4))
    T[3, 3] = 1
    omega, v = twistToWV(twist)

    if (omega == np.zeros((3, 1))).all():  # theta == 0
        R = np.eye(3)
        p = v
    else:
        R = wToR(omega)

        omega_hat, theta = normalize(omega)
        skewW_hat = wToSkew(omega_hat)
        # TODO: this equation seems wrong
        p = 1/theta * (theta * np.eye(3) +
                       (1-np.cos(theta)*skewW_hat + (theta-np.sin(theta))
                        * skewW_hat @ skewW_hat)
                       ) @ v
        # TODO: this seems to work
        v_hat = v/theta
        p = (np.eye(3) - R)@(skewW_hat@v_hat) + \
            omega_hat @ (omega_hat.T) @ v_hat * theta

    T[0:3, 0:3] = R
    T[0:3, [3]] = p

    return T


def TToTwist(T):
    """
    turn homogenous matrix T to twist V
    """
    R = T[0:3, 0:3]
    p = T[0:3, [3]]

    if np.all(R == np.eye(3)):
        omega = 0
        v = p
    else:
        theta, omega_hat = rToW(R)
        omega = omega_hat * theta

        skewW = wToSkew(omega)
        A_1 = np.eye(3)-1/2*skewW +\
            (2*np.sin(theta)-theta*(1+np.cos(theta))) / \
            (2*theta**2*np.sin(theta))*skewW@skewW
        v = A_1 @ p

    return wVToTwist(omega, v)


def TtoAdt(T):
    """
    turn T to Ad_T
    """
    R = T[0:3, 0:3]
    p = T[0:3, [3]]

    Ad_T = np.zeros((6, 6))
    Ad_T[0:3, 0:3] = R
    Ad_T[3:, 3:] = R
    Ad_T[3:, 0:3] = wToSkew(p) @ R

    return Ad_T
