import math

class Quaternion:

    def __init__(self, s=0, x=0, y=0, z=0):
        self.s = s
        self.x = x
        self.y = y
        self.z = z
    
    def __abs__(self):
        self.abs = math.sqrt(s ** 2 + x ** 2 + y ** 2 + z ** 2)
        return self.abs

    def __add__(self, other):
        return Quaternion(
                self.s + other.s,
                self.x + other.x
                self.y + other.y
                self.y + other.y
            )

    def __mul__(self, scalar):
        return Quaternion(
                scalar * self.s,
                scalar * self.x,
                scalar * self.y,
                scalar * self.z,
            )

    def __conjugate__(self):
        return Quaternion(
                self.s,
                -self.x,
                -self.y,
                -self.z
            )
    
    def __inner__(self, other):
        return (self.s * other.s
                + self.x * other.x
                + self.y * other.y
                + self.z * other.z
            )
    #x 1 × x 2 = (y 1 z 2 − y 2 z 1 , z 1 x 2 − z 2 x 1 , x 1 y 2 − x 2 y 1 )
    def __cross__(self, other):
        return Quaternion(
                0,
                self.y * other.z - other.y * self.z,
                self.z * other.x - other.z * self.x,
                self.x * other.y - other.x * self.y
            )
