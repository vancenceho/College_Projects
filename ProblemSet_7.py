# Problem 7.01 - PolarComplex
import math

class PolarComplex:

    def __init__(self, r, theta):
        self._r = r
        self._theta = theta
        
    def get_modulus(self):
        return self._r
    
    def get_argument(self):
        return self._theta

    def get_real(self):
        re = self._r * math.cos(self._theta)
        return re

    def get_imaginary(self):
        im = self._r * math.sin(self._theta)
        return im

    def get_cartesian_form(self):
        re = a.get_real()
        im = a.get_imaginary()
        return complex(re, im)

    def get_polar_form_string(self):
        r = self._r
        theta = self._theta
        return("{0:.3f} exp( j*{1:.3f} )".format(r, theta))

# TEST CASES
a = PolarComplex(5, math.pi/3)
print(a.get_modulus())                                                      # 5
print(a.get_argument())                                                     # 1.0471975511965976
print(a.get_real())                                                         # 2.5000000000000004
print(a.get_imaginary())                                                    # 4.330127018922193
print(a.get_cartesian_form())                                               # (2.5000000000000004+4.330127018922193j)
print(a.get_polar_form_string())                                            # 5.000 exp( j*1.047 )

b = PolarComplex(10/3, math.pi/6)                                           
print(b.get_polar_form_string())                                            # 3.333 exp( j*0.524 )
## If 0 < r < 10 and 0 < theta < 2pi, 
## each string below has length 20.