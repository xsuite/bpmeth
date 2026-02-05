version='0.0.0'
from .frames import *
from .generate_expansion import *
from .numerical_solver import *
from .track4d import *
from .fringefieldmaps import *
from .fieldmaps import *
from .magnet import *
from . import poly_fit

from pathlib import Path
ab_order, sorder = 3, 3
package_dir = Path(__file__).resolve().parent
filename = f"polyhamiltonian_{ab_order}_{sorder}_h.py"
file_path = package_dir / filename
if not file_path.exists():
    print("Building tracking code...")
    from .fast_hamilton_solver_create_sourcecode import mk_field
    mk_field(ab_order=ab_order, sorder=sorder, h=True, nphi=5, out=str(file_path))

from .fast_hamilton_solver import *
