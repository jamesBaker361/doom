help(
[[
This module loads the GNU Compiler Collection (GCC). 
This provides Fortran, C, and C++ Compilers.
]])

whatis("Description: GCC: the GNU Compiler Collection")
whatis("URL: https://gcc.gnu.org/")

conflict("gcc")
load("gmp/6.2.0-gc563")
load("mpfr/4.1.0-bz186")
load("mpc/1.2.0-bz186")

local base = pathJoin("/projects/community", "gcc", "10.3.0", "pgarias")
prepend_path("PATH", pathJoin(base, "bin"))
prepend_path("C_INCLUDE_PATH", pathJoin(base, "include"))
prepend_path("CPLUS_INCLUDE_PATH", pathJoin(base, "include"))
prepend_path("LIBRARY_PATH", pathJoin(base, "lib64"))
prepend_path("LD_LIBRARY_PATH", pathJoin(base, "lib64"))
prepend_path("MANPATH", pathJoin(base, "share/man"))
