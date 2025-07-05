help(
[[
This module loads bzip2 
A Program for Compressing Files
]])

whatis("Description: A Program for Compressing Files")
whatis("URL: ://downloads.sourceforge.net/project/bzip2/bzip2-1.0.6.tar.gz")

conflict("bzip2")
local base = pathJoin("/projects/community/bzip2/1.0.6/bd387/")
prepend_path("PATH", pathJoin(base, "bin"))
prepend_path("LIBRARY_PATH", pathJoin(base, "lib"))
prepend_path("CPATH", pathJoin(base, "include"))
prepend_path("LD_LIBRARY_PATH", pathJoin(base, "lib"))

load("gcc/5.4")
