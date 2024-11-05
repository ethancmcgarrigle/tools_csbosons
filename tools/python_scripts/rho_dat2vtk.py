# script to read in vector quantities
import sys
import re

outname = sys.argv[1]

px = []

# TODO: make this adjustable
dx = 0.5936363636363636 
#dx = sys.argv[2]
dy = dx 
dz = dx 

#dx = 0.78125
#dy = 0.78125
#dz = 0.78125
#    0.000000     0.000000     0.000000      3.372766864      3.864859935 
#    0.000000     0.000000     0.593636      2.584396411    -0.5513510687 
#    0.000000     0.000000     1.187273       -7.4319093      3.112904444 
#    0.000000     0.000000     1.780909      2.930862311      1.753413138 

#with open ("density_0_FINAL.dat") as pxin:
with open ("density_1_FINAL.dat") as pxin:
  pxin.readline() # read format version line
  pxin.readline() # read nfields line
  pxin.readline() # read ndim line
  line = pxin.readline() # read pw grid line
  tup = re.split(r"\s+", line)
  nx = tup[4]
  ny = tup[5]
  nz = tup[6]
  pxin.readline() # consume kspace data line
  pxin.readline() # consume comment newline
  for line in pxin:
    if line == "" or line == "\n":
      pass
    else:
      tup = re.split(r"\s+", line)
      #tup = tup[3:5]
      tup = tup[4:6] # correct for real and imaginary part 
      tup = tuple(map(float, tup))
      px.append(tup)

with open (outname + "_real.vtk", "w") as fout:
  fout.write("# vtk DataFile Version 2.0\n")
  fout.write("CT Density\n")
  fout.write("ASCII\n")
  fout.write("\n")
  fout.write("DATASET STRUCTURED_POINTS\n")
  fout.write("DIMENSIONS " + nz + " " + ny + " " + nx + "\n")
  fout.write("ORIGIN 0.000000 0.000000 0.000000\n")
  fout.write("SPACING " + str(dz) + " " + str(dy) + " " + str(dx) +"\n")
  fout.write("\n")
  fout.write("POINT_DATA " + str(int(nx)*int(ny)*int(nz)) + "\n")
  fout.write("SCALARS scalars float\n")
  fout.write("LOOKUP_TABLE default\n\n")

  for x in px:
    fout.write(str(x[0]) + "\n")

with open (outname + "_imag.vtk", "w") as fout:
  fout.write("# vtk DataFile Version 2.0\n")
  fout.write("CT Density\n")
  fout.write("ASCII\n")
  fout.write("\n")
  fout.write("DATASET STRUCTURED_POINTS\n")
  fout.write("DIMENSIONS " + nz + " " + ny + " " + nx + "\n")
  fout.write("ORIGIN 0.000000 0.000000 0.000000\n")
  fout.write("SPACING " + str(dz) + " " + str(dy) + " " + str(dx) +"\n")
  fout.write("\n")
  fout.write("POINT_DATA " + str(int(nx)*int(ny)*int(nz)) + "\n")
  fout.write("SCALARS scalars float\n")
  fout.write("LOOKUP_TABLE default\n\n")

  for x in px:
    fout.write(str(x[1]) + "\n")

