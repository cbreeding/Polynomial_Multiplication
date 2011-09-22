IDIR=/cygdrive/c/Program\ Files\ \(x86\)/AMD\ APP/include/
LDIR=/cygdrive/c/Program\ Files\ \(x86\)/AMD\ APP/lib/x86_64/gcc/

# NOTE: Had to regenerate the libOpenCL.a library so it would work
#       with gcc. Performed the following steps:
#
#  cp /c/Windows/System32/OpenCL.dll .
#  gendef OpenCL.dll
#  dlltool -l libOpenCL.a -d OpenCL.def -k -A

CC=gcc
CFLAGS=-I$(IDIR) -Wall -L$(LDIR)
LIBS=-lOpenCL
OUTFILE=vector_add
SOURCE=main.c

all: 
	$(CC) $(CFLAGS) $(SOURCE) $(LIBS) -o $(OUTFILE)

clean:
	rm -f *.o $(OUTFILE)
   