IDIR=/Developer/GPU\ Computing/OpenCL/common/inc/
CC=gcc
CFLAGS=-Wall 
LFLAGS=-framework OpenCL
OUTFILE=parallel_fft
SOURCE=main.c

all: $(SOURCE) 
	$(CC) $(CFLAGS) $(SOURCE) $(LFLAGS) -o $(OUTFILE)

clean:
	rm -f *.o $(OUTFILE)
   
