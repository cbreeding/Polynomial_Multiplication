CC=gcc
CFLAGS=-Wall -O2
LIBS=-lm
SRC=$(wildcard *.c)

all: timed_fft recursive_fft polymul

timed_fft: $(SRC)
	$(CC) $(CFLAGS) $^ $(LIBS) -DTIMED_FFT -o $@
   
recursive_fft: $(SRC)
	$(CC) $(CFLAGS) $^ $(LIBS) -DREC_FFT -DDEBUG_TRACE -o $@
   
polymul: $(SRC)
	$(CC) $(CFLAGS) $^ $(LIBS) -o $@

clean:
	rm -f *.o *.exe
   