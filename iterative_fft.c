#include "common_defs.h"

/* iterative_fft - see common_defs.h for more details */
#if 0
void iterative_fft(complex* a, int n, int inv)
{
   complex* A;
   
   A = (complex*)malloc(n * sizeof(complex));
   bit_reverse_copy(a,A);
   
   free(A);
}
#endif
