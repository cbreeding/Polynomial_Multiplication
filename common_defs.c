#include <stdlib.h>
#include "common_defs.h"

/* complex_mul - see common_defs.h for more details */
complex complex_mul(complex a, complex b)
{
   complex ans;
   
   ans.r = a.r * b.r - a.i * b.i;
   ans.i = a.r * b.i + a.i * b.r;
   
   return ans;
}

/* complex_add - see common_defs.h for more details */
complex complex_add(complex a, complex b)
{
   complex ans;
   
   ans.r = a.r + b.r;
   ans.i = a.i + b.i;
   
   return ans;
}

/* complex_sub - see common_defs.h for more details */
complex complex_sub(complex a, complex b)
{
   complex ans;
   
   ans.r = a.r - b.r;
   ans.i = a.i - b.i;
   
   return ans;
}

#if 0
/* bit_reverse_copy - see common_defs.h for more details */
void bit_reverse_copy(complex* a, complex* a_rev_copy)
{
   a = 0;
}
#endif

/* poly_mul - see common_defs.h for more details */
void poly_mul(complex* a, complex* b, int n)
{
   complex* ya;
   complex* yb;
   int j;
   
   /* Allocate storage for fft results */
   ya = (complex*)malloc(n * sizeof(complex));
   yb = (complex*)malloc(n * sizeof(complex));
   
   /* DFT of A and B */
   recursive_fft(a, ya, n, 0);
   recursive_fft(b, yb, n, 0);
   
   /* Pointwise Multiplication */
   for (j = 0; j < n; j++)
      ya[j] = complex_mul(ya[j], yb[j]);
      
   /* Inverse DFT (swapped input and output arrays) */
   recursive_fft(ya, a, n, 1);
   
   /* Divide real part by n */
   for (j = 0; j < (n-1); j++)
      a[j].r = a[j].r/n;
      
   free(ya);
   free(yb);
}
