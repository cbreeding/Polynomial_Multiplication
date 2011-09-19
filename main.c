#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "common_defs.h"

/* #define RUN_EXAMPLE */

/*
** NOTE: Can check results in octave, e.g. abs(ifft(fft([4 3 2 1], 8).*fft([4 3 2 1], 8), 8))
*/
int main(int argc, char* argv[])
{
   int n = 1;
   complex* a;
   complex* b;
   int j;
#ifndef RUN_EXAMPLE
   int i;
   const int max_coeff = 10;
   const int num_iter = 100;
   const int max_n = (1<<15);
#endif

   srand(time(NULL));
   
#ifdef RUN_EXAMPLE
   n = 4;
#else 
   while ((n = (n<<1)) <= max_n)
#endif
   {
      clock_t start = clock();
      
      a = (complex*)malloc(2 * n * sizeof(complex));
      b = (complex*)malloc(2 * n * sizeof(complex));
      
 
#ifndef RUN_EXAMPLE 
      for (i = 0; i < num_iter; i++)
#endif
      {
         for (j = 0; j < (2*n); j++)
         {
            if (j < n)
            {
#ifdef RUN_EXAMPLE
               a[j].r = n-j; a[j].i = 0.0;
               b[j].r = n-j; b[j].i = 0.0;
#else
               a[j].r = rand()%max_coeff; a[j].i = 0.0;
               b[j].r = rand()%max_coeff; b[j].i = 0.0;
#endif
            }
            else
            {
               a[j].r = 0.0; a[j].i = 0.0;
               b[j].r = 0.0; b[j].i = 0.0;
            }
         }
         
         /* Perform polynomial multiplication and place results in a */
         poly_mul(a, b, 2*n);
         
#ifdef RUN_EXAMPLE
         /* Print results of polynomial multiplication */
         for (j = 0; j < (2*n-1); j++)
         {
            if (j > 100)
               break;
            printf("[%d] = %d\n", j, (int)a[j].r);
         }
#endif
      }
      
      free(a);
      free(b);
#ifndef RUN_EXAMPLE      
      printf("[N = %d] Time elapsed: %.9f\n", n, ((double)clock() - start) / CLOCKS_PER_SEC);
#endif
   }
   
   return 0;
}
