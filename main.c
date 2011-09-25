#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "common_defs.h"

#define MAX_COEFF    10
#define MAX_N        (1<<20)

int main(int argc, char* argv[])
{
   int n;
   int i;
   int next_power_of_2;
   int timed_test;
   int shift_val;
   int coeff;
   int ret_val;
   complex* a;
   complex* b;
   
#ifdef TIMED_FFT
      timed_test = 1;
#else 
      timed_test = 0;
#endif

   if (timed_test)
   {
      srand(time(NULL));
      
      n = 1;
      shift_val = 0;
      while ((n = (n<<1)) <= MAX_N)
      {
         shift_val++;
         clock_t start = clock();
         
         a = (complex*)malloc(2 * n * sizeof(complex));
         b = (complex*)malloc(2 * n * sizeof(complex));
         
         /* Randomize polynomials to multiply */
         for (i = 0; i < (2*n); i++)
         {
            if (i < n)
            {  
               a[i].r = rand()%MAX_COEFF; a[i].i = 0.0;
               b[i].r = rand()%MAX_COEFF; b[i].i = 0.0;
            }
            else /* pad with zeros */
            {
               a[i].r = 0.0; a[i].i = 0.0;
               b[i].r = 0.0; b[i].i = 0.0;
            }
         }
         
         /* Perform polynomial multiplication and place results in a */
         poly_mul(a, b, 2*n);
         
         free(a);
         free(b);
         
         printf("[N = 2^%-2d = %-7d] Time elapsed: %.9f sec\n", 
            shift_val, n, ((double)clock() - start) / CLOCKS_PER_SEC);
      }
   }
   else
   {
      /* Read size of coefficient array from stdin */
      ret_val = scanf("%d",&n);
      
      /* Determine the next biggest power of two */
      next_power_of_2 = 1;
      while (next_power_of_2 < n)
         next_power_of_2 <<= 1;
      
      /* Allocate space for polynomials */
      a = (complex*)malloc(2 * next_power_of_2 * sizeof(complex));
      b = (complex*)malloc(2 * next_power_of_2 * sizeof(complex));
      
      /* Read coefficients from stdin */
      for (i = 0; i < n; i++)
      {
         ret_val = scanf("%d",&coeff);
         a[i].r = (double)coeff;
         a[i].i = 0.0;
      }
      
#ifndef REC_FFT
      for (i = 0; i < n; i++)
      {
         ret_val = scanf("%d",&coeff);
         b[i].r = (double)coeff;
         b[i].i = 0.0;
      }
#endif
         
      /* Pad the rest with zeros */
      for (i = n; i < (2 * next_power_of_2); i++)
      {
         a[i].r = 0.0; a[i].i = 0.0;
         b[i].r = 0.0; b[i].i = 0.0;
      }
      
      n = next_power_of_2; 
         
#ifdef REC_FFT
      /* Execute recursive FFT (results placed in b) */
      recursive_fft(a, b, n, 0);
      
      printf("\nPrinting coefficient evaluations at w_n^k = e^(2*PI*i*k/n):\n");
      for (i = 0; i < n; i++)
      {
         if (i > 100)
            break;

         printf("[%d] = %.4f %s %.4fi\n", 
            i,
            b[i].r,
            b[i].i < 0.0 ? "-" : "+",
            b[i].i < 0.0 ? (b[i].i*-1.0) : b[i].i);
      }
#else
      /* Multiply polynomials */
      poly_mul(a, b, 2*n);
      
      printf("\nPrinting coefficients for x^k:\n");
      for (i = 0; i < (2*n - 1); i++)
      {
         if (i > 100)
            break;

         printf("[%d] = %.0f\n", i, a[i].r);
      }
#endif

      free(a);
      free(b);
   }

   return 0;
}
