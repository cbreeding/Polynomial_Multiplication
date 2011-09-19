#define _USE_MATH_DEFINES

#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>

/* #define RUN_EXAMPLE */

#define PI M_PI /* from math.h */

typedef struct
{
   double r; /* real */
   double i; /* imaginary */
} complex;

complex complex_mul(complex a, complex b)
{
   complex ans;
   
   ans.r = a.r * b.r - a.i * b.i;
   ans.i = a.r * b.i + a.i * b.r;
   
   return ans;
}

complex complex_add(complex a, complex b)
{
   complex ans;
   
   ans.r = a.r + b.r;
   ans.i = a.i + b.i;
   
   return ans;
}

complex complex_sub(complex a, complex b)
{
   complex ans;
   
   ans.r = a.r - b.r;
   ans.i = a.i - b.i;
   
   return ans;
}

/*---------------------------------------------------------
** NAME: bit_reverse_copy
**
** PURPOSE:
**    Copy input array into output array by bit-reversed
**    indices.
**
**    e.g. n = 8, a[4] = A[100.b] copied to A[1] = A[001.b]
**
** INPUTS:
**    a     Complex array of polynomial coefficients
**
** OUTPUTS:
**    A     Bit-reversed copy of a
**
** RETURNS: void
**
**-------------------------------------------------------*/
void bit_reverse_copy(complex* a, complex* a_rev_copy)
{
   
}

/*---------------------------------------------------------
** NAME: iterative_fft
**
** PURPOSE:
**    Implement the in-place iterative FFT algorithm in CLRS
**    for evaluating polynomials at complex roots of unity.
**    Used as a helper function for multiplying polynomials.
**
** INPUTS:
**    a     Complex array of polynomial coefficients
**    n     Length of array (must be a power of 2)
**    inv   1 if performing inverse DFT, 0 otherwise
**
** OUTPUTS:
**    a     An array of n complex numbers representing the 
**          evaluation of the input polynomial at complex 
**          roots of unity.
**
** RETURNS: void
**
**-------------------------------------------------------*/
void iterative_fft(complex* a, int n, int inv)
{
   complex* A;
   
   A = (complex*)malloc(n * sizeof(complex));
   bit_reverse_copy(a,A);
   
   free(A);
}

/*---------------------------------------------------------
** NAME: recursive_fft
**
** PURPOSE:
**    Implement the recursive FFT algorithm in CLRS for
**    evaluating polynomials at complex roots of unity.
**    Used as a helper function for multiplying polynomials.
**
** INPUTS:
**    a     Complex array of polynomial coefficients
**    n     Length of array (must be a power of 2)
**    inv   1 if performing inverse DFT, 0 otherwise
**
** OUTPUTS:
**    y     An array of n complex numbers representing the 
**          evaluation of the input polynomial at complex 
**          roots of unity. User must allocate this memory.  
**
** RETURNS: void
**
**-------------------------------------------------------*/
void recursive_fft(complex* a, complex* y, int n, int inv)
{
   complex w, wn, twiddle;
   complex* a0;
   complex* a1;
   complex* y0;
   complex* y1;
   int i, k;
   
   /* Base Case */
   if (n == 1)
   {
      y[0] = a[0];
      return;
   }
   
   /* Calculate principal nth root of unity (i.e. exp(2*PI*i/n)) */
   if (inv)
   {
      wn.r = cos(-2*PI/(double)n);
      wn.i = sin(-2*PI/(double)n);
   }
   else
   {
      wn.r = cos(2*PI/(double)n);
      wn.i = sin(2*PI/(double)n); 
   }
   w.r = 1.0;
   w.i = 0.0;
   
   /* allocate memory for even/odd coefficients and corresponding FFTs */
   a0 = (complex*)malloc((n/2) * sizeof(complex));
   a1 = (complex*)malloc((n/2) * sizeof(complex));
   y0 = (complex*)malloc((n/2) * sizeof(complex));
   y1 = (complex*)malloc((n/2) * sizeof(complex));
   
   /* Extract even and odd coefficients */
   for (i = 0; i < (n/2); i++)
   {
      a0[i] = a[2*i];
      a1[i] = a[2*i+1];
   }
   
   /* Calculate 2 FFTs of size n/2 */
   recursive_fft(a0, y0, n/2, inv);
   recursive_fft(a1, y1, n/2, inv);
   
   /* Combine results from half-size FFTs */
   for (k = 0; k < (n/2); k++)
   {
      twiddle  = complex_mul(w, y1[k]);
      y[k]     = complex_add(y0[k], twiddle);
      y[k+n/2] = complex_sub(y0[k], twiddle);
      w        = complex_mul(w, wn);
   }
   
   free(a0);
   free(a1);
   free(y0);
   free(y1);
   
   return;
}

/*---------------------------------------------------------
** NAME: poly_mul
**
** PURPOSE:
**    Perform polynomial multiplication per CLRS algorithm.
**    Utilizes the recursive (or iterative fft) for converting
**    to/from coefficient and point-value forms.
**
**    NOTE: It is assumed that the coefficient arrays are
**    already padded with zeros.
**
** INPUTS:
**    a     Complex array of polynomial coefficients
**    b     Complex array of polynomial coefficients
**    n     Size of coefficient arrays a and b
**
** OUTPUTS:
**    a     Coefficient vector resulting from polynomial
**          multiplication of a and b.  
**
** RETURNS: void
**
**-------------------------------------------------------*/
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
