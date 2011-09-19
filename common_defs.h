#ifndef COMMON_DEFS_H
#define COMMON_DEFS_H

#define _USE_MATH_DEFINES /* for M_PI constant */

#include <math.h>

#define PI M_PI /* from math.h */

typedef struct
{
   double r; /* real */
   double i; /* imaginary */
} complex;

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
void bit_reverse_copy(complex* a, complex* a_rev_copy);

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
void recursive_fft(complex* a, complex* y, int n, int inv);

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
void iterative_fft(complex* a, int n, int inv);

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
void poly_mul(complex* a, complex* b, int n);

/* 
** complex_mul
**    
** Multiply two polynomials. For example:
**
**    (a + ix) * (b + iy) = ab + iay + ixb - xy
**                        = (ab - xy) + i(ay + xb)
*/
complex complex_mul(complex a, complex b);

/* complex_add - add real and imaginary parts of two complex numbers */
complex complex_add(complex a, complex b);

/* complex_add - subtract real and imaginary parts of two complex numbers */
complex complex_sub(complex a, complex b);

#endif
