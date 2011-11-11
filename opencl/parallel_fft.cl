///////////////////////////////////////////////////////////////////////////////
// Complex Arithmetic Functions
///////////////////////////////////////////////////////////////////////////////

float2 complex_mul(float2 a, float2 b);
float2 complex_add(float2 a, float2 b);
float2 complex_sub(float2 a, float2 b);

float2 complex_mul(float2 a, float2 b)
{
   float2 ans;
   ans.x = a.x * b.x - a.y * b.y;
   ans.y = a.x * b.y + a.y * b.x;
   return ans;
}

float2 complex_add(float2 a, float2 b)
{
   float2 ans;
   ans.x = a.x + b.x;
   ans.y = a.y + b.y;
   return ans;
}

float2 complex_sub(float2 a, float2 b)
{
   float2 ans;
   ans.x = a.x - b.x;
   ans.y = a.y - b.y;
   return ans;
}


///////////////////////////////////////////////////////////////////////////////
// Kernel Functions
///////////////////////////////////////////////////////////////////////////////

//-----------------------------------------------------------------------------
// NAME: bitrev_permute_x2
//
// PURPOSE:
//    Perform bit-reverse permutation on two input vectors. The work item
//    index is bit-reversed and is used as the index to the output array.
//
//    NOTE: This algorithm was inspired by the "bit hacks" from
//          http://graphics.stanford.edu/~seander/bithacks.html
//
// INPUT: 
//    in1     First vector to permute
//    in2     Second vector to permute
//    k       Number of bits to bit-reverse  
//
// OUTPUT: 
//    out1    Permutation of first input vector
//    out2    Permutation of second input vector
//
// RETURNS: void
//-----------------------------------------------------------------------------
__kernel void bitrev_permute_x2(__global const float2 *in1, 
                                __global const float2 *in2, 
                                __global float2 *out1,
                                __global float2 *out2,
                                unsigned int k) 
{
   unsigned int gid = get_global_id(0); 
   unsigned int v = gid;

   // swap odd and even bits
   v = ((v >> 1) & 0x55555555) | ((v & 0x55555555) << 1);
   //swap consecutive pairs
   v = ((v >> 2) & 0x33333333) | ((v & 0x33333333) << 2);
   //swap nibbles
   v = ((v >> 4) & 0x0F0F0F0F) | ((v & 0x0F0F0F0F) << 4);
   //swap bytes
   v = ((v >> 8) & 0x00FF00FF) | ((v & 0x00FF00FF) << 8);
   //swap 2-byte pairs
   v = (v >> 16) | (v << 16);
   //shift right to account for a k-bit number (not the full 32-bits)
   v >>= (32-k);

   out1[v] = in1[gid];
   out2[v] = in2[gid];
}


//-----------------------------------------------------------------------------
// NAME: parallel_fft_x2
//
// PURPOSE:
//    Perform forward fft on two input coefficient vectors. Uses work item
//    index to calculate the twiddle factor then performs either the top
//    or bottom half of the size-n butterfly operation.
//
// INPUT: 
//    in1     First vector to perform forward fft on
//    in2     Second vector to perform forward fft on
//    n       Size of butterfly operation
//
// OUTPUT: 
//    out1    Output of forward fft on first input vector
//    out2    Output of forward fft on second input vector
//
// RETURNS: void
//-----------------------------------------------------------------------------
__kernel void parallel_fft_x2(__global const float2 *in1,
                              __global const float2 *in2,
                              __global float2 *out1,
                              __global float2 *out2,
                              unsigned int n)
{
   unsigned int gid = get_global_id(0);
   unsigned int n_div2 = n>>1;
   
   // Determine exponent of twiddle factor for butterfly operation
   int exp = gid & (n_div2 - 1); // gid mod (n/2)
   
   // Determine if this work item will be the bottom part of the butterfly operation
   unsigned int is_bottom = gid & n_div2;
   
   // calculate twiddle factor ( e ^ (2*pi*exp/n) )
   float2 twiddle;
   twiddle.x = cospi(2*exp/(float)n);
   twiddle.y = sinpi(2*exp/(float)n);
   
   float2 t1, t2;
   float2 input1 = in1[gid];
   float2 input2 = in2[gid];
   float2 top_part1, top_part2;
   float2 bottom_part1, bottom_part2;
   
   // Perform portion of butterfly operation
   if (is_bottom)
   {
      bottom_part1 = input1;
      bottom_part2 = input2;
      top_part1 = in1[gid - n_div2];
      top_part2 = in2[gid - n_div2];
      t1 = complex_mul(twiddle, bottom_part1);
      t2 = complex_mul(twiddle, bottom_part2);
      out1[gid] = complex_sub(top_part1, t1);
      out2[gid] = complex_sub(top_part2, t2);
   }
   else
   { 
      bottom_part1 = in1[gid + n_div2];
      bottom_part2 = in2[gid + n_div2];
      top_part1 = input1;
      top_part2 = input2;
      t1 = complex_mul(twiddle, bottom_part1);
      t2 = complex_mul(twiddle, bottom_part2);
      out1[gid] = complex_add(top_part1, t1);
      out2[gid] = complex_add(top_part2, t2);
   }
}


//-----------------------------------------------------------------------------
// NAME: pointwise_mul
//
// PURPOSE:
//    Performs pointwise complex multiplication on two input vectors
//
// INPUT: 
//    in1     First vector
//    in2     Second vector
//
// OUTPUT: 
//    out     Pointwise multiplication of in1 and in2
//
// RETURNS: void
//-----------------------------------------------------------------------------
__kernel void pointwise_mul(__global const float2 *in1,
                            __global const float2 *in2,
                            __global float2 *out)
{
   unsigned int gid = get_global_id(0);
   out[gid] = complex_mul(in1[gid], in2[gid]);
}


//-----------------------------------------------------------------------------
// NAME: inverse_parallel_fft
//
// PURPOSE:
//    Perform inverse fft. Uses work item index to calculate the twiddle 
//    factor then performs either the top or bottom half of the size-n
//    butterfly operation.
//
//    This is similar to the forward fft with two exceptions:
//       1. The exponent of the twiddle factor is negated
//       2. The output vector is divided by n at the final stage
//
// INPUT: 
//    in      Vector to perform inverse fft on
//    n       Size of butterfly operation
//
// OUTPUT: 
//    out     Output of inverse fft on first input vector
//
// RETURNS: void
//-----------------------------------------------------------------------------
__kernel void inverse_parallel_fft(__global const float2 *in,
                                   __global float2 *out,
                                   unsigned int n)
{
   unsigned int gid = get_global_id(0);
   unsigned int n_div2 = n>>1;
   
   // Determine exponent of twiddle factor for butterfly operation
   int exp = gid & (n_div2 - 1); // gid mod (n/2)
   
   // Determine if this work item will be the bottom part of the butterfly operation
   unsigned int is_bottom = gid & n_div2;
   
   // calculate twiddle factor ( e ^ (-2*pi*exp/n) )
   float2 twiddle;
   twiddle.x = cospi(-2*exp/(float)n);
   twiddle.y = sinpi(-2*exp/(float)n);
   
   float2 t;
   float2 input = in[gid];
   float2 top_part;
   float2 bottom_part;
   
   // Perform butterfly operation
   if (is_bottom)
   {
      bottom_part = input;
      top_part = in[gid - n_div2];
      t = complex_mul(twiddle, bottom_part);
      out[gid] = complex_sub(top_part, t);
   }
   else
   { 
      bottom_part = in[gid + n_div2];
      top_part = input;
      t = complex_mul(twiddle, bottom_part);
      out[gid] = complex_add(top_part, t);
   }
   
   // Divide real part by n if this is the last stage
   if (n == get_global_size(0))
      out[gid].x /= (float)n;
}
