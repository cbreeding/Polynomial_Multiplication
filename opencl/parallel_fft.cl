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

__kernel void bitrev_permute(__global const float2 *in, __global float2 *out, unsigned int k) 
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

   out[v] = in[gid];
}

__kernel void parallel_fft(__global const float2 *in, __global float2 *out, unsigned int n)
{
   unsigned int gid = get_global_id(0);
   unsigned int n_div2 = n>>1;
   
   // Determine exponent of twiddle factor for butterfly operation
   unsigned int exp = gid & (n_div2 - 1); // gid mod (n/2)
   
   // Determine if this work item will be the bottom part of the butterfly operation
   unsigned int is_bottom = gid & n_div2;
   
   // calculate twiddle factor ( e ^ (2*pi*exp/n) )
   float2 twiddle;
   twiddle.x = cospi(2*exp/(float)n);
   twiddle.y = sinpi(2*exp/(float)n);
   
   float2 t;
   float2 input = in[gid];
   float2 top_part;
   float2 bottom_part;
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
}
