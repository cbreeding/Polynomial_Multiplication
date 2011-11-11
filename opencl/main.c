#include <stdio.h>
#include <stdlib.h>
#if defined __APPLE__ || defined(MACOSX)
   #include <OpenCL/opencl.h>
#else
   #include <CL/cl.h>
#endif
#include <time.h>

#define MAX_SOURCE_SIZE (0x100000)

// Function prototypes
static void print_device_info(cl_platform_id p, cl_device_id d);
static int get_input_polynomials();
static void swap_mem_ptr(cl_mem* a, cl_mem* b);
static int gen_polynomials();

// Global coefficient arrays
cl_float2* poly1;
cl_float2* poly2;

// max work group size
size_t group_size;


//-----------------------------------------------------------------------------
// NAME: main
//
// PURPOSE:
//    Compiles OpenCL kernels, creates device memory buffers, 
//    creates kernel instances, deploys kernels to GPU, and 
//    verifies results.
//
//    High-Level Algorithm:
//       1. Get polynomials from user
//       2. Compile the kernels and create device memory
//       3. Create kernel instances
//             a. Bit-reverse permutation
//             b. lg(n) FFT stages for both polynomials
//             c. Point-wise multiplication of two polynomials
//             d. Bit-reverse permutation
//             e. lg(n) inverse-FFT stages
//       4. Deploy kernel instances to GPU
//       5. Verify results
//       6. Clean up
//
// INPUT: void
//
// OUTPUT: Prints device specs and results of polynomial multiplication
//
// RETURNS: 0 on success, error code otherwise
//----------------------------------------------------------------------------- 
int main(int argc, char* argv[]) 
{
   int i;
   
   ////////////////////////////////////
   //
   // Get polynomials from user
   //
   ////////////////////////////////////
   //int fft_size = 2 * get_input_polynomials();
   int fft_size = 2 * gen_polynomials((1<<24));
   
   // Calculate log (base 2) of fft_size
   int lg_n = 0;
   i = 1;
   while (i < fft_size)
   {
      lg_n++;
      i <<= 1;
   }
   
   ////////////////////////////////////
   //
   // Compile the kernels and create device memory
   //
   ////////////////////////////////////
    
   // Load kernel code into buffer
   FILE *fp;
   char *source_str;
   size_t source_size;

   fp = fopen("parallel_fft.cl", "r");
   if (!fp) 
   {
      fprintf(stderr, "Failed to load kernel.\n");
      exit(1);
   }
   source_str = (char*)malloc(MAX_SOURCE_SIZE);
   source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
   fclose( fp );
 
   // Get platform and device information
   cl_platform_id platform_id = NULL;
   cl_device_id device_id = NULL;   
   cl_uint ret_num_devices;
   cl_uint ret_num_platforms;
   cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
   ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);
   print_device_info(platform_id, device_id);
 
   // Create an OpenCL context and command queue
   cl_context context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
   cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

   // Create and build the program from the kernel source
   cl_program program = clCreateProgramWithSource(context, 1, 
         (const char **)&source_str, (const size_t *)&source_size, &ret);
   ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

   // Show the build log
   char* build_log;
   size_t log_size;
   clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
   build_log = malloc(log_size+1);
   clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
   build_log[log_size] = 0;
   printf("%s\n",build_log);
   free(build_log);
   
   // Allocate memory on device for coefficient array input and output
   cl_mem in_mem_obj1 = clCreateBuffer(context, CL_MEM_READ_WRITE, 
         fft_size * sizeof(cl_float2), NULL, &ret);
   cl_mem in_mem_obj2 = clCreateBuffer(context, CL_MEM_READ_WRITE, 
         fft_size * sizeof(cl_float2), NULL, &ret);
   cl_mem out_mem_obj1 = clCreateBuffer(context, CL_MEM_READ_WRITE,
         fft_size * sizeof(cl_float2), NULL, &ret);
   cl_mem out_mem_obj2 = clCreateBuffer(context, CL_MEM_READ_WRITE,
         fft_size * sizeof(cl_float2), NULL, &ret);
   cl_mem initial_input1 = in_mem_obj1;
   cl_mem initial_input2 = in_mem_obj2;

   ////////////////////////////////////
   //
   // Create kernel instances
   //
   ////////////////////////////////////
   
   //---------------------------
   // Bit-Reverse Permutation
   //---------------------------
   cl_kernel bitrev_kernel1 = clCreateKernel(program, "bitrev_permute_x2", &ret);

   // Set the arguments of the kernel
   ret = clSetKernelArg(bitrev_kernel1, 0, sizeof(cl_mem), (void *)&in_mem_obj1);
   ret = clSetKernelArg(bitrev_kernel1, 1, sizeof(cl_mem), (void *)&in_mem_obj2);
   ret = clSetKernelArg(bitrev_kernel1, 2, sizeof(cl_mem), (void *)&out_mem_obj1);
   ret = clSetKernelArg(bitrev_kernel1, 3, sizeof(cl_mem), (void *)&out_mem_obj2);
   ret = clSetKernelArg(bitrev_kernel1, 4, sizeof(unsigned int), (void*)&lg_n);
         
   // Swap device memory pointers (output of this will be input to parallel-fft)
   swap_mem_ptr(&in_mem_obj1, &out_mem_obj1);
   swap_mem_ptr(&in_mem_obj2, &out_mem_obj2);
 
   //---------------------------
   // lg(n) FFT stages
   //---------------------------
   unsigned int n;
   
   cl_kernel* fft_kernel = (cl_kernel*)malloc(lg_n * sizeof(cl_kernel));
   for (i=0; i<lg_n; i++)
   {
      fft_kernel[i] = clCreateKernel(program, "parallel_fft_x2", &ret);
      
      // Set the arguments of the kernel
      n = (1 << (i+1)); // double n for each stage of FFT (2,4,8,16...)
      ret = clSetKernelArg(fft_kernel[i], 0, sizeof(cl_mem), (void *)&in_mem_obj1);
      ret = clSetKernelArg(fft_kernel[i], 1, sizeof(cl_mem), (void *)&in_mem_obj2);
      ret = clSetKernelArg(fft_kernel[i], 2, sizeof(cl_mem), (void *)&out_mem_obj1);
      ret = clSetKernelArg(fft_kernel[i], 3, sizeof(cl_mem), (void *)&out_mem_obj2);
      ret = clSetKernelArg(fft_kernel[i], 4, sizeof(unsigned int), (void*)&n);
      
      // Swap memory pointers (output of this stage will be input of next)
      swap_mem_ptr(&in_mem_obj1, &out_mem_obj1);
      swap_mem_ptr(&in_mem_obj2, &out_mem_obj2);
   }
   
   //---------------------------
   // Point-wise multiplication
   //---------------------------
   cl_kernel mul_kernel = clCreateKernel(program, "pointwise_mul", &ret);
   
   // Set the arguments of the kernel
   ret = clSetKernelArg(mul_kernel, 0, sizeof(cl_mem), (void *)&in_mem_obj1);
   ret = clSetKernelArg(mul_kernel, 1, sizeof(cl_mem), (void *)&in_mem_obj2);
   ret = clSetKernelArg(mul_kernel, 2, sizeof(cl_mem), (void *)&out_mem_obj1);
   
   // Swap memory pointers (output of this stage will be input of next)
   swap_mem_ptr(&in_mem_obj1, &out_mem_obj1);
   
   //---------------------------
   // Bit-Reverse Permutation
   //---------------------------
   cl_kernel bitrev_kernel2 = clCreateKernel(program, "bitrev_permute_x2", &ret);

   // Set the arguments of the kernel
   // NOTE: We're using the bitrev_permute_x2 kernel function, but we 
   //       are only concerned with bit reversing in_mem_obj1.
   ret = clSetKernelArg(bitrev_kernel2, 0, sizeof(cl_mem), (void *)&in_mem_obj1);
   ret = clSetKernelArg(bitrev_kernel2, 1, sizeof(cl_mem), (void *)&in_mem_obj2);
   ret = clSetKernelArg(bitrev_kernel2, 2, sizeof(cl_mem), (void *)&out_mem_obj1);
   ret = clSetKernelArg(bitrev_kernel2, 3, sizeof(cl_mem), (void *)&out_mem_obj2);
   ret = clSetKernelArg(bitrev_kernel2, 4, sizeof(unsigned int), (void*)&lg_n);
         
   // Swap device memory pointers (output of this will be input to inverse parallel-fft)
   swap_mem_ptr(&in_mem_obj1, &out_mem_obj1);
 
   //---------------------------
   // lg(n) inverse-FFT stages
   //---------------------------
   cl_kernel* inv_fft_kernel = (cl_kernel*)malloc(lg_n * sizeof(cl_kernel));
   for (i=0; i<lg_n; i++)
   {
      inv_fft_kernel[i] = clCreateKernel(program, "inverse_parallel_fft", &ret);
      
      // Set the arguments of the kernel
      n = (1 << (i+1)); // double n for each stage of FFT (2,4,8,16...)
      ret = clSetKernelArg(inv_fft_kernel[i], 0, sizeof(cl_mem), (void *)&in_mem_obj1);
      ret = clSetKernelArg(inv_fft_kernel[i], 1, sizeof(cl_mem), (void *)&out_mem_obj1);
      ret = clSetKernelArg(inv_fft_kernel[i], 2, sizeof(unsigned int), (void*)&n);
            
      // Swap memory pointers (output of this stage will be input of next)
      swap_mem_ptr(&in_mem_obj1, &out_mem_obj1);
   }
   
   // one final swap
   swap_mem_ptr(&in_mem_obj1, &out_mem_obj1);
   cl_mem final_output = out_mem_obj1;
   
   ////////////////////////////////////
   //
   // Deploy kernel instances to GPU
   //
   ////////////////////////////////////
   printf("starting\n");
   
   size_t global_item_size = fft_size;
   size_t local_item_size;
   if (fft_size >= group_size)
      local_item_size = group_size;
   else
      local_item_size = fft_size;
   
   // Transfer host memory to device
   ret = clEnqueueWriteBuffer(command_queue, initial_input1, CL_TRUE, 0,
         fft_size * sizeof(cl_float2), poly1, 0, NULL, NULL);
   ret = clEnqueueWriteBuffer(command_queue, initial_input2, CL_TRUE, 0,
         fft_size * sizeof(cl_float2), poly2, 0, NULL, NULL);
         
   // Bit-Reverse Permutation
   ret = clEnqueueNDRangeKernel(command_queue, bitrev_kernel1, 1, NULL, 
         &global_item_size, &local_item_size, 0, NULL, NULL);
         
   // FFT stages
   for (i=0; i<lg_n; i++)
      ret = clEnqueueNDRangeKernel(command_queue, fft_kernel[i], 1, NULL, 
            &global_item_size, &local_item_size, 0, NULL, NULL);
            
   // Pointwise Multiplication
   ret = clEnqueueNDRangeKernel(command_queue, mul_kernel, 1, NULL, 
         &global_item_size, &local_item_size, 0, NULL, NULL);
         
   // Bit-Reverse Permutation
   ret = clEnqueueNDRangeKernel(command_queue, bitrev_kernel2, 1, NULL, 
         &global_item_size, &local_item_size, 0, NULL, NULL);
         
   // Inverse FFT stages
   for (i=0; i<lg_n; i++)
      ret = clEnqueueNDRangeKernel(command_queue, inv_fft_kernel[i], 1, NULL, 
            &global_item_size, &local_item_size, 0, NULL, NULL);
            
   // Transfer device memory to host
   ret = clEnqueueReadBuffer(command_queue, final_output, CL_TRUE, 0, 
         fft_size * sizeof(cl_float2), poly1, 0, NULL, NULL);
   printf("done\n");
   
   ////////////////////////////////////
   //
   // Verify results
   //
   ////////////////////////////////////
#if 0
   printf("\nPrinting coefficients for x^k:\n");
   for (i=0; i<(fft_size-1); i++)
      printf("[k = %d]: %.0f\n", 
             i, 
             // eliminates "-0" floating-point artifact in output
             poly1[i].x < 0 ? -poly1[i].x : poly1[i].x); 
#endif
 
   ////////////////////////////////////
   //
   // Clean up
   //
   ////////////////////////////////////
   
   // Flush command queue
   ret = clFlush(command_queue);
   ret = clFinish(command_queue);
   
   // Release kernels
   ret = clReleaseKernel(bitrev_kernel1);
   ret = clReleaseKernel(bitrev_kernel2);
   for (i=0; i<lg_n; i++)
   {
      clReleaseKernel(fft_kernel[i]);
      clReleaseKernel(inv_fft_kernel[i]);
   }
   ret = clReleaseKernel(mul_kernel);
   ret = clReleaseProgram(program);
   
   // Release device memory
   ret = clReleaseMemObject(in_mem_obj1);
   ret = clReleaseMemObject(in_mem_obj2);
   ret = clReleaseMemObject(out_mem_obj1);
   ret = clReleaseMemObject(out_mem_obj2);
   
   // Release command queue and context
   ret = clReleaseCommandQueue(command_queue);
   ret = clReleaseContext(context);
   
   // Free host memory
   free(poly1);
   free(poly2);
   free(fft_kernel);
   free(inv_fft_kernel);
   
   return 0;
}


//-----------------------------------------------------------------------------
// NAME: get_input_polynomials
//
// PURPOSE:
//    Reads input polynomial size and coefficients from stdin and allocates
//    memory for coefficients. 
//
//    If the size of the polynomial is not a power of 2, the coefficient 
//    array will be padded with zeroes until the next biggest power of 2.
//
//    Coefficient array sizes are double the polynomial size. This is because
//    these arrays will be used during polynomial multiplication, and
//    multiplying two polynomials of size n will result in a polynomial of
//    (at most) size 2n-1.
//
// INPUT: void
//
// OUTPUT: Memory allocation for arrays poly1 and poly2
//
// RETURNS: Polynomial size rounded to the next biggest power of 2
//-----------------------------------------------------------------------------
static int get_input_polynomials()
{
   int ret_val;
   int n;
   int i;
   int coeff;
   int next_power_of_2;

   // Read size of coefficient array from stdin
   printf("Enter size of polynomial: ");
   ret_val = scanf("%d",&n);
   printf("\n");

   // Determine the next biggest power of two
   next_power_of_2 = 1;
   while (next_power_of_2 < n)
      next_power_of_2 <<= 1;
   
   // Allocate space for polynomials
   poly1 = (cl_float2*)malloc(2 * next_power_of_2 * sizeof(cl_float2));
   poly2 = (cl_float2*)malloc(2 * next_power_of_2 * sizeof(cl_float2));
   
   // Read coefficients from stdin
   printf("Enter %d coefficients for first polynomial (x^0 coeff first): ", n);
   for (i = 0; i < n; i++)
   {
      ret_val = scanf("%d",&coeff);
      poly1[i].x = (float)coeff;
      poly1[i].y = 0.0;
   }
   printf("\n");
   printf("Enter %d coefficients for second polynomial (x^0 coeff first): ", n);
   for (i = 0; i < n; i++)
   {
      ret_val = scanf("%d",&coeff);
      poly2[i].x = (float)coeff;
      poly2[i].y = 0.0;
   }
   printf("\n");
   
   // Pad the rest with zeros
   for (i = n; i < (2 * next_power_of_2); i++)
   {
      poly1[i].x = 0.0; poly1[i].y = 0.0;
      poly2[i].x = 0.0; poly2[i].y = 0.0;
   }
   
   // Return size of polynomial rounded to next biggest power of 2
   return next_power_of_2;
}


//-----------------------------------------------------------------------------
// NAME: gen_polynomials
//
// PURPOSE: Generates two polynomial coefficients arrays of size (2*size)
//
// INPUT:
//    size     Size of polynomial
//
// OUTPUT:
//    poly1    First generated coefficient array
//    poly2    Second generated coefficient array
//
// RETURNS: size of polynomials generated
//-----------------------------------------------------------------------------
static int gen_polynomials(int size)
{
   int i;
   const int MAX_COEFF = 10;
   
   srand(time(NULL));
   
   poly1 = (cl_float2*)malloc(2 * size * sizeof(cl_float2));
   poly2 = (cl_float2*)malloc(2 * size * sizeof(cl_float2));
   
   for (i = 0; i < size; i++)
   {
      poly1[i].x = rand()%MAX_COEFF; poly1[i].y = 0.0;
      poly2[i].x = rand()%MAX_COEFF; poly2[i].y = 0.0;
   }
   
   for (i = size; i < (2*size); i++)
   {
      poly1[i].x = 0.0; poly1[i].y = 0.0;
      poly2[i].x = 0.0; poly2[i].y = 0.0;
   }
   
   return size;
}


//-----------------------------------------------------------------------------
// NAME: print_device_info
//
// PURPOSE: Prints the specifications of a particular compute device
//
// INPUT:
//    p     Platform ID
//    d     Compute Device
//
// OUTPUT:
//    Prints device specifications to stdout
//
// RETURNS: void
//-----------------------------------------------------------------------------
static void print_device_info(cl_platform_id p, cl_device_id d)
{
   char vendor[1024];
   char device_ver[1024];
   char device_name[1024];
   cl_uint num_cores;
   cl_long global_mem;
   cl_long local_mem;
   cl_uint clk_freq;
   cl_uint item_dim;
   size_t* item_sizes;
   int i;

   clGetPlatformInfo(p, CL_PLATFORM_VENDOR, sizeof(vendor), vendor, NULL);
   printf("-------------------------------------------------\n");
   printf("Platform Vendor: %s\n", vendor);

   clGetDeviceInfo(d, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
   clGetDeviceInfo(d, CL_DEVICE_VENDOR, sizeof(vendor), vendor, NULL);
   clGetDeviceInfo(d, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(num_cores), &num_cores, NULL);
   clGetDeviceInfo(d, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem), &global_mem, NULL);
   clGetDeviceInfo(d, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clk_freq), &clk_freq, NULL);
   clGetDeviceInfo(d, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_mem), &local_mem, NULL);
   clGetDeviceInfo(d, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(group_size), &group_size, NULL);
   clGetDeviceInfo(d, CL_DEVICE_VERSION, sizeof(device_ver), device_ver, NULL);
   clGetDeviceInfo(d, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(item_dim), &item_dim, NULL);
   item_sizes = malloc(item_dim * sizeof(size_t));
   clGetDeviceInfo(d, CL_DEVICE_MAX_WORK_ITEM_SIZES, item_dim * sizeof(size_t), item_sizes, NULL);

   printf("   Name: %s\n", device_name);
   printf("   Vendor: %s\n", vendor);
   printf("   Compute Units: %u\n", num_cores);
   printf("   Global Memory: %d bytes\n", (int)global_mem);
   printf("   Max Clock Freq: %d MHz\n", (int)clk_freq);
   printf("   Local Memory: %d bytes\n", (int)local_mem);
   printf("   Work Group Size: %d\n", (int)group_size);
   for (i=0; i < item_dim; i++)
      printf("   Dim %d Work Items: %d\n", i+1, (int)item_sizes[i]);
   printf("   Device Version: %s\n", device_ver);
   printf("-------------------------------------------------\n");
}

// Swap cl_mem pointers
static void swap_mem_ptr(cl_mem* a, cl_mem* b)
{
   cl_mem temp = *a;
   *a = *b;
   *b = temp;
}
