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

// Global coefficient arrays
cl_float2* poly1;
cl_float2* poly2;


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
//             d. lg(n) inverse-FFT stages
//       4. Deploy kernel instances to GPU
//       5. Verify results
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
   int fft_size = get_input_polynomials();
   
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
   
   cl_mem in_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, 
         fft_size * sizeof(cl_float2), NULL, &ret);
   cl_mem out_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
         fft_size * sizeof(cl_float2), NULL, &ret);
   cl_mem initial_input = in_mem_obj;

   ////////////////////////////////////
   //
   // Create kernel instances
   //
   ////////////////////////////////////
   
   //---------------------------
   // Bit-Reverse Permutation
   //---------------------------
   cl_kernel bitrev_kernel = clCreateKernel(program, "bitrev_permute", &ret);

   // Set the arguments of the kernel
   ret = clSetKernelArg(bitrev_kernel, 0, sizeof(cl_mem), (void *)&in_mem_obj);
   ret = clSetKernelArg(bitrev_kernel, 1, sizeof(cl_mem), (void *)&out_mem_obj);
   ret = clSetKernelArg(bitrev_kernel, 2, sizeof(unsigned int), (void*)&lg_n);
         
   // Swap device memory pointers (output of this will be input to parallel-fft)
   cl_mem temp_mem = in_mem_obj;
   in_mem_obj = out_mem_obj;
   out_mem_obj = temp_mem;
         
   //---------------------------
   // lg(n) FFT stages
   //---------------------------
   unsigned int n;
   
   cl_kernel* fft_kernel = (cl_kernel*)malloc(lg_n * sizeof(cl_kernel));
   for (i=0; i<lg_n; i++)
   {
      fft_kernel[i] = clCreateKernel(program, "parallel_fft", &ret);
      
      // Set the arguments of the kernel
      n = (1 << (i+1)); // double n for each stage of FFT (2,4,8,16...)
      ret = clSetKernelArg(fft_kernel[i], 0, sizeof(cl_mem), (void *)&in_mem_obj);
      ret = clSetKernelArg(fft_kernel[i], 1, sizeof(cl_mem), (void *)&out_mem_obj);
      ret = clSetKernelArg(fft_kernel[i], 2, sizeof(unsigned int), (void*)&n);
            
      // If not last stage, swap device memory pointers 
      // (output of this stage will be input of next).
      if (i != (lg_n-1))
      {
         temp_mem = in_mem_obj;
         in_mem_obj = out_mem_obj;
         out_mem_obj = temp_mem;
      }
   }
   
   //---------------------------
   // Point-wise multiplication
   //---------------------------
   
   //---------------------------
   // lg(n) inverse-FFT stages
   //---------------------------
   
   cl_mem final_output = out_mem_obj;
   
   ////////////////////////////////////
   //
   // Deploy kernel instances to GPU
   //
   ////////////////////////////////////
   
   clock_t start = clock();
   
   size_t global_item_size = fft_size;
   size_t local_item_size = 1;
   
   // Transfer host memory to device
   ret = clEnqueueWriteBuffer(command_queue, initial_input, CL_TRUE, 0,
         fft_size * sizeof(cl_float2), poly1, 0, NULL, NULL);
         
   // Bit-Reverse Perumtation
   ret = clEnqueueNDRangeKernel(command_queue, bitrev_kernel, 1, NULL, 
         &global_item_size, &local_item_size, 0, NULL, NULL);
         
   // FFT stages
   for (i=0; i<lg_n; i++)
      ret = clEnqueueNDRangeKernel(command_queue, fft_kernel[i], 1, NULL, 
            &global_item_size, &local_item_size, 0, NULL, NULL);
            
   // Transfer device memory to host
   ret = clEnqueueReadBuffer(command_queue, final_output, CL_TRUE, 0, 
         fft_size * sizeof(cl_float2), poly2, 0, NULL, NULL);
         
   printf("Time elapsed: %.9f sec\n", ((double)clock() - start) / CLOCKS_PER_SEC);
   
   ////////////////////////////////////
   //
   // Verify results
   //
   ////////////////////////////////////
         
   // Print results
   for (i=0; i<fft_size; i++)
      printf("output[%d] = %f %c %fi\n", 
         i, 
         poly2[i].x, 
         poly2[i].y < 0 ? '-' : '+',
         poly2[i].y < 0 ? -poly2[i].y : poly2[i].y);
 
   // Clean up
   ret = clFlush(command_queue);
   ret = clFinish(command_queue);
   ret = clReleaseKernel(bitrev_kernel);
   ret = clReleaseProgram(program);
   ret = clReleaseMemObject(in_mem_obj);
   ret = clReleaseMemObject(out_mem_obj);
   ret = clReleaseCommandQueue(command_queue);
   ret = clReleaseContext(context);
   free(poly1);
   free(poly2);
   free(fft_kernel);
   
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
   printf("Enter %d polynomial coefficients (x^0 coeff first): ", n);
   for (i = 0; i < n; i++)
   {
      ret_val = scanf("%d",&coeff);
      poly1[i].x = (float)coeff;
      poly1[i].y = 0.0;
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
   size_t group_size;
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
