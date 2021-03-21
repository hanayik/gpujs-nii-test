const readLocalFileSync = require('itk/readLocalFileSync')
const writeLocalFileSync = require('itk/writeLocalFileSync')
const { GPU, input } = require('gpu.js')
const fs = require('fs')
const {performance} = require('perf_hooks')

// test fslmaths with box blur 13
//for i in {1..10}; do time fslmaths brain.nii.gz -kernel boxv 13 -fmean fsl_maths_out.nii; done

// test niimath with box blur 13
//for i in {1..10}; do time ./niimath brain.nii.gz -kernel boxv 13 -fmean niimath_out.nii; done

// test this code box blur (voxels) set my m,n,p
//for i in {1..10}; do node main.js; done

// uses itkjs to read nifti files in nodejs
const object = readLocalFileSync('./brain.nii.gz')

// kernel dimensions (keep uniform for now)
// use large kernels to see more speed benefit compared to CPU
let m = 13
let n = 13
let p = 13
let kw = 1/(m*n*p) // weights, box blur
// make the kernel array
const kernel = new Array(m*n*p).fill(kw) // box blur (1D array)

// initialise a gpu object
const gpu = new GPU({mode: 'gpu'})

// create a kernel function. This gets converted to a shader
const gpuConvolve = gpu.createKernel(function (src, kernel, radius) {
  // src:     input array (will be flattened if not flat)
  // kernel:  convolution kernel to use (box blur for now)
  // radius:  assumes uniform kernel dims, needed to caclulate kSize within shader
  let sum = 0.0 //output value for this thread loop
  let kSize = 2 * radius + 1; // kSize = m = n = p from above
  // loop through kernel
  for (let i = 0; i < kSize; i++){
    for (let j = 0; j < kSize; j++) {
      for (let k = 0; k < kSize; k++) {
        // accumulate new voxel value. Multiply kernel at index i,j,k by src at index z,y,x
        // using src z,y,x versus x,y,z still blows my mind, but it works
        // https://github.com/gpujs/gpu.js#creating-and-running-functions
        sum += kernel[i][j][k] * src[this.thread.z + i][this.thread.y + j][this.thread.x + k]
      }
    }
  }
  // return new voxel value after convolution
  return sum
}).setOutput([object.data.length]).setTactic('precision').setPipeline(true) //for faster GPU kernel use pipeline true. if 'cpu' mode then remove .setPipeline()

// uncomment if you want to time the performance of running the GPU kernel.
// this is the real time the gpu takes to accomplish the task. If using time command on unix, the values will be
// inflated by many other slow javascript things.
const startTime = performance.now();
const result = gpuConvolve(input(object.data, object.size), input(kernel, [m, n, p]), (m-1)/2);
const endTime = performance.now();
console.log(endTime - startTime) //milliseconds

// initalise new output typed array. Demo image is Int16 so use that here. Change for other images
let out = new Int16Array
// fill out array
out = Int16Array.from(result.toArray()) // result.toArray() if using pipeline mode in GPU kernel. Converts texture to 1D array
// use same nifti object from input image, replace data array
object.data = out
// write output file
writeLocalFileSync(false, object, './out.nii')
