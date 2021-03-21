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
let t1 = performance.now()
const object = readLocalFileSync('./brain.nii.gz')
console.log("read in data ms: ", performance.now()-t1)
const w = object.size[0]
const h = object.size[1]
const d = object.size[2]

// kernel dimensions (keep uniform for now)
// use large kernels to see more speed benefit compared to CPU
let k = 13
let m = k
let n = k
let p = k
let kw = 1/(m*n*p) // weights, box blur
// make the kernel array
t1 = performance.now()
const kernel = new Array(m*n*p).fill(kw) // box blur (1D array)

// edge kernel attempt
// set radius to 1 in gpuConvolve call
//const kernel = [-1, -1, -1, -1, 8, -1, -1, -1, -1, -1, -1, -1, -1, 8, -1, -1, -1, -1, -1, -1, -1, -1, 8, -1, -1, -1, -1]
console.log("kernel build ms: ", performance.now()-t1)

// initialise a gpu object
const gpu = new GPU({mode: 'gpu'})

// create a kernel function. This gets converted to a shader
const gpuConvolve = gpu.createKernel(function (src, w, h, d, kernel, radius) {
  // src:     input array (will be flattened if not flat)
  // w:       dim [1] from nifti header, not used
  // h:       dim [2] from nifti header, not used
  // d:       dim [3] from nifti header, not used
  // kernel:  convolution kernel to use (box blur for now)
  // radius:  assumes uniform kernel dims, needed to caclulate kSize within shader
  let sum = 0.0 //output value for this thread loop
  let kSize = 2 * radius + 1; // kSize = m = n = p from above
  let x = this.thread.z // rename for clarity
  let y = this.thread.y
  let z = this.thread.x
  // loop through kernel
  for (let i = -radius; i <= radius; i++){
    for (let j = -radius; j <= radius; j++) {
      for (let k = -radius; k <= radius; k++) {
        // accumulate new voxel value. Multiply kernel at index i,j,k by src at index z,y,x
        // using src z,y,x versus x,y,z still blows my mind, but it works
        // https://github.com/gpujs/gpu.js#creating-and-running-functions
        let xi = x + i
        let yj = y + j
        let zk = z + k
        sum += kernel[i][j][k] * src[xi][yj][zk] // might have some OOB image artifacts
      }
    }
  }
  // return new voxel value after convolution
  return sum
}).setOutput([object.data.length]).setTactic('precision').setPipeline(true) //for faster GPU kernel use pipeline true. if 'cpu' mode then remove .setPipeline()
// setPipeline(true) keeps array in gpu memory.

// uncomment if you want to time the performance of running the GPU kernel.
// this is the real time the gpu takes to accomplish the task. If using time command on unix, the values will be
// inflated by many other slow javascript things.
const startTime = performance.now();
const result = gpuConvolve(input(object.data, object.size), w, h, d, input(kernel, [m, n, p]), (k-1)/2);
const endTime = performance.now();
console.log("gpu processing ms: ", endTime - startTime) //milliseconds

// initalise new output typed array. Demo image is Int16 so use that here. Change for other images
t1 = performance.now()
let out = new Int16Array
// fill out array
// there is a significant slow down here. Takes about 4-5 seconds just to complete this step! Transfer to/from GPU mem?
out = Int16Array.from(result.toArray()) // result.toArray() if using pipeline mode in GPU kernel. Converts texture to 1D array
console.log("fill output array ms: ", performance.now() - t1)
// use same nifti object from input image, replace data array
object.data = out
// write output file
t1 = performance.now()
writeLocalFileSync(false, object, './out.nii')
console.log("write ouput ms: ", performance.now() - t1)
