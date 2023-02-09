export const switchHardwareUsage = (shouldUseGPU?: boolean): void => {
  if (shouldUseGPU) {
    console.log('Attempting to use GPU.');
    require('@tensorflow/tfjs-node-gpu');
  } else {
    console.log('Using CPU.');
    require('@tensorflow/tfjs-node');
  }
};
