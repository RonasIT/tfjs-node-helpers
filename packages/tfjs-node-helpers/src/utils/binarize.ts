import { onesLike, Tensor, tidy, where, zerosLike } from '@tensorflow/tfjs-node';

export const binarize = (tensor: Tensor, threshold = 0.5): Tensor => tidy(
  () => where(
    tensor.greater(threshold),
    onesLike(tensor),
    zerosLike(tensor)
  )
);
