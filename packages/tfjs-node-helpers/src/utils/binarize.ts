import { onesLike, Rank, Tensor, tidy, where, zerosLike } from '@tensorflow/tfjs-node';

export const binarize = <R extends Rank = Rank>(tensor: Tensor<R>, threshold = 0.5): Tensor<R> => tidy(
  () => where(
    tensor.greater(threshold),
    onesLike(tensor),
    zerosLike(tensor)
  )
);
