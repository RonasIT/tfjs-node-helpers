import { data, TensorContainer } from '@tensorflow/tfjs-node';
import { Sample } from '../training/sample';

export const makeDataset = (samples: Array<Sample>, batchSize: number = samples.length): data.Dataset<TensorContainer> => {
  const xs = data.array(samples.map((sample) => sample.input));
  const ys = data.array(samples.map((sample) => sample.output));

  return data.zip({ xs, ys }).batch(batchSize);
}
