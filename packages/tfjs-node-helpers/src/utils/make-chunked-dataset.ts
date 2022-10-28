import { data, TensorContainer } from '@tensorflow/tfjs-node';
import { Sample } from '../training/sample';
import { makeChunkedDatasetGenerator } from '../utils/make-chunked-dataset-generator';

export const makeChunkedDataset = ({
  loadChunk,
  chunkSize,
  batchSize
}: {
  loadChunk: (skip: number, take: number) => Promise<Array<Sample>>,
  chunkSize: number,
  batchSize: number
}): data.Dataset<TensorContainer> => {
  return data
    .generator(
      () => makeChunkedDatasetGenerator({
        loadChunk,
        chunkSize
      }) as any
    )
    .batch(batchSize);
};
