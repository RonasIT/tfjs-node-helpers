import { TensorContainer } from '@tensorflow/tfjs-node';
import { Sample } from '../training/sample';

export const makeChunkedDatasetGenerator = async function* ({
  loadChunk,
  chunkSize
}: {
  loadChunk: (skip: number, take: number) => Promise<Array<Sample>>,
  chunkSize: number
}): AsyncGenerator<{ xs: TensorContainer, ys: TensorContainer }> {
  let skip = 0;
  const take = chunkSize;

  while (true) {
    const samples = await loadChunk(skip, take);

    for (const sample of samples) {
      yield {
        xs: sample.input,
        ys: sample.output
      };
    }

    if (samples.length < take) {
      break;
    }

    skip += take;
  }
};
