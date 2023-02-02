import { Tensor, tensor } from '@tensorflow/tfjs-node';
import { Sample } from '../training/sample';

export const makeChunkedDatasetGenerator = async function* ({
  loadChunk,
  chunkSize
}: {
  loadChunk: (skip: number, take: number) => Promise<Array<Sample>>,
  chunkSize: number
}): AsyncGenerator<{ xs: Tensor, ys: Tensor }> {
  let skip = 0;
  const take = chunkSize;

  while (true) {
    const samples = await loadChunk(skip, take);

    for (const sample of samples) {
      yield {
        xs: tensor(sample.input),
        ys: tensor(sample.output)
      };
    }

    if (samples.length < take) {
      break;
    }

    skip += take;
  }
};
