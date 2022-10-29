import { Sample } from '../training/sample';
import { FeatureExtractor } from './feature-extractor';

export const extractFeatures = async <D, T>({
  data,
  inputFeatureExtractors,
  outputFeatureExtractor
}: {
  data: Array<D>;
  inputFeatureExtractors: Array<FeatureExtractor<D, T>>;
  outputFeatureExtractor: FeatureExtractor<D, T>;
}): Promise<Array<Sample>> => {
  const samples = [];

  for (const dataItem of data) {
    const [inputFeatures, outputFeature] = await Promise.all([
      Promise.all(
        inputFeatureExtractors.map((featureExtractor) => {
          return featureExtractor.extract(dataItem);
        })
      ),
      outputFeatureExtractor.extract(dataItem)
    ]);

    const input = inputFeatures.map((feature) => feature.value);
    const output = [outputFeature.value];

    samples.push({ input, output });
  }

  return samples;
}
