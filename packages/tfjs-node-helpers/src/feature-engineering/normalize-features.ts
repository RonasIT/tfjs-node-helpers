import { FeatureNormalizer } from './feature-normalizer';
import { Sample } from '../training/sample';
import { DataItemExtract } from './extract-features';
import { TensorContainerObject } from '@tensorflow/tfjs-node';

export const normalizeFeatures = async <T>({
  extracts,
  inputFeatureNormalizers
}: {
  extracts: Array<DataItemExtract<T>>;
  inputFeatureNormalizers: Array<FeatureNormalizer<T>>;
}): Promise<Array<Sample>> => {
  const samples = [];

  for (const extractItem of extracts) {
    const inputNormalizedFeatures = await Promise.all(
      extractItem.inputFeatures.map((feature) => {
        const desiredNormalizer = inputFeatureNormalizers.find((normalizer) => normalizer.featureType === feature.type);

        return (desiredNormalizer !== undefined) ? desiredNormalizer.normalize(feature) : feature;
      })
    );

    const input: TensorContainerObject = inputNormalizedFeatures.reduce((input, feature) => {
      input[feature.id] = feature.value;

      return input;
    }, {} as TensorContainerObject);
    const output: TensorContainerObject = { [extractItem.outputFeature.id]: extractItem.outputFeature.value };

    samples.push({ input, output });
  }

  return samples;
};
