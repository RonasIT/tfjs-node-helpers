import { FeatureExtractor } from './feature-extractor';
import { Feature } from './feature';

export type DataItemExtract<T> = {
  inputFeatures: Array<Feature<T>>;
  outputFeature: Feature<T>;
};

export const extractFeatures = async <D, T>({
  data,
  inputFeatureExtractors,
  outputFeatureExtractor
}: {
  data: Array<D>;
  inputFeatureExtractors: Array<FeatureExtractor<D, T>>;
  outputFeatureExtractor: FeatureExtractor<D, T>;
}): Promise<Array<DataItemExtract<T>>> => {
  const extracts = [];

  for (const dataItem of data) {
    const [inputFeatures, outputFeature] = await Promise.all([
      Promise.all(
        inputFeatureExtractors.map((featureExtractor) => {
          return featureExtractor.extract(dataItem);
        })
      ),
      outputFeatureExtractor.extract(dataItem)
    ]);

    extracts.push({ inputFeatures, outputFeature });
  }

  return extracts;
};
