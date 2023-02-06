import { Feature, FeatureEngineer } from '@ronas-it/tfjs-node-helpers';

export abstract class MinMaxNormalizedFeatureEngineer<T, D> extends FeatureEngineer<T, D> {
  protected normalizeFeature({
    featureValues: values,
    feature
  }: {
    featureValues: Array<number>;
    feature: Feature<T>;
  }): Feature<T> {
    let min = values[0];
    let max = values[0];
    let i = values.length;

    while (i--) {
      min = (values[i] < min) ? values[i] : min;
      max = (values[i] > max) ? values[i] : max;
    }

    const normalizedValue = (feature.value - min) / (max - min);

    return new Feature({ ...feature, value: normalizedValue });
  }
}
