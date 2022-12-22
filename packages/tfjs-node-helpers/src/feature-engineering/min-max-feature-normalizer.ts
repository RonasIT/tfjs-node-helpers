import { FeatureNormalizer } from './feature-normalizer';
import { Feature } from './feature';

export abstract class MinMaxFeatureNormalizer<T> extends FeatureNormalizer<T> {
  private min: number;
  private max: number;

  constructor({ min, max }: { min: number; max: number }) {
    super();

    this.min = min;
    this.max = max;
  }

  public normalize(feature: Feature<T>): Feature<T> | Promise<Feature<T>> {
    return new Feature({
      ...feature,
      value: (feature.value - this.min) / (this.max - this.min)
    });
  }
}
