import { Shape } from '@tensorflow/tfjs-node';
import { FeatureExtractor } from './feature-extractor';
import { FeatureNormalizer } from './feature-normalizer';

export type FeatureOptions = {
  extractor: FeatureExtractor;
  normalizer?: FeatureNormalizer;
};

export abstract class Feature {
  public abstract get shape(): Shape;

  public extractor: FeatureExtractor;
  public normalizer?: FeatureNormalizer;

  constructor(options: FeatureOptions) {
    this.extractor = options.extractor;
    this.normalizer = options.normalizer;
  }
}
