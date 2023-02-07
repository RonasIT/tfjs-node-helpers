import { Shape } from '@tensorflow/tfjs-node';
import { Feature, FeatureOptions } from './feature';
import { ImageFeatureNormalizer } from './image-feature-normalizer';

export type ImageFeatureOptions = FeatureOptions & {
  normalizer: ImageFeatureNormalizer;
};

export class ImageFeature extends Feature {
  public get shape(): Shape {
    return this.normalizer.shape;
  }

  public override normalizer: ImageFeatureNormalizer;

  constructor(options: ImageFeatureOptions) {
    super(options);

    this.normalizer = options.normalizer;
  }
}
