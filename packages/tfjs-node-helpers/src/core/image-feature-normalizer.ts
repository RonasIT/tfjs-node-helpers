import { Shape } from '@tensorflow/tfjs-node';
import { FeatureNormalizer } from './feature-normalizer';

export type ImageSize = {
  width: number;
  height: number;
};

export type ImageFeatureNormalizerOptions = {
  imageSize: ImageSize;
};

export class ImageFeatureNormalizer extends FeatureNormalizer {
  public get shape(): Shape {
    return []; // TODO: calculate shape
  }

  public imageSize: ImageSize;

  constructor(options: ImageFeatureNormalizerOptions) {
    super();

    this.imageSize = options.imageSize;
  }
}
