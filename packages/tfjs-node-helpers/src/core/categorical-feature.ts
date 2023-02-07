import { Shape } from '@tensorflow/tfjs-node';
import { Feature } from './feature';

export class CategoricalFeature extends Feature {
  public get shape(): Shape {
    return []; // TODO: calculate shape
  }

  // TODO: implement
}
