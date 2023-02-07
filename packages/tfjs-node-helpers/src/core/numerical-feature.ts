import { Shape } from '@tensorflow/tfjs-node';
import { Feature } from './feature';

export class NumericalFeature extends Feature {
  public get shape(): Shape {
    return []; // TODO: calculate shape
  }
}
