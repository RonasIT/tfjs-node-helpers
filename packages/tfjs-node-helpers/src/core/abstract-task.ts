import { LayersModel, Optimizer, SymbolicTensor } from '@tensorflow/tfjs-node';
import { Feature } from './feature';

export type AbstractTaskOptions = {
  inputFeatures: Array<Feature>;
  outputFeatures: Array<Feature>;

  optimizer?: string | Optimizer;
};

export abstract class AbstractTask {
  public inputFeatures: Array<Feature>;
  public outputFeatures: Array<Feature>;

  public optimizer?: string | Optimizer;

  public abstract model: LayersModel;
  public abstract lastSymbolicTensor: SymbolicTensor;

  constructor(options: AbstractTaskOptions) {
    this.inputFeatures = options.inputFeatures;
    this.outputFeatures = options.outputFeatures;

    this.optimizer = options.optimizer;
  }

  public abstract train(): void;
  public abstract test(): void;
  public abstract use(): void;
}
