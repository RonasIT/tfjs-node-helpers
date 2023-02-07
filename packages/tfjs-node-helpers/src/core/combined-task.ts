import { LayersModel, SymbolicTensor } from '@tensorflow/tfjs-node';
import { AbstractTask, AbstractTaskOptions } from './abstract-task';

export type CombinedTaskOptions = AbstractTaskOptions & {
  model: LayersModel;
  lastSymbolicTensor: SymbolicTensor;
};

export class CombinedTask extends AbstractTask {
  public model: LayersModel;
  public lastSymbolicTensor: SymbolicTensor;

  constructor(options: CombinedTaskOptions) {
    super(options);

    this.model = options.model;
    this.lastSymbolicTensor = options.lastSymbolicTensor;
  }

  public train(): void {
    // TODO: implement
  }

  public test(): void {
    // TODO: implement
  }

  public use(): void {
    // TODO: implement
  }
}
