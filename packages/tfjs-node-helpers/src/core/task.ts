import { Feature } from './feature';

export type TaskOptions = {
  inputFeatures: Array<Feature>;
  outputFeatures: Array<Feature>;
};

export class Task {
  private inputFeatures: Array<Feature>;
  private outputFeatures: Array<Feature>;

  constructor(options: TaskOptions) {
    this.inputFeatures = options.inputFeatures;
    this.outputFeatures = options.outputFeatures;
  }

  public train(): void {

  }

  public test(): void {

  }

  public evaluate(): void {

  }
}
