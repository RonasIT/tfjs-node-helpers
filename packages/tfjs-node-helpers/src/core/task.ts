import { input, layers, LayersModel, model as tfModel, SymbolicTensor } from '@tensorflow/tfjs-node';
import { AbstractTask, AbstractTaskOptions } from './abstract-task';

export type TaskOptions = AbstractTaskOptions & {
  hiddenLayers: Array<layers.Layer>;
};

export class Task extends AbstractTask {
  public model: LayersModel;
  public lastSymbolicTensor: SymbolicTensor;

  private hiddenLayers: Array<layers.Layer>;

  constructor(options: TaskOptions) {
    super(options);

    this.hiddenLayers = options.hiddenLayers;

    const { model, lastSymbolicTensor } = this.createModelFromHiddenLayers();

    this.model = model;
    this.lastSymbolicTensor = lastSymbolicTensor;
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

  private createModelFromHiddenLayers(): { model: LayersModel, lastSymbolicTensor: SymbolicTensor } {
    const { model, lastSymbolicTensor } = this.createModel();

    this.compileModel(model);

    return { model, lastSymbolicTensor };
  }

  private createModel(): { model: LayersModel, lastSymbolicTensor: SymbolicTensor } {
    const { inputs, lastSymbolicTensor } = this.createInputsAndApplyHiddenLayers();
    const outputs = this.createOutputsAndApplyTo(lastSymbolicTensor);
    const model = tfModel({ inputs, outputs });;

    return {
      model,
      lastSymbolicTensor
    };
  }

  private createInputsAndApplyHiddenLayers(): { inputs: Array<SymbolicTensor>, lastSymbolicTensor: SymbolicTensor } {
    const inputs = this.createInputs();
    const inputSymbolicTensor = layers.concatenate().apply(inputs) as SymbolicTensor; // TODO: check if it's possible to remove this
    const lastSymbolicTensor = this.applyHiddenLayersSequentially(inputSymbolicTensor);

    return { inputs, lastSymbolicTensor };
  }

  private createInputs(): Array<SymbolicTensor> {
    return this.inputFeatures.map((feature) => input({
      shape: feature.shape
    }));
  }

  private createOutputsAndApplyTo(applyTo: SymbolicTensor): Array<SymbolicTensor> {
    return this.outputFeatures.map((feature) => layers.dense({
      units: 1,             // TODO: adjust this based on the feature
      activation: 'sigmoid' // TODO: adjust this based on the feature
    }).apply(applyTo) as SymbolicTensor);
  }

  private applyHiddenLayersSequentially(applyTo: SymbolicTensor): SymbolicTensor {
    let symbolicTensor = applyTo;

    for (const layer of this.hiddenLayers) {
      symbolicTensor = layer.apply(symbolicTensor) as SymbolicTensor;
    }

    return symbolicTensor;
  }

  private compileModel(model: LayersModel): void {
    model.compile({
      optimizer: this.optimizer ?? 'adam', // TODO: adjust this based on the task and features
      loss: 'binaryCrossentropy'           // TODO: adjust this based on the task and features
    });
  }
}
